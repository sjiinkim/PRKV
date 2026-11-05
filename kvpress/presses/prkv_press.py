"""
Unified PRKV cache scoring.

1) Transition:
   - "nodes_attention": For PRKV. PPR over selected nodes (landmark per chunk + last window tokens) where the node-to-node
     transition is attention between those nodes (N x N).
   - "global_coattn": For PRKV-GLB. Transition over all keys (T) using low-rank co-attention transition induced by
     window attention A (W queries x T keys). No explicit T x T matrix is built.
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv

from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.utils import get_query_states

logger = logging.getLogger(__name__)

TransitionMode = Literal["global_coattn", "nodes_attention"]


def align_rope_to_hidden_states(module: nn.Module, hidden_states: torch.Tensor, kwargs: dict) -> dict:
    """
    Make kwargs["position_embeddings"] cover every row of `hidden_states`.

    Under DecodingPress, `hidden_states` is a buffer of the last L decoding tokens, while `kwargs` comes from the
    current step and covers only the current token. RoPE is recomputed at the buffered tokens' own positions, which
    are consecutive and end at the current absolute position in `position_ids`. `cache_position` cannot be used
    because it is relative to the compressed cache.
    """
    pe = kwargs.get("position_embeddings")
    if pe is None:
        return kwargs
    L = int(hidden_states.shape[1])
    if int(pe[0].shape[1]) >= L:
        return kwargs  # prefill: already aligned

    position_ids = kwargs.get("position_ids")
    if position_ids is None or getattr(module, "rotary_emb", None) is None:
        raise ValueError(
            "Buffered hidden_states are longer than position_embeddings, and RoPE for the buffered positions "
            "cannot be recomputed without kwargs['position_ids'] and module.rotary_emb."
        )
    last = position_ids[:, -1:].to(hidden_states.device)  # [bsz, 1]
    buffer_pos = last - (L - 1) + torch.arange(L, device=hidden_states.device)  # [bsz, L]
    cos, sin = module.rotary_emb(hidden_states, buffer_pos)
    kwargs = dict(kwargs)
    # With device_map="auto", rotary_emb can sit on another GPU and returns its output there.
    kwargs["position_embeddings"] = (cos.to(hidden_states.device), sin.to(hidden_states.device))
    return kwargs


@dataclass
class PRKVPress(ScorerPress):
    # Basic compression parameters
    compression_ratio: float = 0.0
    n_sink: int = 4

    # Window parameters (used by global_coattn and also to define the "recent tail" for nodes_attention)
    window_size: int = 64

    # Node/landmark parameters (nodes_attention)
    chunk_size: int = 64

    # PageRank parameters
    alpha: float = 0.85
    n_hops: int = 3

    # Transition / personalization modes
    transition_mode: TransitionMode = "nodes_attention"

    # Recency personalization params (recent)
    recency_frac: float = 0.25
    recency_boost: float = 2.0
    
    # Technical parameters
    epsilon: float = 1e-2

    def __post_init__(self):
        super().__post_init__()
        assert self.window_size > 0
        assert self.chunk_size > 0
        assert 0.0 < float(self.alpha) < 1.0
        assert int(self.n_hops) >= 0
        assert 0.0 <= float(self.recency_frac) <= 1.0
        assert float(self.recency_boost) >= 1.0

    def _apply_rope_to_queries(
        self,
        q_states: torch.Tensor,
        kwargs: dict,
        cos_sin_slice: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Apply RoPE to query states if position embeddings are provided."""
        if "position_embeddings" not in kwargs or kwargs["position_embeddings"] is None:
            return q_states

        from transformers.models.llama.modeling_llama import rotate_half

        if cos_sin_slice is not None:
            cos_l, sin_l = cos_sin_slice
        else:
            cos, sin = kwargs["position_embeddings"]
            # Default to last L positions (window-style)
            L = int(q_states.shape[2])
            cos_l = cos[:, -L:]
            sin_l = sin[:, -L:]

        return (q_states * cos_l.unsqueeze(1)) + (rotate_half(q_states) * sin_l.unsqueeze(1))

    def _compute_window_attention(
        self,
        keys: torch.Tensor,
        module: nn.Module,
        hidden_states: torch.Tensor,
        seq_len: int,
        kwargs: dict,
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute window attention for the last W queries attending to ALL keys.
        Returns:
          window_attn: [Hkv, W, T]
          W: int
        """
        bsz, num_kv_heads, _T, head_dim = keys.shape
        num_heads = int(module.config.num_attention_heads)
        num_kv_groups = num_heads // int(num_kv_heads)

        W = min(int(self.window_size), int(seq_len))
        window_hidden = hidden_states[:, -W:, :]

        q_states = get_query_states(module, window_hidden)  # [bsz, H, W, d]
        keys_expanded = repeat_kv(keys, num_kv_groups)  # [bsz, H, T, d]

        # Apply RoPE to queries if available (keys assumed already RoPE'd)
        if "position_embeddings" in kwargs and kwargs["position_embeddings"] is not None:
            cos, sin = kwargs["position_embeddings"]
            q_states = self._apply_rope_to_queries(q_states, kwargs, cos_sin_slice=(cos[:, -W:], sin[:, -W:]))

        attn_logits = torch.matmul(q_states, keys_expanded.transpose(2, 3)) / math.sqrt(float(head_dim))  # [bsz,H,W,T]

        # Causal mask for window queries against all keys
        attention_mask = torch.ones_like(attn_logits) * float("-inf")
        attention_mask = torch.triu(attention_mask, diagonal=int(seq_len) - int(W) + 1)
        attn_logits = attn_logits + attention_mask

        attn = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(dtype=keys.dtype)

        # Reduce to KV heads: average over kv-groups, then over batch
        attn = attn.view(bsz, num_kv_heads, num_kv_groups, W, seq_len).mean(dim=2).mean(dim=0)  # [Hkv,W,T]
        return attn, W

    def _recent_vector_1d(self, L: int, device: torch.device) -> torch.Tensor:
        v = torch.ones(int(L), device=device, dtype=torch.float32)
        recent = max(1, int(round(int(L) * float(self.recency_frac))))
        v[-recent:] *= float(self.recency_boost)
        v = v / (v.sum() + float(self.epsilon))
        return v
    
    # -----------------------------
    # PRKV-GLB
    # -----------------------------
    def _personalization_global(
        self,
        keys: torch.Tensor,
        W: int,
        seq_len: int,
    ) -> torch.Tensor:
        """
        v over all key positions: [Hkv, T]
        """
        _bsz, Hkv, _T, _head_dim = keys.shape
        device = keys.device
        T = int(seq_len)
        w = self._recent_vector_1d(int(W), device=device)  # [W], sum=1
        if int(W) < T:
            w = F.pad(w, (T - int(W), 0), value=0.0)  # [T]
        v = w.unsqueeze(0).expand(int(Hkv), -1)  # [Hkv, T]
        return v.to(dtype=keys.dtype)


    def _pagerank_global_from_window_attn(
        self,
        window_attn: torch.Tensor,  # [Hkv, W, T]
        v: torch.Tensor,  # [Hkv, T]
        seq_len: int,
        W: int,
    ) -> torch.Tensor:
        device = window_attn.device
        dtype = window_attn.dtype
        Hkv = int(window_attn.shape[0])
        T = int(seq_len)
        eps = float(self.epsilon)
        alpha = float(self.alpha)
        A = window_attn[:, :W, :T].to(torch.float32)  # [Hkv,W,T]
        c = A.sum(dim=1).clamp_min(eps)  # [Hkv,T]

        v = v[:, :T].to(device=device, dtype=torch.float32)
        v = v / (v.sum(dim=-1, keepdim=True) + eps)

        r = v.clone()
        for _ in range(int(self.n_hops)):
            wgt = r / c  # [Hkv,T]
            tmp = torch.matmul(wgt.unsqueeze(1), A.transpose(1, 2)).squeeze(1)  # [Hkv,W]
            rP = torch.matmul(tmp.unsqueeze(1), A).squeeze(1)  # [Hkv,T]
            r = alpha * rP + (1.0 - alpha) * v
            r = r / (r.sum(dim=-1, keepdim=True) + eps)

        return r.to(dtype=dtype)

    # -----------------------------
    # Nodes transition (landmarks + window nodes)
    # -----------------------------
    def _chunk_layout(self, endT: int) -> tuple[int, int, int]:
        endT = int(max(0, endT))
        start0 = int(min(max(0, int(self.n_sink)), endT))
        if start0 >= endT:
            return start0, endT, 0
        cs = int(self.chunk_size)
        L = int(endT - start0)
        C = (L + cs - 1) // cs
        return start0, endT, int(C)

    def _pick_landmarks(self, keys: torch.Tensor, start0: int, endT: int) -> torch.Tensor:
        """
        Pick 1 landmark per chunk from the prefix (fixed: cosine_dissimilarity).

        Returns:
          landmark_pos: [C]
        """
        device = keys.device
        start0 = int(start0)
        endT = int(endT)
        if endT <= start0:
            return torch.empty((0,), device=device, dtype=torch.long)

        keys_avg = keys.to(dtype=torch.float32).mean(dim=(0, 1))  # [T, d]
        prefix = keys_avg[start0:endT]  # [L, d]
        L = int(prefix.shape[0])
        if L <= 0:
            return torch.empty((0,), device=device, dtype=torch.long)

        cs = int(self.chunk_size)
        C = (L + cs - 1) // cs
        if C <= 0:
            return torch.empty((0,), device=device, dtype=torch.long)

        pad_len = int(C * cs - L)
        prefix_n = F.normalize(prefix, p=2, dim=-1)  # [L, d]
        anchor = F.normalize(prefix_n.mean(dim=0), p=2, dim=-1)  # [d]
        sims_vec = torch.matmul(prefix_n, anchor)  # [L]

        # Chunk-wise argmin cosine similarity, pad last chunk with +inf.
        if pad_len > 0:
            sims_vec = F.pad(sims_vec, (0, pad_len), value=float("inf"))
        sims_chunks = sims_vec.view(int(C), int(cs))  # [C, cs]
        idx_in_chunk = sims_chunks.argmin(dim=1)

        landmark_pos = start0 + torch.arange(int(C), device=device, dtype=torch.long) * int(cs) + idx_in_chunk.to(torch.long)
        return landmark_pos

    def _nodes_attention_matrix(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        node_pos: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        """
        Node-to-node attention transition, reduced to KV heads.
        Returns A: [Hkv, N, N]
        """
        bsz, Hkv, _T, head_dim = keys.shape
        N = int(node_pos.numel())
        device = keys.device
        if N == 0:
            return torch.empty((int(Hkv), 0, 0), device=device, dtype=keys.dtype)

        hidden_nodes = hidden_states.index_select(dim=1, index=node_pos)  # [bsz,N,hidden]
        q_states = get_query_states(module, hidden_nodes)  # [bsz,H,N,d]

        keys_nodes = keys.index_select(dim=2, index=node_pos)  # [bsz,Hkv,N,d]
        num_heads = int(module.config.num_attention_heads)
        groups = num_heads // int(Hkv)
        keys_expanded = repeat_kv(keys_nodes, groups)  # [bsz,H,N,d]

        if "position_embeddings" in kwargs and kwargs["position_embeddings"] is not None:
            cos, sin = kwargs["position_embeddings"]  # [bsz,T,d]
            cos_n = cos.index_select(dim=1, index=node_pos)
            sin_n = sin.index_select(dim=1, index=node_pos)
            q_states = self._apply_rope_to_queries(q_states, kwargs, cos_sin_slice=(cos_n, sin_n))

        attn_logits = torch.matmul(q_states, keys_expanded.transpose(2, 3)) / math.sqrt(float(head_dim))  # [bsz,H,N,N]

        if N > 1:
            pos_i = node_pos.view(N, 1)
            pos_j = node_pos.view(1, N)
            causal = (pos_j <= pos_i).to(attn_logits.device)
            causal_mask = torch.where(causal,torch.zeros((N, N), device=attn_logits.device),torch.full((N, N), float("-inf"), device=attn_logits.device),)
            attn_logits = attn_logits + causal_mask.view(1, 1, N, N)

        attn = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(dtype=keys.dtype)  # [bsz,H,N,N]
        attn = attn.view(bsz, int(Hkv), int(groups), N, N).mean(dim=2).mean(dim=0)  # [Hkv,N,N]
        return attn

    def _personalization_nodes(
        self,
        keys: torch.Tensor,
        node_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        v over nodes: [Hkv, N]
        """
        _bsz, Hkv, _T, _head_dim = keys.shape
        N = int(node_pos.numel())
        if N == 0:
            return torch.empty((int(Hkv), 0), device=keys.device, dtype=keys.dtype)

        v1 = self._recent_vector_1d(N, device=keys.device)  # [N]
        return v1.unsqueeze(0).expand(int(Hkv), -1).to(dtype=keys.dtype)

    def _pagerank_on_transition(self, A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        PageRank over nodes with attention transition A.
        A: [Hkv,N,N], v: [Hkv,N]
        returns r: [Hkv,N]
        """
        Hkv, N, _ = A.shape
        if N == 0:
            return torch.empty((int(Hkv), 0), device=A.device, dtype=A.dtype)

        eps = float(self.epsilon)
        alpha = float(self.alpha)
        P = A.float()
        P = P / (P.sum(dim=-1, keepdim=True) + eps)

        v = v.to(device=A.device, dtype=torch.float32)
        v = v / (v.sum(dim=-1, keepdim=True) + eps)
        r = v.clone()

        for _ in range(int(self.n_hops)):
            r = alpha * torch.matmul(r.unsqueeze(-2), P).squeeze(-2) + (1.0 - alpha) * v
            r = r / (r.sum(dim=-1, keepdim=True) + eps)

        return r.to(dtype=A.dtype)

    # -----------------------------
    # Main scoring
    # -----------------------------
    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        assert keys.size(2) > int(self.n_sink), f"Input should contain more tokens than n_sink={self.n_sink}"
        kwargs = align_rope_to_hidden_states(module, hidden_states, kwargs)

        bsz, Hkv, seq_len, _ = keys.shape
        device = keys.device

        with torch.no_grad():
            if self.transition_mode == "global_coattn":
                window_attn, W = self._compute_window_attention(
                    keys=keys,
                    module=module,
                    hidden_states=hidden_states,
                    seq_len=int(seq_len),
                    kwargs=kwargs,
                )  # [Hkv,W,T], W

                v = self._personalization_global(keys=keys, W=W, seq_len=int(seq_len)).to(dtype=torch.float32)  # [Hkv,T]
                pr = self._pagerank_global_from_window_attn(window_attn=window_attn, v=v, seq_len=int(seq_len), W=W)  # [Hkv,T]
                pr = pr.to(dtype=keys.dtype)
                score_2d = pr  # [Hkv,T]

            elif self.transition_mode == "nodes_attention":
                W = min(int(self.window_size), int(seq_len))
                prefix_end = max(0, int(seq_len) - int(W))
                start0, endT, _C = self._chunk_layout(endT=int(prefix_end))

                landmark_pos = self._pick_landmarks(keys=keys, start0=start0, endT=endT)
                window_pos = torch.arange(int(seq_len) - int(W), int(seq_len), device=device, dtype=torch.long)
                node_pos = torch.cat([landmark_pos, window_pos], dim=0)

                A = self._nodes_attention_matrix(
                    module=module,
                    hidden_states=hidden_states,
                    keys=keys,
                    node_pos=node_pos,
                    kwargs=kwargs,
                )  # [Hkv,N,N]

                v_nodes = self._personalization_nodes(keys=keys, node_pos=node_pos)  # [Hkv,N]

                node_scores = self._pagerank_on_transition(A, v_nodes)  # [Hkv,N]

                score_2d = torch.zeros((int(Hkv), int(seq_len)), device=device, dtype=node_scores.dtype)

                # Scatter landmark scores (sparse in prefix)
                C = int(landmark_pos.numel())
                if C > 0:
                    # index_copy_ avoids expanding indices to [Hkv, C] and is typically faster than scatter_ here.
                    score_2d.index_copy_(dim=1, index=landmark_pos.to(dtype=torch.long), source=node_scores[:, :C])

                # Fill window tail densely with node scores
                score_2d[:, int(seq_len) - int(W) : int(seq_len)] = node_scores[:, C : C + int(W)]

            else:
                raise ValueError(f"Unknown transition_mode: {self.transition_mode}")

        v_norms = values.norm(dim=-1)  # [bsz,Hkv,T]
        if bsz == 1:
            score_2d = (score_2d + float(self.epsilon)) * v_norms.squeeze(0)
        else:
            score_2d = (score_2d + float(self.epsilon)) * v_norms.mean(dim=0)

        scores = score_2d.unsqueeze(0).expand(int(bsz), -1, -1)

        # Protect sink tokens
        sink_score = torch.finfo(scores.dtype).max if scores.dtype.is_floating_point else torch.iinfo(scores.dtype).max
        scores[:, :, : int(self.n_sink)] = sink_score
        return scores



