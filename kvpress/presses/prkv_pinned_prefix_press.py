"""
Decoding-friendly PRKV variant with prefix pinning.

This press is intended to be used as the `base_press` inside `DecodingPress`.

Mechanism:
- Compute normal PRKV scores (delegated to `PRKVPress.score`).
- If we're in decoding, pick landmark indices from the current prefix 
  and force their scores to +inf (per KV head), so they are always kept by top-k pruning.
"""

from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.prkv_press import PRKVPress, align_rope_to_hidden_states

@dataclass
class PRKVPressPinnedPrefix(PRKVPress):
    """
    PRKV with prefix landmark pinning.

    Parameters
    ----------
    prefix_keep_frac : float
        Fraction of *kept tokens* to reserve for prefix landmarks.
        In decoding with `DecodingPress(target_size=...)`, this is effectively a fraction of target_size.
    """

    prefix_keep_frac: float = 0.05

    def __post_init__(self):
        super().__post_init__()
        assert 0.0 <= float(self.prefix_keep_frac) <= 1.0

    def _select_prefix_landmarks(self, keys: torch.Tensor) -> torch.Tensor:
        """
        Select one landmark per chunk from the current prefix (excluding sink and window tail).

        Returns
        -------
        torch.Tensor
            landmark positions [C] on device, dtype long.
        """
        seq_len = int(keys.shape[2])
        W = min(int(self.window_size), seq_len)
        prefix_end = max(0, seq_len - W)
        start0, endT, _C = self._chunk_layout(endT=int(prefix_end))
        landmark_pos = self._pick_landmarks(keys=keys, start0=start0, endT=endT)
        return landmark_pos

    def _score_nodes_attention_decoding_window_only(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        """
        Decoding-only scoring for `transition_mode="nodes_attention"` that uses *only window nodes*
        for the node graph (so it can run with buffered decoding `hidden_states`), and relies on
        prefix landmark *pinning* to keep prefix representatives alive.

        Returns scores: [bsz, Hkv, T]
        """
        assert keys.size(2) > int(self.n_sink), f"Input should contain more tokens than n_sink={self.n_sink}"

        bsz, Hkv, seq_len, _ = keys.shape
        device = keys.device

        # Use at most what we actually have in the decoding hidden-state buffer.
        W = min(int(self.window_size), int(seq_len), int(hidden_states.shape[1]))
        if W <= 0:
            # Shouldn't happen in practice; fall back to base behavior.
            return super().score(module, hidden_states, keys, values, attentions, kwargs)

        # Tail-only tensors; we score ONLY within this tail window.
        hidden_tail = hidden_states[:, -W:, :]  # [bsz,W,hidden]
        keys_tail = keys[:, :, -W:, :]  # [bsz,Hkv,W,d]

        # Slice RoPE for the tail window. score() has already aligned position_embeddings to hidden_states.
        kwargs_local = kwargs
        if "position_embeddings" in kwargs and kwargs["position_embeddings"] is not None:
            cos, sin = kwargs["position_embeddings"]
            kwargs_local = dict(kwargs)
            kwargs_local["position_embeddings"] = (cos[:, -W:], sin[:, -W:])

        with torch.no_grad():
            window_attn, W2 = self._compute_window_attention(
                keys=keys_tail,
                module=module,
                hidden_states=hidden_tail,
                seq_len=int(W),
                kwargs=kwargs_local,
            )  # window_attn: [Hkv,W,W]

            A = window_attn  # [Hkv,W,W]
            v1 = self._recent_vector_1d(int(W2), device=keys.device)  # [W]
            v_nodes = v1.unsqueeze(0).expand(int(Hkv), -1).to(dtype=keys.dtype)  # [Hkv,W]

            node_scores = self._pagerank_on_transition(A, v_nodes)  # [Hkv,W]

            score_2d = torch.zeros((int(Hkv), int(seq_len)), device=device, dtype=node_scores.dtype)
            score_2d[:, int(seq_len) - int(W) : int(seq_len)] = node_scores

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

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        kwargs = align_rope_to_hidden_states(module, hidden_states, kwargs)

        scores = self._score_nodes_attention_decoding_window_only(
            module=module,
            hidden_states=hidden_states,
            keys=keys,
            values=values,
            attentions=attentions,
            kwargs=kwargs,
        )
        if float(self.prefix_keep_frac) <= 0.0:
            return scores

        k_len = int(keys.shape[2])
        n_kept = int(k_len * (1.0 - float(self.compression_ratio)))
        if n_kept <= 0:
            return scores

        # Budget for pinned prefix landmarks
        k_pin = int(round(float(self.prefix_keep_frac) * float(n_kept)))
        if k_pin <= 0:
            return scores

        landmark_pos = self._select_prefix_landmarks(keys=keys)  # [C]
        if landmark_pos.numel() == 0:
            return scores
        
        if landmark_pos.dtype != torch.long:
            landmark_pos = landmark_pos.to(dtype=torch.long)
        valid = (landmark_pos >= 0) & (landmark_pos < int(k_len))
        if not bool(valid.all().item()):
            # Keep only valid indices; if none remain, just skip pinning this round.
            landmark_pos = landmark_pos[valid]
            if landmark_pos.numel() == 0:
                return scores

        # Cap pinned count (can't pin more than kept tokens minus sinks).
        max_pin = max(0, n_kept - int(self.n_sink))
        if max_pin <= 0:
            return scores
        if k_pin > max_pin:
            k_pin = max_pin

        # If too many chunks, subsample landmarks uniformly for coverage.
        if int(landmark_pos.numel()) > int(k_pin):
            step = max(1, int(landmark_pos.numel()) // int(k_pin))
            landmark_pos = landmark_pos[::step][: int(k_pin)]

        # Force these positions to be kept by setting them to dtype max per head.
        sink_score = (
            torch.finfo(scores.dtype).max
            if scores.dtype.is_floating_point
            else torch.iinfo(scores.dtype).max
        )
        scores[:, :, landmark_pos] = sink_score
        return scores


