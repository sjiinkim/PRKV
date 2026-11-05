# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from benchmarks.aime25.calculate_metrics import calculate_metrics as aime25_scorer
from benchmarks.longbench.calculate_metrics import calculate_metrics as longbench_scorer
from benchmarks.math500.calculate_metrics import calculate_metrics as math500_scorer
from benchmarks.needle_in_haystack.calculate_metrics import calculate_metrics as needle_in_haystack_scorer
from benchmarks.ruler.calculate_metrics import calculate_metrics as ruler_scorer
from kvpress.presses import (
    PRKVPress,
    PRKVPressPinnedPrefix,
)
from kvpress.presses.decoding_press import DecodingPress

# These dictionaries define the available datasets, scorers, and KVPress methods for evaluation.
DATASET_REGISTRY = {
    "ruler": "simonjegou/ruler",
    "longbench": "Xnhyacinth/LongBench",
    "needle_in_haystack": "alessiodevoto/paul_graham_essays",
    # Datasets used to be used for decoding compression
    "aime25": "alessiodevoto/aime25",
    "math500": "alessiodevoto/math500",
}

# Dataset revisions used in the paper. Later upstream revisions append a \boxed{} instruction to every question.
DATASET_REVISION = {
    "aime25": "6c6853611089d679bf20453b40fca854977eb03e",
    "math500": "5c06bb9ee02a115fff716c0a1be1a18b1005e63e",
}

SCORER_REGISTRY = {
    "ruler": ruler_scorer,
    "longbench": longbench_scorer,
    "needle_in_haystack": needle_in_haystack_scorer,
    "aime25": aime25_scorer,
    "math500": math500_scorer,
}


PRESS_REGISTRY = {
    "no_press": None,
    
    ## Prefill
    "PRKV": PRKVPress(transition_mode="nodes_attention"),
    "PRKV_GLB": PRKVPress(transition_mode="global_coattn"),
    ## Decoding
    "decoding_PRKV": DecodingPress(base_press=PRKVPressPinnedPrefix(transition_mode="nodes_attention", window_size=128)),
    "decoding_PRKV_GLB": DecodingPress(base_press=PRKVPress(transition_mode="global_coattn", window_size=128)),
}
