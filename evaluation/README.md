# Evaluation

We follow evaluation scheme provided by kvpress.

## How to evaluate

1. **Configure your evaluation** - Edit `evaluate_config.yaml` to specify your *method*, *press*, and *dataset*
2. **Run the evaluation** - Execute the script: ```python evaluate.py```

Note: this repo vendors `kvpress/` locally; make sure you created the conda env from the repo root so `pip -e .` installed the local `kvpress` package (see the top-level `README.md`).

The script will read from `evaluate_config.yaml` and run inference accordingly. 
If you want, you can override the settings via command line, for instance:

```bash
python evaluate.py --dataset longbench --data_dir narrativeqa --model meta-llama/Meta-Llama-3.1-8B-Instruct --press_name PRKV --compression_ratio 0.5
```

For decoding eviction, set the KV budget with `--target_size`:

```bash
python evaluate.py --dataset math500 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --press_name decoding_PRKV_GLB --target_size 1024 --compression_interval 128
```

Results (predictions & metrics) are automatically saved to the `output_dir` directory .
You can also run evaluation by .sh file. We provide example .sh file for RULER.

```bash
bash evaluate_ruler.sh
```

### Presses and Datasets 
We provide our configuration in `PRESS_REGISTRY`.
- Prefill
    - PRKV: PRKVPress(transition_mode="nodes_attention")
    - PRKV-GLB: PRKVPress(transition_mode="global_coattn")
- Decoding
    - PRKV: DecodingPress(base_press=PRKVPressPinnedPrefix(transition_mode="nodes_attention", window_size=128))
    - PRKV-GLB: DecodingPress(base_press=PRKVPress(transition_mode="global_coattn", window_size=128))
    
For benchmarks, we follow dataset directory listed in `DATASET_REGISTRY`.

At the moment, we evaluated on the following standard popular benchmarks:

- [RULER](benchmarks/ruler/README.md) ([hf link](https://huggingface.co/datasets/simonjegou/ruler))
- [longbench](benchmarks/longbench/README.md)([hf link](https://huggingface.co/datasets/Xnhyacinth/LongBench))
- [Needle in a Haystack](benchmarks/needle_in_haystack/README.md)([hf link][Paul Graham's essays](https://huggingface.co/datasets/alessiodevoto/paul_graham_essays))
- [AIME25](benchmarks/aime25/README.md) ([hf link](https://huggingface.co/datasets/opencompass/AIME2025))
- [MATH-500](benchmarks/math500/README.md) ([hf link](https://huggingface.co/datasets/HuggingFaceH4/MATH-500))

### Dataset revisions

AIME25 and MATH-500 results use the revisions below, which are pinned in `DATASET_REVISION`. Later upstream revisions append a `\boxed{}` instruction to every question.

| Dataset | Revision |
|---|---|
| `alessiodevoto/aime25` | `6c6853611089d679bf20453b40fca854977eb03e` |
| `alessiodevoto/math500` | `5c06bb9ee02a115fff716c0a1be1a18b1005e63e` |

**For detailed information** about each dataset or implementing custom benchmarks, see the individual README files in the benchmarks directory.