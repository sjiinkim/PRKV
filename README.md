# [NeurIPS 2026] Global Importance Estimation for KV Cache Eviction

## TL;DR
We propose PRKV, a training-free Personalized PageRank-based KV eviction strategy that captures global token importance.

## Installation

We build our code implementation on [kvpress library](https://github.com/NVIDIA/kvpress.git).
You can install environment using our [environment.yaml](environment.yaml) file.


```bash
conda env create -f environment.yaml
conda activate kvpress
```

## Evaluation
We follow evaluation scheme also provided by kvpress.

Please refer to the [evaluation](evaluation/README.md) directory in this repo for more details and results.

## License

This repository is released under the [Apache License 2.0](LICENSE). It is built on [kvpress](https://github.com/NVIDIA/kvpress.git) (Apache License 2.0, Copyright 2024 NVIDIA Corporation), and we modified its code in `kvpress/` and `evaluation/`.
