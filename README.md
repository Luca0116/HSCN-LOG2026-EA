# HSCN — Hyperbolic Spectral Convolutional Network

Code release for the LoG 2026 Extended Abstract.

Topology is mapped to a **Sarkar-inspired BFS-tree hyperbolic support** on the Poincaré disk; node features are ordinary Euclidean signals filtered by Helgason–Fourier spectral convolution (multi-scale heat multipliers), then pooled and classified by an MLP.

## Method (paper)

- **Support:** deterministic BFS-tree hyperbolic embedding inspired by Sarkar’s construction (depth → radius, subtree mass → angle)
- **Layers:** Helgason plane-wave analysis → multi-scale heat spectral multiplier → synthesis → real/imaginary channel projection (`reim`)
- **Architecture:** pre-normalized HSCN layers with fixed diffusion scales and a residual post-convolution FFN
- **Readout:** `mean`, `sum`, or `layerwise_hybrid`
- **Classifier:** MLP with **two hidden layers** of width **256**

Paper ablations (Appendix Table 2) are also supported:

| Mode | Flags |
|------|--------|
| Full HSCN | `geometry_mode=hyperbolic`, `multiplier_mode=full` |
| Flat multiplier | `geometry_mode=hyperbolic`, `multiplier_mode=flat` |
| Euclidean counterpart | `geometry_mode=euclidean`, `multiplier_mode=full` |

## Install

```bash
# Create an environment with PyTorch matching your CUDA setup, then:
pip install -e .
# or
pip install -r requirements.txt
```

Requires Python ≥ 3.9, PyTorch ≥ 2.1, and PyTorch Geometric ≥ 2.4.

## Data

TU datasets download automatically via PyTorch Geometric into `./data`.

- For datasets with native node labels, we use the **default PyG `TUDataset` node features** (`use_node_attr=False`).
- For IMDB-BINARY / IMDB-MULTI (no native labels), we use degree-based one-hot features.

## Evaluation protocol

Matches the paper:

1. Stratified **10-fold** CV; within each fold, hold out **10%** of trainval for validation.
2. Select the checkpoint by **validation** accuracy; report the corresponding **test** accuracy.
3. Repeat over seeds `{42, 0, 1, 2, 3}`.
4. Table 1 numbers are **mean ± std across seeds** of the per-seed 10-fold means.

Hardware used for the reported runs: **NVIDIA RTX 4090**.

### One 10-fold run (single seed; ± is fold std)

```bash
PYTHONPATH=$PWD python scripts/run_graph_10fold.py \
  --config configs/paper/mutag.yaml \
  --seed 42
```

### Full five-seed paper evaluation (± is seed std)

```bash
PYTHONPATH=$PWD python scripts/run_paper_seeds.py \
  --config configs/paper/mutag.yaml \
  --device cuda
```

### Single-fold debugging

```bash
PYTHONPATH=$PWD python scripts/run_graph_10fold.py \
  --config configs/paper/mutag.yaml \
  --fold 0 \
  --seed 42
```

## Paper configs

| Dataset | Config |
|---------|--------|
| MUTAG | `configs/paper/mutag.yaml` |
| PROTEINS | `configs/paper/proteins.yaml` |
| DD | `configs/paper/dd.yaml` |
| NCI1 | `configs/paper/nci1.yaml` |
| IMDB-BINARY | `configs/paper/imdb_binary.yaml` |
| IMDB-MULTI | `configs/paper/imdb_multi.yaml` |
| ENZYMES | `configs/paper/enzymes.yaml` |

Ablations (same HPs as the corresponding paper config):

```
configs/ablations/{mutag,proteins,enzymes}_{flat,euclidean}.yaml
```


## Expected results (Table 1, five seeds)

Mean ± std across seeds `{42,0,1,2,3}` (test @ best val):

| Dataset | Acc (%) |
|---------|---------|
| MUTAG | 81.55 ± 2.22 |
| PROTEINS | 75.36 ± 1.14 |
| DD | 75.98 ± 0.49 |
| NCI1 | 73.35 ± 0.41 |
| IMDB-BINARY | 73.02 ± 0.61 |
| IMDB-MULTI | 49.98 ± 0.45 |
| ENZYMES | 36.77 ± 1.50 |

A single `run_graph_10fold.py` call reports fold std for one seed and will **not** match the ± column above.

## Package layout

```
hscn/
  embeddings/sarkar.py   # Sarkar-inspired / Euclidean BFS-tree supports
  nn/helgason.py         # Helgason + Euclidean plane waves
  nn/hscn_layer.py       # Spectral convolution (full / flat)
  nn/graph_readout.py
  models/hscn_graph.py
  data/graph_loaders.py
  graph_task/            # train; test evaluated only on best-val ckpt
scripts/run_graph_10fold.py
scripts/run_paper_seeds.py
configs/paper/
configs/ablations/
```

## Citation

Please cite the accompanying LoG 2026 Extended Abstract (details upon publication).

## License

See `LICENSE`.
