# HSCN — Hyperbolic Spectral Convolutional Network

Anonymous code release accompanying the LoG 2026 Extended Abstract submission.

**GitHub:** [Luca0116/HSCN-LOG2026-EA](https://github.com/Luca0116/HSCN-LOG2026-EA)

This repository implements **HSCN** for **TU graph classification**: topology is mapped into the Poincaré disk with a Sarkar-style BFS embedding, node signals are filtered with Helgason–Fourier spectral convolution (multi-scale heat multipliers), and graphs are pooled with mean / sum / layerwise hybrid readout before an MLP classifier.

## Method (paper)

- **Support:** Sarkar Poincaré-disk embedding from the graph topology
- **Layers:** Helgason plane-wave analysis → multi-scale heat spectral multiplier → synthesis → real/imaginary mixing (`reim`)
- **Defaults:** `sigma=0`, fixed diffusion scales (`learnable_scales=false`), pre-norm, feature MLP + optional post-conv FFN
- **Readout:** `mean`, `sum`, or `layerwise_hybrid`
- **Classifier:** 2-layer MLP with hidden width 256

Node features are ordinary Euclidean signals; geometry enters through the hyperbolic support and Helgason transform.

## Install

```bash
# Create an environment with PyTorch matching your CUDA setup, then:
pip install -e .
# or
pip install -r requirements.txt
```

Requires Python ≥ 3.9, PyTorch ≥ 2.1, and PyTorch Geometric ≥ 2.4.

## Data

TU datasets are downloaded automatically by PyTorch Geometric into `./data` (configurable via `data_root`).

Social graphs without native attributes (e.g. IMDB-BINARY / IMDB-MULTI) use one-hot degree features, following common TU benchmark practice.

## Reproduce 10-fold CV

```bash
PYTHONPATH=$PWD python scripts/run_graph_10fold.py \
  --config configs/paper_unified/mean_readout/mutag_mean_10fold.yaml
```

Other paper configs:

| Dataset | Config |
|---------|--------|
| MUTAG | `configs/paper_unified/mean_readout/mutag_mean_10fold.yaml` |
| NCI1 | `configs/paper_unified/paper_main_sigma0_fixed/nci1_10fold.yaml` |
| DD | `configs/paper_unified/sigma0/dd_sum_sigma0_10fold.yaml` |
| ENZYMES | `configs/paper_unified/sigma0/enzymes_sum_sigma0_fixed_10fold.yaml` |
| IMDB-BINARY | `configs/paper_unified/sigma0/imdb_binary_layerwise_sigma0_10fold.yaml` |
| IMDB-MULTI | `configs/paper_unified/sigma0/imdb_multi_layerwise_sigma0_10fold.yaml` |
| PROTEINS | `configs/paper_unified/sigma0/proteins_layerwise_sigma0_10fold.yaml` |

Protocol: stratified 10-fold CV, validation taken as 10% of each trainval split, early stopping on validation, report test accuracy at the best validation epoch. Results are written under `./runs/`.

Single-fold debug:

```bash
PYTHONPATH=$PWD python scripts/run_graph_10fold.py \
  --config configs/paper_unified/mean_readout/mutag_mean_10fold.yaml \
  --fold 0
```

## Package layout

```
hscn/
  embeddings/sarkar.py   # Sarkar Poincaré support
  nn/helgason.py         # Helgason plane waves
  nn/hscn_layer.py       # Spectral convolution layer
  nn/graph_readout.py    # mean / sum / layerwise helpers
  models/hscn_graph.py   # HSCNGraphClassifier
  data/graph_loaders.py  # TU loaders + k-fold masks
  graph_task/            # training loop
scripts/run_graph_10fold.py
configs/paper_unified/   # paper hyperparameters
```

## Citation

If you use this code, please cite the accompanying LoG 2026 Extended Abstract (citation details to be added upon publication).

## License

See [LICENSE](LICENSE).
