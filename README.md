# DistoMove

Predicting structural movements from AlphaFold distograms.

## Overview

DistoMove trains a neural network to predict per-residue-pair structural mobility from AlphaFold2 distogram logits. Given a set of AlphaFold2 pickle outputs for a target protein, it learns to classify residue pairs into movement classes based on structural labels, enabling prediction of conformational flexibility without running MD simulations.

## Method

Input features are the 64-bin distogram logits from AlphaFold2 (optionally extended with the predicted aligned error, PAE, to 65 features). Two network architectures are supported:

- **2dconv** (default): linear projection → 2D convolution over the L×L residue-pair map → linear classifier
- **mlp**: purely feedforward, operating on each residue pair independently

Labels are quantized into movement classes using distance thresholds `[0, 1, 3, 10]` Å and trained with binary cross-entropy loss. Validation metrics include AUROC, AUPR, and top-10 accuracy.

## Installation

```bash
conda env create -f environment.yml
conda activate DistoMove
```

Requires a CUDA-capable GPU. The environment is based on PyTorch 2.1.0 + CUDA 11.8.

## Data

Training labels (`multiclass_labels.pkl`) are available on Zenodo. Place the file in the working directory before training.

AlphaFold2 pickle files (containing `distogram.logits` and optionally `predicted_aligned_error`) should be organized as:

```
<pkl_dir>/<target>/afsample2/result*.pkl
```

## Usage

```bash
python train.py <target> [options]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `target` | *(required)* | Target protein identifier (used for leave-one-out validation) |
| `--pkl-dir` | *(hardcoded path)* | Directory containing AlphaFold2 pickle files |
| `--training-to-use` | `10` | Number of AF2 samples per target to use (`1`, `5`, or `10`) |
| `--network-type` | `2dconv` | Network architecture: `2dconv` or `mlp` |
| `--no-pae` | `False` | Exclude PAE features (uses 64 input channels instead of 65) |
| `--no-half-precision` | `False` | Disable float16 training |
| `--device` | auto | `cuda` or `cpu` |

### Example

```bash
python train.py 1ABC --training-to-use 10 --network-type 2dconv
```

Training uses leave-one-out validation: the specified `target` is held out as the validation set and all other proteins in the label dictionary are used for training.

## Output

Training produces the following directories (named by run configuration):

- `checkpoints_<prefix>/` — model checkpoints saved every 10 epochs
- `plots_<prefix>/` — per-epoch PR/ROC curves and residue-pair maps (PDF)
- `pickles_<prefix>/` — raw model outputs and labels per epoch
- `metrics_<prefix>/` — CSV files with per-epoch AUC, AUPR, and top-10 accuracy

## Label generation

```bash
python make_labels.py
```

Generates `multiclass_labels.pkl` from structural data. See script for input format details.

## License

Apache 2.0
