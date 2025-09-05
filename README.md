## Adaptive Segmentation for Steered Mixture of Experts (SMoE)

This repository implements adaptive segmentation for Steered Mixture of Experts (SMoE). The workflow contains four main steps: Segmentation, SMoE Training on individual segments, Global Exploration, and Global Training.

---

## Table of Contents

* [Step 1: Segmentation](#step-1-segmentation)
* [Step 2: Train SMoE on Individual Segments](#step-2-train-smoe-on-individual-segments)
* [Step 3: Global Exploration](#step-3-global-exploration)
* [Step 4: Global Training](#step-4-global-training)

---

## Step 1: Segmentation

Run the MATLAB code:

```matlab
test.m
```

in the `MDBSCAN-old` folder.

The resulting segmentation will be stored in:

```
../data/seg/mdbscan/$thres/$partition
```

* `$thres` — Threshold setting for the segmentation algorithm
* `$partition` — Dataset name

---

## Step 2: Train SMoE on Individual Segments

Generate fixed-size image patches for each individual segment:

```bash
python data_generation.py --partition <dataset_name> --p <dataset_name> --diff <threshold>
```

* `--partition, --p` — Dataset name
* `--diff` — Threshold used in the segmentation algorithm

These fixed-size patches are then used to train multiple SMoE models.

Train SMoE models:

```bash
python Final_train_seg.py
```

The resulting SMoE models will be stored. You can then aggregate the parameters of each model to formulate the proposed initialization.

---

## Step 3: Global Exploration

Run:

```bash
python load_init_para.py
```

This generates aggregated initial parameters, ready for global optimization.

---

## Step 4: Global Training

Run:

```bash
python train_with_init_seperate_grid.py \
    --steer \
    --init_para_path init_para/ \
    --data_path data/img/$partition/ \
    --map_path data/seg/ \
    --partition $partition \
    --result_path seperate_grid \
    --data_name cameraman
```

**Parameters:**

* `--steer` — Use steered kernel
* `--init_para_path` — Path to aggregated initial parameters (output of Step 3)
* `--data_path` — Path to image data
* `--map_path` — Path to segmentation maps (default: `data/seg/`)
* `--partition` — Dataset name
* `--result_path` — Output folder for results
* `--data_name` — Name of the dataset

---
