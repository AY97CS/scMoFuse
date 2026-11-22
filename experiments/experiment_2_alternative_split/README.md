# Experiment 2: site-matched validation

Single run, seed 42.

## Configuration

The Multiome partition differs from the baseline. Two of the three site-4
batches (S4D1 and S4D8) form the test set and the third (S4D9) becomes the
validation set, so base-model checkpoint selection and the fusion gate both
draw on cells from the site used for testing. The former validation cells
(4,702) return to the training set, which grows to 47,025 cells. The CITE-seq
split, the models and the hyperparameters are unchanged.

| Profile | Partition | Source | Cells |
|---|---|---|---|
| CITE-seq | Train | Sites 1 to 3 | 66,175 |
| CITE-seq | Validation | Holdout from sites 1 to 3 | 9,020 |
| CITE-seq | Test | S4D1, S4D8, S4D9 | 15,066 |
| Multiome | Train | Sites 1 to 3 | 47,025 |
| Multiome | Validation | S4D9 | 4,325 |
| Multiome | Test | S4D1, S4D8 | 17,899 |

## Results

 RMSE, lower is better.

| Model | ATAC2GEX | GEX2ATAC | GEX2ADT | ADT2GEX | Overall |
|---|---|---|---|---|---|
| LSIRidge | 0.2462 | 0.1654 | 0.4237 | 0.3257 | 0.2902 |
| CycleAE | 0.2407 | 0.1668 | 0.4221 | 0.3188 | 0.2871 |
| ModDropAE | 0.2350 | 0.1660 | 0.4121 | 0.3156 | 0.2822 |
| DeepMLP | 0.2427 | 0.1657 | 0.4170 | 0.3188 | 0.2860 |
| TransformerMLP | 0.2401 | 0.1668 | 0.4085 | 0.3161 | 0.2829 |
| AttentionFusion | 0.2340 | 0.1654 | 0.3856 | 0.3135 | 0.2746 |

Multiome control predictors for this partition, from
`results/controls_multiome.json`:

| Control | ATAC2GEX | GEX2ATAC |
|---|---|---|
| All-zeros | 0.2440 | 0.1763 |
| Training mean | 0.2433 | 0.1665 |

The controls depend on both the training and the test population, so they are
computed separately for this partition rather than carried over from
Experiment 1. The CITE-seq controls are unchanged.

Full metrics, including per-cell-type breakdowns and feature-wise correlation,
are in `results/results.json`.
