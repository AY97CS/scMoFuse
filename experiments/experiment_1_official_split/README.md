# Experiment 1: baseline under the official protocol

Single run, seed 42.

## Configuration

Site 4 is withheld for testing and validation is drawn only from the training
sites. The five base predictors are trained on the training partition and
early-stopped on the validation partition, and the fusion gate is fitted on
that same validation partition.

## Results

Per-cell RMSE, lower is better.

| Model | ATAC2GEX | GEX2ATAC | GEX2ADT | ADT2GEX | Overall |
|---|---|---|---|---|---|
| All-zeros control | 0.2408 | 0.1767 | 1.1890 | 0.5004 | 0.5267 |
| Training-mean control | 0.2406 | 0.1666 | 0.6551 | 0.3549 | 0.3543 |
| LSIRidge | 0.2437 | 0.1656 | 0.4237 | 0.3257 | 0.2896 |
| CycleAE | 0.2602 | 0.1661 | 0.4238 | 0.3182 | 0.2921 |
| ModDropAE | 0.2616 | 0.1653 | 0.4215 | 0.3206 | 0.2923 |
| DeepMLP | 0.2610 | 0.1659 | 0.4217 | 0.3185 | 0.2918 |
| TransformerMLP | 0.2544 | 0.1660 | 0.4112 | 0.3155 | 0.2868 |
| AttentionFusion | 0.2557 | 0.1654 | 0.3893 | 0.3135 | 0.2810 |

Full metrics, including per-cell-type breakdowns and feature-wise correlation,
are in `results/results.json`.
