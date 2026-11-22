# Experiment 4: gate fitted on a held-out site

Five runs, model seeds 42 to 46. This is the configuration reported in the
paper.

## Configuration

The Multiome fusion gate is fitted on a held-out training site (site 3), unseen
by the base models during both training and checkpoint selection, instead of on
the same validation cells used to early-stop them. The base models therefore
train on sites 1 and 2 only, which reduces the Multiome training set from
42,323 to 29,222 cells. The CITE-seq partition is unchanged.

## Results

RMSE, mean plus or minus one sample standard deviation over the five
seeds. Overall 0.2799 ± 0.0012.

| Model | ATAC2GEX | GEX2ATAC | GEX2ADT | ADT2GEX |
|---|---|---|---|---|
| LSIRidge | 0.2548 ± 0.0000 | 0.1678 ± 0.0000 | 0.4237 ± < 0.0001 | 0.3257 ± 0.0000 |
| CycleAE | 0.2561 ± 0.0010 | 0.1664 ± 0.0002 | 0.4227 ± 0.0043 | 0.3187 ± 0.0018 |
| ModDropAE | 0.2564 ± 0.0006 | 0.1653 ± 0.0001 | 0.4188 ± 0.0040 | 0.3170 ± 0.0021 |
| DeepMLP | 0.2571 ± 0.0014 | 0.1685 ± 0.0002 | 0.4211 ± 0.0044 | 0.3167 ± 0.0022 |
| TransformerMLP | 0.2536 ± 0.0019 | 0.1683 ± 0.0003 | 0.4062 ± 0.0092 | 0.3193 ± 0.0023 |
| AttentionFusion | 0.2511 ± 0.0008 | 0.1654 ± < 0.0001 | 0.3892 ± 0.0045 | 0.3138 ± 0.0008 |

Mean gate weight on ATAC2GEX, compared with Experiment 3:

| Base model | Experiment 3 | Experiment 4 |
|---|---|---|
| LSIRidge | 3.96 % | 35.23 % |
| CycleAE | 16.21 % | 5.14 % |
| ModDropAE | 15.25 % | 16.88 % |
| DeepMLP | 44.83 % | 12.41 % |
| TransformerMLP | 19.76 % | 30.34 % |

Per-seed metrics are in `results/results_seed42.json` through
`results_seed46.json`; the aggregate is in `results/metrics_multiseed.json`.
