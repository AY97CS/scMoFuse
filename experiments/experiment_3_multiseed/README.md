# Experiment 3: seed stability

Five runs, model seeds 42 to 46.

## Configuration

The Experiment 1 configuration repeated five times with the data partition
frozen, so the only quantity varying between runs is the model's own
randomness: weight initialisation, dropout masks and batch order.

SPLIT_SEED governs which cells land in which partition and is frozen at 42.
MODEL_SEED governs initialisation, dropout and batch order, and is read from
the environment.

## Results

RMSE, mean plus or minus one sample standard deviation over the five
seeds.

| Model | ATAC2GEX | GEX2ADT | Overall |
|---|---|---|---|
| LSIRidge | 0.2437 ± < 0.0001 | 0.4237 ± < 0.0001 | 0.2896 ± 0.0000 |
| CycleAE | 0.2616 ± 0.0016 | 0.4205 ± 0.0051 | 0.2919 ± 0.0015 |
| ModDropAE | 0.2577 ± 0.0026 | 0.4161 ± 0.0045 | 0.2890 ± 0.0019 |
| DeepMLP | 0.2623 ± 0.0023 | 0.4192 ± 0.0054 | 0.2916 ± 0.0012 |
| TransformerMLP | 0.2559 ± 0.0025 | 0.4109 ± 0.0053 | 0.2878 ± 0.0016 |
| AttentionFusion | 0.2565 ± 0.0012 | 0.3886 ± 0.0013 | 0.2809 ± 0.0005 |

LSIRidge is fitted in closed form on a basis seeded from SPLIT_SEED, so its
standard deviation is exactly zero in every direction. A non-zero spread there
would indicate that the model seed was reaching the data partition.

Per-seed metrics are in `results/results_seed42.json` through
`results_seed46.json`; the aggregate is in `results/metrics_multiseed.json`.
