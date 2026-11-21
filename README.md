# scMoFuse

Per-cell attention fusion of heterogeneous base predictors for single-cell
cross-modality prediction.

scMoFuse predicts one molecular modality of a cell from another. It trains five
structurally different base predictors and learns a gating network that assigns
every individual cell its own weights over those five predictions.

Evaluated on all four prediction directions of the NeurIPS 2021 Open Problems
multimodal benchmark (GSE194122), under the official site-level hold-out.

## Results

Overall per-cell RMSE across the four directions, lower is better:

| Configuration | Overall RMSE |
|---|---|
| Official split, single run | 0.2810 |
| Official split, five model seeds | 0.2809 ± 0.0005 |
| Gate fitted on a held-out training site, five seeds | **0.2799 ± 0.0012** |

Per direction, for the reported configuration:

| Direction | scMoFuse | Best published |
|---|---|---|
| ADT2GEX | 0.3138 ± 0.0008 | 0.3221 |
| GEX2ADT | 0.3892 ± 0.0045 | 0.3809 |
| GEX2ATAC | 0.1654 ± < 0.0001 | 0.1774 |
| ATAC2GEX | 0.2511 ± 0.0008 | 0.2169 |

## Method

Five base predictors, each with a different inductive bias:

| Model | Description |
|---|---|
| LSIRidge | Truncated SVD followed by ridge regression, solved in closed form |
| CycleAE | Encoder and decoder per modality over a shared latent space, with reconstruction and cycle-consistency terms |
| ModDropAE | Single joint encoder with modality dropout and no divergence term |
| DeepMLP | Five hidden layers, no bottleneck and no shared latent space |
| TransformerMLP | Feed-forward stem into 32 tokens, transformer encoder, feed-forward head |

A gating network reads each cell's source-modality vector and emits five softmax
weights for that cell:

```
h = GELU(LayerNorm(W1 x))
a = softmax(W3 GELU(W2 h))        a_k >= 0,  sum_k a_k = 1
y = sum_k a_k * pred_k
```

The weights form a convex combination, so the fused prediction stays within the
range spanned by its members.

## Layout

```
scMoFuse/
├── README.md
├── requirements.txt
├── LICENSE
└── experiments/
    ├── experiment_1_official_split/      baseline under the official protocol
    ├── experiment_3_multiseed/           baseline repeated over five seeds
    └── experiment_4_heldout_gate/        reported configuration
        ├── code/
        │   ├── stacking_4dir.py          data, training, fusion and scoring
        │   ├── models_lsi_ridge.py
        │   ├── models_autoencoders.py    CycleAE and ModDropAE
        │   ├── models_mlp.py
        │   ├── models_transformer.py
        │   ├── attention_fusion.py       the gating network
        │   └── validate_local.py         pre-flight checks, no GPU needed
        └── results/
            ├── results_*.json            metrics for every model and direction
            └── metrics_multiseed.json    five-seed means and standard deviations
```

Experiment numbering follows the paper. Each folder holds the exact code that
produced its results, so any one of them can be run on its own.

## Data

Accession [GSE194122](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122),
NCBI Gene Expression Omnibus. Human bone marrow mononuclear cells from twelve
donors across four sequencing sites.

| Assay | Cells | Modalities | Features |
|---|---|---|---|
| CITE-seq | 90,261 | GEX + ADT | 13,953 genes, 134 proteins |
| 10x Multiome | 69,249 | GEX + ATAC | 13,431 genes, 116,490 peaks |

Download the processed `.h5ad` files, which carry the official `is_train`
partition annotation, and place them in a directory of your choice. The scripts
read that directory from the `DATA_DIR` environment variable, defaulting to
`./data`.

Splits under the official protocol:

| Assay | Train | Validation | Test |
|---|---|---|---|
| CITE-seq | 66,175 | 9,020 | 15,066 (site 4) |
| Multiome | 42,323 | 4,702 (10% of sites 1-3) | 22,224 (site 4) |

Site 4 is withheld entirely, so the test cells come from donors, reagent lots
and operators not seen during training.

## Running

```bash
pip install -r requirements.txt
```

Tested on Python 3.12 with PyTorch 2.5.1 and CUDA 12.1, on a single GPU with
about 20 GB of memory.

Check the setup first. This needs no GPU:

```bash
cd experiments/experiment_4_heldout_gate/code
python validate_local.py
```

Then:

```bash
DATA_DIR=/path/to/h5ad python stacking_4dir.py          # all four directions
SMOKE=1 DATA_DIR=/path/to/h5ad python stacking_4dir.py  # short run on a subsample
```

For the multi-seed configurations, set `MODEL_SEED` and run once per seed. A
full four-direction run takes a little over 1.5 hours per seed on the hardware
above.

The four neural models hold roughly 72 million parameters per modality pair:
12.0 M for CycleAE, 5.6 M for ModDropAE, 26.2 M for DeepMLP and 28.0 M for
TransformerMLP. LSIRidge is closed-form and fits in under a minute.

## Evaluation

Two metrics and two controls, all computed over the complete target feature
space. Chromatin predictions are made in a 512-component LSI basis and expanded
back to all 116,490 peaks before scoring.

- Per-cell RMSE, the benchmark's ranking metric.
- Feature-wise Pearson correlation, which measures whether a prediction responds
  to the individual cell.
- A training-mean control, which returns the per-feature training average for
  every cell.
- An all-zeros control, which shows how much of a score is explained by the
  target being sparse.

Each configuration is trained five times under different model seeds with the
data partition frozen, and reported as mean ± one sample standard deviation.

## Citation

```bibtex
@article{yousri_scmofuse,
  title  = {scMoFuse: Per-Cell Attention Fusion of Heterogeneous Models
            for Single-Cell Cross-Modality Prediction},
  author = {Yousri, Ahmed},
  note   = {Faculty of Computers and Information, Mansoura University},
  year   = {2026}
}
```

Benchmark and data:

- Luecken, M.D. et al. (2021) A sandbox for prediction and integration of DNA,
  RNA, and proteins in single cells. NeurIPS Datasets and Benchmarks Track.
- Lance, C. et al. (2022) Multimodal single cell data integration challenge:
  results and lessons learned. PMLR 176, 162-176.

## License

MIT, see `LICENSE`.

## Contact

Ahmed Yousri, ahmedyousri@mans.edu.eg
Faculty of Computers and Information, Mansoura University
