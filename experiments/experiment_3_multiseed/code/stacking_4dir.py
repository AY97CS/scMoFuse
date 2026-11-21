"""
scMoFuse - four-direction modality prediction on GSE194122.

Baseline configuration repeated under five model seeds with the data partition
frozen, so the only quantity varying between runs is the model's own
randomness: weight initialisation, dropout masks and batch order.

Two seeds are kept strictly separate. SPLIT_SEED governs which cells land in
which partition and is frozen at 42. MODEL_SEED governs initialisation, dropout
and batch order, and is read from the environment.

LSIRidge is fitted in closed form on a basis seeded from SPLIT_SEED, so its
spread across model seeds must be exactly zero. A non-zero spread would mean
the model seed was reaching the data partition, and is used here as an
automatic check on the multi-seed harness.

Usage
-----
    MODEL_SEED=42 DATA_DIR=/path/to/h5ad python stacking_4dir.py
    SMOKE=1 python stacking_4dir.py      # short run on a subsample
"""

import os, gc, sys, time, json, gzip, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
import torch.nn.functional as F

import anndata

sys.path.insert(0, str(Path(__file__).parent))
from models_autoencoders        import CycleAE, ModDropAE, train_cycleae, train_moddropae, cycleae_predict, moddropae_predict
from models_mlp         import DeepMLP, train_mlp, mlp_predict
from models_transformer import TransformerMLP, train_transformer, transformer_predict
from models_lsi_ridge   import LSIRidge, train_lsiridge, lsiridge_predict
from attention_fusion   import AttentionFusion, train_attention

SPLIT_SEED = 42
MODEL_SEED = int(os.environ.get("MODEL_SEED", "42"))
SEED = MODEL_SEED
np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
print(f"  seeds: SPLIT_SEED={SPLIT_SEED} (frozen)  MODEL_SEED={MODEL_SEED} (varies)")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ imports OK · device={DEVICE} · torch={torch.__version__}")
if DEVICE.type == "cuda":
    try:
        print(f"  GPU: {torch.cuda.get_device_name(0)}  VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
    except Exception as e:
        print(f"  (GPU info unavailable: {e}) — continuing")

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))

WORK_DIR = Path(os.environ.get("WORK_DIR", "./work"))

MULTIOME_FILE = "GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz"
CITE_FILE     = "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz"

CONFIG = {
    "DATA_DIR":     str(DATA_DIR),

    "WORK_DIR":      str(WORK_DIR),
    "CACHE_DIR":    str(WORK_DIR / "cache"),
    "PRED_DIR":     str(WORK_DIR / "predictions"),
    "CKPT_DIR":     str(WORK_DIR / "ckpts"),
    "OUTPUT_DIR":   str(WORK_DIR / "outputs"),
    "MULTIOME_FILE":    MULTIOME_FILE,
    "CITE_FILE":        CITE_FILE,

    "MULTIOME_TEST_BATCHES": ["s4d1", "s4d8", "s4d9"],

    "MULTIOME_VAL_FRAC":     0.10,
    "MULTIOME_VAL_STRATIFY": True,

    "GEX_FULL_TARGET":    True,
    "GEX_HVG_TOP_K":      5000,
    "ATAC_SVD_COMPONENTS": 512,

    "ATAC_PRE_REDUCED":   True,
    "LSI_TMP_ROOT":       str(WORK_DIR / "lsi_tmp"),
    "ATAC_EVAL_FULL":     True,
    "EVAL_CHUNK_CELLS":   2048,
    "REFERENCE_BASELINES": True,
    "PER_CELL_TYPE":       True,
    "CELLTYPE_GATE_RATIO": 1.5,
    "CELLTYPE_RARE_FRAC":  0.01,
    "USE_CACHE":          False,

    "ATAC_TOP_PEAKS":    20000,
    "ATAC_MIN_FREQ":     0.02,
    "GEX_APPLY_LOG1P":   True,

    "ATAC_USE_RAW_COUNTS": False,
    "ATAC_BINARIZE":      True,
    "ADT_KEEP_ALL":      True,

    "USE_LR_SCHEDULER": True,

    "CYCLEAE_HIDDEN":     384,
    "CYCLEAE_LATENT":      48,
    "CYCLEAE_LAYERS":       3,
    "CYCLEAE_DROPOUT":    0.2,
    "CYCLEAE_LR":         5e-4,
    "CYCLEAE_EPOCHS":     200,
    "CYCLEAE_BATCH":      128,
    "CYCLEAE_LAMBDA_RECON": 0.5,
    "CYCLEAE_LAMBDA_CYCLE": 0.5,
    "CYCLEAE_PATIENCE":    25,

    "MODDROPAE_LATENT":      48,
    "MODDROPAE_LAYERS":       3,
    "MODDROPAE_HIDDEN":     192,
    "MODDROPAE_EPOCHS":     250,
    "MODDROPAE_LR":        1e-3,
    "MODDROPAE_BATCH":      128,
    "MODDROPAE_DROPOUT_P":  0.40,
    "MODDROPAE_CROSS_WEIGHT": 3.0,
    "MODDROPAE_PATIENCE":    25,

    "DEEP_MLP_HIDDEN":   768,
    "DEEP_MLP_LAYERS":     5,
    "DEEP_MLP_DROPOUT":  0.2,
    "DEEP_MLP_LR":      5e-4,
    "DEEP_MLP_EPOCHS":   150,
    "DEEP_MLP_BATCH":    128,
    "DEEP_MLP_PATIENCE": 20,

    "TRANSFORMER_N_TOKENS": 32,
    "TRANSFORMER_D_MODEL":  64,
    "TRANSFORMER_N_HEADS":   4,
    "TRANSFORMER_N_LAYERS":  4,
    "TRANSFORMER_FFN_MULT":  2,
    "TRANSFORMER_DROPOUT": 0.1,
    "TRANSFORMER_LR":     3e-4,
    "TRANSFORMER_EPOCHS": 150,
    "TRANSFORMER_BATCH":  128,
    "TRANSFORMER_PATIENCE": 20,

    "ATTN_HIDDEN":     128,
    "ATTN_DROPOUT":    0.1,
    "ATTN_LR":        5e-4,
    "ATTN_EPOCHS":     50,
    "ATTN_BATCH":     256,
    "ATTN_PATIENCE":   10,

    "FP16":         True,
    "AMP_DTYPE":   "float16",
    "NUM_WORKERS":   4,
    "SEED":          SEED,
    "SPLIT_SEED":    SPLIT_SEED,

    "LSI_RIDGE_N_COMPONENTS": 100,
    "LSI_RIDGE_ALPHA":         1.0,
    "LSI_RIDGE_MIN_DIM":       200,
}

SMOKE = os.environ.get("SMOKE", "0") == "1"
if SMOKE:
    CONFIG.update({
        "SMOKE": True,
        "SMOKE_N_CELLS":        1500,
        "ATAC_SVD_COMPONENTS":   64,
        "EVAL_CHUNK_CELLS":     512,
        "CYCLEAE_EPOCHS": 2, "MODDROPAE_EPOCHS": 2, "DEEP_MLP_EPOCHS": 2,
        "TRANSFORMER_EPOCHS": 2, "ATTN_EPOCHS": 2,
        "CYCLEAE_PATIENCE": 99, "MODDROPAE_PATIENCE": 99, "DEEP_MLP_PATIENCE": 99,
        "TRANSFORMER_PATIENCE": 99, "ATTN_PATIENCE": 99,
        "LSI_RIDGE_N_COMPONENTS": 32,
    })
else:
    CONFIG["SMOKE"] = False

for k in ("CACHE_DIR", "PRED_DIR", "CKPT_DIR", "OUTPUT_DIR"):
    Path(CONFIG[k]).mkdir(parents=True, exist_ok=True)
print("✓ config loaded" + ("   ***** SMOKE MODE *****" if SMOKE else ""))
print(f"  data   = {CONFIG['DATA_DIR']}")
print(f"  work   = {CONFIG['WORK_DIR']}")
print(f"  bases (order): LSIRidge → CycleAE -> ModDropAE → DeepMLP → TransformerMLP")
print(f"  meta  : AttentionFusion (per-cell soft routing)")

def pooled_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def per_cell_rmse(y_true, y_pred):
    sq = (y_true - y_pred) ** 2
    return float(np.sqrt(sq.mean(axis=1)).mean())

def feature_pcc(y_true, y_pred):
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    num = (yt * yp).sum(axis=0)
    den = np.sqrt((yt**2).sum(axis=0) * (yp**2).sum(axis=0)) + 1e-9
    return float(np.nan_to_num(num / den).mean())

def cell_pcc(y_true, y_pred):
    yt = y_true - y_true.mean(axis=1, keepdims=True)
    yp = y_pred - y_pred.mean(axis=1, keepdims=True)
    num = (yt * yp).sum(axis=1)
    den = np.sqrt((yt**2).sum(axis=1) * (yp**2).sum(axis=1)) + 1e-9
    return float((num / den).mean())

def evaluate_all(y_true, y_pred, model_name):
    return {
        "model":         model_name,
        "RMSE_pooled":   pooled_rmse(y_true, y_pred),
        "RMSE_per_cell": per_cell_rmse(y_true, y_pred),
        "MSE":           float(np.mean((y_true - y_pred) ** 2)),
        "feature_PCC":   feature_pcc(y_true, y_pred),
        "cell_PCC":      cell_pcc(y_true, y_pred),
    }

class TargetSpace:
    """Maps model-space predictions to the full evaluation feature space.

    kind='identity' : model space IS the full space (GEX, ADT).
    kind='svd'      : model predicts SVD coefficients; decode via `components_`
                      and compare against the true sparse matrix.
    """

    def __init__(self, kind, name, n_features, dense=None,
                 components=None, true_sparse=None):
        self.kind = kind
        self.name = name
        self.n_features = int(n_features)
        self._dense = dense
        self._components = components
        self._true = true_sparse

    def truth(self, rows):
        if self.kind == "identity":
            return self._dense[rows]
        return np.asarray(self._true[rows].todense(), dtype=np.float32)

    def decode(self, pred_model_space):
        if self.kind == "identity":
            return pred_model_space
        return (pred_model_space @ self._components).astype(np.float32)

    def svd_floor(self, rows, encoder_out):
        """RMSE of passing the TRUE target through the basis and back.

        This is the error the model cannot beat given the chosen basis. Reporting
        it separates 'the basis is too small' from 'the model is bad'.
        """
        if self.kind != "svd":
            return None
        return _chunked_metrics(self, encoder_out, rows,
                                CONFIG["EVAL_CHUNK_CELLS"])["RMSE_per_cell"]

def _chunked_metrics(space, pred_model_space, rows, chunk, return_per_cell=False):
    """Accumulate RMSE / MSE / feature-PCC / cell-PCC over the full feature space.

    Streams cell-chunks so peak RAM stays at O(chunk x n_features) rather than
    O(n_cells x n_features). For ATAC that is 2048x116,490x4 = 0.95 GB per array
    instead of 5.1 GB.

    feature-PCC needs column statistics over ALL cells, so per-feature sums are
    accumulated in float64 and combined at the end.
    """
    n = len(rows)
    F = space.n_features
    sse = 0.0
    n_elem = 0
    per_cell_sq = np.empty(n, dtype=np.float64)
    cell_pcc_vals = np.empty(n, dtype=np.float64)
    s_t = np.zeros(F, np.float64); s_p = np.zeros(F, np.float64)
    s_tt = np.zeros(F, np.float64); s_pp = np.zeros(F, np.float64)
    s_tp = np.zeros(F, np.float64)

    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        yt = space.truth(rows[sl]).astype(np.float32, copy=False)

        if callable(pred_model_space):
            yp = np.asarray(pred_model_space(yt.shape), dtype=np.float32)
        else:
            yp = space.decode(pred_model_space[sl]).astype(np.float32, copy=False)

        diff = yt - yp
        sq = diff.astype(np.float64) ** 2
        sse += sq.sum()
        n_elem += sq.size
        per_cell_sq[sl] = sq.mean(axis=1)

        tc = yt - yt.mean(axis=1, keepdims=True)
        pc = yp - yp.mean(axis=1, keepdims=True)
        num = (tc * pc).sum(axis=1)
        den = np.sqrt((tc ** 2).sum(axis=1) * (pc ** 2).sum(axis=1)) + 1e-9
        cell_pcc_vals[sl] = num / den

        s_t += yt.sum(axis=0, dtype=np.float64)
        s_p += yp.sum(axis=0, dtype=np.float64)
        s_tt += (yt.astype(np.float64) ** 2).sum(axis=0)
        s_pp += (yp.astype(np.float64) ** 2).sum(axis=0)
        s_tp += (yt.astype(np.float64) * yp.astype(np.float64)).sum(axis=0)
        del yt, yp, diff, sq, tc, pc

    cov = s_tp - s_t * s_p / n
    vt = np.maximum(s_tt - s_t ** 2 / n, 0.0)
    vp = np.maximum(s_pp - s_p ** 2 / n, 0.0)
    fpcc = np.nan_to_num(cov / (np.sqrt(vt * vp) + 1e-9))

    out = {
        "RMSE_pooled":   float(np.sqrt(sse / n_elem)),
        "RMSE_per_cell": float(np.sqrt(per_cell_sq).mean()),
        "MSE":           float(sse / n_elem),
        "feature_PCC":   float(fpcc.mean()),
        "cell_PCC":      float(cell_pcc_vals.mean()),
    }
    if return_per_cell:

        out["_per_cell_rmse"] = np.sqrt(per_cell_sq)
    return out

def make_all_zeros_predictor():
    return lambda shape: np.zeros(shape, dtype=np.float32)

def make_train_mean_predictor(space, train_rows, chunk):
    """Per-feature mean of the TRAINING targets, broadcast to every test cell.

    Computed by streaming the training rows so the full dense matrix is never
    materialised.
    """
    F = space.n_features
    acc = np.zeros(F, dtype=np.float64)
    n = len(train_rows)
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        acc += space.truth(train_rows[sl]).sum(axis=0, dtype=np.float64)
    mean_vec = (acc / n).astype(np.float32)
    return (lambda shape: np.broadcast_to(mean_vec, shape).astype(np.float32)), mean_vec

def evaluate_full_space(space, pred_model_space, rows, model_name,
                        cell_types=None):
    """Evaluate one model over the FULL target space.

    If `cell_types` is given (aligned with `rows`), also returns the per-cell-type
    RMSE breakdown and the pass/fail gate:
      - no major cell type may exceed 1.5x the population mean RMSE
      - rare cell types (<1% of test cells) are reported explicitly, never folded
        into an "other" bucket
    """
    m = _chunked_metrics(space, pred_model_space, rows,
                         CONFIG["EVAL_CHUNK_CELLS"],
                         return_per_cell=cell_types is not None)
    m["model"] = model_name
    m["n_target_features"] = space.n_features
    m["target_space"] = space.kind

    if cell_types is not None:
        pcr = m.pop("_per_cell_rmse")
        pop_mean = float(pcr.mean())
        gate = CONFIG["CELLTYPE_GATE_RATIO"] * pop_mean
        rare_frac = CONFIG["CELLTYPE_RARE_FRAC"]
        n_total = len(pcr)

        breakdown, worst_ratio, failures = {}, 0.0, []
        for ct in np.unique(cell_types):
            mask = cell_types == ct
            n_ct = int(mask.sum())
            rmse_ct = float(pcr[mask].mean())
            ratio = rmse_ct / pop_mean if pop_mean > 0 else float("nan")
            is_rare = (n_ct / n_total) < rare_frac
            breakdown[str(ct)] = {
                "n_cells": n_ct,
                "frac": n_ct / n_total,
                "rare": bool(is_rare),
                "RMSE_per_cell": rmse_ct,
                "ratio_to_population": ratio,
            }
            if not is_rare:
                worst_ratio = max(worst_ratio, ratio)
                if rmse_ct > gate:
                    failures.append(str(ct))

        m["per_cell_type"] = breakdown
        m["celltype_gate"] = {
            "population_mean_RMSE": pop_mean,
            "threshold": gate,
            "ratio_limit": CONFIG["CELLTYPE_GATE_RATIO"],
            "worst_major_ratio": worst_ratio,
            "failing_major_types": failures,
            "PASS": len(failures) == 0,
        }
    return m

def load_dataset(filename):
    """Resolve a dataset file under DATA_DIR, decompressing a .gz copy if needed."""
    data_dir = Path(CONFIG["DATA_DIR"]); data_dir.mkdir(parents=True, exist_ok=True)
    h5ad = data_dir / filename.replace(".h5ad.gz", ".h5ad")
    if h5ad.exists():
        print(f"  using {h5ad.name} ({h5ad.stat().st_size/1e9:.2f} GB)")
        return h5ad
    gz = data_dir / filename
    if not gz.exists():
        raise FileNotFoundError(
            f"{filename} not found in {data_dir}. Download the processed GSE194122 "
            f"files from GEO and place them there, or set DATA_DIR."
        )
    print(f"  decompressing {gz.name}")
    with gzip.open(gz, "rb") as fin, open(h5ad, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    print(f"  using {h5ad.name} ({h5ad.stat().st_size/1e9:.2f} GB)")
    return h5ad

def select_hvg(X_sparse, top_k):
    if not sp.issparse(X_sparse): X_sparse = sp.csr_matrix(X_sparse)
    if top_k >= X_sparse.shape[1]:
        return X_sparse.toarray().astype(np.float32), np.arange(X_sparse.shape[1])
    sq = X_sparse.copy(); sq.data = sq.data ** 2
    mean = np.asarray(X_sparse.mean(axis=0)).ravel()
    var  = np.maximum(np.asarray(sq.mean(axis=0)).ravel() - mean ** 2, 0.0)
    keep = np.argsort(-var)[:top_k]; keep.sort()
    return X_sparse[:, keep].toarray().astype(np.float32), keep

def filter_peaks(X_sparse, top_n, min_freq):
    if not sp.issparse(X_sparse): X_sparse = sp.csr_matrix(X_sparse)
    n = X_sparse.shape[0]
    freq = (X_sparse > 0).sum(axis=0).A1 / n
    keep_mask = freq >= min_freq
    X_sparse = X_sparse[:, keep_mask]; freq = freq[keep_mask]
    keep_idx = np.where(keep_mask)[0]
    if X_sparse.shape[1] > top_n:
        order = np.argpartition(-freq, top_n - 1)[:top_n]; order.sort()
        X_sparse = X_sparse[:, order]; keep_idx = keep_idx[order]
    return X_sparse, keep_idx

def tfidf_svd_atac(atac_sparse, n_components, seed):
    """TF-IDF + TruncatedSVD (LSI) — the standard scATAC representation.

    This gives the models a tractable chromatin space, while `components_`
    supports projecting predictions back to all 116,490 peaks for scoring.

    Must run before any CUDA call: scikit-learn's TruncatedSVD deadlocks if
    OpenMP/BLAS thread pools are inherited after CUDA initialisation.
    """
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.decomposition import TruncatedSVD

    print(f"  ATAC -> TF-IDF + SVD({n_components}) ...")
    t0 = time.time()
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
    Xt = tfidf.fit_transform(atac_sparse)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    Z = svd.fit_transform(Xt).astype(np.float32)
    evr = float(svd.explained_variance_ratio_.sum())
    print(f"    Z={Z.shape}  explained_var={evr:.4f}  ({time.time()-t0:.1f}s)")

    print("    fitting back-projection to original peak space ...")
    t0 = time.time()
    ZtZ = Z.T @ Z + 1e-3 * np.eye(Z.shape[1], dtype=np.float32)
    ZtZ_inv = np.linalg.inv(ZtZ).astype(np.float32)
    F_total = atac_sparse.shape[1]
    comp = np.empty((Z.shape[1], F_total), dtype=np.float32)
    step = 20000
    for j in range(0, F_total, step):
        blk = np.asarray(atac_sparse[:, j:j + step].todense(), dtype=np.float32)
        comp[:, j:j + step] = ZtZ_inv @ (Z.T @ blk)
        del blk
    print(f"    back-projection {comp.shape}  ({time.time()-t0:.1f}s)")
    return Z, comp, evr

def load_multiome():
    if CONFIG["USE_CACHE"]:
        cache = Path(CONFIG["CACHE_DIR"]) / "multiome_processed.npz"
        if cache.exists():
            print("  ✓ using cached multiome processed data")
            d = np.load(cache)
            return (d["gex"], d["atac"], d["train_idx"], d["val_idx"], d["test_idx"])

    h5ad = load_dataset(CONFIG["MULTIOME_FILE"])
    print("  reading Multiome AnnData ...")
    a = anndata.read_h5ad(h5ad)
    a.var_names_make_unique()
    ft = a.var["feature_types"].astype(str).values
    rna_a  = a[:, ft == "GEX" ].copy()
    atac_a = a[:, ft == "ATAC"].copy()
    print(f"  Multiome shape: {a.shape}  rna: {rna_a.shape}  atac: {atac_a.shape}")

    rna_X = rna_a.X
    if not sp.issparse(rna_X): rna_X = sp.csr_matrix(rna_X)

    if CONFIG["GEX_APPLY_LOG1P"] and rna_X.data.min() >= 0:
        rna_X = rna_X.copy(); rna_X.data = np.log1p(rna_X.data)

    if CONFIG["GEX_FULL_TARGET"] and not CONFIG["SMOKE"]:
        gex_dense = np.asarray(rna_X.todense(), dtype=np.float32)
        print(f"  GEX FULL space: {gex_dense.shape}  ({gex_dense.nbytes/1e9:.2f} GB)")
    else:
        gex_dense, _ = select_hvg(rna_X, CONFIG["GEX_HVG_TOP_K"])
        print(f"  GEX (HVG subset — SMOKE only): {gex_dense.shape}")
    del rna_X, rna_a; gc.collect()

    atac_X = atac_a.X
    if not sp.issparse(atac_X): atac_X = sp.csr_matrix(atac_X)
    atac_X = atac_X.tocsr()
    atac_X = atac_X.copy(); atac_X.data = (atac_X.data > 0).astype(np.float32)
    atac_X.eliminate_zeros()
    print(f"  ATAC FULL sparse: {atac_X.shape}  nnz={atac_X.nnz:,}  "
          f"density={atac_X.nnz/(atac_X.shape[0]*atac_X.shape[1]):.4f}")
    del atac_a; gc.collect()

    cell_types = a.obs["cell_type"].astype(str).values.copy()
    print(f"  cell types: {len(np.unique(cell_types))} distinct")

    batches = a.obs["batch"].astype(str).values
    sites = a.obs["Site"].astype(str).values if "Site" in a.obs.columns else None

    test_mask = np.isin(batches, list(CONFIG["MULTIOME_TEST_BATCHES"]))

    val_mask = np.zeros(len(batches), dtype=bool)
    rng_split = np.random.RandomState(CONFIG["SPLIT_SEED"])
    frac = CONFIG["MULTIOME_VAL_FRAC"]
    pool = np.where(~test_mask)[0]
    if CONFIG["MULTIOME_VAL_STRATIFY"]:
        for b in np.unique(batches[pool]):
            idx_b = pool[batches[pool] == b]
            n_v = int(round(frac * len(idx_b)))
            if n_v > 0:
                val_mask[rng_split.choice(idx_b, n_v, replace=False)] = True
    else:
        n_v = int(round(frac * len(pool)))
        val_mask[rng_split.choice(pool, n_v, replace=False)] = True

    train_mask = (~test_mask) & (~val_mask)
    assert not (test_mask & val_mask).any(), "val leaked into the test site"
    if sites is not None:
        leaked = set(sites[val_mask]) & set(sites[test_mask])
        assert not leaked, f"validation contains test-site cells: {leaked}"
        print(f"  official split: test site = {sorted(set(sites[test_mask]))}, "
              f"train/val sites = {sorted(set(sites[train_mask]))}")
    print(f"  test batches : {sorted(set(batches[test_mask]))}")
    print(f"  val  batches : {sorted(set(batches[val_mask]))} "
          f"({CONFIG['MULTIOME_VAL_FRAC']:.0%} i.i.d. from training sites)")
    train_idx = np.where(train_mask)[0]
    val_idx   = np.where(val_mask)[0]
    test_idx  = np.where(test_mask)[0]
    print(f"  split: train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")

    if CONFIG["SMOKE"]:
        keep = np.concatenate([
            train_idx[:CONFIG["SMOKE_N_CELLS"]],
            val_idx[:CONFIG["SMOKE_N_CELLS"] // 4],
            test_idx[:CONFIG["SMOKE_N_CELLS"] // 4]])
        remap = {o: i for i, o in enumerate(keep)}
        gex_dense = gex_dense[keep]
        atac_X = atac_X[keep]
        cell_types = cell_types[keep]
        train_idx = np.array([remap[i] for i in train_idx if i in remap])
        val_idx = np.array([remap[i] for i in val_idx if i in remap])
        test_idx = np.array([remap[i] for i in test_idx if i in remap])
        print(f"  SMOKE subsample: {gex_dense.shape[0]:,} cells "
              f"(train={len(train_idx)} val={len(val_idx)} test={len(test_idx)})")

    atac_Z, atac_comp, evr = tfidf_svd_atac(
        atac_X, CONFIG["ATAC_SVD_COMPONENTS"], CONFIG["SPLIT_SEED"])
    CONFIG["_ATAC_SVD_EXPLAINED_VAR"] = evr

    gex_space = TargetSpace("identity", "GEX", gex_dense.shape[1], dense=gex_dense)
    atac_space = TargetSpace("svd", "ATAC", atac_X.shape[1],
                             components=atac_comp, true_sparse=atac_X)

    return (gex_dense, atac_Z, train_idx, val_idx, test_idx,
            gex_space, atac_space, cell_types)

def load_citeseq():
    if CONFIG["USE_CACHE"]:
        cache = Path(CONFIG["CACHE_DIR"]) / "citeseq_processed.npz"
        if cache.exists():
            print("  ✓ using cached CITE-seq processed data")
            d = np.load(cache)
            return (d["gex"], d["adt"], d["train_idx"], d["val_idx"], d["test_idx"])

    h5ad = load_dataset(CONFIG["CITE_FILE"])
    print("  reading CITE-seq AnnData ...")
    a = anndata.read_h5ad(h5ad)
    a.var_names_make_unique()
    ft = a.var["feature_types"].astype(str).values
    rna_a = a[:, ft == "GEX"].copy()
    adt_a = a[:, ft == "ADT"].copy()

    rna_X = rna_a.X
    if not sp.issparse(rna_X): rna_X = sp.csr_matrix(rna_X)
    if CONFIG["GEX_APPLY_LOG1P"] and rna_X.data.min() >= 0:
        rna_X = rna_X.copy(); rna_X.data = np.log1p(rna_X.data)

    if CONFIG["GEX_FULL_TARGET"] and not CONFIG["SMOKE"]:
        gex_dense = np.asarray(rna_X.todense(), dtype=np.float32)
        print(f"  GEX FULL space: {gex_dense.shape}  ({gex_dense.nbytes/1e9:.2f} GB)")
    else:
        gex_dense, _ = select_hvg(rna_X, CONFIG["GEX_HVG_TOP_K"])
        print(f"  GEX (HVG subset — SMOKE only): {gex_dense.shape}")
    del rna_X, rna_a; gc.collect()

    adt_X = adt_a.X
    if sp.issparse(adt_X): adt_X = adt_X.toarray()
    adt_dense = adt_X.astype(np.float32)
    print(f"  ADT shape: {adt_dense.shape}  (all {adt_dense.shape[1]} proteins, untransformed)")

    cell_types = a.obs["cell_type"].astype(str).values.copy()
    print(f"  cell types: {len(np.unique(cell_types))} distinct")

    is_train = a.obs["is_train"].astype(str).values
    train_idx = np.where(is_train == "train")[0]
    val_idx   = np.where(is_train == "iid_holdout")[0]
    test_idx  = np.where(is_train == "test")[0]
    print(f"  split: train={len(train_idx):,}  val(iid_holdout)={len(val_idx):,}  test={len(test_idx):,}")

    if CONFIG["SMOKE"]:
        keep = np.concatenate([
            train_idx[:CONFIG["SMOKE_N_CELLS"]],
            val_idx[:CONFIG["SMOKE_N_CELLS"] // 4],
            test_idx[:CONFIG["SMOKE_N_CELLS"] // 4]])
        remap = {o: i for i, o in enumerate(keep)}
        gex_dense = gex_dense[keep]; adt_dense = adt_dense[keep]
        cell_types = cell_types[keep]
        train_idx = np.array([remap[i] for i in train_idx if i in remap])
        val_idx = np.array([remap[i] for i in val_idx if i in remap])
        test_idx = np.array([remap[i] for i in test_idx if i in remap])
        print(f"  SMOKE subsample: {gex_dense.shape[0]:,} cells "
              f"(train={len(train_idx)} val={len(val_idx)} test={len(test_idx)})")

    gex_space = TargetSpace("identity", "GEX", gex_dense.shape[1], dense=gex_dense)
    adt_space = TargetSpace("identity", "ADT", adt_dense.shape[1], dense=adt_dense)

    return (gex_dense, adt_dense, train_idx, val_idx, test_idx,
            gex_space, adt_space, cell_types)

print("✓ data loaders defined")

def run_modality_pair(label, a_arr, b_arr, train_idx, val_idx, test_idx,
                      direction_a2b_name, direction_b2a_name,
                      a_space=None, b_space=None, cell_types=None):
    print("\n" + "═" * 78)
    print(f"  MODALITY PAIR: {label}  ({direction_a2b_name} + {direction_b2a_name})")
    print("═" * 78)
    print(f"  A: {a_arr.shape}  B: {b_arr.shape}")
    print(f"  train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")

    ckpt_dir = Path(CONFIG["CKPT_DIR"]) / label; ckpt_dir.mkdir(exist_ok=True)
    preds = {}

    def _save_preds(name, va2b, vb2a, ta2b, tb2a):
        preds[name] = {"val_a2b": va2b, "val_b2a": vb2a, "test_a2b": ta2b, "test_b2a": tb2a}
        print(f"  ✓ {name} predictions stored in memory")

    print("\n--- LSIRidge (TF-IDF + SVD-100 + Ridge) [FIRST — before GPU] ---")
    lsi = train_lsiridge(a_arr, b_arr, train_idx, val_idx, CONFIG, label=label)
    v_a2b, v_b2a = lsiridge_predict(lsi, a_arr, b_arr, val_idx)
    t_a2b, t_b2a = lsiridge_predict(lsi, a_arr, b_arr, test_idx)
    _save_preds("lsiridge", v_a2b, v_b2a, t_a2b, t_b2a)
    del lsi; gc.collect()

    print("\n--- CycleAE ---")
    babel = train_cycleae(a_arr, b_arr, train_idx, val_idx, DEVICE, CONFIG,
                        ckpt_path=ckpt_dir / "babel.pt", modality_pair_label=label)
    v_a2b, v_b2a = cycleae_predict(babel, a_arr, b_arr, val_idx,  DEVICE)
    t_a2b, t_b2a = cycleae_predict(babel, a_arr, b_arr, test_idx, DEVICE)
    _save_preds("cycleae", v_a2b, v_b2a, t_a2b, t_b2a)
    del babel; gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    print("\n--- ModDropAE ---")
    multivi = train_moddropae(a_arr, b_arr, train_idx, val_idx, DEVICE, CONFIG,
                            ckpt_path=ckpt_dir / "multivi.pt", modality_pair_label=label)
    v_a2b, v_b2a = moddropae_predict(multivi, a_arr, b_arr, val_idx,  DEVICE)
    t_a2b, t_b2a = moddropae_predict(multivi, a_arr, b_arr, test_idx, DEVICE)
    _save_preds("moddropae", v_a2b, v_b2a, t_a2b, t_b2a)
    del multivi; gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    print("\n--- DeepMLP ---")
    deepmlp = DeepMLP(a_dim=a_arr.shape[1], b_dim=b_arr.shape[1],
                      hidden=CONFIG["DEEP_MLP_HIDDEN"], n_layers=CONFIG["DEEP_MLP_LAYERS"],
                      dropout=CONFIG["DEEP_MLP_DROPOUT"])
    deepmlp = train_mlp(deepmlp, "deepmlp", a_arr, b_arr, train_idx, val_idx,
                        DEVICE=DEVICE, cfg=CONFIG, ckpt_name=f"{label}/deep_mlp.pt")
    v_a2b, v_b2a = mlp_predict(deepmlp, a_arr, b_arr, val_idx,  DEVICE=DEVICE)
    t_a2b, t_b2a = mlp_predict(deepmlp, a_arr, b_arr, test_idx, DEVICE=DEVICE)
    _save_preds("deepmlp", v_a2b, v_b2a, t_a2b, t_b2a)
    del deepmlp; gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    print("\n--- TransformerMLP ---")
    tfm = TransformerMLP(a_dim=a_arr.shape[1], b_dim=b_arr.shape[1],
                         n_tokens=CONFIG["TRANSFORMER_N_TOKENS"],
                         d_model=CONFIG["TRANSFORMER_D_MODEL"],
                         n_heads=CONFIG["TRANSFORMER_N_HEADS"],
                         n_layers=CONFIG["TRANSFORMER_N_LAYERS"],
                         ffn_mult=CONFIG["TRANSFORMER_FFN_MULT"],
                         dropout=CONFIG["TRANSFORMER_DROPOUT"])
    tfm = train_transformer(tfm, "transformermlp", a_arr, b_arr, train_idx, val_idx,
                            DEVICE=DEVICE, cfg=CONFIG, ckpt_path=ckpt_dir / "transformer.pt")
    v_a2b, v_b2a = transformer_predict(tfm, a_arr, b_arr, val_idx,  DEVICE=DEVICE)
    t_a2b, t_b2a = transformer_predict(tfm, a_arr, b_arr, test_idx, DEVICE=DEVICE)
    _save_preds("transformermlp", v_a2b, v_b2a, t_a2b, t_b2a)
    del tfm; gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    base_models = ["lsiridge", "cycleae", "moddropae", "deepmlp", "transformermlp"]
    val_a2b  = [preds[m]["val_a2b"]  for m in base_models]
    test_a2b = [preds[m]["test_a2b"] for m in base_models]
    val_b2a  = [preds[m]["val_b2a"]  for m in base_models]
    test_b2a = [preds[m]["test_b2a"] for m in base_models]

    y_val_b  = b_arr[val_idx];  y_test_b = b_arr[test_idx]
    y_val_a  = a_arr[val_idx];  y_test_a = a_arr[test_idx]

    print(f"\n--- AttentionFusion ({direction_a2b_name}) ---")
    stack_a2b, alpha_a2b_test, alpha_a2b_val, _ = train_attention(
        direction_name=direction_a2b_name.lower(),
        input_arr=a_arr, target_arr=b_arr,
        base_preds_val=val_a2b, base_preds_test=test_a2b,
        val_idx=val_idx, test_idx=test_idx,
        DEVICE=DEVICE, cfg=CONFIG,
    )

    print(f"\n--- AttentionFusion ({direction_b2a_name}) ---")
    stack_b2a, alpha_b2a_test, alpha_b2a_val, _ = train_attention(
        direction_name=direction_b2a_name.lower(),
        input_arr=b_arr, target_arr=a_arr,
        base_preds_val=val_b2a, base_preds_test=test_b2a,
        val_idx=val_idx, test_idx=test_idx,
        DEVICE=DEVICE, cfg=CONFIG,
    )

    print(f"\n--- Test-set evaluation (FULL target space) ---")
    print(f"  {direction_a2b_name}: target = {b_space.name} "
          f"({b_space.n_features:,} features, {b_space.kind})")
    print(f"  {direction_b2a_name}: target = {a_space.name} "
          f"({a_space.n_features:,} features, {a_space.kind})")

    results = {direction_a2b_name: [], direction_b2a_name: []}

    ct_test = cell_types[test_idx] if (cell_types is not None and CONFIG["PER_CELL_TYPE"]) else None

    def _report(direction, space, rows, name, pred, with_ct=True):
        r = evaluate_full_space(space, pred, rows, name,
                                cell_types=ct_test if with_ct else None)
        results[direction].append(r)
        mark = "★ " if name == "attentionfusion" else "  "
        line = (f"  {direction:>10s}  {mark}{name:>12s}  "
                f"RMSE_pc={r['RMSE_per_cell']:.4f}  RMSE_pool={r['RMSE_pooled']:.4f}  "
                f"fPCC={r['feature_PCC']:.4f}  cPCC={r['cell_PCC']:.4f}")
        if "celltype_gate" in r:
            g = r["celltype_gate"]
            line += f"  ct_gate={'PASS' if g['PASS'] else 'FAIL'}(worst {g['worst_major_ratio']:.2f}x)"
        print(line)
        return r

    if CONFIG["REFERENCE_BASELINES"]:
        print("\n  -- reference baselines (not models; excluded from Overall) --")
        for direction, space in ((direction_a2b_name, b_space),
                                 (direction_b2a_name, a_space)):
            r = _report(direction, space, test_idx, "ALL_ZEROS",
                        make_all_zeros_predictor())
            r["note"] = ("trivial predictor: all zeros. On a sparse binary target "
                         "this is strong under RMSE - see NOTE_gex2atac_metric_degenerate.md")
            tm_fn, tm_vec = make_train_mean_predictor(space, train_idx,
                                                      CONFIG["EVAL_CHUNK_CELLS"])
            r = _report(direction, space, test_idx, "TRAIN_MEAN", tm_fn)
            r["note"] = "trivial predictor: per-feature mean of the training targets"
            del tm_fn, tm_vec; gc.collect()

    if b_space.kind == "svd":
        r = _report(direction_a2b_name, b_space, test_idx, "_SVD_FLOOR", b_arr[test_idx])
        r["note"] = "lower bound from the SVD target basis; no model can score better"
    if a_space.kind == "svd":
        r = _report(direction_b2a_name, a_space, test_idx, "_SVD_FLOOR", a_arr[test_idx])
        r["note"] = "lower bound from the SVD target basis; no model can score better"

    for m, p in zip(base_models, test_a2b):
        _report(direction_a2b_name, b_space, test_idx, m, p)
    _report(direction_a2b_name, b_space, test_idx, "attentionfusion", stack_a2b)

    for m, p in zip(base_models, test_b2a):
        _report(direction_b2a_name, a_space, test_idx, m, p)
    _report(direction_b2a_name, a_space, test_idx, "attentionfusion", stack_b2a)

    np.savez_compressed(Path(CONFIG["OUTPUT_DIR"]) / f"alpha_{label}_seed{CONFIG['SEED']}.npz",
        a2b_alpha_test=alpha_a2b_test, a2b_alpha_val=alpha_a2b_val,
        b2a_alpha_test=alpha_b2a_test, b2a_alpha_val=alpha_b2a_val,
        base_models=np.asarray(base_models),
        direction_a2b=direction_a2b_name, direction_b2a=direction_b2a_name,
    )
    print(f"  attention α (test, mean):")
    print(f"    {direction_a2b_name}: {dict(zip(base_models, alpha_a2b_test.mean(0).tolist()))}")
    print(f"    {direction_b2a_name}: {dict(zip(base_models, alpha_b2a_test.mean(0).tolist()))}")

    return results, (alpha_a2b_test, alpha_b2a_test)

print("\n" + "█" * 78)
print("  PHASE 1: MULTIOME (ATAC ↔ GEX)")
print("█" * 78)
(gex_m, atac_m, train_m, val_m, test_m,
 gex_space_m, atac_space_m, ct_m) = load_multiome()
mult_results, (alpha_a2g, alpha_g2a) = run_modality_pair(
    label="multiome",
    a_arr=atac_m, b_arr=gex_m,
    train_idx=train_m, val_idx=val_m, test_idx=test_m,
    direction_a2b_name="ATAC2GEX",
    direction_b2a_name="GEX2ATAC",
    a_space=atac_space_m, b_space=gex_space_m, cell_types=ct_m,
)
del gex_space_m, atac_space_m, ct_m
del gex_m, atac_m; gc.collect()
if DEVICE.type == "cuda": torch.cuda.empty_cache()

print("\n" + "█" * 78)
print("  PHASE 2: CITE-SEQ (ADT ↔ GEX)")
print("█" * 78)
(gex_c, adt_c, train_c, val_c, test_c,
 gex_space_c, adt_space_c, ct_c) = load_citeseq()
cite_results, (alpha_a2adt, alpha_adt2g) = run_modality_pair(
    label="citeseq",
    a_arr=adt_c, b_arr=gex_c,
    train_idx=train_c, val_idx=val_c, test_idx=test_c,
    direction_a2b_name="ADT2GEX",
    direction_b2a_name="GEX2ADT",
    a_space=adt_space_c, b_space=gex_space_c, cell_types=ct_c,
)
del gex_space_c, adt_space_c, ct_c
del gex_c, adt_c; gc.collect()

all_results = {**mult_results, **cite_results}
print("\n" + "█" * 78)
print("  OVERALL RESULTS (Open Problems-style RMSE)")
print("█" * 78)

model_overall = {}
for direction, model_list in all_results.items():
    for r in model_list:
        m = r["model"]
        if m in ("_SVD_FLOOR", "ALL_ZEROS", "TRAIN_MEAN"):
            continue
        if m not in model_overall:
            model_overall[m] = {"directions": {}, "rmse_per_cell_list": [], "rmse_pooled_list": []}
        model_overall[m]["directions"][direction] = r
        model_overall[m]["rmse_per_cell_list"].append(r["RMSE_per_cell"])
        model_overall[m]["rmse_pooled_list"].append(r["RMSE_pooled"])

print(f"\n{'Model':>14s}  {'ATAC2GEX':>10s}  {'GEX2ATAC':>10s}  {'GEX2ADT':>10s}  {'ADT2GEX':>10s}  {'Overall':>10s}")
print("-" * 75)
for m, info in model_overall.items():
    d = info["directions"]
    marker = "★ " if m == "attentionfusion" else "  "
    row = "  ".join(f"{d.get(k,{}).get('RMSE_per_cell', float('nan')):.4f}"
                    for k in ["ATAC2GEX","GEX2ATAC","GEX2ADT","ADT2GEX"])
    overall = np.mean(info["rmse_per_cell_list"])
    print(f"{marker}{m:>12s}  {row}  {overall:.4f}")

out_dir = Path(CONFIG["OUTPUT_DIR"]); out_dir.mkdir(parents=True, exist_ok=True)

summary = {
    "dataset":    "GSE194122 — NeurIPS 2021 Open Problems Multimodal",
    "experiment": "scMoFuse",
    "configuration":    "run",
    "split_seed": SPLIT_SEED,
    "model_seed": MODEL_SEED,
    "protocol":   ("FULL target feature space (GEX 13,431/13,953 dense; ADT 134; "
                   "ATAC all 116,490 via SVD back-projection) + log1p GEX + "
                   "untransformed ADT + 4-direction Overall RMSE. "
                   ""),
    "config":     {k: v for k, v in CONFIG.items() if isinstance(v, (str, int, float, bool, list))},
    "results_per_direction": all_results,
    "model_overall": {m: {"directions": info["directions"],
                           "Overall_RMSE_per_cell": float(np.mean(info["rmse_per_cell_list"])),
                           "Overall_RMSE_pooled":   float(np.mean(info["rmse_pooled_list"]))}
                      for m, info in model_overall.items()},
    "published_baselines_overall_RMSE": {
        "GNN_Wen2022":             0.2780,
        "SCJoint_Chen2021":        0.2841,
        "KRR_Deng2021":            0.2845,
        "SimpleMLP_2021":          0.2847,
        "Ensemble_Ubingazhibov":   0.2868,
        "Novel_MLP_Ryazantsev":    0.2911,
        "CAJAL_Laddach":           0.2918,
        "ZZKEEPCycleAEWUZZ2021":            0.3054,
        "SCMM_Minoura":            0.4153,
    },
}

print("\n" + "█" * 78)
print("  TRIVIALITY CHECK - is each direction's RMSE metric informative?")
print("█" * 78)
PUB_BEST = {"ATAC2GEX": ("GNN", 0.2310), "GEX2ATAC": ("Ensemble", 0.1774),
            "GEX2ADT":  ("GNN", 0.3809), "ADT2GEX":  ("GNN", 0.3223)}
triviality = {}
for direction, model_list in all_results.items():
    by = {r["model"]: r["RMSE_per_cell"] for r in model_list}
    zeros = by.get("ALL_ZEROS"); tmean = by.get("TRAIN_MEAN")
    stack = by.get("attentionfusion"); pubn, pubv = PUB_BEST.get(direction, (None, None))
    trivial_best = min([v for v in (zeros, tmean) if v is not None], default=None)
    verdict = "OK"
    if trivial_best is not None and pubv is not None and trivial_best < pubv:
        verdict = "DEGENERATE - a trivial predictor beats the best published method"
    elif trivial_best is not None and stack is not None and trivial_best < stack:
        verdict = "SUSPECT - a trivial predictor beats our stack"
    triviality[direction] = {
        "ALL_ZEROS": zeros, "TRAIN_MEAN": tmean, "attentionfusion": stack,
        "best_published": {"method": pubn, "RMSE": pubv},
        "verdict": verdict,
    }
    print(f"\n  {direction}")
    print(f"    ALL_ZEROS      {zeros if zeros is None else format(zeros, '.4f')}")
    print(f"    TRAIN_MEAN     {tmean if tmean is None else format(tmean, '.4f')}")
    print(f"    ours STACK     {stack if stack is None else format(stack, '.4f')}")
    print(f"    best published {pubv:.4f} ({pubn})" if pubv else "    best published  n/a")
    print(f"    --> {verdict}")
summary["triviality_check"] = triviality

degen = [d for d, v in triviality.items() if v["verdict"].startswith("DEGENERATE")]
if degen:
    keep = [d for d in all_results if d not in degen]
    so = np.mean([triviality[d]["attentionfusion"] for d in keep])
    po = np.mean([PUB_BEST[d][1] for d in keep if d in PUB_BEST])
    summary["overall_excluding_degenerate"] = {
        "excluded": degen, "kept": keep,
        "ours": float(so), "best_published_per_direction_mean": float(po),
        "delta": float(so - po),
    }
    print(f"\n  Excluding {degen}:  ours={so:.4f}  vs per-direction-best published={po:.4f}  "
          f"delta={so - po:+.4f}")

res_name = f"results_seed{CONFIG['SEED']}.json"
csv_name = f"summary_seed{CONFIG['SEED']}.csv"
with open(out_dir / res_name, "w") as f:
    json.dump(summary, f, indent=2, default=str)

rows = []
for direction, model_list in all_results.items():
    for r in model_list:
        rows.append({"direction": direction, **r})
df = pd.DataFrame(rows)
df.to_csv(out_dir / csv_name, index=False)

stack_overall = summary["model_overall"].get("attentionfusion", {}).get("Overall_RMSE_per_cell")
print(f"\n  ★ AttentionFusion Overall RMSE = {stack_overall:.4f}")
for name, val in summary["published_baselines_overall_RMSE"].items():
    beat = "✅" if stack_overall < val else "❌"
    print(f"  {beat} {name} = {val:.4f}  (Δ = {stack_overall - val:+.4f})")

print(f"\n  ✓ results.json  → {out_dir / res_name}")
print(f"  ✓ summary.csv   → {out_dir / csv_name}")
print(f"  ✓ alpha_*.npz   → {out_dir}")

print("\n=== END OF 4-DIRECTION PIPELINE ===")
