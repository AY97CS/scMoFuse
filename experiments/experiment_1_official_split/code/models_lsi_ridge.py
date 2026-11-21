"""
LSIRidge: a closed-form linear baseline.

A truncated SVD reduces the source modality, and ridge regression maps the
reduced representation onto the target in a single step. There is no iterative
training and no checkpoint, so with a fixed SVD seed the predictions are
reproducible to the last digit - which makes this model useful as a check on
the multi-seed harness as well as a reference for how much of the signal is
linearly accessible.
"""
import time, os, sys, subprocess, numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from pathlib import Path
import tempfile

def _tfidf(X, idf=None):
    """Signac-style TF-IDF. If idf is None, compute it; else use provided.
    Returns (X_tfidf_sparse, idf).
    """
    if sp.issparse(X):
        Xs = X.astype(np.float32)
    else:
        Xs = sp.csr_matrix(X.astype(np.float32))
    n_cells = Xs.shape[0]

    if idf is None:
        binary = Xs.copy(); binary.data = np.ones_like(binary.data, dtype=np.float32)
        n_with = np.array(binary.sum(axis=0)).ravel().astype(np.float32)
        idf = np.log(1.0 + n_cells / (1.0 + n_with)).astype(np.float32)

    cell_totals = np.array(Xs.sum(axis=1)).ravel().astype(np.float32)
    cell_totals = np.maximum(cell_totals, 1.0)
    inv_totals = (1.0 / cell_totals).astype(np.float32)
    Xs = sp.diags(inv_totals).dot(Xs)
    Xs = Xs.dot(sp.diags(idf))
    Xs = Xs.multiply(1e4)
    Xs = sp.csr_matrix(Xs)
    Xs.data = np.log1p(Xs.data)
    return Xs, idf

class LSIRidge:
    """Dual-direction Ridge regression on LSI/PCA reduced features."""
    def __init__(self, a_is_atac=False, b_is_atac=False,
                 n_components=100, ridge_alpha=1.0, min_dim_for_svd=200):
        self.a_is_atac = a_is_atac
        self.b_is_atac = b_is_atac
        self.n_components = n_components
        self.ridge_alpha = ridge_alpha
        self.min_dim_for_svd = min_dim_for_svd

    def _fit_reducer(self, X, is_atac):
        if is_atac:
            X_tfidf, idf = _tfidf(X, idf=None)
        else:
            X_tfidf = (X if sp.issparse(X) else sp.csr_matrix(X.astype(np.float32)))
            idf = None
        if X_tfidf.shape[1] < self.min_dim_for_svd:
            return None, idf
        nc = min(self.n_components, X_tfidf.shape[1] - 1)
        svd = TruncatedSVD(n_components=nc, random_state=42)
        svd.fit(X_tfidf)
        return svd, idf

    def _transform(self, X, svd, idf, is_atac):
        if is_atac:
            X_tfidf, _ = _tfidf(X, idf=idf)
        else:
            X_tfidf = X if sp.issparse(X) else sp.csr_matrix(X.astype(np.float32))
        if svd is None:
            return X_tfidf.toarray().astype(np.float32)
        return svd.transform(X_tfidf).astype(np.float32)

    def fit(self, a, b, label=""):
        t0 = time.time()
        print(f"  [LSIRidge {label}] fitting a→b reducer (a_is_atac={self.a_is_atac}) ...")
        self.svd_a, self.idf_a = self._fit_reducer(a, is_atac=self.a_is_atac)
        a_red = self._transform(a, self.svd_a, self.idf_a, is_atac=self.a_is_atac)
        print(f"    a reduced: {a.shape} → {a_red.shape}  ({time.time()-t0:.1f}s)")

        t1 = time.time()
        self.ridge_a2b = Ridge(alpha=self.ridge_alpha)
        self.ridge_a2b.fit(a_red, b)
        print(f"    a2b Ridge fit  ({time.time()-t1:.1f}s)")

        t2 = time.time()
        print(f"  [LSIRidge {label}] fitting b→a reducer (b_is_atac={self.b_is_atac}) ...")
        self.svd_b, self.idf_b = self._fit_reducer(b, is_atac=self.b_is_atac)
        b_red = self._transform(b, self.svd_b, self.idf_b, is_atac=self.b_is_atac)
        print(f"    b reduced: {b.shape} → {b_red.shape}  ({time.time()-t2:.1f}s)")

        t3 = time.time()
        self.ridge_b2a = Ridge(alpha=self.ridge_alpha)
        self.ridge_b2a.fit(b_red, a)
        print(f"    b2a Ridge fit  ({time.time()-t3:.1f}s)")
        print(f"  [LSIRidge {label}] total fit time: {time.time()-t0:.1f}s")

    def predict_a2b(self, a):
        a_red = self._transform(a, self.svd_a, self.idf_a, is_atac=self.a_is_atac)
        return self.ridge_a2b.predict(a_red).astype(np.float32)

    def predict_b2a(self, b):
        b_red = self._transform(b, self.svd_b, self.idf_b, is_atac=self.b_is_atac)
        return self.ridge_b2a.predict(b_red).astype(np.float32)

_WORKER_SCRIPT = r'''
import sys, os, time, pickle
import numpy as np
import scipy.sparse as sp

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

code_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else sys.argv[1]
sys.path.insert(0, code_dir)
from models_lsi_ridge import LSIRidge

tmpdir, a_is_atac_str, b_is_atac_str = sys.argv[2], sys.argv[3], sys.argv[4]
a_is_atac = (a_is_atac_str == "True")
b_is_atac = (b_is_atac_str == "True")

print(f"[worker PID={os.getpid()}] loading data from {tmpdir}", flush=True)
t = time.time()
a = np.load(os.path.join(tmpdir, "a.npy"), mmap_mode=None)
b = np.load(os.path.join(tmpdir, "b.npy"), mmap_mode=None)
train_idx = np.load(os.path.join(tmpdir, "idx.npy"))
with open(os.path.join(tmpdir, "cfg.pkl"), "rb") as f:
    cfg = pickle.load(f)
print(f"[worker] loaded in {time.time()-t:.1f}s. a={a.shape} b={b.shape} train_idx={train_idx.shape}", flush=True)

model = LSIRidge(
    a_is_atac=a_is_atac, b_is_atac=b_is_atac,
    n_components=cfg["LSI_RIDGE_N_COMPONENTS"],
    ridge_alpha=cfg["LSI_RIDGE_ALPHA"],
    min_dim_for_svd=cfg.get("LSI_RIDGE_MIN_DIM", 200),
)
model.fit(a[train_idx], b[train_idx], label="worker")

with open(os.path.join(tmpdir, "model.pkl"), "wb") as f:
    pickle.dump(model, f, protocol=4)
print(f"[worker] done, model saved", flush=True)
'''

def _train_lsiridge_subprocess(a, b, train_idx, cfg, label):
    """Train LSIRidge in an isolated subprocess (subprocess.Popen of a
    standalone python -c script). No CUDA/OpenMP state inherited."""
    import pickle, tempfile, shutil

    print(f"  [LSIRidge {label}] CUDA detected — running SVD in isolated subprocess (no torch import)", flush=True)
    tmp_root = cfg.get("LSI_TMP_ROOT", "./work/lsi_tmp")
    os.makedirs(tmp_root, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="lsi_ridge_", dir=tmp_root)

    print(f"  [LSIRidge {label}] saving data to {tmpdir} (uncompressed .npy for speed)", flush=True)
    t0 = time.time()
    np.save(os.path.join(tmpdir, "a.npy"), a)
    np.save(os.path.join(tmpdir, "b.npy"), b)
    np.save(os.path.join(tmpdir, "idx.npy"), train_idx)
    with open(os.path.join(tmpdir, "cfg.pkl"), "wb") as f:
        pickle.dump(dict(cfg), f, protocol=4)
    print(f"  [LSIRidge {label}] data saved ({time.time()-t0:.1f}s)", flush=True)

    a_is_atac = (label == "multiome") and not cfg.get("ATAC_PRE_REDUCED", False)
    b_is_atac = False
    code_dir = os.path.dirname(os.path.abspath(__file__))

    cmd = [sys.executable, "-u", "-c", _WORKER_SCRIPT,
           code_dir, tmpdir, str(a_is_atac), str(b_is_atac)]
    print(f"  [LSIRidge {label}] launching subprocess: {sys.executable} -u -c <worker_script>", flush=True)
    t1 = time.time()

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, env={**os.environ})

    for line in proc.stdout:
        print(line.rstrip(), flush=True)
    proc.wait()

    print(f"  [LSIRidge {label}] subprocess exited rc={proc.returncode} after {time.time()-t1:.1f}s", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"LSIRidge subprocess failed (rc={proc.returncode}). tmpdir kept for inspection: {tmpdir}")

    result_path = os.path.join(tmpdir, "model.pkl")
    if not os.path.exists(result_path):
        raise RuntimeError(f"LSIRidge subprocess did not produce result at {result_path}")

    with open(result_path, "rb") as f:
        model = pickle.load(f)
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"  [LSIRidge {label}] model loaded from subprocess result", flush=True)
    return model

def train_lsiridge(a, b, train_idx, val_idx, cfg, label=""):
    """Train LSIRidge on train_idx subset.

    AUTO-DETECTS whether CUDA has been initialized:
      - If no CUDA context: runs TruncatedSVD in-process (fast, normal path).
      - If CUDA context present: runs in a fresh 'spawn' subprocess to avoid
        the post-training OpenMP deadlock that killed attempts 01-09.
    """
    print(f"\n=== Training LSIRidge ({label}) ===")

    try:
        import torch
        cuda_initialized = torch.cuda.is_initialized()
    except ImportError:
        cuda_initialized = False

    if cuda_initialized:
        model = _train_lsiridge_subprocess(a, b, train_idx, cfg, label)
    else:

        a_is_atac = (label == "multiome") and not cfg.get("ATAC_PRE_REDUCED", False)
        b_is_atac = False
        model = LSIRidge(
            a_is_atac=a_is_atac, b_is_atac=b_is_atac,
            n_components=cfg["LSI_RIDGE_N_COMPONENTS"],
            ridge_alpha=cfg["LSI_RIDGE_ALPHA"],
            min_dim_for_svd=cfg.get("LSI_RIDGE_MIN_DIM", 200),
        )
        model.fit(a[train_idx], b[train_idx], label=label)

    return model

def lsiridge_predict(model, a, b, indices):
    """Returns (pred_a2b, pred_b2a) on given indices."""
    pred_a2b = model.predict_a2b(a[indices])
    pred_b2a = model.predict_b2a(b[indices])
    return pred_a2b.astype(np.float32), pred_b2a.astype(np.float32)
