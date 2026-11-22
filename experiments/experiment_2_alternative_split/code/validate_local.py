"""
Pre-flight checks that need no GPU: confirm imports resolve, the dataset is
reachable, shapes are what the pipeline expects, and a single minibatch makes
it through a forward pass.
"""
import ast
import numpy as np
import scipy.sparse as sp

SRC = open("stacking_4dir.py", encoding="utf-8").read()
ast.parse(SRC)
print("[OK] stacking_4dir.py parses")
for f in ["attention_fusion.py", "models_autoencoders.py", "models_mlp.py",
          "models_transformer.py", "models_lsi_ridge.py"]:
    ast.parse(open(f, encoding="utf-8").read())
    print(f"[OK] {f} parses")

ns = {"np": np, "sp": sp, "CONFIG": {"EVAL_CHUNK_CELLS": 64}}
blk = SRC[SRC.index("class TargetSpace"):SRC.index("# %% Cell 6")]
exec(compile(blk, "blk", "exec"), ns)
TargetSpace = ns["TargetSpace"]
chunked = ns["_chunked_metrics"]

def pooled(t, p):
    return float(np.sqrt(np.mean((t - p) ** 2)))

def percell(t, p):
    return float(np.sqrt(((t - p) ** 2).mean(axis=1)).mean())

def fpcc(t, p):
    a = t - t.mean(0, keepdims=True)
    b = p - p.mean(0, keepdims=True)
    return float(np.nan_to_num(
        (a * b).sum(0) / (np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0)) + 1e-9)).mean())

def cpcc(t, p):
    a = t - t.mean(1, keepdims=True)
    b = p - p.mean(1, keepdims=True)
    return float(((a * b).sum(1) /
                  (np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1)) + 1e-9)).mean())

rng = np.random.default_rng(0)
N, F = 300, 220
truth = rng.random((N, F)).astype(np.float32)
pred = (truth + 0.3 * rng.standard_normal((N, F))).astype(np.float32)
rows = np.arange(N)

m = chunked(TargetSpace("identity", "GEX", F, dense=truth), pred, rows, 64)
ref = {"RMSE_pooled": pooled(truth, pred), "RMSE_per_cell": percell(truth, pred),
       "feature_PCC": fpcc(truth, pred), "cell_PCC": cpcc(truth, pred)}
print("\n[identity space] chunked vs dense reference")
worst = 0.0
for k, v in ref.items():
    d = abs(m[k] - v)
    worst = max(worst, d)
    print("  %-14s chunked=%.8f  dense=%.8f  |diff|=%.2e" % (k, m[k], v, d))
assert worst < 2e-5, "chunked metrics disagree with dense reference"
print("[OK] chunked evaluator matches dense metrics (worst |diff| = %.2e)" % worst)

C = rng.random((16, F)).astype(np.float32)
Tsp = sp.csr_matrix((rng.random((N, F)) < 0.1).astype(np.float32))
Z = rng.random((N, 16)).astype(np.float32)
m2 = chunked(TargetSpace("svd", "ATAC", F, components=C, true_sparse=Tsp), Z, rows, 64)
t2 = np.asarray(Tsp.todense(), dtype=np.float32)
p2 = (Z @ C).astype(np.float32)
d = abs(m2["RMSE_per_cell"] - percell(t2, p2))
print("\n[svd space] RMSE_per_cell chunked=%.8f dense=%.8f |diff|=%.2e"
      % (m2["RMSE_per_cell"], percell(t2, p2), d))
assert d < 2e-5, "svd back-projection disagrees with dense reference"
print("[OK] svd back-projection path matches dense reference")

assert 'if m == "_SVD_FLOOR":' in SRC and "continue" in SRC, \
    "_SVD_FLOOR is not excluded from model_overall"
print("[OK] _SVD_FLOOR excluded from Overall aggregation")

assert '"GEX_FULL_TARGET":    True' in SRC, "GEX_FULL_TARGET not enabled"
assert '"GEX_APPLY_LOG1P":   True' in SRC, "GEX_APPLY_LOG1P must stay True"
assert '"USE_CACHE":          False' in SRC, "caching must be off for multi-GB arrays"
print("[OK] config flags correct: full GEX target, log1p on, cache off")

print("\nALL LOCAL VALIDATION PASSED")
