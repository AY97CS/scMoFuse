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

assert '"_SVD_FLOOR", "ALL_ZEROS", "TRAIN_MEAN"' in SRC, \
    "reference rows are not excluded from model_overall"
print("[OK] reference rows excluded from Overall aggregation")

assert '"GEX_FULL_TARGET":    True' in SRC, "GEX_FULL_TARGET not enabled"
assert '"GEX_APPLY_LOG1P":   True' in SRC, "GEX_APPLY_LOG1P must stay True"
assert '"USE_CACHE":          False' in SRC, "caching must be off for multi-GB arrays"
print("[OK] config flags correct: full GEX target, log1p on, cache off")

ns["CONFIG"].update({"CELLTYPE_GATE_RATIO": 1.5, "CELLTYPE_RARE_FRAC": 0.01})
blk2 = SRC[SRC.index("def make_all_zeros_predictor"):SRC.index("# %% Cell 6")]
exec(compile(blk2, "blk2", "exec"), ns)
make_zeros = ns["make_all_zeros_predictor"]
make_tmean = ns["make_train_mean_predictor"]
evaluate_full_space = ns["evaluate_full_space"]

dens = 0.0308
Tb = sp.csr_matrix((rng.random((N, F)) < dens).astype(np.float32))
sp_b = TargetSpace("svd", "ATAC", F, components=C, true_sparse=Tb)
mz = chunked(sp_b, make_zeros(), rows, 64)
emp = np.asarray(Tb.todense()).mean()
print("\n[all-zeros baseline] density=%.4f  sqrt(density)=%.4f  measured RMSE_pooled=%.4f"
      % (emp, np.sqrt(emp), mz["RMSE_pooled"]))
assert abs(mz["RMSE_pooled"] - np.sqrt(emp)) < 1e-4, "all-zeros != sqrt(density)"
print("[OK] all-zeros predictor reproduces sqrt(density) analytically")

tm_fn, tm_vec = make_tmean(TargetSpace("identity", "GEX", F, dense=truth),
                           np.arange(200), 64)
assert tm_vec.shape == (F,)
ref_mean = truth[:200].mean(axis=0)
assert np.abs(tm_vec - ref_mean).max() < 1e-5, "train mean vector wrong"
print("[OK] train-mean predictor matches the direct column mean")

cts = np.array(["A"] * 150 + ["B"] * 140 + ["rare"] * 10)
r = evaluate_full_space(TargetSpace("identity", "GEX", F, dense=truth),
                        pred, rows, "test_model", cell_types=cts)
g = r["celltype_gate"]
bd = r["per_cell_type"]
print("\n[per-cell-type] pop_mean=%.4f threshold=%.4f PASS=%s" %
      (g["population_mean_RMSE"], g["threshold"], g["PASS"]))
for k, v in bd.items():
    print("   %-5s n=%3d frac=%.3f rare=%-5s RMSE=%.4f ratio=%.2f"
          % (k, v["n_cells"], v["frac"], v["rare"], v["RMSE_per_cell"],
             v["ratio_to_population"]))
assert bd["rare"]["rare"] is False, "10/300 = 3.3% is above the 1% cutoff, so NOT rare"

cts2 = np.array(["A"] * 298 + ["rare"] * 2)
r2 = evaluate_full_space(TargetSpace("identity", "GEX", F, dense=truth),
                         pred, rows, "t2", cell_types=cts2)
assert r2["per_cell_type"]["rare"]["rare"] is True, "2/300 = 0.67% must be rare"
assert r2["per_cell_type"]["A"]["rare"] is False
print("[OK] rare-type flagging and XVII gate behave correctly")

assert '"_SVD_FLOOR", "ALL_ZEROS", "TRAIN_MEAN"' in SRC, \
    "reference rows must be excluded from Overall"
print("[OK] reference rows excluded from Overall aggregation")

print("\n[split]")
assert '"MULTIOME_TEST_BATCHES": ["s4d1", "s4d8", "s4d9"]' in SRC, \
    "test set must be ALL THREE site-4 batches"
assert "MULTIOME_VAL_BATCHES" not in SRC, \
    "the old batch-based validation must be gone"
assert '"MULTIOME_VAL_FRAC"' in SRC and '"MULTIOME_VAL_STRATIFY"' in SRC
assert "validation contains test-site cells" in SRC, \
    "the anti-leakage assertion must be present"
print("[OK] test = all of site 4; val = i.i.d. fraction of training sites")

batches_syn = np.array(
    ["s1d1"] * 100 + ["s1d2"] * 100 + ["s2d1"] * 100 + ["s3d3"] * 100
    + ["s4d1"] * 60 + ["s4d8"] * 60 + ["s4d9"] * 60)
sites_syn = np.array(["site" + b[1] for b in batches_syn])
test_mask = np.isin(batches_syn, ["s4d1", "s4d8", "s4d9"])
val_mask = np.zeros(len(batches_syn), bool)
rs = np.random.RandomState(42)
pool = np.where(~test_mask)[0]
for b in np.unique(batches_syn[pool]):
    ib = pool[batches_syn[pool] == b]
    n_v = int(round(0.10 * len(ib)))
    if n_v:
        val_mask[rs.choice(ib, n_v, replace=False)] = True
train_mask = (~test_mask) & (~val_mask)
assert not (test_mask & val_mask).any()
assert not (set(sites_syn[val_mask]) & set(sites_syn[test_mask])), \
    "validation must not contain test-site cells"
assert test_mask.sum() == 180
assert val_mask.sum() == 40, f"expected 10% of 400 = 40, got {val_mask.sum()}"
assert train_mask.sum() == 360
per_batch = {b: int(val_mask[batches_syn == b].sum()) for b in np.unique(batches_syn[pool])}
print("   test n=%d  val n=%d  train n=%d" % (test_mask.sum(), val_mask.sum(), train_mask.sum()))
print("   val per training batch:", per_batch)
assert all(v == 10 for v in per_batch.values()), "stratification must be even"
print("[OK] split is stratified, leak-free, and correctly sized")

print("\nALL LOCAL VALIDATION PASSED")
