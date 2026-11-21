"""
Per-cell attention fusion.

A small gating network reads a cell's source-modality vector and emits a
softmax distribution over the base predictors. The fused prediction is the
weighted sum of their outputs, so it is a convex combination and can never
fall outside the range its members span.
"""
import time, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from contextlib import nullcontext

class AttentionFusion(nn.Module):
    """Compute per-cell soft routing weights over K base predictions.

    Args:
        input_dim: dimension of the SOURCE modality (ATAC for a2g, GEX for g2a).
        n_bases:   number of base predictors (K).
        hidden:    hidden width of the routing MLP.
        dropout:   dropout in the routing MLP.

    Forward:
        x:          (B, input_dim) source modality tensor (the model's input)
        base_preds: list of K tensors each (B, n_out), one per base model
    Returns:
        fused:  (B, n_out)
        alpha:  (B, K)   — for diagnostics / per-cell routing analysis
    """
    def __init__(self, input_dim, n_bases, hidden=128, dropout=0.1):
        super().__init__()
        self.n_bases = n_bases
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, n_bases),
        )

    def forward(self, x, base_preds):
        alpha = F.softmax(self.gate(x), dim=-1)
        stack = torch.stack(base_preds, dim=-1)
        fused = (stack * alpha.unsqueeze(1)).sum(-1)
        return fused, alpha

def train_attention(
    direction_name, input_arr, target_arr, base_preds_val, base_preds_test,
    val_idx, test_idx, DEVICE, cfg,
):
    """Train an AttentionFusion head for one direction.

    Args:
        direction_name:    'atac2gex' or 'gex2atac'
        input_arr:         (N, input_dim) the source modality, full population
        target_arr:        (N, out_dim) the target modality, full population (for val truth)
        base_preds_val:    list of K (n_val, out_dim) per-base val predictions
        base_preds_test:   list of K (n_test, out_dim) per-base test predictions
        val_idx:           (n_val,) cell indices into input_arr / target_arr
        test_idx:          (n_test,) cell indices

    Returns:
        test_fused:        (n_test, out_dim) blended test predictions
        alpha_test:        (n_test, K) per-test-cell routing weights
        alpha_val:         (n_val, K) per-val-cell routing weights
    """
    print(f"\n=== Training AttentionFusion ({direction_name}) ===")
    K       = len(base_preds_val)
    in_dim  = input_arr.shape[1]
    out_dim = target_arr.shape[1]
    n_val   = len(val_idx)
    n_test  = len(test_idx)
    print(f"  K bases  : {K}")
    print(f"  in_dim   : {in_dim}")
    print(f"  out_dim  : {out_dim}")
    print(f"  n_val    : {n_val}   n_test: {n_test}")

    model = AttentionFusion(in_dim, K,
                              hidden=cfg["ATTN_HIDDEN"],
                              dropout=cfg["ATTN_DROPOUT"]).to(DEVICE)
    print(f"  params   : {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["ATTN_LR"], weight_decay=1e-4)

    x_val_t      = torch.from_numpy(input_arr [val_idx]).float()
    y_val_t      = torch.from_numpy(target_arr[val_idx]).float()
    base_val_t   = [torch.from_numpy(p).float() for p in base_preds_val]

    n_train_attn = int(n_val * 0.85)
    perm         = np.random.RandomState(cfg["SPLIT_SEED"]).permutation(n_val)
    train_attn_i = perm[:n_train_attn]
    val_attn_i   = perm[n_train_attn:]

    BATCH = cfg["ATTN_BATCH"]

    def make_loader(idx_subset, shuffle):
        ds = TensorDataset(
            x_val_t[idx_subset], y_val_t[idx_subset],
            *[bp[idx_subset] for bp in base_val_t],
        )

        return DataLoader(ds, batch_size=BATCH, shuffle=shuffle,
                            num_workers=0, pin_memory=(DEVICE.type=="cuda"))

    train_loader = make_loader(train_attn_i, True)
    val_loader   = make_loader(val_attn_i,   False)

    use_fp16 = cfg["FP16"] and DEVICE.type == "cuda" and cfg["AMP_DTYPE"] == "float16"
    use_bf16 = cfg["FP16"] and DEVICE.type == "cuda" and cfg["AMP_DTYPE"] == "bfloat16"
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None
    def amp_ctx():
        if use_fp16: return torch.amp.autocast("cuda", dtype=torch.float16)
        if use_bf16: return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    best_val, patience = float("inf"), 0
    for ep in range(1, cfg["ATTN_EPOCHS"] + 1):
        model.train(); t0 = time.time(); tr_losses = []
        for batch in train_loader:
            x, y, *bp = [b.to(DEVICE, non_blocking=True) for b in batch]
            opt.zero_grad()
            with amp_ctx():
                fused, alpha = model(x, bp)
                loss = F.mse_loss(fused, y)
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tr_losses.append(loss.item())

        model.eval(); v_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x, y, *bp = [b.to(DEVICE) for b in batch]
                fused, _ = model(x, bp)
                v_losses.append(F.mse_loss(fused, y).item())
        v = float(np.mean(v_losses)); tr = float(np.mean(tr_losses))
        print(f"  ep{ep:3d}/{cfg['ATTN_EPOCHS']}  train={tr:.4f}  val={v:.4f}  ({time.time()-t0:.1f}s)")

        if v < best_val:
            best_val = v; patience = 0
            ckpt_path = Path(cfg["CKPT_DIR"]) / f"attn_{direction_name}.pt"
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience += 1
            if patience >= cfg["ATTN_PATIENCE"]:
                print(f"  → early stop @ ep{ep}"); break

    model.load_state_dict(torch.load(Path(cfg["CKPT_DIR"]) / f"attn_{direction_name}.pt",
                                       map_location="cpu", weights_only=True))
    model = model.to(DEVICE)

    model.eval()
    x_test_t    = torch.from_numpy(input_arr[test_idx]).float()
    base_test_t = [torch.from_numpy(p).float() for p in base_preds_test]
    fused_chunks, alpha_chunks = [], []
    with torch.no_grad():
        for i in range(0, n_test, BATCH):
            sl = slice(i, min(i + BATCH, n_test))
            x = x_test_t[sl].to(DEVICE)
            bp = [b[sl].to(DEVICE) for b in base_test_t]
            fused, alpha = model(x, bp)
            fused_chunks.append(fused.cpu().numpy())
            alpha_chunks.append(alpha.cpu().numpy())
    test_fused = np.concatenate(fused_chunks).astype(np.float32)
    alpha_test = np.concatenate(alpha_chunks).astype(np.float32)

    val_fused_chunks, val_alpha_chunks = [], []
    with torch.no_grad():
        for i in range(0, n_val, BATCH):
            sl = slice(i, min(i + BATCH, n_val))
            x = x_val_t[sl].to(DEVICE)
            bp = [b[sl].to(DEVICE) for b in base_val_t]
            fused, alpha = model(x, bp)
            val_fused_chunks.append(fused.cpu().numpy())
            val_alpha_chunks.append(alpha.cpu().numpy())
    val_fused = np.concatenate(val_fused_chunks).astype(np.float32)
    alpha_val = np.concatenate(val_alpha_chunks).astype(np.float32)

    print(f"  attention weights (test, mean per base): {alpha_test.mean(0).tolist()}")
    return test_fused, alpha_test, alpha_val, val_fused
