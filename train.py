import os
import json
import random
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from mm_lego.config import Config
from mm_lego.model import load_dataset, initialize_model, GatingNetwork, run_leiden

# ====== Loss Functions ======
def spatial_contrastive_loss_batch(emb: torch.Tensor, coords: torch.Tensor, k: int = 6, margin: float = 1.0) -> torch.Tensor:
    B = emb.size(0)
    if B <= 1: return torch.tensor(0., device=emb.device)

    diff = coords.unsqueeze(1) - coords.unsqueeze(0) 
    dist_spatial = diff.pow(2).sum(-1).sqrt()        
    dist_spatial = dist_spatial + torch.eye(B, device=emb.device) * 1e6
    
    knn_idx = dist_spatial.topk(k=min(k, B-1), dim=1, largest=False).indices 
    
    center_emb = emb.unsqueeze(1) 
    nbr_emb = emb[knn_idx]        
    
    pos_dist = (center_emb - nbr_emb).norm(dim=-1).mean()

    rand_idx = torch.randperm(B, device=emb.device)
    rand_emb = emb[rand_idx]
    
    neg_dist = (emb - rand_emb).norm(dim=-1)
    
    neg_loss = F.relu(margin - neg_dist).mean()
    
    return pos_dist + neg_loss

def variance_loss(z: torch.Tensor, min_std: float = 0.1) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return torch.mean(F.relu(min_std - std))

# ====== Training Functions ======
def pretrain_aes(AES, X_tensors, device, num_epochs=100, batch_size=512):
    print("=== Stage 1: Pre-training AutoEncoders ===")
    N = next(iter(X_tensors.values())).size(0)
    keys = list(X_tensors.keys())
    
    ae_params = []
    for name in keys:
        ae_params += list(AES[name].parameters())
    optimizer_ae = optim.AdamW(ae_params, lr=Config.PRETRAIN_LR, weight_decay=Config.PRETRAIN_WEIGHT_DECAY)

    for ep in range(1, num_epochs + 1):
        idx = np.random.permutation(N)
        total_loss_ep = 0.0
        
        for start in range(0, N, batch_size):
            sel = idx[start:start + batch_size]
            if len(sel) < 2: continue
            
            loss_batch = 0.0
            for name in keys:
                x_cpu = X_tensors[name][sel]
                x = x_cpu.to(device)
                x_flat = x.squeeze(1)
                
                emb, rec = AES[name](x_flat)
                loss_batch += F.mse_loss(rec, x_flat)
            
            optimizer_ae.zero_grad()
            loss_batch.backward()
            optimizer_ae.step()
            total_loss_ep += loss_batch.item()
            
        if ep % 10 == 0:
            print(f"[Pretrain] Epoch {ep}/{num_epochs} | Loss: {total_loss_ep:.4f}")
            
def eval_stage1_only(AES, BLOCKS, FUSE, X_tensors, true_labels, device):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    for m in AES.values():
        m.eval()

    keys = list(X_tensors.keys())
    N = X_tensors[keys[0]].shape[0]

    zs = []
    with torch.no_grad():
        for start in range(0, N, 512):
            end = min(start + 512, N)
            batch_zs = []
            for name in keys:
                x = X_tensors[name][start:end].to(device)    # (B,1,D)
                x_flat = x.squeeze(1)                        # (B,D)
                emb, _ = AES[name](x_flat)                   # (B,emb_dim)
                batch_zs.append(emb.cpu().numpy())
            zs.append(np.concatenate(batch_zs, axis=1))       # (B, emb_dim * num_mod)

    Z_all = np.concatenate(zs, axis=0)                        # (N, emb_total)
    Z_all = np.log1p(np.abs(Z_all)) * np.sign(Z_all)
    Z_all = StandardScaler().fit_transform(Z_all)

    # KMeans
    clusters = KMeans(n_clusters=Config.TARGET_K, random_state=100).fit_predict(Z_all)
    ari_kmeans = adjusted_rand_score(true_labels, clusters)
    print(f"[Stage1 only] KMeans ARI = {ari_kmeans:.4f}")

    # Leiden
    ari_leiden = -1.0
    try:
        clusters_leiden = run_leiden(Z_all, target_k=Config.TARGET_K)
        ari_leiden = adjusted_rand_score(true_labels, clusters_leiden)
        print(f"[Stage1 only] Leiden  ARI = {ari_leiden:.4f}")
    except Exception as e:
        print(f"[Stage1 only] Leiden failed: {e}")
        
    best_ari = max(ari_kmeans, ari_leiden)
    print(f"[Stage1 only] Best ARI = {best_ari:.4f}, saving model (epoch=0) to {Config.BEST_MODEL_NAME} ...")
    
    torch.save({
        'AES': {k: v.state_dict() for k, v in AES.items()},
        'BLOCKS': {k: v.state_dict() for k, v in BLOCKS.items()},
        'FUSE': FUSE.state_dict(),
        'epoch': 0,
        'ari': best_ari
    }, Config.BEST_MODEL_NAME)
    
    return best_ari


def train_joint(AES, BLOCKS, FUSE, X_tensors, coords_all, true_labels, device, num_epochs=200, batch_size=512, start_ari=-1.0):
    print("=== Stage 2: Joint Training (Fusion) ===")
    N = next(iter(X_tensors.values())).size(0)
    keys = list(X_tensors.keys())

    ae_params_set = set()
    ae_params = []
    for name in keys:
        for p in BLOCKS[name].parameters():
            if p not in ae_params_set:
                ae_params.append(p)
                ae_params_set.add(p)
    
    fusion_params = []
    for p in FUSE.parameters():
        if p not in ae_params_set:
            fusion_params.append(p)
            
    gate_net = GatingNetwork(input_dim=Config.EMB_DIM * len(keys), num_modalities=len(keys)).to(device)
    fusion_params += list(gate_net.parameters())
    
    optimizer = optim.AdamW([
        {'params': ae_params, 'lr': Config.TRAIN_LR_AE},
        {'params': fusion_params, 'lr': Config.TRAIN_LR_FUSION}
    ], weight_decay=Config.TRAIN_WEIGHT_DECAY)

    best_ari = start_ari
    
    # Helper for evaluation
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler
    
    def get_fuse_embeddings_eval(batch=512):
        outs = []
        for i in range(0, N, batch):
            xs = [X_tensors[k][i:i+batch].to(device) for k in keys]
            
            if FUSE.method == "weave":
                xs_emb = []
                for k, x in zip(keys, xs):
                    x_flat = x.squeeze(1)
                    emb, _ = AES[k](x_flat)
                    emb = F.normalize(emb, p=2, dim=1)
                    xs_emb.append(emb.unsqueeze(1))
                L = FUSE(xs_emb, return_embeddings=True)
            else:
                L = FUSE(xs, return_embeddings=True)
                
            Z = L.flatten(1)
            outs.append(Z.detach().cpu().numpy())
        Z = np.concatenate(outs, axis=0)
        Z = np.log1p(np.abs(Z)) * np.sign(Z)
        Z = StandardScaler().fit_transform(Z)
        return Z

    for ep in range(1, num_epochs + 1):
        idx = np.random.permutation(N)
        total_loss_ep = 0.0

        for start in range(0, N, batch_size):
            sel = idx[start:start + batch_size]
            if len(sel) < 2: continue

            loss_rec_total = 0.0
            latents_clean = {}

            for name in keys:
                x_cpu = X_tensors[name][sel]
                x = x_cpu.to(device)
                ae = AES[name]
                
                x_flat = x.squeeze(1)
                emb, rec = ae(x_flat)
                loss_rec_total += F.mse_loss(rec, x_flat)
                latents_clean[name] = emb

            for name in keys:
                latents_clean[name] = F.normalize(latents_clean[name], p=2, dim=1)
            
            latents_list = [latents_clean[k] for k in keys]

            contrastive = 0.0
            for i in range(len(latents_list)):
                for j in range(i + 1, len(latents_list)):
                    contrastive += F.mse_loss(latents_list[i], latents_list[j])

            xs_batch = [X_tensors[k][sel].to(device) for k in keys]
            
            if FUSE.method == "weave":
                xs_input = [latents_clean[k].unsqueeze(1) for k in keys]
                fused_L = FUSE(xs_input, return_embeddings=True)
            else:
                fused_L = FUSE(xs_batch, return_embeddings=True)
                
            fused_Z = fused_L.flatten(1)
            
            if not torch.isfinite(fused_Z).all():
                fused_Z = torch.nan_to_num(fused_Z, nan=0.0)
                
            fused_Z = F.normalize(fused_Z, p=2, dim=1)

            z_cat = torch.cat(latents_list, dim=1)
            weights = gate_net(z_cat)
            
            align = 0.0
            for i, z_mod in enumerate(latents_list):
                w = weights[:, i].unsqueeze(1)
                align += (w * (fused_Z - z_mod).pow(2)).sum(dim=1).mean()
            
            avg_weights = weights.mean(dim=0)
            loss_entropy = (avg_weights * torch.log(avg_weights + 1e-6)).sum()

            coords_batch = coords_all[sel].to(device)
            loss_spatial = spatial_contrastive_loss_batch(fused_Z, coords_batch, k=Config.SPATIAL_K, margin=1.0)

            loss_var = variance_loss(fused_Z, min_std=0.1)

            total_loss = (
                Config.COEF_REC * loss_rec_total +
                Config.COEF_CONTRASTIVE * contrastive +
                Config.COEF_ALIGN * align +
                Config.COEF_SPATIAL * loss_spatial +
                Config.COEF_VAR * loss_var + 
                Config.COEF_ENTROPY * loss_entropy
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(ae_params + fusion_params, max_norm=5.0)
            optimizer.step()

            total_loss_ep += total_loss.item()

        if ep % Config.EVAL_INTERVAL == 0:
            print(
                f"[{ep}/{num_epochs}] Total: {total_loss_ep:.3f} | "
                f"Rec: {loss_rec_total.item():.3f} | "
                f"Contr: {contrastive.item():.3f} | "
                f"Align: {align.item():.3f} | "
                f"Spat: {loss_spatial.item():.3f} | "
                f"Var: {loss_var.item():.3f} | "
                f"Ent: {loss_entropy.item():.3f}"
            )
            
            # Evaluation
            for m in AES.values(): m.eval()
            for b in BLOCKS.values(): b.eval()
            FUSE.eval()
            
            Z = get_fuse_embeddings_eval(batch=batch_size)
            
            if Config.CLUSTERING_ALGO == "kmeans":
                from sklearn.cluster import KMeans
                clusters = KMeans(n_clusters=Config.TARGET_K, random_state=100).fit_predict(Z)
                print(f"[eval] Using KMeans clustering (k={Config.TARGET_K})")
            else:
                try:
                    clusters = run_leiden(Z, target_k=Config.TARGET_K)
                    n_c = len(np.unique(clusters))
                    print(f"[eval] Using Leiden clustering (found {n_c} clusters)")
                except Exception as e:
                    print(f"[eval] Leiden failed ({e}), falling back to KMeans")
                    from sklearn.cluster import KMeans
                    clusters = KMeans(n_clusters=Config.TARGET_K, random_state=100).fit_predict(Z)
            
            ari = adjusted_rand_score(true_labels, clusters)
            print(f"[eval] epoch={ep} | ARI={ari:.4f}")
            
            if ari > best_ari:
                best_ari = ari
                print(f"New best ARI: {best_ari:.4f}, saving model to {Config.BEST_MODEL_NAME} ...")
                torch.save({
                    'AES': {k: v.state_dict() for k, v in AES.items()},
                    'BLOCKS': {k: v.state_dict() for k, v in BLOCKS.items()},
                    'FUSE': FUSE.state_dict(),
                    'epoch': ep,
                    'ari': best_ari
                }, Config.BEST_MODEL_NAME)
                
            for m in AES.values(): m.train()
            for b in BLOCKS.values(): b.train()
            FUSE.train()

def main():
    root = Path(__file__).resolve().parent
    configs = json.loads((root / "configs/datasets.json").read_text())
    parser = argparse.ArgumentParser(description="Train ILR-SMO on a main-experiment dataset.")
    parser.add_argument("--dataset", required=True, choices=list(configs))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--pretrain-epochs", type=int)
    parser.add_argument("--train-epochs", type=int)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu.")
    if args.threads < 1:
        parser.error("--threads must be positive.")
    if any(n is not None and n < 0 for n in [args.pretrain_epochs, args.train_epochs]):
        parser.error("Epoch counts must be nonnegative.")
    torch.set_num_threads(args.threads)
    cfg = configs[args.dataset]
    Config.set_dataset(cfg["dataset"])
    Config.DEVICE = args.device
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    Config.BEST_MODEL_NAME = str(out / "best_model_ari.pth")
    if args.pretrain_epochs is not None:
        Config.PRETRAIN_EPOCHS = args.pretrain_epochs
    if args.train_epochs is not None:
        Config.TRAIN_EPOCHS = args.train_epochs

    random.seed(Config.SEED)
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
    features, coords, labels, _ = load_dataset(cfg["dataset"], str(root / cfg["data_path"]), args.device)
    tensors = {name: torch.tensor(arr, device="cpu")[:, None, :] for name, arr in features.items()}
    coords = torch.tensor(coords, device="cpu", dtype=torch.float32)
    AES, BLOCKS, FUSE = initialize_model(features, args.device)
    pretrain_aes(AES, tensors, args.device, num_epochs=Config.PRETRAIN_EPOCHS, batch_size=Config.PRETRAIN_BATCH_SIZE)
    best = eval_stage1_only(AES, BLOCKS, FUSE, tensors, labels, args.device)
    train_joint(AES, BLOCKS, FUSE, tensors, coords, labels, args.device,
                num_epochs=Config.TRAIN_EPOCHS, batch_size=Config.TRAIN_BATCH_SIZE, start_ari=best)


if __name__ == "__main__":
    main()
