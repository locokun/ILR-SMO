import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score, homogeneity_score, mutual_info_score,
    v_measure_score, adjusted_mutual_info_score, normalized_mutual_info_score,
)
from mm_lego.config import Config
from mm_lego.model import load_dataset, initialize_model, run_leiden

def get_concat_embeddings(AES, X_tensors, device, batch=512):
    for ae in AES.values():
        ae.eval()
        
    outs = []
    N = next(iter(X_tensors.values())).size(0)
    keys = list(X_tensors.keys())
    with torch.no_grad():
        for i in range(0, N, batch):
            batch_zs = []
            for k in keys:
                x = X_tensors[k][i:i+batch].to(device)
                x_flat = x.squeeze(1)
                emb, _ = AES[k](x_flat)
                # Note: No normalization here to match eval_stage1_only
                batch_zs.append(emb.cpu().numpy())
            outs.append(np.concatenate(batch_zs, axis=1))
            
    Z = np.concatenate(outs, axis=0)
    Z = np.log1p(np.abs(Z)) * np.sign(Z)
    Z = StandardScaler().fit_transform(Z)
    return Z

def get_fuse_embeddings(FUSE, AES, X_tensors, device, batch=512):
    FUSE.eval()
    for ae in AES.values():
        ae.eval()
        
    outs = []
    N = next(iter(X_tensors.values())).size(0)
    keys = list(X_tensors.keys())
    with torch.no_grad():
        for i in range(0, N, batch):
            xs = [X_tensors[k][i:i+batch].to(device) for k in keys]
            
            if hasattr(FUSE, 'method') and FUSE.method == "weave":
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

def main():
    root = Path(__file__).resolve().parent
    configs = json.loads((root / "configs/datasets.json").read_text())
    parser = argparse.ArgumentParser(description="Test an ILR-SMO checkpoint.")
    parser.add_argument("--dataset", required=True, choices=list(configs))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-csv", type=Path, help="Optionally save the six raw clustering scores.")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu.")
    if args.threads < 1:
        parser.error("--threads must be positive.")
    if args.output_csv and args.output_csv.exists():
        parser.error("--output-csv must point to a new file.")
    torch.set_num_threads(args.threads)
    cfg = configs[args.dataset]
    Config.set_dataset(cfg["dataset"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if args.device == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])

    features, _, labels, _ = load_dataset(cfg["dataset"], str(root / cfg["data_path"]), args.device)
    AES, BLOCKS, FUSE = initialize_model(features, args.device)
    checkpoint = torch.load(args.checkpoint or root / cfg["checkpoint"], map_location=args.device, weights_only=True)
    for name, state in checkpoint["AES"].items():
        AES[name].load_state_dict(state, strict=True)
    for name, state in checkpoint["BLOCKS"].items():
        BLOCKS[name].load_state_dict(state, strict=True)
    FUSE.load_state_dict(checkpoint["FUSE"], strict=True)
    tensors = {name: torch.tensor(arr, device="cpu")[:, None, :] for name, arr in features.items()}
    if checkpoint.get("epoch") == 0:
        Z = get_concat_embeddings(AES, tensors, args.device, batch=cfg["batch_size"])
        print("Using Stage 1 concatenated AE embeddings.")
    else:
        Z = get_fuse_embeddings(FUSE, AES, tensors, args.device, batch=cfg["batch_size"])
    if cfg["cluster_method"] == "kmeans":
        clusters = KMeans(n_clusters=cfg["target_k"], random_state=100).fit_predict(Z)
    else:
        clusters = run_leiden(Z, target_k=cfg["target_k"])
    scores = {
        "ARI": adjusted_rand_score(labels, clusters),
        "Homogeneity": homogeneity_score(labels, clusters),
        "MI": mutual_info_score(labels, clusters),
        "V_measure": v_measure_score(labels, clusters),
        "AMI": adjusted_mutual_info_score(labels, clusters),
        "NMI": normalized_mutual_info_score(labels, clusters),
    }
    print(args.dataset)
    for name, value in scores.items():
        print(f"{name}: {value:.10f}")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset"] + list(scores))
            writer.writeheader()
            writer.writerow(dict(dataset=args.dataset, **scores))


if __name__ == "__main__":
    main()
