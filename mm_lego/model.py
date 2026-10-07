import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
import h5py
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import pandas as pd

# mm_lego
from mm_lego.models import LegoBlock, LegoMerge, LegoFuse
from mm_lego.config import Config

# ====== AutoEncoders ======
class ImageAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, output_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)  # [B,1,D]
        emb = self.encoder(x)
        rec = self.decoder(emb)
        return emb, rec

    @torch.no_grad()
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        return self.encoder(x)

class MLP_AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 128):
        super().__init__()
        enc_layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        enc_layers += [nn.Linear(prev, output_dim)]
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: List[nn.Module] = []
        prev = output_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(x)
        rec = self.decoder(emb)
        return emb, rec

    @torch.no_grad()
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class AEEncoderWrapper(nn.Module):
    def __init__(self, ae: MLP_AutoEncoder, emb_dim: int):
        super().__init__()
        self.ae = ae
        self.proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, data, **kwargs):
        if isinstance(data, (list, tuple)):
            x = data[0]
        else:
            x = data
            
        if x.dim() == 3:
            B, C, D = x.shape
            x_flat = x.view(B, C * D)
        elif x.dim() == 2:
            x_flat = x
        else:
             raise ValueError(f"Unexpected input shape: {x.shape}")

        emb, _ = self.ae(x_flat)
        emb = self.proj(emb)
        return emb.unsqueeze(1) # (B, 1, emb_dim)


class ImgAEEncoderWrapper(nn.Module):
    def __init__(self, ae: ImageAutoEncoder, emb_dim: int):
        super().__init__()
        self.ae = ae
        self.proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, data, **kwargs):
        if isinstance(data, (list, tuple)):
            x = data[0]
        else:
            x = data
            
        if x.dim() == 3:
            B, C, D = x.shape
            x_flat = x.view(B, C * D)
        elif x.dim() == 2:
            x_flat = x
        else:
             raise ValueError(f"Unexpected input shape: {x.shape}")

        emb, _ = self.ae(x_flat)
        emb = self.proj(emb)
        return emb.unsqueeze(1) # (B, 1, emb_dim)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_modalities=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_modalities),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

# ====== Utils ======
def compute_hidden_dims(input_dim: int, out_dim: int, max_layers: int = 2) -> List[int]:
    hidden = []
    cur = min(1024, max(out_dim*4, input_dim // 2))
    for _ in range(max_layers):
        if cur <= out_dim: break
        hidden.append(cur)
        cur = max(out_dim*2, cur // 2)
    if not hidden:
        hidden = [max(out_dim*2, input_dim // 2)]
    return hidden

def run_leiden(data, target_k=10):
    # Create temporary AnnData
    adata_temp = sc.AnnData(data)
    sc.pp.neighbors(adata_temp, use_rep='X', n_neighbors=15)
    
    # Binary search for resolution
    res_min, res_max = 0.1, 2.0
    best_res = 0.5
    best_k_diff = 999
    
    # Quick search
    for _ in range(10): 
        res = (res_min + res_max) / 2
        try:
            sc.tl.leiden(adata_temp, resolution=res, key_added='leiden')
        except:
             sc.tl.leiden(adata_temp, resolution=res, key_added='leiden', flavor='igraph', n_iterations=2, directed=False)
             
        n_clusters = len(adata_temp.obs['leiden'].unique())
        
        if n_clusters == target_k:
            best_res = res
            break
        elif n_clusters > target_k:
            res_max = res
        else:
            res_min = res
            
        if abs(n_clusters - target_k) < best_k_diff:
            best_k_diff = abs(n_clusters - target_k)
            best_res = res
            
    try:
        sc.tl.leiden(adata_temp, resolution=best_res, key_added='leiden')
    except:
        sc.tl.leiden(adata_temp, resolution=best_res, key_added='leiden', flavor='igraph', n_iterations=2, directed=False)
        
    return adata_temp.obs['leiden'].astype(int).values

# ====== Data Loading ======

def load_dataset(dataset_name: str, data_path: str, device: str = 'cuda'):
    features_dict = {}
    coords = None
    true_labels = None
    adata_rna = None
    
    if dataset_name == "MVC_simulated":
        # Load CSVs
        counts_path = os.path.join(data_path, 'MVC_counts.csv')
        meta_path = os.path.join(data_path, 'MVC_meta.csv')
        
        df_data = pd.read_csv(counts_path, sep=",", header=0, na_filter=False, index_col=0)
        df_meta = pd.read_csv(meta_path, sep=",", header=0, na_filter=False, index_col=0)
        
        df_pixels = df_meta.iloc[:,2:4]
        df_labels = list(df_meta.iloc[:,1])
        
        adata = sc.AnnData(X = df_data)
        adata.obs['LayerName'] = df_labels
        adata.obsm['spatial'] = np.array(df_pixels)
        
        label_type = ['L1','L2/3','L4','L5','L6','HPC/CC']
        
        # Simulation Logic
        index_all = [np.array([i for i in range(len(df_labels)) if df_labels[i] == label_type[0]])]
        for k in range(1,len(label_type)):
            temp_idx = np.array([i for i in range(len(df_labels)) if df_labels[i] == label_type[k]])
            index_all.append(temp_idx)
        index_int1 = np.array(list(index_all[2]) + list(index_all[3]))
        index_int2 = np.array(list(index_all[4]) + list(index_all[3]))

        random_seed = 42
        adata1 = adata.copy()
        np.random.seed(random_seed)
        data_noise_1 = 1 + np.random.normal(0,0.05,adata.shape)
        X_arr = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
        adata1.X = X_arr.copy()
        adata1.X[index_int1,:] = np.multiply(X_arr, data_noise_1)[np.random.permutation(index_int1),:]

        adata2 = adata.copy()
        np.random.seed(random_seed+1)
        data_noise_2 = 1 + np.random.normal(0,0.05,adata.shape)
        adata2.X = X_arr.copy()
        adata2.X[index_int2,:] = np.multiply(X_arr, data_noise_2)[np.random.permutation(index_int2),:]

        # Preprocessing
        from gsmo.utils import preprocess
        feat_mod1 = preprocess(adata1, modality='rna')
        feat_mod2 = preprocess(adata2, modality='rna')
        
        # Standardization
        def zscore(x):
            sca = StandardScaler().fit(x)
            return sca.transform(x).astype('float32')

        feat_mod1_z = zscore(feat_mod1)
        feat_mod2_z = zscore(feat_mod2)

        features_dict = {
            "mod1": feat_mod1_z,
            "mod2": feat_mod2_z
        }
        
        coords = adata.obsm['spatial']
        
        # Convert labels to integers
        unique_labels = np.unique(df_labels)
        label_map = {l: i for i, l in enumerate(unique_labels)}
        true_labels = np.array([label_map[l] for l in df_labels])
        
        adata_rna = adata1

    elif dataset_name == "MouseBrain":
        # Existing logic for Mouse Brain
        h5_path = os.path.join(data_path, 'ATAC_RNA_Seq_MouseBrain_RNA_ATAC.h5')
        data_mat = h5py.File(h5_path, 'r')

        X_RNA  = np.array(data_mat['X_RNA']).astype('float64')
        X_ATAC = np.array(data_mat['X_ATAC']).astype('float64')
        loc    = np.array(data_mat['Pos']).astype('float64')
        LayerName = [s.decode("utf-8") for s in list(data_mat['LayerName'])]

        adata_rna = sc.AnnData(X_RNA, dtype="float64")
        adata_rna.obsm['spatial'] = loc
        adata_rna.obs['LayerName'] = LayerName

        adata_atac = sc.AnnData(X_ATAC, dtype="float64")
        adata_atac.obsm['spatial'] = loc
        adata_atac.obs['LayerName'] = LayerName

        # Preprocess
        from gsmo.utils import preprocess
        # Check if processed files exist to save time
        rna_proc_path = os.path.join(data_path, 'rna_processed.h5ad')
        atac_proc_path = os.path.join(data_path, 'atac_processed.h5ad')
        
        if os.path.exists(rna_proc_path):
            adata_rna2 = sc.read(rna_proc_path)
        else:
            adata_rna_proc = preprocess(adata_rna, modality='rna', save_path=rna_proc_path)
            adata_rna2 = sc.read(rna_proc_path)
            
        if os.path.exists(atac_proc_path):
            adata_atac2 = sc.read(atac_proc_path)
        else:
            adata_atac_proc = preprocess(adata_atac, modality='atac', n_dim=50, save_path=atac_proc_path)
            adata_atac2 = sc.read(atac_proc_path)

        feat_rna  = adata_rna2.X.toarray() if hasattr(adata_rna2.X, "toarray") else adata_rna2.X
        feat_atac = adata_atac2.obsm['X_lsi'][:, :50]

        # Standardization
        scalers = {}
        def zscore(x):
            sca = StandardScaler().fit(x)
            return sca.transform(x).astype('float32')

        feat_rna_z  = zscore(feat_rna)
        feat_atac_z = zscore(feat_atac)

        features_dict = {
            "rna":  feat_rna_z,
            "atac": feat_atac_z,
        }
        coords = loc
        true_labels = np.array(LayerName)
        
    elif dataset_name == "HumanLymphNodeNoImage":
        # Logic for Human Lymph Node (No Image)
        
        # Load RNA
        rna_path = os.path.join(data_path, "adata_RNA.h5ad")
        rna = sc.read(rna_path)
        
        # Load Positions
        pos_path = os.path.join(data_path, "GSM8195494_A1LN_tissue_positions.csv.gz")
        positions = pd.read_csv(pos_path, header=None)
        positions.columns = ["barcode", "in_tissue", "array_row", "array_col", "image_row", "image_col"]
        positions = positions.set_index("barcode")
        
        common_barcodes = rna.obs_names.intersection(positions.index)
        rna = rna[common_barcodes]
        positions = positions.loc[common_barcodes]
        
        coords = positions[["image_col", "image_row"]].astype(float).values
        
        # Process RNA (High Var Genes)
        sc.pp.filter_genes(rna, min_cells=10)
        sc.pp.highly_variable_genes(rna, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(rna, target_sum=1e4)
        sc.pp.log1p(rna)
        sc.pp.scale(rna)
        adata_omics1_high = rna[:, rna.var['highly_variable']]
        
        # Process Protein
        protein_path = os.path.join(data_path, 'adata_ADT.h5ad')
        protein = sc.read(protein_path)
        
        # CLR Normalization
        def clr_normalize_each_cell(adata, inplace=True):
            import scipy
            def seurat_clr(x):
                s = np.sum(np.log1p(x[x > 0]))
                exp = np.exp(s / len(x))
                return np.log1p(x / exp)
            if not inplace: adata = adata.copy()
            adata.X = np.apply_along_axis(
                seurat_clr, 1, (adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X))
            )
            return adata
            
        protein = clr_normalize_each_cell(protein)
        sc.pp.scale(protein)
        
        features_dict = {
            'rna': adata_omics1_high.X.astype(np.float32) if not hasattr(adata_omics1_high.X, "toarray") else adata_omics1_high.X.toarray().astype(np.float32),
            'protein': protein.X.astype(np.float32) if not hasattr(protein.X, "toarray") else protein.X.toarray().astype(np.float32),
        }
        
        # Load Annotations
        anno_path = os.path.join(data_path, "annotation.csv")
        df = pd.read_csv(anno_path, sep=',')
        true_labels = pd.factorize(df['manual-anno'])[0]
        
        adata_rna = rna # For plotting later

    elif dataset_name == "HumanHippocampus":
        # Load RNA
        rna_path = os.path.join(data_path, "Human_RNA.h5ad")
        rna = sc.read(rna_path)
        
        # Load ATAC
        atac_path = os.path.join(data_path, "Human_ATAC_lsi.h5ad")
        atac = sc.read(atac_path)
        
        # Load Labels
        label_path = os.path.join(data_path, "Human_RNA_true_label.h5ad")
        label_adata = sc.read(label_path)
        true_labels = label_adata.obs["true_label"].values
        # Convert to int if they are strings/categorical
        if true_labels.dtype == 'O' or isinstance(true_labels[0], str) or hasattr(true_labels, 'categories'):
             true_labels = pd.factorize(true_labels)[0]
        
        # Coords
        coords = rna.obsm["spatial"].copy()
        coords[:, 1] *= -1
        
        # Process RNA
        if 'highly_variable' in rna.var:
             rna = rna[:, rna.var['highly_variable']]
        else:
             sc.pp.filter_genes(rna, min_cells=10)
             sc.pp.highly_variable_genes(rna, flavor="seurat_v3", n_top_genes=3000)
             rna = rna[:, rna.var['highly_variable']]
             
        if 'log1p' not in rna.uns:
            sc.pp.normalize_total(rna, target_sum=1e4)
            sc.pp.log1p(rna)
            
        sc.pp.scale(rna)
        
        # Process ATAC
        from gsmo.utils import preprocess
        atac_proc_path = os.path.join(data_path, 'atac_processed.h5ad')
        
        if os.path.exists(atac_proc_path):
            atac_proc = sc.read(atac_proc_path)
            if 'X_lsi' in atac_proc.obsm:
                feat_atac = atac_proc.obsm['X_lsi'][:, :50]
            else:
                feat_atac = preprocess(atac, modality='atac', n_dim=50, save_path=atac_proc_path)
        else:
            feat_atac = preprocess(atac, modality='atac', n_dim=50, save_path=atac_proc_path)
            
        # Standardization
        def zscore(x):
            sca = StandardScaler().fit(x)
            return sca.transform(x).astype('float32')

        feat_rna = rna.X.toarray() if hasattr(rna.X, "toarray") else rna.X
        feat_rna_z = zscore(feat_rna)
        feat_atac_z = zscore(feat_atac)
        
        features_dict = {
            "rna": feat_rna_z,
            "atac": feat_atac_z
        }
        
        adata_rna = rna

    elif dataset_name == "TripletOmics":
        from gsmo.utils import preprocess
        
        # Load RNA
        rna_path = os.path.join(data_path, "adata_RNA.h5ad")
        rna = sc.read(rna_path)
        rna.var_names_make_unique()
        feat_rna = preprocess(rna, modality='rna')
        
        # Load ATAC
        atac_path = os.path.join(data_path, "adata_ATAC.h5ad")
        atac = sc.read(atac_path)
        atac.var_names_make_unique()
        feat_atac = preprocess(atac, modality='atac', n_dim=1000)
        
        # Load Protein
        protein_path = os.path.join(data_path, "adata_ADT.h5ad")
        protein = sc.read(protein_path)
        protein.var_names_make_unique()
        feat_protein = preprocess(protein, modality='protein')
        
        # Load Labels
        label_path = os.path.join(data_path, "ground_truth_clusters.csv")
        df = pd.read_csv(label_path)
        true_labels = df['cluster'].values
        
        # Coords
        coords = rna.obsm['spatial'].copy()
        
        # Standardization
        def zscore(x):
            sca = StandardScaler().fit(x)
            return sca.transform(x).astype('float32')
            
        features_dict = {
            "rna": zscore(feat_rna),
            "atac": zscore(feat_atac),
            "protein": zscore(feat_protein)
        }
        
        adata_rna = rna

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    return features_dict, coords, true_labels, adata_rna

def initialize_model(features_dict: Dict[str, np.ndarray], device: str):
    BLOCKS = {}
    AES = {}
    IN_SHAPES = {}  
    
    # Hardcoded for now as per user request
    FUSE_METHOD = "weave"

    for name, arr in features_dict.items():
        D = arr.shape[1]

        if name.lower() == "image":
            ae = ImageAutoEncoder(input_dim=D, output_dim=Config.EMB_DIM).to(device)
            encoder_for_lego = ImgAEEncoderWrapper(ae, emb_dim=Config.EMB_DIM).to(device)
        else:
            hidden = compute_hidden_dims(D, Config.EMB_DIM, max_layers=Config.HIDDEN_DIMS_MAX_LAYERS)
            ae = MLP_AutoEncoder(input_dim=D, hidden_dims=hidden, output_dim=Config.EMB_DIM).to(device)
            encoder_for_lego = AEEncoderWrapper(ae, emb_dim=Config.EMB_DIM).to(device)

        AES[name] = ae
        IN_SHAPES[name] = (1, D)
        
        if FUSE_METHOD == "weave":
            # For weave, we pass embeddings, so block input is (1, EMB_DIM) and no encoder
            block = LegoBlock(
                in_shape=(1, Config.EMB_DIM),
                encoder=None,
                l_c=1,
                l_d=Config.EMB_DIM,
                frequency_domain=False
            ).to(device)
        else:
            block = LegoBlock(
                in_shape=IN_SHAPES[name],
                encoder=encoder_for_lego,
                l_c=1,
                l_d=Config.EMB_DIM,
                frequency_domain=False
            ).to(device)
            
        BLOCKS[name] = block

    FUSE = LegoFuse(
        blocks=[BLOCKS[name] for name in features_dict.keys()],
        fuse_method=FUSE_METHOD,
        head_method="slerp"
    ).to(device)
    
    return AES, BLOCKS, FUSE
