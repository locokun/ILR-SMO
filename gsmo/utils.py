import numpy as np
import scipy
import scanpy as sc

def tfidf_transform(adata):
    """TF-IDF transformation for sparse ATAC matrix"""
    from sklearn.preprocessing import normalize

    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    idf = np.log(1 + X.shape[0] / (1 + np.sum(X > 0, axis=0)))  # inverse document freq
    tf = normalize(X, norm='l1', axis=1)  # term frequency
    tfidf = tf * idf
    return tfidf

def lsi_from_tfidf(adata, n_components=3000):
    """Perform TF-IDF + TruncatedSVD to compute LSI"""
    from sklearn.decomposition import TruncatedSVD
    tfidf = tfidf_transform(adata)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    lsi = svd.fit_transform(tfidf)
    return lsi

def preprocess(adata, modality, n_dim=1024, save_path=None):
    """
    Preprocess adata based on modality.
    If save_path is provided, save the processed adata to a .h5ad file.
    """
    adata.var_names_make_unique()

    if modality == 'rna':
        sc.pp.filter_genes(adata, min_cells=10)

        # 只在有 "vf_vst_counts_variable" 标志的情况下筛选高变基因
        if 'vf_vst_counts_variable' in adata.var.columns:
            hvg_mask = adata.var['vf_vst_counts_variable'].values.astype(bool)
            adata = adata[:, hvg_mask]

        if not hasattr(adata, 'raw') or adata.raw is None:
            sc.pp.log1p(adata)
        
        # 保护机制：在scale之前过滤掉标准差为0的基因
        X_dense = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
        gene_std = np.std(X_dense, axis=0)
        non_zero_std_genes = np.where(gene_std > 1e-6)[0]  # 保留标准差大于很小阈值的基因
        adata = adata[:, non_zero_std_genes]  # 只保留这些基因

        sc.pp.scale(adata)  # 现在scale不会炸了

        # 最保险：scale完再清理一遍NaN（虽然应该没有了）
        X_dense = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
        X_dense = np.nan_to_num(X_dense, nan=0.0, posinf=1e5, neginf=-1e5)

        if save_path is not None:
            adata.write_h5ad(save_path)
            print(f"✅ Processed RNA data saved to {save_path}")
        
        return X_dense

    elif modality in ['atac', 'histone', 'H3K27ac', 'H3K27me3', 'H3K4me3']:
        if 'X_lsi' in adata.obsm:
            lsi = adata.obsm['X_lsi']
            if lsi.shape[1] >= n_dim:
                if save_path is not None:
                    adata.write_h5ad(save_path)
                    print(f"✅ ATAC/histone data (existing LSI) saved to {save_path}")
                return lsi[:, :n_dim]
            else:
                print(f"⚠️ obsm['X_lsi'] has only {lsi.shape[1]} dims, recomputing LSI.")
        else:
            print("⚠️ X_lsi not found, performing LSI in Python from count matrix.")

        lsi = lsi_from_tfidf(adata, n_components=n_dim)
        adata.obsm['X_lsi'] = lsi

        if save_path is not None:
            adata.write_h5ad(save_path)
            print(f"✅ Recomputed and saved ATAC/histone data with new LSI to {save_path}")

        return lsi

    elif modality == 'protein':
        if scipy.sparse.issparse(adata.X):
            adata.X = adata.X.toarray()
        sc.pp.scale(adata)

        if save_path is not None:
            adata.write_h5ad(save_path)
            print(f"✅ Protein data saved to {save_path}")

        return adata.X
    elif modality == "metabolite":
        sc.pp.log1p(adata)
        if scipy.sparse.issparse(adata.X):
            adata.X = adata.X.toarray()
        if save_path is not None:
            adata.write_h5ad(save_path)
            print(f"✅ metabolite data saved to {save_path}")
        return adata.X

    else:
        raise ValueError(f"Unsupported modality: {modality}")
