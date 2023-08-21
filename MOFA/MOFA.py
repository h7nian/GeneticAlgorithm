from orion.client import build_experiment
import anndata
import torch
import torch.optim as optim
import numpy as np
import anndata
import scvi
import random
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans
import multiprocessing


def processing(adata):
    
    sc.pp.filter_cells(adata, min_genes=20)
    sc.pp.filter_genes(adata, min_cells=3)
    
    data = adata.copy()
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    data = data[:, adata.var.highly_variable]
    
    return data

def metrics(vae,adata,clusters_true):
    
    mark = False
    
    adata.obsm["X_scVI"] = vae.get_latent_representation()
    adata.obsm["X_normalized_scVI"] = vae.get_normalized_expression()
    
    sc.pp.neighbors(adata,use_rep='X_scVI', random_state=123)
    sc.tl.leiden(adata, resolution=0.5, random_state=123)
    clusters_pred = adata.obs['leiden'].astype('int').values
    
    ari = adjusted_rand_score(clusters_true, clusters_pred)
    nmi = adjusted_mutual_info_score(clusters_true,clusters_pred)
    
    print(f"Current ARI:{ari}, Current NMI:{nmi}")
    
    return mark

def scVITraining(lr,n_hidden,n_latent,n_layers,dropout_rate,batch_size,epochs,noise=None):
    
    device = 'cuda:2'
    
    torch.set_float32_matmul_precision('high')
    
    adata = anndata.read_h5ad('SinianZhang/Mcgill/GAVAE/data/cortex.h5ad')
    adata = processing(adata)
    
    clusters_true = adata.obs['label']
    
    adata = adata.copy()
    
    scvi.model.SCVI.setup_anndata(adata)
    
    vae = scvi.model.SCVI(adata,n_hidden=n_hidden,n_latent=n_latent,n_layers=n_layers,dropout_rate=dropout_rate)
    vae.train(plan_kwargs={'lr':lr},use_gpu=device,batch_size=batch_size,max_epochs=epochs)
    
    mark = metrics(vae,adata,clusters_true)
    
    history = vae.history
    reconstruction_loss = history['reconstruction_loss_train'].iloc[-1]

    return [{"name": "objective", "type": "objective", "value": reconstruction_loss[0]}]

if __name__ == "__main__":
    
    random.seed(20)
    np.random.seed(20)
    torch.manual_seed(20)
    
    storage = {
    "type": "legacy",
    "database": {
        "type": "pickleddb",
        "host": "./db.pkl",
    },}
    
    space = {"lr": "uniform(1e-4,1e-2)",
         "n_hidden":"uniform(50,500,discrete=True)",
         "n_latent":"uniform(5,50,discrete=True)",
         "epochs":"uniform(300,600,discrete=True)",
         "n_layers":"uniform(1,2,discrete=True)",
         "batch_size":"uniform(16,512,discrete=True)",
         "dropout_rate":"uniform(0,0.2)"}
    
    experiment = build_experiment(
    # "TEPscVI20",
    # space=space,
    "MOFAscVI20",
    algorithm={"MOFA":{"n_levels":7}},
    space=space,
    # algorithms={"tpe": {"n_initial_points": 5}},
    storage=storage)
    
    multiprocessing.set_start_method('spawn')
    
    experiment.workon(scVITraining, max_trials=100)


