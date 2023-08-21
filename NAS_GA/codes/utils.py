import torch
from torch.utils.data import Dataset,DataLoader
import scanpy as sc

from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.model_selection import train_test_split

class MyDataset(Dataset):
    def __init__(self,gene_matrix):
        
        self.gene_matrix = torch.from_numpy(gene_matrix)
    
    def __len__(self):
        
        return self.gene_matrix.shape[0]

    def __getitem__(self,idx):
        
        X = self.gene_matrix[idx]
        
        return X
    
def Create_Dataloader(args):
    
    adata = sc.read(args.data_path,dtype='float64')
    
    gene_matrix = adata.X
    
    dataset = MyDataset(gene_matrix)
    dataloader = DataLoader(dataset=dataset,batch_size=args.batch_size,shuffle=True)
    
    return adata,dataloader


def cal_fitness(model,dataloader,clusters_true,adata,device,args):
    
    model.eval()
    
    gene_matrix = torch.from_numpy(adata.X).to(device)
    
    if args.IFVQVAE:
        x_q,z_e,z_q  = model(gene_matrix)
        
    else:
        x_e,z_e = model(gene_matrix)
    
    z_e = z_e.detach().cpu().numpy()
    
    adata.obsm['X_unifan'] = z_e
    
    sc.pp.neighbors(adata, n_pcs=32,use_rep='X_unifan', random_state=123)
    
    sc.tl.leiden(adata, resolution=1, random_state=123)
    clusters_pre = adata.obs['leiden'].astype('int').values 
    
    ari = adjusted_rand_score(clusters_pre, clusters_true)
    nmi = adjusted_mutual_info_score(clusters_pre, clusters_true)
    
    model.train()

    return ari,nmi

def cal_loss(model,dataloader,adata,device,non_blocking=True):
    
    model.to(device)
    
    total_loss = 0.0
    
    for batch_idx, X_batch in enumerate(dataloader):
        
        X_batch = X_batch.to(device, non_blocking=non_blocking).float()

        x_e, z_e = model(X_batch)

        l = model.loss(X_batch,x_e)
        
        total_loss += l
        
    model.cpu()

    return float(total_loss)