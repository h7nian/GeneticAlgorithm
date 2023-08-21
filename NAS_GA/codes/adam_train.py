import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc

from utils import cal_fitness

global best_ari,best_nmi

best_ari = 0
best_nmi = 0

def Adam_Train(model,dataloader,adata,G,clusters_true,args,device):
    
    model.to(device)
    
    model.train()
    
    best_ari = best_nmi = 0
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,50,args.decay_factor)
    
    if args.IFVQVAE:
        
        for epoch in range(args.Adam_epochs):

            total_loss = 0
            
            for batch_idx,X_batch in enumerate(dataloader):
                
                X_batch = X_batch.to(device).float()
                optimizer.zero_grad()

                x_q,z_e,z_q  = model(X_batch)

                loss = model.loss(X_batch.float(),x_q.float(),z_e,z_q)
                
                total_loss += loss
                
                loss.backward()
                
                optimizer.step()
                scheduler.step()
                
            ari,nmi = cal_fitness(model,dataloader,clusters_true,adata,device,args)
            
            if args.ari < ari and args.nmi < nmi:
                
                args.ari = ari
                
                args.nmi = nmi
                
            print(f"epoch:{epoch+1} total_loss:{total_loss/len(dataloader.sampler)} \n"
                    f"ari:{ari} nmi:{nmi} \n"
                    f"best ari:{args.ari} best nmi:{args.nmi}")
            
        model.cpu()    
    
    else:
        
        for epoch in range(args.Adam_epochs):

            total_loss = 0
            
            for batch_idx,X_batch in enumerate(dataloader):
                
                X_batch = X_batch.to(device).float()
                optimizer.zero_grad()

                x_e, z_e = model(X_batch)

                loss = model.loss(X_batch.float(), x_e.float())
                
                total_loss += loss
                
                loss.backward()
                
                optimizer.step()
                scheduler.step()
                
            ari,nmi = cal_fitness(model,dataloader,clusters_true,adata,device,args)
            
            if args.ari < ari and args.nmi < nmi:
                
                args.ari = ari
                
                args.nmi = nmi
                
            print(f"epoch:{epoch+1} total_loss:{total_loss/len(dataloader.sampler)} \n"
                    f"ari:{ari} nmi:{nmi} \n"
                    f"best ari:{args.ari} best nmi:{args.nmi}")
            
        model.cpu()    
        
    return model