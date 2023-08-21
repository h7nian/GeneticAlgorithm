import torch
import torch.optim as optim
import numpy as np
import random
import anndata
import argparse
import time
import scvi
import optuna
import scanpy as sc
from orion.client import build_experiment
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans
import multiprocessing

from utils import processing

def GA_Search(pop_hyperparams,adata,clusters_true,args,device):
    
    fitness_values = []
    
    adata = adata.copy()

    for hyperparams in pop_hyperparams:
        
        scvi.model.SCVI.setup_anndata(adata)
        
        if args.use_scvi:
            vae = scvi.model.SCVI(adata)
            vae.train()
        else:
            vae = scvi.model.SCVI(adata,n_hidden=hyperparams['n_hidden'],n_latent=hyperparams['n_latent'],n_layers=hyperparams['n_layers'],dropout_rate=hyperparams['dropout_rate'])
            vae.train(max_epochs=hyperparams['epochs'],use_gpu=device,batch_size=hyperparams['batch_size'],log_every_n_steps=args.log_every_n_steps,plan_kwargs={'lr':hyperparams['learning_rate']})
    
        mark = metrics(vae,adata,clusters_true,args)

        history = vae.history
        reconstruction_loss = history['reconstruction_loss_train'].iloc[-1]['reconstruction_loss_train']
        fitness_values.append(reconstruction_loss)
    
    new_pop_hyperparams = crossover(pop_hyperparams, args)
    new_pop_hyperparams = mutation(new_pop_hyperparams, args)
    new_pop_hyperparams = selection(new_pop_hyperparams, fitness_values)

    return new_pop_hyperparams

def selection(pop_hyperparams, fitness_values):
    
    selected_indices = np.argsort(fitness_values)[:len(pop_hyperparams) // 2]
    new_pop_hyperparams = [pop_hyperparams[i] for i in selected_indices]
    new_pop_hyperparams = new_pop_hyperparams * 2
    
    return new_pop_hyperparams

def crossover(pop_hyperparams, args):
    
    random.shuffle(pop_hyperparams)
    
    for i in range(0, len(pop_hyperparams), 2):
        
        if random.random() < args.crossover_prob:
            crossover_point = random.choice(['n_hidden', 'n_latent', 'n_layers', 'batch_size', 'learning_rate','epochs','dropout_rate'])
            pop_hyperparams[i][crossover_point], pop_hyperparams[i+1][crossover_point] = pop_hyperparams[i+1][crossover_point], pop_hyperparams[i][crossover_point]
            
    return pop_hyperparams

def mutation(pop_hyperparams, args):
    
    for i, hyperparams in enumerate(pop_hyperparams):
        if random.random() < args.mutation_prob:
            mutation_choice = random.choice(['n_hidden', 'n_latent', 'n_layers', 'batch_size', 'learning_rate','dropout_rate'])
            if mutation_choice == 'n_hidden':
                hyperparams['n_hidden'] = np.random.randint(50,500)
            elif mutation_choice == 'n_latent':
                hyperparams['n_latent'] = np.random.randint(5, 50)
            elif mutation_choice == 'n_layers':
                hyperparams['n_layers'] = np.random.randint(1, 2)
            elif mutation_choice == 'batch_size':
                hyperparams['batch_size'] = np.random.randint(16,512)
            elif mutation_choice == 'learning_rate':
                hyperparams['learning_rate'] = np.random.uniform(1e-4, 1e-2)
            elif mutation_choice == 'epochs':
                hyperparams['epochs'] = np.random.randint(300,600)
            elif mutation_choice == 'dropout_rate':
                hyperparams['dropout_rate'] = np.random.uniform(0,0.2)

    return pop_hyperparams

def Grid_Search(pop_hyperparams,adata,clusters_true,args,device):
        
    adata = adata.copy()

    for hyperparams in pop_hyperparams:
        
        scvi.model.SCVI.setup_anndata(adata)
        
        if args.use_scvi:
            vae = scvi.model.SCVI(adata)
            vae.train()
        else:
            vae = scvi.model.SCVI(adata,n_hidden=hyperparams['n_hidden'],n_latent=hyperparams['n_latent'],n_layers=hyperparams['n_layers'],dropout_rate=hyperparams['dropout_rate'])
            vae.train(max_epochs=hyperparams['epochs'],use_gpu=device,batch_size=hyperparams['batch_size'],log_every_n_steps=args.log_every_n_steps,plan_kwargs={'lr':hyperparams['learning_rate']})
    
        mark = metrics(vae,adata,clusters_true,args)
    
        if mark:
            
            best_ind = hyperparams
    
    return best_ind

def Greedy_Search(adata,clusters_true,args,device):
        
    adata = adata.copy()
    scvi.model.SCVI.setup_anndata(adata)
    
    default = {
        'n_hidden': 128,
        'n_latent': 10,
        'n_layers': 1,
        'batch_size': 128,
        'learning_rate': 0.001,
        'epochs': int(np.min([round((20000 / args.num_cell) * 400),400])),
        'dropout_rate': 0.1
    }
    best_ind = default
    
    ## Fixed Value
    # print("---------Fixed Value---------")
    # n_hidden_list = [default['n_hidden']] + list(range(50, 501, 30))
    # n_latent_list = [default['n_latent']] + list(range(5, 51, 3))
    # n_layers_list = [default['n_layers']] + list(range(1, 3))
    # batch_size_list = [default['batch_size']] + list(range(16, 513, 32))
    # learning_rate_list = [default['learning_rate']] + [round(i * 1e-4, 6) for i in range(1, 101, 5)]
    # epochs_list = [default['epochs']] + list(range(300, 601, 25))
    # dropout_rate_list = [default['dropout_rate']] + [round(i * 0.01, 6) for i in range(21)]

    ## Random Generate
    print("---------Random Value---------")
    n_hidden_list = [default['n_hidden']] + np.random.randint(50,500, size=15).tolist()
    n_latent_list = [default['n_latent']] + np.random.randint(5, 50, size=15).tolist()
    n_layers_list = [default['n_layers']] + np.random.randint(1, 4, size=15).tolist()
    batch_size_list = [default['batch_size']] + np.random.randint(16,1024, size=15).tolist()
    learning_rate_list = [default['learning_rate']] + np.random.uniform(1e-5, 1e-2,size=15).tolist()
    epochs_list = [default['epochs']] + np.random.randint(200, 600,size=15).tolist()
    dropout_rate_list = [default['dropout_rate']] + np.random.uniform(0, 0.4,size=15).tolist()

    for n_hidden in n_hidden_list:
        vae = scvi.model.SCVI(adata,n_hidden=n_hidden)
        vae.train(use_gpu=device,log_every_n_steps=args.log_every_n_steps)
        mark = metrics(vae,adata,clusters_true,args)
        if mark: 
            best_ind['n_hidden'] = n_hidden
    for n_latent in n_latent_list:
        vae = scvi.model.SCVI(adata,n_hidden=best_ind['n_hidden'],n_latent=n_latent)
        vae.train(use_gpu=device,log_every_n_steps=args.log_every_n_steps)
        mark = metrics(vae,adata,clusters_true,args)
        if mark: 
            best_ind['n_latent'] = n_latent
    for n_layers in n_layers_list:
        vae = scvi.model.SCVI(adata,n_hidden=best_ind['n_hidden'],n_latent=best_ind['n_latent'],n_layers=n_layers)
        vae.train(use_gpu=device,log_every_n_steps=args.log_every_n_steps)
        mark = metrics(vae,adata,clusters_true,args)
        if mark: 
            best_ind['n_layers'] = n_layers
    for batch_size in batch_size_list:
        vae = scvi.model.SCVI(adata,n_hidden=best_ind['n_hidden'],n_latent=best_ind['n_latent'],n_layers=best_ind['n_layers'])
        vae.train(batch_size=batch_size,use_gpu=device,log_every_n_steps=args.log_every_n_steps)
        mark = metrics(vae,adata,clusters_true,args)
        if mark: 
            best_ind['batch_size'] = batch_size
    for learning_rate in learning_rate_list:
        vae = scvi.model.SCVI(adata,n_hidden=best_ind['n_hidden'],n_latent=best_ind['n_latent'],n_layers=best_ind['n_layers'])
        vae.train(batch_size=best_ind['batch_size'],use_gpu=device,log_every_n_steps=args.log_every_n_steps,plan_kwargs={'lr':learning_rate})
        mark = metrics(vae,adata,clusters_true,args)
        if mark: 
            best_ind['learning_rate'] = learning_rate
    for epochs in epochs_list:
        vae = scvi.model.SCVI(adata,n_hidden=best_ind['n_hidden'],n_latent=best_ind['n_latent'],n_layers=best_ind['n_layers'])
        vae.train(max_epochs = epochs,batch_size=best_ind['batch_size'],use_gpu=device,log_every_n_steps=args.log_every_n_steps,plan_kwargs={'lr':best_ind['learning_rate']})
        mark = metrics(vae,adata,clusters_true,args)
        if mark: 
            best_ind['epochs'] = epochs
    for dropout_rate in dropout_rate_list:
        vae = scvi.model.SCVI(adata,n_hidden=best_ind['n_hidden'],n_latent=best_ind['n_latent'],n_layers=best_ind['n_layers'],dropout_rate=dropout_rate)
        vae.train(max_epochs =best_ind['epochs'], batch_size=best_ind['batch_size'],use_gpu=device,log_every_n_steps=args.log_every_n_steps,plan_kwargs={'lr':best_ind['learning_rate']})
        mark = metrics(vae,adata,clusters_true,args)
        if mark: 
            best_ind['dropout_rate'] = dropout_rate
    
    return best_ind

def GA_init(args):
    
    pop_hyperparams = []
    for _ in range(args.pop_size):
        
        hyperparams = {
        'n_hidden': np.random.randint(50,500),
        'n_latent': np.random.randint(5, 50),
        'n_layers': np.random.randint(1,2),
        'batch_size': np.random.randint(16,512),
        'learning_rate': np.random.uniform(1e-4, 1e-2),
        'epochs': np.random.randint(200,600),
        'dropout_rate': np.random.uniform(0,0.2)
        }
        pop_hyperparams.append(hyperparams)
    
    default = {
        'n_hidden': 128,
        'n_latent': 10,
        'n_layers': 1,
        'batch_size': 128,
        'learning_rate': 0.001,
        'epochs': int(np.min([round((20000 / args.num_cell) * 400), 400])),
        'dropout_rate': 0.1
        }
    
    pop_hyperparams.append(default)
    
    return pop_hyperparams

def Grid_init(args):
    
    pop_hyperparams = []
    n_hidden = [50, 500]
    n_latent = [5, 50]
    n_layers = [1, 2]
    batch_size = [64,256]
    learning_rate = [1e-3, 1e-2]
    epochs = [300,600]
    dropout_rate = [0.1, 0.2]

    for h in n_hidden:
        for l in n_latent:
            for layers in n_layers:
                for bs in batch_size:
                    for lr in learning_rate:
                        for ep in epochs:
                            for dr in dropout_rate:
                                hyperparams = {
                                    'n_hidden': h,
                                    'n_latent': l,
                                    'n_layers': layers,
                                    'batch_size': bs,
                                    'learning_rate': lr,
                                    'epochs': ep,
                                    'dropout_rate': dr
                                }
                                pop_hyperparams.append(hyperparams)
                                
    default = {
        'n_hidden': 128,
        'n_latent': 10,
        'n_layers': 1,
        'batch_size': 128,
        'learning_rate': 0.001,
        'epochs': int(np.min([round((20000 / args.num_cell) * 400), 400])),
        'dropout_rate': 0.1
        }
    
    pop_hyperparams.append(default)

    return pop_hyperparams

def Bayes_Search(adata,clusters_true,args,device):
    
    adata = adata.copy()
    
    def Bayes_Objective(trial):
    
        ## The Parameters
        n_hidden = trial.suggest_int("n_hidden",50,500)
        n_latent = trial.suggest_int("n_latent",5, 50)
        n_layers = trial.suggest_int("n_layers",1,2)
        batch_size = trial.suggest_int("batch_size",16,512)
        epochs = trial.suggest_int("epochs",300,600)
        learning_rate = trial.suggest_uniform("learning_rate",1e-4, 1e-2)
        dropout_rate = trial.suggest_uniform("dropout_rate",0,0.2)
        
        scvi.model.SCVI.setup_anndata(adata)
        
        vae = scvi.model.SCVI(adata,n_hidden=n_hidden,n_latent=n_latent,n_layers=n_layers,dropout_rate=dropout_rate)
        vae.train(max_epochs=epochs,use_gpu=device,batch_size=batch_size,log_every_n_steps=args.log_every_n_steps,plan_kwargs={'lr':learning_rate})

        mark = metrics(vae,adata,clusters_true,args)

        history = vae.history
        reconstruction_loss = history['reconstruction_loss_train'].iloc[-1]['reconstruction_loss_train']

        return reconstruction_loss
    
    study = optuna.create_study(direction='minimize')
    
    study.optimize(Bayes_Objective, n_trials=100)
    
    param = study.best_params
    
    return(param)

def scVITraining(lr,n_hidden,n_latent,n_layers,dropout_rate,batch_size,epochs,noise=None):
    
    device = 'cuda:0'
    
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

def metrics(vae,adata,clusters_true,args):
    
    mark = False
    
    adata.obsm["X_scVI"] = vae.get_latent_representation()
    adata.obsm["X_normalized_scVI"] = vae.get_normalized_expression()
    
    sc.pp.neighbors(adata,use_rep='X_scVI', random_state=123)
    sc.tl.leiden(adata, resolution=0.5, random_state=123)
    clusters_pred = adata.obs['leiden'].astype('int').values
    
    ari = adjusted_rand_score(clusters_true, clusters_pred)
    nmi = adjusted_mutual_info_score(clusters_true,clusters_pred)
    
    if ari > args.ari and nmi > args.nmi:
        
        args.ari = ari
        args.nmi = nmi
        mark = True
    
    print(f"Current ARI:{ari}, Current NMI:{nmi}")
    print(f"Best ARI {args.ari}, Best NMI {args.nmi}")
    
    return mark

def parse_args(args=None):

    parser = argparse.ArgumentParser(description='PyTorch Hyperparameter GA Training')
    
    #random seed
    parser.add_argument('--seed', default=1, type=int) 

    #hyperparameters of GA
    parser.add_argument('--pop_size', default=9, type=int) 
    parser.add_argument('--max_generations', default=10, type=int) 
    parser.add_argument('--log_every_n_steps', default=1, type=int) 
    parser.add_argument('--mutation_prob',default=0.9,type=int)
    parser.add_argument('--crossover_prob',default=0.9,type=int)

    #hyperparameters of the data
    #parser.add_argument('--data_path', default='SinianZhang/Mcgill/GAVAE/data/IPF_control.h5ad',type=str)
    parser.add_argument('--data_path', default='SinianZhang/Mcgill/GAVAE/data/cortex.h5ad',type=str)
    parser.add_argument('--label', default='label',type=str)
    parser.add_argument('--num_cell', default=0,type=int)
    parser.add_argument('--num_gene', default=0,type=int)

    #hyperparameters of the model
    parser.add_argument('--input_dim', default=500, type=int)
    
    #hyperparameters of the VAE
    parser.add_argument('--n_clusters',default=10,type=int)
    
    #hyperparameters of the ari and nmi
    parser.add_argument('--ari', default=0, type=float)
    parser.add_argument('--nmi', default=0, type=float)
    
    #use scvi or not
    parser.add_argument('--use_scvi',default=False,type=bool)
    
    #use which gpu
    parser.add_argument('--gpu',default='0',type=str)
    
    #use grid, GA or greedy
    parser.add_argument('--method',default="GA",type=str)

    return parser.parse_args(args)


if __name__ == "__main__":
    
    args = parse_args()
    
    ## some setting
    torch.set_float32_matmul_precision('high')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.gpu != '-1' and torch.cuda.is_available():
        device = 'cuda:'+args.gpu
    else:
        device = 'cpu'
    
    adata = anndata.read_h5ad(args.data_path)
    adata = processing(adata)
    
    args.num_cell = adata.X.shape[0]
    args.num_gene = adata.X.shape[1]
    clusters_true = adata.obs[args.label]
    
    Train_start = time.time()
    
    if args.method == "GA":
        print("---------GA Search---------")
        pop_hyperparams = GA_init(args)
        for i in range(args.max_generations):
            print(f"\n Generation {i + 1} \n")
            pop_hyperparams = GA_Search(pop_hyperparams,adata,clusters_true,args,device)
    elif args.method == "Grid":
        print("---------Grid Search---------")
        pop_hyperparams = Grid_init(args)
        best_ind = Grid_Search(pop_hyperparams,adata,clusters_true,args,device)
        
    elif args.method == "Greedy":
        print("---------Greedy Search---------")
        best_ind = Greedy_Search(adata,clusters_true,args,device)
        
    elif args.method == "Bayes":
        print("---------Bayes Search---------")
        best_ind = Bayes_Search(adata,clusters_true,args,device)
        
    Train_end = time.time()
        
    print(f"All Time:{(Train_end-Train_start)*1000}ms")
    