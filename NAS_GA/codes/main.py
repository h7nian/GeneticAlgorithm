import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import random

from model import AutoEncoder,VQVAE
from utils import Create_Dataloader
from adam_train import Adam_Train
from ga_train import GA_Train


 
def parse_args(args=None):

    parser = argparse.ArgumentParser(description='PyTorch NAS Training')
    
    #AE or VQVAE
    parser.add_argument('--IFVQVAE', default=False, type=bool) 

    #hyperparameters of GA
    parser.add_argument('--pop_size', default=10, type=int) 
    parser.add_argument('--offspring_size', default=6, type=int) 
    parser.add_argument('--max_generations', default=50, type=int) 
    parser.add_argument('--Adam_epochs', default=300, type=int) 
    parser.add_argument('--GA_epochs', default=2, type=int) 
    parser.add_argument('--elitist_level', default=0.6, type=float) 
    parser.add_argument('--sigma', default=0.01, type=float) 
    parser.add_argument('--rho', default=2, type=float) 
    parser.add_argument('--gene_size',default=69,type=int)
    parser.add_argument('--mutation_prob',default=0.8,type=int)
    parser.add_argument('--crossover_prob',default=0.8,type=int)

    #hyperparameters of the data
    parser.add_argument('--batch_size', default=512, type=int) 
    parser.add_argument('--num_classes', default=10,type=int)
    parser.add_argument('--data_path', default='SinianZhang/Mcgill/GAVAE/data/cortex.h5ad',type=str)

    #hyperparameters of optimizer
    parser.add_argument('--weight_decay', default=5e-4,type=float)
    parser.add_argument('--momentum', default=0.9,type=float)
    parser.add_argument('--optimizer', default='Adam',type=str)
    parser.add_argument('--learning_rate', default=1e-3, type=float) 
    parser.add_argument('--decay_factor', default=0.9, type=float) 

    #hyperparameters of the model
    parser.add_argument('--input_dim', default=500, type=int)
    parser.add_argument('--hidden_dim', default=32, type=int)
    parser.add_argument('--dropout', default=0.1, type=float) 
    
    #hyperparameters of the VQVAE
    parser.add_argument('--n_clusters',default=10,type=int)
    
    #hyperparameters of the ari and nmi
    parser.add_argument('--ari', default=0, type=float)
    parser.add_argument('--nmi', default=0, type=float)

    return parser.parse_args(args)

def GA_Neural_Train(encoder_population,decoder_population,
                    dataloader,
                    adata,G,clusters_true,
                    args,device):
    
    print(f"Starting with population of size: {args.pop_size}")
    
    for k in range(args.max_generations):
        
        print(f"Currently in generation {k+1}")
        
        print("-----Decoding the Chromosome------")
        
        ## decode the chromosome to model
        population = []
        for pop in range(args.pop_size):
            
            if args.IFVQVAE:
                model = VQVAE(args)
            else:
                model = AutoEncoder(args)
                
            encoder_num_layers = decoder_num_layers = 2

            Encoder_Linear = []
            Encoder_Activation = []
            
            Decoder_Linear = []
            Decoder_Activation = []  
            
            ## the encoder in the model
            
            for i in range(4,10):
                
                if encoder_population[pop][i] == '0':           
                    Encoder_Activation.append(nn.LeakyReLU())   
                else:                    
                    Encoder_Activation.append(nn.ReLU())
                    
            for i in range(10,70,10): 
                
                if int(encoder_population[pop][i:i+10],2) > 0:
                                
                    Encoder_Linear.append(int(encoder_population[pop][i:i+10],2))
                    
                else: 
                    
                    Encoder_Linear.append(random.randint(100,5000))
                
            for i in range(encoder_num_layers):
                
                if i==0:
                    
                    model.add_encoder(nn.Linear(args.input_dim,Encoder_Linear[0]))
                    model.add_encoder(Encoder_Activation[0])
                    
                else:
                    model.add_encoder(nn.Linear(Encoder_Linear[i-1],Encoder_Linear[i]))
                    model.add_encoder(Encoder_Activation[i])
                    
            model.add_encoder(nn.Linear(Encoder_Linear[encoder_num_layers - 1],args.hidden_dim)) 
            model.add_encoder(nn.ReLU())      
                    
            ## the decoder in the model
              
            for i in range(4,10):
                
                if decoder_population[pop][i] == '0':
                    Decoder_Activation.append(nn.LeakyReLU())                
                else:
                    Decoder_Activation.append(nn.ReLU())
                    
            for i in range(10,70,10):
                
                if int(decoder_population[pop][i:i+10],2) > 0:
                    Decoder_Linear.append(int(decoder_population[pop][i:i+10],2))
                    
                else:
                    Decoder_Linear.append(random.randint(100,5000))
                
            for i in range(decoder_num_layers):
                
                if i==0:
                    model.add_decoder(nn.Linear(args.hidden_dim,Decoder_Linear[0]))
                    model.add_decoder(Decoder_Activation[0])
                    
                else:
                    model.add_decoder(nn.Linear(Decoder_Linear[i-1],Decoder_Linear[i]))
                    model.add_decoder(Decoder_Activation[i])
            
            model.add_decoder(nn.Linear(Decoder_Linear[encoder_num_layers - 1],args.input_dim))
            model.add_decoder(nn.ReLU())
            
            population.append(model)

        print("-----Finish the Decode-----")

        #Adam Train
        print(f"--- Starting Adam ---")
        
        population = [Adam_Train(population[i],dataloader,adata,G,clusters_true,args,device) for i in range(args.pop_size)]
        
        print(f"--- Finished Adam")
         
        # GA
        print(f"--- Starting GA")

        GA_start = time.time()

        for i in range(0, args.GA_epochs):

            encoder_population,decoder_population = GA_Train(encoder_population,decoder_population,population,dataloader,adata,clusters_true,args,device)
        
        GA_end = time.time()

        print(f"--- Finished GA,Time:{(GA_end-GA_start)*1000}ms")
        
        
    print(f"Finished training process")
    
    return population


def init_population(args):

    '''
    Initialize the population
    '''
    
    encoder_population = []
    decoder_population = []
    
    for pop in range(args.pop_size):
        
        encoder_tmp = ''
        
        for gene in range(args.gene_size):
            
            encoder_tmp += str(random.randint(0,1))

        encoder_population.append(encoder_tmp)
        
        decoder_tmp = ''
        for gene in range(args.gene_size):
            decoder_tmp += str(random.randint(0,1))

        decoder_population.append(decoder_tmp)
        
    return encoder_population,decoder_population
    


if __name__ == '__main__':

    args = parse_args()

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    encoder_population,decoder_population = init_population(args)

    adata,dataloader = Create_Dataloader(args)
    
    G = adata.X.shape[1]
    args.input_dim = G
    
    clusters_true = adata.obs["label"]
    
    args.n_clusters = adata.obs["label"].nunique(dropna=True)

    Train_start = time.time()

    trained_population = GA_Neural_Train(encoder_population,decoder_population,dataloader,adata,G,clusters_true,args,device)
    
    Train_end = time.time()

    print(f"All Time:{(Train_end-Train_start)*1000}ms")