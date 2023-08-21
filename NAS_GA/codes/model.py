from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self,args):
        
        super(AutoEncoder, self).__init__()
        self.args = args
        self.encoder = nn.Sequential()
        self.encoder_num_layder = 0
        self.decoder = nn.Sequential()
        self.decoder_num_layder = 0
        
        self.mse_loss = nn.MSELoss()
        
    def add_encoder(self,layer):
        
        self.encoder.append(layer)
        self.encoder_num_layder += 1
        
    def add_decoder(self,layer):
        
        self.decoder.append(layer)
        self.decoder_num_layder += 1
        
    def forward(self,x):
        
        z_e = self.encoder(x)
        x_e = self.decoder(z_e)

        return x_e, z_e
    
    def loss(self,x,x_e):
        
        return self.mse_loss(x,x_e)
    
    
class VQVAE(nn.Module):
    def __init__(self,args):
        
        super(VQVAE, self).__init__()
        self.args = args
        
        self.embeddings = nn.Parameter(torch.randn(self.args.n_clusters,self.args.hidden_dim)*0.05,requires_grad=True)
        
        self.encoder = nn.Sequential()
        self.encoder_num_layder = 0
        self.decoder = nn.Sequential()
        self.decoder_num_layder = 0
        
        self.mse_loss = nn.MSELoss()
        
    def add_encoder(self,layer):
        
        self.encoder.append(layer)
        self.encoder_num_layder += 1
        
    def add_decoder(self,layer):
        
        self.decoder.append(layer)
        self.decoder_num_layder += 1
        
    def forward(self,x):
        
        z_e = self.encoder(x)
        
        k = self._get_clusters(z_e)
        
        z_q = self._get_embeddings(k)
        
        x_q = self.decoder(z_e + (z_q - z_e).detach())

        return x_q, z_e,z_q
    
    def _get_clusters(self,z_e):
        
        _z_dist = (z_e.unsqueeze(1) - self.embeddings.unsqueeze(0)) ** 2
        z_dist = torch.sum(_z_dist,dim=-1)
        
        k = torch.argmin(z_dist,dim=-1)
        
        return k
        
    def _get_embeddings(self,k):
        
        k = k.long()
        
        _z_q = []
        
        for i in range(len(k)):
            
            _z_q.append(self.embeddings[k[i]])
            
        z_q = torch.stack(_z_q)
        
        return z_q
    
    def loss(self,x,x_q,z_e,z_q):
        
        return self.mse_loss(x,x_q) + self.mse_loss(z_e.detach(),z_q) + self.mse_loss(z_e,z_q.detach())