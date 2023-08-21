import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
import random
 
from utils import cal_fitness,cal_loss

def GA_Train(encoder_population,decoder_population,population,dataloader,adata,clusters_true,args,device):
    
    fitness = [1/cal_loss(population[i],dataloader,adata,device) for i in range(args.pop_size)]
    
    best_gene = torch.argmin(torch.Tensor(fitness))
    
    encoder_population,decoder_population = crossover(encoder_population,decoder_population,args)
    
    encoder_population,decoder_population = mutation(encoder_population,decoder_population,args)
    
    return encoder_population,decoder_population
    
def mutation(encoder_population,decoder_population,args):
    
    new_encoder_population = []
    for individual in encoder_population:
        if random.random() < args.mutation_prob:
            rand_ = random.randint(0, len(individual) - 1)
            if individual[rand_] == '0':
                individual = individual[:rand_] + '1' + individual[rand_ + 1:]
            else:
                individual = individual[:rand_] + '0' + individual[rand_ + 1:]
        new_encoder_population.append(individual)
        
    new_decoder_population = []
    for individual in decoder_population:
        if random.random() < args.mutation_prob:
            rand_ = random.randint(0, len(individual) - 1)
            if individual[rand_] == '0':
                individual = individual[:rand_] + '1' + individual[rand_ + 1:]
            else:
                individual = individual[:rand_] + '0' + individual[rand_ + 1:]
        new_decoder_population.append(individual)
        
    return new_encoder_population,new_decoder_population

def crossover(encoder_population,decoder_population,args):
    
    for idx, individual in enumerate(encoder_population):
        if random.random() < args.crossover_prob:
            rand_spouse = random.randint(0, len(encoder_population) - 1)
            rand_point = random.randint(0, len(individual) - 1)

            new_parent_1 = individual[:rand_point] + encoder_population[rand_spouse][rand_point:]
            new_parent_2 = encoder_population[rand_spouse][:rand_point] + individual[rand_point:]

            encoder_population[idx] = new_parent_1
            encoder_population[rand_spouse] = new_parent_2
            
    for idx, individual in enumerate(decoder_population):
        if random.random() < args.crossover_prob:
            rand_spouse = random.randint(0, len(decoder_population) - 1)
            rand_point = random.randint(0, len(individual) - 1)

            new_parent_1 = individual[:rand_point] + decoder_population[rand_spouse][rand_point:]
            new_parent_2 = decoder_population[rand_spouse][:rand_point] + individual[rand_point:]

            decoder_population[idx] = new_parent_1
            decoder_population[rand_spouse] = new_parent_2
            
            
    return encoder_population,decoder_population