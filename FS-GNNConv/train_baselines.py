"""
---------------------------------------------------------------------------------------
Code Implementation to train the graph-based baselines on Tox21 and SIDER to 
obtain the best ROC-AUC results and obtain the model weigths for evaluation on several
experiments with different support sets.
Don't forget to choose 'gin', 'gcn' or 'graphsage' as the model and the correct 
pre-trained weights.In this case, always choose the supervised_contextpred weigths 
to return the best result.
Model weights are saved by uncommenting save_ckp() command variable.
Use the variable ckp_path_gnn in gnn_comp.py to load the saved models
in case you want to resume training using another checkpoint.
Additionally, consider using batch_size = 10 for #support set = 5 and batch_size = 20
for #support set = 10.
This example trains a pre-trained GCN model using a previously saved model 
checkpoint specificied in the variable ckp_path_gnn on gnn_comp.py.
---------------------------------------------------------------------------------------
"""

import argparse
import torch
from gnn_comp import Meta_train
import shutil

def save_ckp(state, is_best, checkpoint_dir, best_model_dir, filename, best_model):
    f_path = checkpoint_dir + filename
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + best_model
        shutil.copyfile(f_path, best_fpath)

    
def train_baselines(dataset, gnn, support_set, pretrained):

    parser = argparse.ArgumentParser(description='train_baselines')
    args = parser.parse_args()
    args.data = dataset   
    args.pretrained = pretrained
    args.gnn = gnn #gin, gcn, graph_sage
    args.n_support = support_set
    args.device = 0
    args.batch_size = 10 #(10 for # support_set = 5, 20 for # support_set = 10)
    args.learning_rate = 0.001
    args.graph_layers = 5
    args.emb_size = 300
    args.n_query = 128
    args.lr_update = 0.4
    args.k_train = 5
    args.k_test = 10
    
    # FS-GNNConv - Two module GNN-CNN architecture
    # GraphSAGE - assumes that nodes that reside in the same neighborhood should have similar embeddings.
    # GIN - Graph Isomorphism Network
    # GCN - Standard Graph Convolutional Network
    
    device = "cuda:0"
    baseline = Meta_train(args).to(device)
    baseline.to(device)
    
    for epoch in range(1, 1000):
        
        baseline.meta_train()
        
        roc_scores, gnn_model, gnn_opt = baseline.meta_test() #baselines
        
        print(roc_scores)
        checkpoint_gnn = {
                'epoch': epoch + 1,
                'state_dict': gnn_model,
                'optimizer': gnn_opt
        }

        checkpoint_dir = 'checkpoints/checkpoints-baselines/GCN'
        model_dir = 'model'
        is_best = epoch
        
        """
        Uncomment the following lines to save the model weigths as a new checkpoint
        """
        
        #save_ckp(checkpoint_gnn, is_best, checkpoint_dir, model_dir, "/checkpoint_GCN_gnn_tox21_5.pt", "/best_gnn.pt")

train_baselines("tox21", "gcn", 5, "pre-trained/gcn_supervised_contextpred.pth")
