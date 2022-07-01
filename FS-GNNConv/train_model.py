"""
---------------------------------------------------------------------------------------
Code Implementation to train the FS-GNNConv model on Tox21 and SIDER to 
obtain the best ROC-AUC results and obtain the model weigths for evaluation on several
experiments with different support sets.
Don't forget to choose 'gin' as the model and the correct pre-trained weights. In this
case, always choose the supervised_contextpred weigths to return the best result.
Model weights are saved by uncommenting save_ckp() command variable.
Use the variable ckp_path_gnn and ckp_path_cnn in fsgnnconv_train.py to load the saved models
in case you want to resume training using another checkpoint.
Additionally, consider using batch_size = 10 for #support set = 5 and batch_size = 20
for #support set = 10. In gnn_cnn.py change inputs shape according to this batch_size.
This example trains a FS-GNNConv model on Tox21 using a previously saved model 
checkpoint specificied in the variable ckp_path_gnn and ckp_path_cnn on fsgnnconv_train.py.
---------------------------------------------------------------------------------------
"""

import argparse
import torch
from fsgnnconv_train import Meta_train
import shutil

def save_ckp(state, is_best, checkpoint_dir, best_model_dir, filename, best_model):
    f_path = checkpoint_dir + filename
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + best_model
        shutil.copyfile(f_path, best_fpath)
        

def train_model(dataset, gnn, support_set, pretrained):
 
    parser = argparse.ArgumentParser(description='train_model')
    args = parser.parse_args()
    args.data = dataset       
    args.pretrained = pretrained
    args.gnn = gnn #gin, gcn, graph_sage
    args.n_support = support_set
    args.batch_size = 10 #(10 for # support_set = 5, 20 for # support_set = 10)
    args.learning_rate = 0.001
    args.graph_layers = 5
    args.emb_size = 300
    args.n_query = 128
    args.lr_update = 0.4
    args.k_train = 5
    args.k_test = 10
    args.p_weight = 35 #Tox21:35, SIDER:1

    # FS-GNNConv - Two module GNN-CNN architecture
    # GraphSAGE - assumes that nodes that reside in the same neighborhood should have similar embeddings.
    # GIN - Graph Isomorphism Network
    # GCN - Standard Graph Convolutional Network
    
    device = "cuda:0"
    model_train = Meta_train(args).to(device)
    model_train.to(device)
   
    for epoch in range(1, 1000):
        
        model_train.meta_train()
        
        roc_scores, gnn_model, cnn_model, gnn_opt, cnn_opt = model_train.meta_test() #FS-GNNConv
        
        print(roc_scores)
        
        checkpoint_gnn = {
                'epoch': epoch + 1,
                'state_dict': gnn_model,
                'optimizer': gnn_opt
        }
        
        checkpoint_cnn = {
                'epoch': epoch + 1,
                'state_dict': cnn_model,
                'optimizer': cnn_opt
        }
        
        checkpoint_dir = 'checkpoints/checkpoints-FSGNNConv'
        model_dir = 'model'
        
        """
        Uncomment the following lines to save the model weigths as a new checkpoint
        """
        
        #save_ckp(checkpoint_gnn, is_best, checkpoint_dir, model_dir, "/check-tox21-5-gnn-new.pt", "/best_gnn.pt")
        #save_ckp(checkpoint_cnn, is_best, checkpoint_dir, model_dir, "/check-tox21-5-cnn-new.pt", "/best_cnn.pt")

train_model("tox21", "gin", 5, "pre-trained/supervised_contextpred.pth")
