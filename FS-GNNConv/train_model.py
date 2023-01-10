import argparse
import torch
from fsgnnconv_train import FSGNNConv
import shutil
import gc

def save_ckp(state, is_best, checkpoint_dir, best_model_dir, filename, best_model):
    f_path = checkpoint_dir + filename
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + best_model
        shutil.copyfile(f_path, best_fpath)

dataset = "tox21"
gnn= "gin" #gin, graphsage, gcn
support_set = 5
pretrained = "pre-trained/supervised_contextpred.pth"
baseline = 0
device = "cuda:0"
model = FSGNNConv(dataset, gnn, support_set, pretrained, baseline)
model.to(device)

if dataset == "tox21":
    exp = [0,0,0]
    labels =  ['SR-HSE', 'SR-MMP', 'SR-p53']
elif dataset == "sider":
    exp = [0,0,0,0,0,0]
    labels =  ['R.U.D.', 'P.P.P.C.', 'E.L.D.', 'C.D.', 'N.S.D.', 'I.P.P.C.']

for epoch in range(1, 10000):
    
    is_best = False
    
    model.meta_train()
    
    roc_scores, gnn_model, cnn_model, gnn_opt, t_opt = model.meta_test() 
    
    print(roc_scores)
    checkpoint_gnn = {
            'epoch': epoch + 1,
            'state_dict': gnn_model,
            'optimizer': gnn_opt
    }
    
    checkpoint_cnn = {
            'epoch': epoch + 1,
            'state_dict': cnn_model,
            'optimizer': t_opt
    }
    

    checkpoint_dir = 'checkpoints/checkpoints-FSGNNConv/'
    model_dir = 'model/'
    is_best = epoch
    
    for i in range(0, len(roc_scores)):
        if exp[i] < roc_scores[i]:
            exp[i] = roc_scores[i]
            is_best = True
            
    filename = "results-exp/name-file.txt"
    file = open(filename, "a")
    file.write("ROC-AUC scores:\t")
    #file.write(str(exp))
    file.write(str(roc_scores)+"\t ; ")
    file.write(str(exp)) 
    file.write("\n")
    file.close()
    
    if baseline == 0:
        save_ckp(checkpoint_gnn, is_best, checkpoint_dir, model_dir, "/FS-GNNConv-GNN_tox21_5.pt", "/FS-GNNConv-GNN_tox21_5.pt")
        save_ckp(checkpoint_cnn, is_best, checkpoint_dir, model_dir, "/FS-GNNConv-CNN_tox21_5.pt", "/FS-GNNConv-CNN_tox21_5.pt")

    elif baseline == 1:
        save_ckp(checkpoint_gnn, is_best, checkpoint_dir, model_dir, "/GNN_tox21_5.pt", "/GNN_tox21_5.pt")
        
   
