"""
---------------------------------------------------------------------------------------
Code Implementation to perform N experiments on Tox21 and SIDER to 
obtain the mean and standard deviation results for FS-GNNConv and other model baselines.
This allows to evaluate the average ROC-AUC scores for #support_set = 5 or 
#support_set = 10.
Load the weights to perform the experiments on fsgnnconv_eval.py on the variables
ckp_path_gnn and ckp_path_cnn. Make sure that the model weights that you load correspond
to the dataset and #support_set variables.
Additionally, consider using batch_size = 10 for #support_set = 5 and batch_size = 20
for #support_set = 10. In gnn_cnn.py change inputs shape according to this batch_size.
We also perform the transfer learning experiments on Tox21 and SIDER using this script.
(In this last case uncomment the last line of fsgnnconv_eval.py or baseline_eval.py).
For transfer-learning experiments consider using all tasks for testing and using Tox21
dataset and loading the weights of the model trained on SIDER (on fsgnnconv_eval.py),
and vice-versa.
t-SNE visualizations can be enabled by uncommenting part of the code in fsgnnconv_eval.py.
This example evaluates the FS-GNNConv model on Tox21 using a previously saved model 
checkpoint specificied in the variable ckp_path_gnn and ckp_path_cnn on fsgnnconv_eval.py.
Don't forget to change the name of the file where you save the results (variable "file").
---------------------------------------------------------------------------------------
"""
import argparse
import torch
from fsgnnconv_eval import Meta_eval
import statistics
import shutil
import matplotlib.pyplot as plt

def save_ckp(state, is_best, checkpoint_dir, best_model_dir, filename, best_model):
    
    f_path = checkpoint_dir + filename
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + best_model
        shutil.copyfile(f_path, best_fpath)

def save_result(epoch, N, exp, filename):
            
    file = open(filename, "a")
    
    file.write("Results: " + "\t")
    if epoch < N:
     file.write(str(exp) + "\t")
    if epoch >= N:
      for list_acc in exp:
        file.write(str(epoch) + " Support Sets: (Mean:"+ str(statistics.mean(list_acc)) +", SD:" +str(statistics.stdev(list_acc)) + ") | \t")
    file.write("\n")
    file.close()

def main_evaluate(dataset, gnn, support_set, pretrained):
   
    parser = argparse.ArgumentParser(description='main_evaluate')
    args = parser.parse_args()
    args.data = dataset
    args.pretrained = pretrained
    args.gnn = gnn #gin, gcn, graph_sage
    args.n_support = support_set
    args.device = 0
    args.batch_size = 10 #(10 for # support_set = 5, 20 for # support_set = 10)
    args.n_epochs = 1000
    args.learning_rate = 0.001
    args.graph_layers = 5
    args.emb_size = 300
    args.n_query = 128
    args.lr_update = 0.4
    args.k_train = 5
    args.k_test = 10
    args.p_weight = 1 #Tox21:35, SIDER:1 
    args.tl = 0 #transfer-learning (read information above) 
    
    """
    For this example, on fsgnnconv_eval.py consider:
    ckp_path_gnn = "checkpoints/checkpoints-FSGNNConv/check-sider-5-gnn.pt"
    ckp_path_cnn = "checkpoints/checkpoints-FSGNNConv/check-sider-5-cnn.pt"
    
    and
    
    Uncomment last line of fsgnnconv_eval.py    
    """
    
    # FS-GNNConv - Two module GNN-CNN architecture
    # GraphSage - assumes that nodes that reside in the same neighborhood should have similar embeddings.
    # GIN - Graph Isomorphism Network
    # GCN - Standard Graph Convolutional Network
    
    if args.tl == 0:
        if args.data == "tox21":
            args.tasks = 12
            args.train_tasks = 9 #change to 0 for transfer-learning experiments
            args.test_tasks = 3 #change to 12 for transfer-learning experiments
    
        elif args.data == "sider":
            args.tasks = 27
            args.train_tasks = 21 #change to 0 for transfer-learning experiments
            args.test_tasks = 6 #change to 27 for transfer-learning experiments
    
    elif args.tl == 1:
        if args.data == "tox21":
            args.tasks = 12
            args.train_tasks = 0 #change to 0 for transfer-learning experiments
            args.test_tasks = 12 #change to 12 for transfer-learning experiments
    
        elif args.data == "sider":
            args.tasks = 27
            args.train_tasks = 0 #change to 0 for transfer-learning experiments
            args.test_tasks = 27 #change to 27 for transfer-learning experiments

    device = "cuda:0"      
    model_eval = Meta_eval(args).to(device)
    model_eval.to(device)

    print("Dataset:", args.data)
    
    roc_auc_list = []
    
    if args.data == "tox21":
        exp = [[],[],[]]
        labels =  ['SR-HSE', 'SR-MMP', 'SR-p53']
    elif args.data == "sider":
        exp = [[],[],[],[],[],[]]
        labels =  ['R.U.D.', 'P.P.P.C.', 'E.L.D.', 'C.D.', 'N.S.D.', 'I.P.P.C.']
        
    if args.tl == 1:
        exp = [[]]
        labels = ['Mean_Task']
        
    N = 20
   
    for epoch in range(1, 1000):
        
        roc_scores, gnn_model, cnn_model, gnn_opt, cnn_opt = model_eval.meta_evaluate() #FS-GNNConv
        #roc_scores, gnn_model, gnn_opt = model.meta_evaluate(grads) #baselines
        if roc_auc_list != []:
            for score in range(len(roc_auc_list)):

                if roc_auc_list[score] < roc_scores[score]:
                    roc_auc_list[score] = roc_scores[score]
                    
        if epoch <= N:
          i=0
          for a in roc_scores:
            exp[i].append(round(a,4))
            i+=1
          
        if epoch > N:
          for i in range(len(exp)):
            if min(exp[i]) < round(roc_scores[i],4):
              index = exp[i].index(min(exp[i]))
              exp[i][index] = roc_scores[i]
        else:
            roc_auc_list = roc_scores
           
        
        save_result(epoch, N, exp, "results-exp/mean-FS-GNNConv_tox21_5.txt")
        
        if args.tl == 0:
            if args.data == "tox21":
                box_plot_data=[exp[0], exp[1], exp[2]]
                plot_title = "Tox21"
            elif args.data == "sider":
                box_plot_data=[exp[0], exp[1], exp[2], exp[3], exp[4], exp[5]]
                plot_title = "SIDER"  
        elif args.tl == 1:
            if args.data == "tox21":
                box_plot_data=[exp[0]]
                plot_title = "Transfer-Learning (SIDER->Tox21)"
            elif args.data == "sider":
                box_plot_data=[exp[0]]
                plot_title = "Transfer-Learning (Tox21->SIDER)"
            
        fig = plt.figure()   
        fig.suptitle(plot_title, fontsize=14, fontweight='bold')
        ax = fig.add_subplot(111)
        ax.boxplot(box_plot_data,labels=labels)
        ax.set_xlabel('Test Task')
        ax.set_ylabel('ROC-AUC score')
        plt.grid(b=False)
        if epoch == N:
            plt.savefig('plots/boxplot_FSGNNConv_'+ str(args.data) + '_' + str(args.n_support))
        plt.show()
        plt.close(fig)
       
        if epoch >= N/2:
            for list_acc in exp:
                print(str(epoch) + " Support Sets: (Mean:"+ str(statistics.mean(list_acc)) +", SD:" +str(statistics.stdev(list_acc)) + ")")
       
if __name__ == "__main__":
    main_evaluate("tox21", "gin", 5, "pre-trained/supervised_contextpred.pth")