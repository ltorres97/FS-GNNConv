import torch
import torch.nn as nn
from gnn_cnn import GNN_prediction
import torch.nn.functional as F
from loader import MoleculeDataset, random_sampler
from torch_geometric.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
from sklearn.manifold import TSNE
# from tsnecuda import TSNE # Use this package if the previous one doesn't work
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statistics

def optimizer_to(optim, device):
    # move optimizer to device
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def load_ckp(checkpoint_fpath, model, optimizer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))
    checkpoint = torch.load(checkpoint_fpath, map_location = device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.to(device)
      
    optimizer_to(optimizer, device)

    return model, optimizer, checkpoint['epoch']


class Meta_train(nn.Module):
    def __init__(self, args):
        super(Meta_train,self).__init__()

        self.data = args.data
        
        if self.data == "tox21":
            self.tasks = 12
            self.train_tasks  = 9
            self.test_tasks = 3

        elif self.data == "sider":
            self.tasks = 27
            self.train_tasks  = 21
            self.test_tasks = 6
      
        self.n_support = args.n_support
        self.n_query = args.n_query
        self.emb_size = args.emb_size
        self.batch_size = args.batch_size
        self.lr_update = args.lr_update
        self.k_train = args.k_train
        self.k_test = args.k_test
        self.device = 0
        self.criterion = nn.BCEWithLogitsLoss()
        self.gnn = GNN_prediction(args.graph_layers, args.emb_size, jk = "last", dropout_prob = 0.5, pooling = "mean", gnn_type = args.gnn)
        self.gnn.from_pretrained(args.pretrained)
       
        graph_params = []
        graph_params.append({"params": self.gnn.gnn.parameters()})
        graph_params.append({"params": self.gnn.graph_pred_linear.parameters(), "lr":args.learning_rate})
        
        self.optimizer = optim.Adam(graph_params, lr=args.learning_rate, weight_decay=0) 
        self.gnn.to(torch.device("cuda:0"))
        ckp_path_gnn = "checkpoints/checkpoints-baselines/GCN/checkpoint_gcn_gnn_tox21_5.pt"
        
        # Model checkpoints:
        # GCN-Tox21-5-GNN: "checkpoints/checkpoints-baselines/GCN/checkpoint_GCN_gnn_tox21_5.pt"
        # GCN-SIDER-5-GNN: "checkpoints/checkpoints-baselines/GCN/checkpoint_GCN_gnn_sider_5.pt"    
        # GCN-Tox21-10-GNN: "checkpoints/checkpoints-baselines/GCN/checkpoint_GCN_gnn_tox21_10.pt"
        # GCN-SIDER-10-GNN: "checkpoints/checkpoints-baselines/GCN/checkpoint_GCN_gnn_sider_10.pt"   
        
        # GraphSAGE-Tox21-5-GNN: "checkpoints/checkpoints-baselines/GraphSAGE/checkpoint_graphsage_gnn_tox21_5.pt"
        # GraphSAGE-SIDER-5-GNN: "checkpoints/checkpoints-baselines/GraphSAGE/checkpoint_graphsage_gnn_sider_5.pt"    
        # GraphSAGE-Tox21-10-GNN: "checkpoints/checkpoints-baselines/GraphSAGE/checkpoint_graphsage_gnn_tox21_10.pt"   
        # GraphSAGE-SIDER-10-GNN: "checkpoints/checkpoints-baselines/GraphSAGE/checkpoint_graphsage_gnn_sider_10.pt" 
        
        # GIN-Tox21-5-GNN: "checkpoints/checkpoints-baselines/GIN/checkpoint_GIN_gnn_tox21_5.pt"
        # GIN-SIDER-5-GNN: "checkpoints/checkpoints-baselines/GIN/checkpoint_GIN_gnn_sider_5.pt"
        # GIN-Tox21-10-GNN: "checkpoints/checkpoints-baselines/GIN/checkpoint_GIN_gnn_tox21_10.pt"
        # GIN-SIDER-10-GNN: "checkpoints/checkpoints-baselines/GIN/checkpoint_GIN_gnn_sider_10.pt"
        
        self.gnn, self.optimizer, start_epoch = load_ckp(ckp_path_gnn, self.gnn, self.optimizer)
            
    def update_graph_params(self, loss, lr_update):
        grads = torch.autograd.grad(loss, self.gnn.parameters())
        return parameters_to_vector(grads), parameters_to_vector(self.gnn.parameters()) - parameters_to_vector(grads) * lr_update

    def meta_train(self):
        support_sets = []
        query_sets = []
        device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")
        self.gnn.train()
        
        for train_task in range(self.train_tasks):
            dataset = MoleculeDataset("Data/" + self.data + "/pre-processed/task_" + str(train_task+1), dataset = self.data)
            support_dataset, query_dataset = random_sampler(dataset, self.data, train_task, self.n_support, self.n_query, train=True)
            support_set = DataLoader(support_dataset, batch_size=self.batch_size, shuffle=False, num_workers = 0, drop_last = True)
            query_set = DataLoader(query_dataset, batch_size=self.batch_size, shuffle=False, num_workers = 0, drop_last = True)
            support_sets.append(support_set)
            query_sets.append(query_set)
        
        for k in range(0, self.k_train):
            graph_params = parameters_to_vector(self.gnn.parameters())
            query_losses = torch.tensor([0.0]).to(device)
            for task in range(self.train_tasks):

                loss_support = torch.tensor([0.0]).to(device)
                for batch_idx, batch in enumerate(tqdm(support_sets[task], desc="Iteration")):
                    batch = batch.to(device)
                    graph_pred, node_emb = self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    label = batch.y.view(graph_pred.shape).to(torch.float64)
                    support_loss = torch.sum(self.criterion(graph_pred.double(), label))/graph_pred.size()[0]
                    loss_support += support_loss

                updated_grad, updated_params = self.update_graph_params(loss_support, lr_update = self.lr_update)
                vector_to_parameters(updated_params, self.gnn.parameters())

                loss_query = torch.tensor([0.0]).to(device)
                for batch_idx, batch in enumerate(tqdm(query_sets[task], desc="Iteration")):
                    batch = batch.to(device)
                    graph_pred, node_emb = self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    label = batch.y.view(graph_pred.shape).to(torch.float64)
                    query_loss = torch.sum(self.criterion(graph_pred.double(), label))/graph_pred.size()[0]
                    loss_query += query_loss

                if task == 0:
                    query_losses = loss_query
                else:
                    query_losses = torch.cat((query_losses, loss_query), 0)

                vector_to_parameters(graph_params, self.gnn.parameters())

            query_losses = torch.sum(query_losses)
            loss_graph = query_losses / self.train_tasks       
            self.optimizer.zero_grad()
            loss_graph.backward()
            self.optimizer.step()
            
        return 

    def meta_test(self):
         roc_scores = []
         t=0
         graph_params = parameters_to_vector(self.gnn.parameters())
         for test_task in range(self.test_tasks):
             dataset = MoleculeDataset("Data/" + self.data + "/pre-processed/task_" + str(self.tasks-test_task), dataset = self.data)
             support_dataset, query_dataset = random_sampler(dataset, self.data, self.tasks-test_task-1, self.n_support, self.n_query, train=False)
             support_set = DataLoader(support_dataset, batch_size=self.batch_size, shuffle=False, num_workers = 0, drop_last=True)
             query_set = DataLoader(query_dataset, batch_size=self.batch_size, shuffle=False, num_workers = 0, drop_last = True)
             device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")
             self.gnn.eval()
        
             for k in range(0, self.k_test):
                 graph_loss = torch.tensor([0.0]).to(device)
                 for batch_idx, batch in enumerate(tqdm(support_set, desc="Iteration")):
                     batch = batch.to(device)
                     graph_pred, node_emb = self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                     y = batch.y.view(graph_pred.shape).to(torch.float64)
                     graph_loss += torch.sum(self.criterion(graph_pred.double(), y))/graph_pred.size()[0]
        
                 updated_grad, updated_params = self.update_graph_params(graph_loss, lr_update = self.lr_update)
                 vector_to_parameters(updated_params, self.gnn.parameters())
             
             nodes=[]
             labels=[]
             y_label = []
             y_pred = []
             labels_tox21 = ['SR-HSE', 'SR-MMP', 'SR-p53']
             #labels_sider = ['R.U.D.', 'P.P.P.C.', 'E.L.D.', 'C.D.', 'N.S.D.', 'I.S.P.C.']
             for batch_idx, batch in enumerate(tqdm(query_set, desc="Iteration")):
                 batch = batch.to(device)
                 graph_pred, node_emb = self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)    
                 graph_pred = F.sigmoid(graph_pred)
                 graph_pred = torch.where(graph_pred>0.5, torch.ones_like(graph_pred), graph_pred)
                 graph_pred = torch.where(graph_pred<=0.5, torch.zeros_like(graph_pred), graph_pred) 
                 node_emb_tsne = node_emb.cpu().detach().numpy() 
                 y_tsne = batch.y.view(graph_pred.shape).cpu().detach().numpy()
        
                 for i in node_emb_tsne:
                     nodes.append(i)
                 for j in y_tsne:
                     labels.append(j)
                     
                 y_pred.append(graph_pred)
                 y_label.append(batch.y.view(graph_pred.shape))
             
             #Plot t-SNE visualizations
             """
             t+=1
             node_emb_tsne = np.asarray(nodes)
       
             
             y_tsne = np.asarray(labels).flatten()
             slipper_colour = pd.DataFrame({'colour': ['Blue', 'Orange'],
                                'label': [0, 1]})
             
             c_dict = {'Positive': '#ff7f0e','Negative': '#1f77b4' }
      
             z = TSNE(n_components=2, init='random').fit_transform(node_emb_tsne)
             label_vals = {0: 'Negative', 1: 'Positive'}
             tsne_result_df = pd.DataFrame({'tsne_dim_1': z[:,0], 'tsne_dim_2': z[:,1], 'label': y_tsne})
             tsne_result_df['label'] = tsne_result_df['label'].map(label_vals)
             fig, ax = plt.subplots(1)
             sns.set_style("ticks",{'axes.grid' : True})
             g1 = sns.scatterplot(x='tsne_dim_1', y='tsne_dim_2', hue='label', data=tsne_result_df, ax=ax,s=10, palette = c_dict, hue_order=('Negative', 'Positive') )
             lim = (z.min()-5, z.max()+5)
             ax.set_xlim(lim)
             ax.set_ylim(lim)
             ax.set_aspect('equal') 
             
             g1.legend(title=labels_tox21[t-1], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)    
             g1.set(xticklabels=[])
             g1.set(yticklabels=[])
             g1.set(xlabel=None)
             g1.set(ylabel=None)
             g1.tick_params(bottom=False) 
             g1.tick_params(left=False)
             plt.savefig('plots/'+labels_tox21[t-1])
             plt.show()
             plt.close(fig)
             """
             
             y_label = torch.cat(y_label, dim = 0).cpu().detach().numpy()
             y_pred = torch.cat(y_pred, dim = 0).cpu().detach().numpy()
             roc_auc_list = []
             roc_auc_list.append(roc_auc_score(y_label, y_pred))
             roc_auc = sum(roc_auc_list)/len(roc_auc_list)
             roc_scores.append(roc_auc)
        
             vector_to_parameters(graph_params, self.gnn.parameters())
             
         return roc_scores, self.gnn.state_dict(), self.optimizer.state_dict()
                