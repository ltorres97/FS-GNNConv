
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from gnn_models import GNN
#from torchmeta.modules import (MetaModule, MetaSequential, MetaConv1d,
                               #MetaBatchNorm1d, MetaLinear)
from torch_geometric.nn import MessagePassing
from einops import rearrange
from torch_geometric import utils
import torch_geometric.utils as utils
from torch_scatter import scatter_add, scatter_mean, scatter_max
import torch_geometric.nn as gnn

from torchmeta.modules import (MetaModule, MetaSequential, MetaConv1d,
                             MetaBatchNorm1d, MetaLinear)



num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

GNN_TYPES = [
    'graph', 'graphsage', 'gcn',
    'gin', 'gine',
    'pna', 'pna2', 'pna3', 'mpnn', 'pna4',
    'rwgnn', 'khopgnn'
]

EDGE_GNN_TYPES = []


def conv3x3(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv1d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs), #kernel_size=10
        MetaBatchNorm1d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU()
    )

def dense(in_channels, out_channels, **kwargs):
    return MetaSequential(
          MetaLinear(in_channels, out_channels, **kwargs), 
          MetaBatchNorm1d(out_channels, momentum=1., track_running_stats=False),
          nn.ReLU()
      )


class FullyConnectedNeuralNetwork(MetaModule):
    def __init__(self, in_channels, out_features, hidden_size=10): #512
        super(FullyConnectedNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            dense(in_channels, 100),
            dense(100, 50),
            dense(50, hidden_size)
        )
       
        self.classifier = MetaLinear(hidden_size, 1) #6144

    def forward(self, inputs, params=None):
        
        inputs = inputs.reshape(10,300) #10,1,300
        #print(inputs.size())
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits

class ConvolutionalNeuralNetwork(MetaModule):
    def __init__(self, in_channels, out_features, hidden_size=64): #512
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            conv3x3(in_channels, 128),
            conv3x3(128, 64),
            conv3x3(64, 64)
        )
        
        self.classifier = MetaLinear(64, 1) #6144

    def forward(self, inputs, params=None):
        
        inputs = inputs.reshape(10,300,1) #change to (20,300,1) for # support set = 10 and batch_size = 20
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        #print(features.size())
        features = features.view((features.size(0), -1))
        
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits, features

class GNN_prediction(torch.nn.Module):

    def __init__(self, layer_number, emb_dim, jk = "last", dropout_prob= 0, pooling = "mean", gnn_type = "gin"):
        super(GNN_prediction, self).__init__()
        
        self.num_layer = layer_number
        self.drop_ratio = dropout_prob
        self.jk = jk
        self.emb_dim = emb_dim

        if self.num_layer < 2:
            raise ValueError("Number of layers must be > 2.")

        self.gnn = GNN(layer_number, emb_dim, jk, dropout_prob, gnn_type = gnn_type)
        
        if pooling == "mean":
            self.pool = global_mean_pool
        else:
            raise ValueError("Invalid pooling.")

        self.mult = 1
        self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, 1)
        
    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file, map_location='cuda:0'), strict = False)
        
    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("The arguments are unmatched!")

        node_embeddings = self.gnn(x, edge_index, edge_attr)
            
        pred_gnn = self.graph_pred_linear(self.pool(node_embeddings, batch))
        
        return pred_gnn, node_embeddings
        
        
if __name__ == "__main__":
    pass
