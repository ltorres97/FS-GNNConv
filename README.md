## Few-Shot Learning via Graph Embeddings with Convolutional Networks for Low-Data Molecular Property Prediction

We introduce a two-module GNN-CNN architecture, FS-GNNConv, that accepts the compound chemical structure to exploit the rich information of graph embeddings. A few-shot learning (FSL) strategy is used to learn from task-transferable knowledge and predict new molecular properties across tasks in Tox21 and SIDER datasets.

The first module is a graph isomorphism network (GIN) to encode the topological structure of molecular graphs as a set of node (atoms) and edge (chemical bonds) features. These graphs are then converted into embedding representations to support further learning. A convolutional neural network (CNN) exploits the rich information of these embedded descriptors to compute deep vectorial representations. These representations are then propagated across convolutional layers to detect global patterns shared among molecules and identify local connections between close and distant neighbors in the graph. Deep representations are later used to predict task-specific molecular properties.

![ScreenShot](results/figures/fsgnnconv.png?raw=true)

A meta-learning framework was explored to optimize model parameters across tasks and quickly adapt to new molecular properties on few-shot data. 

Extensive experiments on real multiproperty prediction data demonstrate the predictive power and stable performances of the proposed model when inferring specific target properties adaptively.

This repository provides the source code and datasets for the proposed work.

Contact Information: (uc2015241578@student.uc.pt, luistorres@dei.uc.pt), if you have any questions about this work.

## Data Availability and Pre-Processing

The Tox21 and SIDER datasets are downloaded from [Data](http://snap.stanford.edu/gnn-pretrain/data/) (chem_dataset.zip). 

We pre-process the data and transform SMILES strings into molecular graphs using RDKit.Chem. 

Data pre-processing and pre-trained models are implemented based on [Strategies for Pre-training Graph Neural Networks (Hu et al.) (2020)](https://arxiv.org/abs/1905.12265).

## Package Installation

We used the following Python packages for core development.

```
- torch = 1.9.0
- torch-cluster = 1.5.9
- torch-geometric = 2.0.4
- torch-scatter = 2.0.9
- torch-sparse = 0.6.12
- torch-spline-conv = 1.2.1
- torchvision = 0.10.0
- torchmeta = 1.8.0
- scikit-learn = 1.0.2
- seaborn = 0.11.2
- scipy = 1.8.0
- numpy = 1.21.5
- tqdm = 4.50.0
- tensorflow = 2.8.0
- keras = 2.8.0
- tsnecuda = 3.0.1
- tqdm = 4.62.3
- matplotlib = 3.5.1
- pandas = 1.4.1
- networkx = 2.7.1
- rdkit
```

## Performance Results

The results below are the mean ROC-AUC scores and standard deviations obtained on 20 experiments with 20 different (5+,5-) random support sets.

| Dataset 	|   Task   	|     GIN     	|     GCN     	|  GraphSAGE  	| FS-GNNConv (GIN+CNN) 	| delta(ROC-AUC) 	|
|:-------:	|:--------:	|:-----------:	|:-----------:	|:-----------:	|:--------------------:	|:----------:	|
|   Tox21  	|  SR-HSE  	| 61.44+-1.17 	| 65.76+-2.49 	| 64.19+-2.50 	|      76.37+-0.48     	|   +10.61   	|
|   Tox21  	|  SR-MMP  	| 57.55+-0.90 	| 64.85+-1.28 	| 63.56+-3.89 	|      77.60+-0.33     	|   +12.75   	|
|   Tox21  	|  SR-p53  	| 59.15+-1.13 	| 63.02+-1.49 	| 61.75+-3.45 	|      72.67+-0.59     	|    +9.65   	|
|     -   	|  Average 	|    59.38    	|    64.54    	|    63.17    	|         75.55        	|   +11.01   	|
|   SIDER  	|  R.U.D.  	| 69.77+-1.08 	| 60.62+-1.56 	| 62.62+-0.64 	|      69.44+-0.49     	|    -0.33   	|
|   SIDER  	| P.P.P.C. 	| 77.05+-0.66 	| 71.89+-1.25 	| 74.16+-1.19 	|      70.88+-0.56     	|    -6.17   	|
|   SIDER  	|  E.L.D.  	| 70.24+-1.03 	| 62.78+-0.96 	| 64.50+-0.75 	|      70.54+-0.63     	|    +0.30   	|
|   SIDER  	|   C.D.   	| 68.66+-0.90 	| 60.82+-1.11 	| 61.81+-0.76 	|      70.52+-0.57     	|    +1.86   	|
|   SIDER  	|  N.S.D.  	| 65.23+-0.70 	| 58.77+-2.27 	| 59.00+-1.41 	|      67.39+-0.81     	|    +2.16   	|
|   SIDER  	| I.P.P.C. 	| 72.92+-1.03 	| 65.62+-1.95 	| 67.01+-0.76 	|      71.38+-0.68     	|    -1.54   	|
|     -   	|  Average 	|    70.63    	|    63.42    	|    64.85    	|         70.03        	|    -0.60   	|

## References

[1] Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V., Leskovec, J.: Strategies for pre-training graph neural networks. CoRR abs/1905.12265 (2020). https://doi.org/10.48550/ARXIV.1905.12265
```
@inproceedings{
hu2020pretraining,
title={Strategies for Pre-training Graph Neural Networks},
author={Hu, Weihua and Liu, Bowen and Gomes, Joseph and Zitnik, Marinka and Liang, Percy and Pande, Vijay and Leskovec, Jure},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=HJlWWJSFDH},
}
```

[2] Finn, C., Abbeel, P., Levine, S.: Model-agnostic meta-learning for fast adaptation of deep networks. In: 34th International Conference on Machine Learning, ICML 2017, vol. 3 (2017). https://doi.org/10.48550/arXiv.1703.03400
```
@article{finn17maml,
  author    = {Chelsea Finn and Pieter Abbeel and Sergey Levine},
  title     = {{Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks}},
  journal   = {International Conference on Machine Learning (ICML)},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.03400}
}
```

[3] Guo, Z., Zhang, C., Yu, W., Herr, J., Wiest, O., Jiang, M., Chawla, N.V.: Few-shot graph learning for molecular property prediction. In: The Web Conference 2021 - Proceedings of the World Wide Web Conference, WWW 2021 (2021). https://doi.org/10.1145/3442381.3450112
```
@article{guo2021few,
  title={Few-Shot Graph Learning for Molecular Property Prediction},
  author={Guo, Zhichun and Zhang, Chuxu and Yu, Wenhao and Herr, John and Wiest, Olaf and Jiang, Meng and Chawla, Nitesh V},
  journal={arXiv preprint arXiv:2102.07916},
  year={2021}
}
```
[4] Tristan Deleu, Tobias Würfl, Mandana Samiei, Joseph Paul Cohen, and Yoshua Bengio. Torchmeta: A Meta-Learning library for PyTorch, 2019. 
https://doi.org/10.48550/arXiv.1909.06576
```
@misc{deleu2019torchmeta,
  title={{Torchmeta: A Meta-Learning library for PyTorch}},
  author={Deleu, Tristan and W\"urfl, Tobias and Samiei, Mandana and Cohen, Joseph Paul and Bengio, Yoshua},
  year={2019},
  url={https://arxiv.org/abs/1909.06576},
  note={Available at: https://github.com/tristandeleu/pytorch-meta}
}
```
