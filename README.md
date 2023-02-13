## Few-Shot Learning via Graph Embeddings with Convolutional Networks for Low-Data Molecular Property Prediction

We introduce a two-module GNN-CNN architecture, FS-GNNConv, that accepts the compound chemical structure to exploit the rich information of graph embeddings. A few-shot learning (FSL) strategy is used to learn from task-transferable knowledge and predict new molecular properties across tasks in Tox21 and SIDER datasets.

The first module is a graph isomorphism network (GIN) to encode the topological structure of molecular graphs as a set of node (atoms) and edge (chemical bonds) features. These graphs are then converted into embedding representations to support further learning. A convolutional neural network (CNN) exploits the rich information of these embedded descriptors to compute deep vectorial representations. These representations are then propagated across convolutional layers to identify local connections between close and distant neighbors in the graph. Deep representations are later used to predict task-specific molecular properties.

![ScreenShot](results/figures/fsgnnconv.png?raw=true)

A two-module meta-learning framework was explored to optimize model parameters across few-shot tasks and quickly adapt to new molecular properties on few-shot data. 

![ScreenShot](results/figures/meta-fs-gnnconv.png?raw=true)

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
