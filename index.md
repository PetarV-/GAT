---
layout: default
title: {{ site.name }}
---

# Overview

A multitude of important real-world datasets come together with some form of graph structure: social networks, citation networks, protein-protein interactions, brain connectome data, etc. Extending neural networks to be able to properly deal with this kind of data is therefore a very important direction for machine learning research, but one that has received comparatively rather low levels of attention until very recently.

| ![](https://www.dropbox.com/s/uip789ds97jvoak/graphs.png?raw=1) | 
| :-------------------------: |
| *Motivating examples of graph-structured inputs: molecular networks, transportation networks, social networks and brain connectome networks.* | 

Here we will present our ICLR 2018 work on [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (GATs), novel neural network architectures that operate on graph-structured data, leveraging masked self-attentional layers ([Vaswani _et al._, 2017](https://arxiv.org/abs/1706.03762)) to address the shortcomings of prior methods based on graph convolutions or their approximations (including, but not limited to: [Bruna _et al._, 2014](https://arxiv.org/abs/1312.6203); [Duvenaud _et al._, 2015](https://arxiv.org/abs/1509.09292); [Li _et al._, 2016](https://arxiv.org/abs/1511.05493); [Defferrard _et al._, 2016](https://arxiv.org/abs/1606.09375); [Kipf and Welling, 2017](https://arxiv.org/abs/1609.02907); [Monti _et al._, 2017](https://arxiv.org/abs/1611.08402); [Hamilton _et al._, 2017](https://arxiv.org/abs/1706.02216)).

# Motivation for graph convolutions

We can think of graphs as encoding a form of _irregular spatial structure_---and therefore, it would be highly appropriate if we could somehow generalise the **convolutional** operator (as used in CNNs) to operate on arbitrary graphs!

CNNs are a major workforce when it comes to working with image data. They exploit the fact that images have a highly rigid and regular connectivity pattern (each pixel "connected" to its eight neighbouring pixels), making such an operator trivial to deploy (as a small kernel matrix which is slid across the image).

| ![](https://camo.githubusercontent.com/3309220c48ab22c9a5dfe7656c3f1639b6b1755d/68747470733a2f2f7777772e64726f70626f782e636f6d2f732f6e3134713930677a386138726278622f32645f636f6e766f6c7574696f6e2e706e673f7261773d31) | 
| :-------------------------: |
| *2D convolutional operator as applied to a grid-structured input (e.g. image).* | 

Arbitrary graphs are a **much harder** challenge! Ideally, we would like to aggregate information across each of the nodes' neighbourhoods in a principled manner, but we are no longer guaranteed such rigidity of structure.

| ![](https://www.dropbox.com/s/zgj3a0hqabe4e4i/graph_conv.png?raw=1) |
| :-------------------------: |
| *A desirable form of a graph convolutional operator.* |

Enumerating the desirable traits of image convolutions, we arrive at the following properties we would ideally like our graph convolutional layer to have:
* **Computational and storage efficiency** (requiring no more than \\(O(V+E)\\) time and memory);
* **Fixed** number of parameters (independent of input graph size);
* **Localisation** (acting on a _local neighbourhood_ of a node);
* Ability to specify **arbitrary importances** to different neighbours;
* Applicability to **inductive problems** (arbitrary, unseen graph structures).

Satisfying all of the above at once has proved to be quite challenging, and indeed, none of the prior techniques have been successful at achieving them simultaneously.

## Towards a viable graph convolution

Consider a graph of \\(n\\) nodes, specified as a set of node features, \\((\vec{h}\_1, \vec{h}\_2, \dots, \vec{h}\_n)\\), and an adjacency matrix \\(\bf A\\), such that \\({\bf A}\_{ij} = 1\\) if \\(i\\) and \\(j\\) are connected, and \\(0\\) otherwise[^1]. A **graph convolutional layer** then computes a set of new node features, \\((\vec{h}'\_1, \vec{h}'\_2, \dots, \vec{h}'\_n)\\), based on the input features as well as the graph structure. 

Every graph convolutional layer starts off with a shared node-wise feature transformation (in order to achieve a higher-level representation), specified by a weight matrix \\({\bf W}\\). This transforms the feature vectors into \\(\vec{g}\_i = {\bf W}\vec{h}\_i\\). After this, the vectors \\(\vec{g}\_i\\) are typically recombined in some way at each node.

In general, to satisfy the localisation property, we will define a graph convolutional operator as an aggregation of features across neighbourhoods; defining \\(\mathcal{N}\_i\\) as the neighbourhood of node \\(i\\) (typically consisting of all first-order neighbours of \\(i\\), including \\(i\\) itself), we can define the output features of node \\(i\\) as:

$$
\vec{h}'_i = \sigma\left(\sum_{j\in\mathcal{N}_i}\alpha_{ij}\vec{g}_j\right)
$$

where \\(\sigma\\) is an activation function, and \\(\alpha\_{ij}\\) specifies the _weighting factor_ (importance) of node \\(j\\)'s features to node \\(i\\). 

Most prior work defines \\(\alpha\_{ij}\\) _explicitly_[^2] (either based on the structural properties of the graph, or as a learnable weight); this requires compromising at least one other desirable property.

# Graph Attention Networks

We instead decide to let \\(\alpha\_{ij}\\) be **implicitly** defined, employing _self-attention_ over the node features to do so. This choice was not without motivation, as self-attention has previously been shown to be self-sufficient for state-of-the-art-level results on machine translation, as demonstrated by the Transformer architecture ([Vaswani _et al._, 2017](https://arxiv.org/abs/1706.03762)).

Generally, we let \\(\alpha\_{ij}\\) be computed as a byproduct of an _attentional mechanism_, \\(a : \mathbb{R}^N \times \mathbb{R}^N \rightarrow \mathbb{R}\\), which computes unnormalised coefficients \\(e\_{ij}\\) across pairs of nodes \\(i, j\\), based on their features:

$$
e_{ij} = a(\vec{h}_i, \vec{h}_j)
$$

We inject the graph structure by only allowing node \\(i\\) to attend over nodes in its neighbourhood, \\(j \in \mathcal{N}\_i\\). These coefficients are then typically normalised using the softmax function, in order to be comparable across different neighbourhoods:

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k\in\mathcal{N}_i}\exp(e_{ik})}
$$

With the setup of the preceding section, this fully specifies a [Graph Attention Network](https://arxiv.org/abs/1710.10903) (GAT) layer!

Our framework is agnostic to the choice of attentional mechanism \\(a\\): in our experiments, we employed a simple single-layer neural network. The parameters of the mechanism are trained jointly with the rest of the network in an end-to-end fashion.

## Properties

A brief analysis of the properties of this layer reveals that it satisfies all of the desirable properties for a graph convolution:

* **Computationally efficient**: the computation of attentional coefficients can be parallelised across all edges of the graph, and the aggregation may be parallelised across all nodes;
* **Storage efficient**: It is possible to implement a GAT layer using sparse matrix operations only, requiring no more than \\(O(V+E)\\) entries to be stored anywhere;
* **Fixed** number of parameters, irrespective of the graph's node or edge count;
* Trivially **localised**, as we only attend over neighbourhoods;
* Allows for (implicitly) specifying **different importances** to **different neighbours**;
* Readily applicable to **inductive problems**, as it is a shared _edge-wise_ mechanism and therefore does not depend on the global graph structure!

To the best of our knowledge, this is the _first_ proposed graph convolution layer to do so.

These theoretical properties have been further validated within [our paper](https://arxiv.org/abs/1710.10903) by matching or exceeding state-of-the-art performance across four challenging transductive and inductive node classification benchmarks (Cora, Citeseer, PubMed and PPI). t-SNE visualisations on the Cora dataset further demonstrate that our model is capable of effectively discriminating between its target classes.

| ![](http://www.cl.cam.ac.uk/~pv273/images/gat_tsne.jpg) |
| :-------------------------: |
| *t-SNE + Attentional coefficients of a pre-trained GAT model, visualised on the Cora citation network dataset.* |

## Regularisation

To stabilise the learning process of self-attention, we have found _multi-head attention_ to be very beneficial (as was the case in [Vaswani _et al._, 2017](https://arxiv.org/abs/1706.03762)). Namely, the operations of the layer are independently replicated \\(K\\) times (each replica with different parameters), and outputs are featurewise aggregated (typically by concatenating or adding).

$$
\vec{h}'_i = {\LARGE \|}_{k=1}^K \sigma\left(\sum_{j\in\mathcal{N}_i}\alpha_{ij}^k{\bf W}^k\vec{h}_j\right)
$$

where \\(\alpha_{ij}^k\\) are the attention coefficients derived by the \\(k\\)-th replica, and \\({\bf W}^k\\) the weight matrix specifying the linear transformation of the \\(k\\)-th replica.

| ![](http://www.cl.cam.ac.uk/~pv273/images/gat.jpg) |
| :-------------------------: |
| *A GAT layer with multi-head attention. Every neighbour \\(i\\) of node \\(1\\) sends its own vector of attentional coefficients, \\(\vec{\alpha}\_{1i}\\) one per each attention head \\(\alpha\_{1i}^k\\). These are used to compute \\(K\\) separate linear combinations of neighbours' features \\(\vec{h}\_i\\), which are then aggregated (typically by concatenation or averaging) to obtain the next-level features of node \\(1\\), \\(\vec{h}'\_1\\).* |

Furthermore, we have found that applying _dropout_ ([Srivastava _et al._, 2014](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)) to the attentional coefficients \\(\alpha_{ij}\\) was a highly beneficial regulariser, especially for small training datasets. This effectively exposes nodes to _stochastically sampled neighbourhoods_ during training, in a manner reminiscent of the (concurrently published) FastGCN method ([Chen _et al._, 2018](https://arxiv.org/abs/1801.10247)).

# Applications

Following the publication of the GAT paper, we have been delighted to witness (as well as contribute to) several new lines of research in which GAT-like architectures have been leveraged to solve challenging problems. Here, we will highlight two subsequent contributions that some of us have developed, and outline some works that have been released by others.

## Mesh-based parcellation of the cerebral cortex

In this work (done in collaboration with the University of Cambridge Department of Psychiatry and the Montréal Neurological Institute), we have considered the task of _cortical mesh segmentation_ (predicting functional regions for locations on a human brain mesh). To do this, we have leveraged functional MRI data from the Human Connectome Project (HCP). We found that graph convolutional methods (such as GCNs and GATs) are capable of setting state-of-the-art results, exploiting the underlying structure of a brain mesh better than all prior approaches, enabling more informed decisions.

| ![](https://www.dropbox.com/s/f4362wl6uxpyj4d/parcellation.png?raw=1) |
| :-------------------------: |
| *Area parcellation qualitative results for several methods on a test subject. The approach of [Jakobsen et al. (2016)](https://www.ncbi.nlm.nih.gov/pubmed/27693796) is the prior state-of-the-art.* |

For more details, please refer to our MIDL publication ([Cucurull _et al._, 2018](https://openreview.net/forum?id=rkKvBAiiz)).

## Neural paratope prediction

Antibodies are a critical part of the immune system, having the function of directly neutralising or tagging undesirable objects (the antigens) for future destruction. Here we consider the task of _paratope prediction_: predicting the amino acids of an antibody that participate in binding to a target antigen. A viable paratope predictor is a significant facilitator to antibody design, which in turn will contribute to the development of personalised medicine. 

In this work, we build on Parapred, the previous state of the art approach of [Liberis _et al._ (2018)](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/bty305/4972995), substituting its convolutional and recurrent layers with à trous convolutional and attentional layers, respectively. These layers have been shown to perform more favourably, and allowed us to, for the first time, positively exploit antigen data. We do this through _cross-modal attention_, allowing amino acids of the antibody to attend over the amino acids of the antigen (which could be seen as a GAT-like model applied to a bipartite antibody-antigen graph). This allowed us to set new state-of-the-art results on this task, along with obtaining insightful interpretations about the model's mechanism of action.

| ![](https://www.dropbox.com/s/hkgy68zjbh8afdi/try2.png?raw=1) | ![](https://www.dropbox.com/s/7zo2co53iaw6ifp/agvis.png?raw=1) |
| :-------------------------: |
| *Antibody amino acid binding probabilities to the antigen (in gold) assigned by our model for a test antibody-antigen complex. Warmer colours indicate higher probabilities.* | *Normalised antigen attention coefficients for a single (binding) antibody amino acid (in red). Warmer colours indicate higher coefficients.* |

For more details, please refer to our ICML WCB publication ([Deac _et al._, 2018](https://arxiv.org/abs/1806.04398)).

## Additional related work

Finally, we outline three interesting relevant pieces of work that leverage or further build on GAT(-like) models. The list is by no means exhaustive!

* _Gated Attention Networks_ (GaAN) ([Zhang _et al._, 2018](https://arxiv.org/abs/1803.07294)), where gating mechanisms are inserted into the multi-head attention system of GATs, in order to give different value to different heads' computations. Several challenging baselines are outperformed on both inductive node classification tasks (Reddit, PPI) and a traffic speed forecasting task (METR-LA).
* _DeepInf_ ([Qiu _et al._, 2018](https://www.haoma.io/pdf/deepinf.pdf)), leveraging graph convolutional layers to modelling _influence locality_ (predicting whether a node will perform a particular action, given the action statuses of its \\(r\\)-hop neighbours at a particular point in time). Notably, this is the first study where attentional mechanisms (GAT) appear to be necessary for surpassing baseline approaches (such as SVMs or logistic regression), given the heterogeneity of the edges. Furthermore, a very nice qualitative analysis is performed on the action mechanism of the various attention heads employed by the GAT model.
* _Attention Solves Your TSP_ ([Kool and Welling, 2018](https://arxiv.org/abs/1803.08475)), where GAT-like layers (using the Transformer-style attention mechanism) have been successfully applied to solving _combinatorial optimisation_ problems, specifically the Travelling Salesman Problem (TSP).

# Conclusions

We have presented graph attention networks (GATs), novel convolution-style neural networks that operate on graph-structured data, leveraging masked self-attentional layers. The graph attentional layer utilised throughout these networks is computationally efficient (does not require costly matrix operations, and is parallelisable across all nodes in the graph), allows for (implicitly) assigning different importances to different nodes within a neighborhood while dealing with different sized neighborhoods, and does not depend on knowing the entire graph structure upfront---thus addressing many of the theoretical issues with approaches. Results, both within our work and the numerous subsequently published work, highlight the importance of such architectures towards building principled graph convolutional neural networks.

## Citation

If you make advantage of the GAT model in your research, please cite the following:

```
@article{
  velickovic2018graph,
  title="{Graph Attention Networks}",
  author={Veli{\v{c}}kovi{\'{c}}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`{o}}, Pietro and Bengio, Yoshua},
  journal={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=rJXMpikCZ},
}
```

---

[^1]: In general, the adjacency matrix may be weighted, contain edges of different types, or the edges may even have features of their own. We do not consider these cases for simplicity, but the GAT model may be trivially extended to handle them, as was done by EAGCN ([Shang _et al._, 2018](https://arxiv.org/abs/1802.04944v1)).
[^2]: A notable exception is the MoNet framework ([Monti _et al._, 2017](https://arxiv.org/abs/1611.08402)) which our work can be considered an instance of. However, in all of the experiments presented in this paper, the weights were implicitly specified based on the graph's structural properties, meaning that two nodes with same local topologies would always necessarily receive the same (learned) weighting---thus still violating one of the desirable properties for a graph convolution.
