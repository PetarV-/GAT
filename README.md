# GAT
Graph Attention Networks (Veličković *et al.*, ICLR 2018): [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)

GAT layer            |  t-SNE + Attention coefficients on Cora
:-------------------------:|:-------------------------:
![](http://www.cl.cam.ac.uk/~pv273/images/gat.jpg)  |  ![](http://www.cl.cam.ac.uk/~pv273/images/gat_tsne.jpg)

## Overview
Here we provide the implementation of a Graph Attention Network (GAT) layer in TensorFlow, along with a minimal execution example (on the Cora dataset). The repository is organised as follows:
- `data/` contains the necessary dataset files for Cora;
- `models/` contains the implementation of the GAT network (`gat.py`);
- `pre_trained/` contains a pre-trained Cora model (achieving 84.4% accuracy on the test set);
- `utils/` contains:
    * an implementation of an attention head, along with an experimental sparse version (`layers.py`);
    * preprocessing subroutines (`process.py`);
    * preprocessing utilities for the PPI benchmark (`process_ppi.py`).

Finally, `execute_cora.py` puts all of the above together and may be used to execute a full training run on Cora.

## Sparse version
An experimental sparse version is also available, working only when the batch size is equal to 1.
The sparse model may be found at `models/sp_gat.py`.

You may execute a full training run of the sparse model on Cora through `execute_cora_sparse.py`.

## Dependencies

The script has been tested running under Python 3.5.2, with the following packages installed (along with their dependencies):

- `numpy==1.14.1`
- `scipy==1.0.0`
- `networkx==2.1`
- `tensorflow-gpu==1.6.0`

In addition, CUDA 9.0 and cuDNN 7 have been used.

## Reference
If you make advantage of the GAT model in your research, please cite the following in your manuscript:

```
@article{
  velickovic2018graph,
  title="{Graph Attention Networks}",
  author={Veli{\v{c}}kovi{\'{c}}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`{o}}, Pietro and Bengio, Yoshua},
  journal={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=rJXMpikCZ},
  note={accepted as poster},
}
```

You may also be interested in the following unofficial ports of the GAT model:
- \[Keras\] [keras-gat](https://github.com/danielegrattarola/keras-gat), currently under development by [Daniele Grattarola](https://github.com/danielegrattarola);
- \[PyTorch\] [pyGAT](https://github.com/Diego999/pyGAT), currently under development by [Diego Antognini](https://github.com/Diego999).

## License
MIT
