# Maximal Coding Rate Reduction
This repository is an unofficial implementation of the following papers:

- [Learning Diverse and Discriminative Representations via the Principle of Maximal Coding Rate Reduction
](https://arxiv.org/abs/2006.08558) (2020)
- [ReduNet: A White-box Deep Network from the Principle of Maximizing Rate Reduction](https://arxiv.org/abs/2105.10446) (2021)
- More on the way.

This also serves as the host repository for [the Pip package](https://pypi.org/project/mcr2/0.0.1/).

## What is Maximal Coding Rate Reduction? 
Our goal is to learn a mapping that maps the high-dimensional data that lies in a low-dimensional manifold to low-dimensional subspaces with the following three properties: 

1. _Between-Class Discriminative_: Features of samples from different classes/clusters should be highly uncorrelatedand belong to different low-dimensional linear subspaces
2. _Within-Class Compressible_: Features of samples from the same class/cluster should be relatively correlated in a sense that they belong to a low-dimensional linear subspace
3. _Maximally Diverse Representation_: Dimension (or variance) of features for each class/cluster should beas large as possibleas long as they stay uncorrelated from the other classes

To achieve this, we propose an objective function called **Maximal Coding Rate Reduction** (MCR<sup>2</sup>). In our paper, we provide not only theoretical guarantees to the desired properties upon convergence, but also practical properties such as robustness to label corruption and empirical results such as state-of-the-art unsupervised clustering performance. For more details on algorithm design, please refer to our paper.

## What is ReduNet?
Our goal is to build a neural network for representation learning with the following properties:

1. _Interpretable_: We should be able to interpret each network operator and assign precise meaning to each layer and parameter.
2. _Forward-Propagation Only_: The network should be trained using much-more interpretable forward-propagation methods, as opposed to back-propagation which tends to create black-boxes.
3. _Use MCR<sup>2</sup>_: The network should seek to optimize MCR<sup>2</sup> loss function, as the purpose is distribution learning.

To achieve this, we propose a neural network architecture and algorithms called **ReduNet**. In our paper, we provide not only theoretical interpretations and a precise derivation of each operator in the network, but also connections to other architectures that form naturally as components of ReduNet. We also provide empirical justification for the power of ReduNet. For more details on algorithm design, please refer to our paper.