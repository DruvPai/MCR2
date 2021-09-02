# Learning Diverse and Discriminative Representations via the Principle of Maximal Coding Rate Reduction
This repository is an unofficial implementation of the following paper,

 [ReduNet: A White-box Deep Network from the Principle of Maximizing Rate Reduction](https://arxiv.org/abs/2105.10446) (2021) 

by [Kwan Ho Ryan Chan](https://ryanchankh.github.io)* (UC Berkeley), [Yaodong Yu](https://yaodongyu.github.io/)* (UC Berkeley), [Chong You](https://sites.google.com/view/cyou)* (UC Berkeley), [Haozhi Qi](https://haozhi.io/) (UC Berkeley), [John Wright](http://www.columbia.edu/~jw2966/) (Columbia), and [Yi Ma](http://people.eecs.berkeley.edu/~yima/) (UC Berkeley),

which includes the implementations of the Maximal Coding Rate Reduction (**MCR<sup>2</sup>**) objective function part ([**MCR<sup>2</sup>** paper link](https://arxiv.org/abs/2006.08558)). This also serves as the host repository for the Pip package.

## What is Maximal Coding Rate Reduction? 
Our goal is to learn a mapping that maps the high-dimensional data that lies in a low-dimensional manifold to low-dimensional subspaces with the following three properties: 

1. _Between-Class Discriminative_: Features of samples from different classes/clusters should be highly uncorrelatedand belong to different low-dimensional linear subspaces
2. _Within-Class Compressible_: Features of samples from the same class/cluster should be relatively correlated in a sense that they belong to a low-dimensional linear subspace
3. _Maximally Diverse Representation_: Dimension (or variance) of features for each class/cluster should beas large as possibleas long as they stay uncorrelated from the other classes

To achieve this, we propose an objective function called **Maximal Coding Rate Reduction** (MCR<sup>2</sup>). In our paper, we provide not only theoretical guarantees to the desired properties upon convergence, but also practical properties such as robustness to label corruption and empirical results such as state-of-the-art unsupervised clustering performance. For more details on algorithm design, please refer to our paper.
