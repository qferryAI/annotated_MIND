# annotated_MIND
Python implementation of [Manifold Inference from Neural Dynamics](https://www.biorxiv.org/content/10.1101/418939v2) (MIND).

## About MIND

MIND is an [unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning) algortihm designed to discover/reveal low-dimensional manifolds from high-dimensional dynamical/time series data. The method was co-invented by Ryan Low and Sam Lewallen to analyze neural recording data and is brilliantly explained in their [BioRxiv publication](https://www.biorxiv.org/content/10.1101/418939v2). For an additional example of MIND in action, check out this excellent work by [Nieh et al. 2021](https://www.nature.com/articles/s41586-021-03652-7).  

The schematic below presents in a nutshell the internal workings of MIND:
<img src="https://github.com/qferryAI/annotated_MIND/blob/main/pics/mind_schematic.png" style='width: 50%; margin-left: auto;margin-right: auto;'>

## About this code

This implementation follows closely the method of the original paper:
> [Low, R. J., Lewallen, S., Aronov, D., Nevers, R. & Tank, D. W. 
Probing variability in a cognitive map using manifold inference from neural dynamics. 
Biorxiv 418939 (2018) doi:10.1101/418939.](https://www.biorxiv.org/content/10.1101/418939v2)

## Module requirements

This code relies on the following Python modules:
- [numpy](https://numpy.org/doc/#) for everything linear algebra.
- [autograd](https://github.com/HIPS/autograd) for gradient-based optimization.
- [scipy](https://scipy.org/) for mutilvariate normal & shortest path algorithm.
- [scikit-learn](https://scikit-learn.org/stable/) for kNN & dimensionality reduction algorithms.
- [matplotlib](https://matplotlib.org/) for visualization.

To make things easier, I am providing the annotated_MIND.yaml file of the conda environment I used for this project. To build and use the same environment, make sure you have [conda](https://docs.conda.io/en/latest/) install on your machine, clone this repo, navigate to the folder, and enter the following command line:
`conda env create -f annotated_MIND.yaml`.

## Enjoy!



