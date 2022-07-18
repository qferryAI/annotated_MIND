# annotated_MIND
Python implementation of [Manifold Inference from Neural Dynamics](https://www.biorxiv.org/content/10.1101/418939v2) (MIND).

## About MIND

MIND is an [unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning) algortihm designed to discover/reveal low-dimensional manifolds from high-dimensional dynamical/time series data (e.g., neural recording data). 

This implementation follows closely the method of the original paper:
> [Low, R. J., Lewallen, S., Aronov, D., Nevers, R. & Tank, D. W. 
Probing variability in a cognitive map using manifold inference from neural dynamics. 
Biorxiv 418939 (2018) doi:10.1101/418939.](https://www.biorxiv.org/content/10.1101/418939v2)

## About this code


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



