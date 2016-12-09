# BoltzmannMachines

This package implements algorithms for training and evaluating several types of Boltzmann Machines (BMs):

* Learning of Restricted Boltzmann Machines (RBMs) using Contrastive Divergence (CD).
* Greedy layerwise pre-training of Deep Boltzmann Machines (DBMs) and Multimodal DBMs
* Learning procedure for general Boltzmann Machines using mean-field inference and stochastic approximation. Applicable to DBMs and Multimodal DBMs and used for fine-tuning the weights after the pre-training.
* Exact calculation of the likelihood of BMs (only suitable for small models)
* Annealed Importance Sampling (AIS) for estimating the likelihood of larger BMs


## References

## Types of Boltzmann Machines

The package contains the following types of RBMs:




## Overview of functions

The following table provides tables with lists of functions of the package together with a short description. The tables are grouped by function. You can find more detailed descriptions for each function using the Julia help mode (entered by typing `? ` at the beginning of the Julia command prompt).

### Functions for Training

#### Training of RBMs

Function name    | Short description
---------------- | -----------------
`fitrbm`         | Fits a RBM model to a dataset using CD.
`trainrbm!`      | Trains a DBM or Multimodal DBM using the learning procedure for a general Boltzmann Machine.
`samplevisible` (`samplehidden`) | Gibbs sampling of visible (hidden) nodes' states given the hidden (visible) nodes' states in an RBM.


#### Training of DBMs and Multimodal DBMs

Function name    | Short description
---------------- | -----------------
`addlayer!`      | Adds an additional layer of nodes to a DBM and pre-trains the new weights.
`fitdbm`         | Fits a DBM model to a dataset. This includes pre-training, followed by the general Boltzmann Machine learning procedure for fine-tuning.
`gibbssample!`   | Performs Gibbs sampling in a DBM or Multimodal DBM.
`meanfield`      | Computes the mean-field inference of the hidden nodes' activations in a DBM or Multimodal DBM.
`traindbm!`      | Trains a DBM or Multimodal DBM using the learning procedure for a general Boltzmann Machine.


### Functions for evaluating a trained model

Function name          | Short description
--------------         | -----------------
`aisimportanceweights` | Performs AIS on a BM.
`freeenergy`           | Computes the free energy of an RBM.
`loglikelihood`        | Estimates the loglikelihood of a dataset in a BM model using AIS.
`logpartitionfunction` | Estimates the log of the partition function of a BM. 
`logproblowerbound`    | Estimates the lower bound of the log probability of a dataset in a DBM model.
`reconstructionerror`  | Computes the reconstruction error of a dataset in am RBM.
`sampleparticles`      | Samples from a BM model.


### Monitoring the learning process
The functions of the form `monitor*!` can be used for monitoring a property of the model during the learning process.
The following words, corresponding to properties, may stand in place of `*`: 

* `freeenergy`
* `exactloglikelihood`
* `loglikelihood`
* `logproblowerbound`
* `reconstructionerror`
* `weightsnorm`

The results of evaluations are stored in `Monitor` objects. The evaluations can be plotted by the function `plotevaluation(monitor, *)` in the submodule `BMPlots`.

For intended usage of these functions, best see the examples.


## Examples
TODO

