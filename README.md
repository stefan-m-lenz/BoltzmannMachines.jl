# BoltzmannMachines

This Julia package implements algorithms for training and evaluating several types of Boltzmann Machines (BMs):

* Learning of Restricted Boltzmann Machines (RBMs) using Contrastive Divergence (CD)
* Greedy layerwise pre-training of Deep Boltzmann Machines (DBMs)
* Learning procedure for general Boltzmann Machines using mean-field inference and stochastic approximation. Applicable to DBMs and used for fine-tuning the weights after the pre-training
* Exact calculation of the likelihood of BMs (only suitable for small models)
* Annealed Importance Sampling (AIS) for estimating the likelihood of larger BMs

## Types of Boltzmann Machines

The package contains the following types of RBMs:

Type                    | Distribution of visible units    | Distribution of hidden units
------------------------|----------------------------------|-----------------------------
`BernoulliRBM`          | Bernoulli                        | Bernoulli
`GaussianBernoulliRBM`  | Gaussian                         | Bernoulli
`Binomial2BernoulliRBM` | Binomial distribution with n = 2 | Bernoulli
`BernoulliGaussianRBM`  | Bernoulli                        | Gaussian

TODO DBMs:


## Overview of functions

The following tables provide an overview of the functions of the package, together with a short description. You can find more detailed descriptions for each function using the Julia help mode (entered by typing `?` at the beginning of the Julia command prompt).

### Functions for Training

#### Training of RBMs

Function name    | Short description
---------------- | -----------------
`fitrbm`         | Fits a RBM model to a dataset using CD.
`samplevisible` (`samplehidden`) | Gibbs sampling of visible (hidden) nodes' states given the hidden (visible) nodes' states in an RBM.
`visiblepotential` (`hiddenpotential`) | Computes the deterministic potential for the activation of the visible (hidden) nodes of an RBM.
`visibleinput` (`hiddeninput`) | Computes the total input received by the visible (hidden) layer of an RBM.
`trainrbm!` | 
`initrbm` | 


#### Training of DBMs

Function name    | Short description
---------------- | -----------------
`addlayer!`      | Adds an additional layer of nodes to a DBM and pre-trains the new weights.
`fitdbm`         | Fits a DBM model to a dataset. This includes pre-training, followed by the general Boltzmann Machine learning procedure for fine-tuning.
`gibbssample!`   | Performs Gibbs sampling in a DBM.
`meanfield`      | Computes the mean-field inference of the hidden nodes' activations in a DBM.
`stackrbms`      | Greedy layerwise pre-training of a DBM model or a Deep Belief Network.
`traindbm!`      | Trains a DBM using the learning procedure for a general Boltzmann Machine.


### Functions for evaluating a trained model

Function name          | Short description
--------------         | -----------------
`aisimportanceweights` | Performs AIS on a BM and calculates the importance weights for estimating the BM's partition function.
`freeenergy`           | Computes the mean free energy of a data set in an RBM model.
`loglikelihood`        | Estimates the mean loglikelihood of a dataset in a BM model using AIS.
`logpartitionfunction` | Estimates the log of the partition function of a BM. 
`logproblowerbound`    | Estimates the mean lower bound of the log probability of a dataset in a DBM model.
`reconstructionerror`  | Computes the mean reconstruction error of a dataset in an RBM model.
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

The results of evaluations are stored in `Monitor` objects. The evaluations can be plotted by calling the function `plotevaluation` in the submodule `BMPlots` as `BMPlots.plotevaluation(monitor, key)`, with the key being one of the constants `monitor*` defined in the module.

For intended usage of these functions, best see the examples.

## Examples

Prerequisite for running the following code snippets is that the `BoltzmannMachines` package is installed and loaded:

    Pkg.add("BoltzmannMachines")
    using BoltzmannMachines
    
If you want to use the plotting functionality in the submodule `BMPlots`, you are required to have the Julia package [Gadfly](http://gadflyjl.org/stable/) installed.

    
### RBMs



### DBMs

<!--TODO: Two ways, fitdbm or addlayer! and traindbm!
 Small dbm, exact, big dbm loglikelihood am Schlus, logproblowerbound während training, alle 2 Schritte.
Partitioned Training-->


## References

[Salakhutdinov, 2015] : Learning Deep Generative Models
[Salakhutdinov+Hinton, 2012] : An Efficient Learning Procedure for Deep Boltzmann Machines
[Salakhutdinov, 2008] : Learning and Evaluating Boltzmann Machines
[Krizhevsky, 2009] : Learning Multiple Layers of Features from Tiny Images
[Srivastava+Salakhutdinov, 2014] : Multimodal Learning with Deep Boltzmann Machines

