# BoltzmannMachines.jl

[![Build Status](https://travis-ci.org/stefan-m-lenz/BoltzmannMachines.jl.svg?branch=master)](https://travis-ci.org/stefan-m-lenz/BoltzmannMachines.jl)

This Julia package implements algorithms for training and evaluating several types of Boltzmann Machines (BMs):

* Learning of Restricted Boltzmann Machines (RBMs) using Contrastive Divergence (CD)
* Greedy layerwise pre-training of Deep Boltzmann Machines (DBMs)
* Learning procedure for general Boltzmann Machines using mean-field inference and stochastic approximation. Applicable to DBMs and used for fine-tuning the weights after the pre-training
* Exact calculation of the likelihood of BMs (only suitable for small models)
* Annealed Importance Sampling (AIS) for estimating the likelihood of larger BMs

## Types of Boltzmann Machines

### Restricted Boltzmann Machines
The package contains the following types of RBMs (subtypes of `AbstractRBM`):

Type                    | Distribution of visible units    | Distribution of hidden units
------------------------|----------------------------------|-----------------------------
`BernoulliRBM`          | Bernoulli                        | Bernoulli
`GaussianBernoulliRBM`, `GaussianBernoulliRBM2` ([6])  | Gaussian                         | Bernoulli
`Binomial2BernoulliRBM` | Binomial distribution with n = 2 | Bernoulli
`BernoulliGaussianRBM`  | Bernoulli                        | Gaussian

### (Multimodal) Deep Boltzmann Machines

DBMs are implemented as vectors of RBMs. `BasicDBM`s have only Bernoulli distributed nodes and therefore consist of a vector of `BernoulliRBM`s.
DBMs with different types of visible units can be constructed
by using the corresponding RBM type in the first layer.
Actual `MultimodalDBM`s can be formed by using `PartitionedRBM`s, which is a type of `AbstractRBM` that is able to encapsulate non-connected RBMs of different types into an RBM-like layer.

All these types of DBMs can be trained using layerwise pre-training and fine-tuning employing the mean-field approximation. It is also possible to estimate or calculate the likelihood for these DBM types.

## Overview of functions

The following tables provide an overview of the functions of the package, together with a short description. You can find more detailed descriptions for each function using the Julia help mode (entered by typing `?` at the beginning of the Julia command prompt).

### Functions for Training

#### Training of RBMs

Function name    | Short description
---------------- | -----------------
`initrbm`        | Initializes an RBM model.
`trainrbm!`      | Performs CD-learning on an RBM model.
`fitrbm`         | Fits a RBM model to a dataset using CD. (Wraps `initrbm` and `trainrbm!`)
`samplevisible`, `samplevisible!` (`samplehidden`, `samplehidden!`) | Gibbs sampling of visible (hidden) nodes' states given the hidden (visible) nodes' states in an RBM.
`visiblepotential`, `visiblepotential!` (`hiddenpotential`, `hiddenpotential!`) | Computes the deterministic potential for the activation of the visible (hidden) nodes of an RBM.
`visibleinput`, `visibleinput!` (`hiddeninput`, `hiddeninput!`) | Computes the total input received by the visible (hidden) layer of an RBM.


#### Training of DBMs

Function name    | Short description
---------------- | -----------------
`fitdbm`         | Fits a DBM model to a dataset. This includes pre-training, followed by the general Boltzmann Machine learning procedure for fine-tuning.
`gibbssample!`   | Performs Gibbs sampling in a DBM.
`meanfield`      | Computes the mean-field inference of the hidden nodes' activations in a DBM.
`stackrbms`      | Greedy layerwise pre-training of a DBM model or a Deep Belief Network.
`traindbm!`      | Trains a DBM using the learning procedure for a general Boltzmann Machine.


### Partitioned training and joining of models

The following functions may be used to join models fitted on partitioned data sets. The weights cross-linking the models are initialized with zeros.

Function name | Short description
--------------|------------------
`joindbms`    | Joins two or more DBM models together.
`joinrbms`    | Joins two or more RBM models to form a joint RBM model of the same type.


### Functions for evaluating a trained model

Function name          | Short description
--------------         | -----------------
`aislogimpweights`     | Performs AIS on a BM and calculates the logarithmised importance weights for estimating the BM's partition function.
`freeenergy`           | Computes the mean free energy of a data set in an RBM model.
`loglikelihood`        | Estimates the mean loglikelihood of a dataset in a BM model using AIS.
`logpartitionfunction` | Estimates the log of the partition function of a BM.
`logproblowerbound`    | Estimates the mean lower bound of the log probability of a dataset in a DBM model.
`reconstructionerror`  | Computes the mean reconstruction error of a dataset in an RBM model.
`samples`              | Generates samples from the distribution defined by a BM model.


### Monitoring the learning process

The functions of the form `monitor*!` can be used for monitoring a property of the model during the learning process.
The following words, corresponding to the denominated properties, may stand in place of `*`:

* `freeenergy`
* `exactloglikelihood`
* `loglikelihood`
* `logproblowerbound`
* `reconstructionerror`
* `weightsnorm`

The results of evaluations are stored in `Monitor` objects. The evaluations can be plotted by calling the function `plotevaluation` in the submodule `BMPlots` as `BMPlots.plotevaluation(monitor, key)`, with the key being one of the constants `monitor*` defined in the module.

For intended usage of these functions, best see the [examples](test/examples.jl).


## Examples

You can find [example code here](test/examples.jl).

If you want to use the plotting functionality, you need to install the package [`BoltzmannMachinesPlots`](https://github.com/stefan-m-lenz/BoltzmannMachinesPlots.jl)
in addition.

### Applications

The package has been used for an approach to uncover patterns in high-dimensional genetic data, described in the article

> Hess M., Lenz S., Blätte T. J., Bullinger L., Binder H. *Partitioned learning of deep Boltzmann machines for SNP data*. Bioinformatics 2017 btx408. doi: https://doi.org/10.1093/bioinformatics/btx408.

The code for the analyses presented there is available in the article supplement.

## References

[1] Salakhutdinov, R. (2015). *Learning Deep Generative Models*. Annual Review of Statistics and Its Application, 2, 361-385.

[2] Salakhutdinov, R. Hinton, G. (2012). *An Efficient Learning Procedure for Deep Boltzmann Machines*. Neural computation, 24(8), 1967-2006.

[3] Salakhutdinov. R. (2008). *Learning and Evaluating Boltzmann Machines*. Technical Report UTML TR 2008-002, Department of Computer Science, University of Toronto.

[4] Krizhevsky, A., Hinton, G. (2009). *Learning Multiple Layers of Features from Tiny Images*.

[5] Srivastava, N., Salakhutdinov R. (2014). *Multimodal Learning with Deep Boltzmann Machines*. Journal of Machine Learning Research, 15, 2949-2980.

[6] Cho, K., Ilin A., Raiko, T. (2011) *Improved learning of Gaussian-Bernoulli restricted Boltzmann machines*. Artificial Neural Networks and Machine Learning – ICANN 2011.

