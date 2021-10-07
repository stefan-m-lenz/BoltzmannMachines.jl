using BoltzmannMachines
import BoltzmannMachinesPlots # only needed for plotting capabilities

# ==============================================================================
# Binary data: BernoulliRBM and BasicDBM
# ------------------------------------------------------------------------------

# Generate data and split it in training and test data set
nsamples = 100;
nvariables = 36;
using Random
Random.seed!(1);
x = barsandstripes(nsamples, nvariables);
x, xtest = splitdata(x, 0.3);
datadict = DataDict("Training data" => x, "Test data" => xtest);


# Fit a BernoulliRBM and collect monitoring data during training.
# For small models, the model's loglikelihood can be computed exactly.
monitor = Monitor();
rbm = fitrbm(x; nhidden = 12,
      epochs = 80, learningrate = 0.006,
      monitoring = (rbm, epoch) -> begin
            monitorexactloglikelihood!(monitor, rbm, epoch, datadict)
            monitorreconstructionerror!(monitor, rbm, epoch, datadict)
      end);

# Plot the collected data (requires Gadfly)
BoltzmannMachinesPlots.plotevaluation(monitor, monitorexactloglikelihood)
BoltzmannMachinesPlots.plotevaluation(monitor, monitorreconstructionerror)

# Simplified monitoring:
monitor, rbm = monitored_fitrbm(x; nhidden = 12,
      epochs = 80, learningrate = 0.006,
      monitoringdata = datadict,
      monitoring = [monitorexactloglikelihood!, monitorreconstructionerror!]);


Random.seed!(12);

# DBM-Fitting: Pretraining and Fine-Tuning combined in one function
dbm = fitdbm(x, nhiddens = [6;2], epochs = 20, learningrate = 0.05);

# .. with extensive monitoring
monitor = Monitor(); monitor1 = Monitor(); monitor2 = Monitor();
dbm = fitdbm(x, epochs = 20, learningrate = 0.05,
      monitoringdatapretraining = datadict,
      pretraining = [
            TrainLayer(nhidden = 6, monitoring = (rbm, epoch, datadict) ->
                  monitorreconstructionerror!(monitor1, rbm, epoch, datadict));
            TrainLayer(nhidden = 2, monitoring = (rbm, epoch, datadict) ->
                  monitorreconstructionerror!(monitor2, rbm, epoch, datadict))
            ],
      monitoring = (dbm, epoch) ->
            monitorexactloglikelihood!(monitor, dbm, epoch, datadict));

# Monitoring plots
BoltzmannMachinesPlots.plotevaluation(monitor1)
BoltzmannMachinesPlots.plotevaluation(monitor2)
BoltzmannMachinesPlots.plotevaluation(monitor)

# Evaluate final result with exact computation of likelihood
exactloglikelihood(dbm, xtest)


# Simplified monitoring
monitors, dbm = monitored_fitdbm(x; nhiddens = [6,2],
      epochs = 20, learningrate = 0.05,
      monitoringpretraining = monitorreconstructionerror!,
      monitoring = monitorexactloglikelihood!,
      monitoringdata = datadict);

# If the model has more parameters, in this case more hidden nodes,
# the exact calculation is not feasible any more:
# We need to calculate the likelihood using AIS.

Random.seed!(12);
monitor = Monitor();
rbm = fitrbm(x, nhidden = 36, epochs = 100,
      learningrates = [0.008*ones(80); 0.001*ones(20)],
      monitoring = (rbm, epoch) ->
            monitorloglikelihood!(monitor, rbm, epoch, datadict));

BoltzmannMachinesPlots.plotevaluation(monitor, monitorloglikelihood; sdrange = 3.0)

# Simplified monitoring:
monitor, rbm = monitored_fitrbm(x, nhidden = 36, epochs = 100,
      learningrates = [0.008*ones(80); 0.001*ones(20)],
      monitoring = monitorloglikelihood!,
      monitoringdata = datadict);


# In the DBM, the estimation of the log likelihood is much slower than in the
# RBM because it needs a AIS run for each sample.
# The lower bound of the log probability is faster to estimate:
# (But on the other hand, in some cases it might be too conservative to be
# useful as it is only a lower bound.)

Random.seed!(2);
monitor = Monitor();
dbm = stackrbms(x; nhiddens = [36;10;5], predbm = true, learningrate = 0.05);
traindbm!(dbm, x; epochs = 50, learningrate = 0.008,
      monitoring = (rbm, epoch) ->
            monitorlogproblowerbound!(monitor, rbm, epoch, datadict));

BoltzmannMachinesPlots.plotevaluation(monitor, monitorlogproblowerbound; sdrange = 3.0)


# Simplified monitoring:
monitor, dbm = monitored_traindbm!(dbm, x; epochs = 50, learningrate = 0.008,
      monitoring = monitorlogproblowerbound!,
      monitoringdata = datadict);

# Evaluate final result with AIS-estimated likelihood
loglikelihood(dbm, xtest)


# ==============================================================================
# Dimension reduction using DBMs
# ------------------------------------------------------------------------------

Random.seed!(1);
# Calculate a two dimensional dimension reduction on the data
x, xlabels = blocksinnoise(500, 50, blocklen = 5, nblocks = 3)
monitor, dbm = monitored_fitdbm(x, nhiddens = [50, 40, 25, 15],
      batchsize = 5,
      epochspretraining = 100,
      learningratepretraining = 0.005,
      epochsfinetuning = 30,
      learningratefinetuning = 0.001)

dimred = top2latentdims(dbm, x)

# Calculate and plot dimension reduction in one command
dimred = BoltzmannMachinesPlots.plottop2latentdims(dbm, x, labels = xlabels)


# ==============================================================================
# Categorical data: Softmax0BernoulliRBM
# ------------------------------------------------------------------------------

# Some data set with values in the categories {0, 1, 2}
# Mixing variables with a different number of categories is also possible
# (see also example below for a MultimodalDBM with binary and categorical
# variables).
xcat = map(v -> max(v, 0),
      barsandstripes(100, 4) .+ barsandstripes(100, 4) .- barsandstripes(100, 4));

# Encode values for Softmax0BernoulliRBM:
# Each of the variables, which have three categories, are translated into
# two binary variables.
# This encoding is to similiar to the one-hot encoding with the deviation
# that a zero is encoded as all-zeros.
# (That way, the encoding of binary variables is not changed.)
x01 = oneornone_encode(xcat, 3) # 3 categories
rbm = fitrbm(x01, rbmtype = Softmax0BernoulliRBM, categories = 3,
      nhidden = 4, epochs = 20);

# The monitoring is omitted here for brevity:
# The same monitoring and training options as those for BernoulliRBMs
# are also available for Softmax0BernoulliRBMs.

# A DBM with a Softmax0BernoulliRBM in the first layer and binary hidden layers:
dbm = fitdbm(x01, pretraining = [
         TrainLayer(rbmtype = Softmax0BernoulliRBM, categories = 3, nhidden = 4);
         TrainLayer(nhidden = 4);
         TrainLayer(nhidden = 2)])

# Getting samples with values in the original sample space:
oneornone_decode(samples(dbm, 5), 3)


# ==============================================================================
# Mixing binary and categorical data: MultimodalDBM
# ------------------------------------------------------------------------------

# x1: binary data
x1 = barsandstripes(100, 4);
# x2: values with categories {0, 1, 2}
x2 = map(v -> max(v, 0), x1 .+ barsandstripes(100, 4) .- barsandstripes(100, 4));
x2 = oneornone_encode(x2, 3)
x = hcat(x1, x2);
x, xtest = splitdata(x, 0.3);
datadict = DataDict("Training data" => x, "Test data" => xtest);
monitor1 = Monitor(); monitor2 = Monitor(); monitor3 = Monitor(); monitor = Monitor();
dbm = fitdbm(x; epochspretraining = 50, epochs = 15,
      learningratepretraining = 0.05, learningrate = 0.1,
      monitoringdatapretraining = datadict,
      pretraining = [
            TrainPartitionedLayer([
               TrainLayer(nhidden = 5, nvisible = 4,
                     monitoring = (rbm, epoch, datadict) ->
                           monitorreconstructionerror!(monitor1, rbm, epoch, datadict));
               TrainLayer(nhidden = 7,
                     rbmtype = Softmax0BernoulliRBM, categories = 3,
                     monitoring = (rbm, epoch, datadict) ->
                           monitorreconstructionerror!(monitor2, rbm, epoch, datadict))
            ]),
            TrainLayer(nhidden = 6,
                  monitoring = (rbm, epoch, datadict) ->
                        monitorreconstructionerror!(monitor3, rbm, epoch, datadict))
      ],
      monitoring =
            (dbm, epoch) -> monitorlogproblowerbound!(monitor, dbm, epoch, datadict));


# The same with simplified monitoring:
Random.seed!(1)
monitors, dbm = monitored_fitdbm(x, epochspretraining = 50, epochs = 15,
      pretraining = [
            TrainPartitionedLayer([
                  TrainLayer(nhidden = 5, nvisible = 4);
                  TrainLayer(nhidden = 7,
                        rbmtype = Softmax0BernoulliRBM, categories = 2)]),
            TrainLayer(nhidden = 6)
      ],
      learningratepretraining = 0.05,
      learningrate = 0.1,
      monitoringdata = datadict,
      monitoringpretraining = monitorreconstructionerror!,
      monitoring = monitorlogproblowerbound!);

# Reconstructionerror of BernoulliRBM in first layer:
BoltzmannMachinesPlots.plotevaluation(monitors[1][1])
# Reconstructionerror of Softmax0BernoulliRBM:
BoltzmannMachinesPlots.plotevaluation(monitors[1][2])
# Reconstructionerror of second layer:
BoltzmannMachinesPlots.plotevaluation(monitors[2])
# Lower bound of likelihood during fine-tuning:
BoltzmannMachinesPlots.plotevaluation(monitors[3])


# It is also possible to combine binary and categorical variables in one RBM,
# as a Softmax0BernoulliRBM with variables having two categories is equivalent
# to a BernoulliRBM.
# Here we do not have a partition then. We can use one Softmax0BernoulliRBM for
# the visible layer and have to specify the number of categories for each of the
# variables separately:

categories = [fill(2, 4); fill(3, 4)];
dbm = fitdbm(x;
      pretraining = [
         TrainLayer(nhidden = 7, categories = categories,
               rbmtype = Softmax0BernoulliRBM);
         TrainLayer(nhidden = 6)
      ])


# ==============================================================================
# Real valued data: Intensities, GaussianBernoulliRBM or GaussianBernoulliRBM2
# ------------------------------------------------------------------------------

# Use "iris" dataset as example data for continuous data
using DelimitedFiles
x = convert(Matrix{Float64},
      readdlm(joinpath(dirname(pathof(BoltzmannMachines)), "..",
            "test/data/iris.csv"), ',', header = true)[1][:,1:4]);
Random.seed!(12);
x, xtest = splitdata(x, 0.3);
datadict = DataDict("Training data" => x, "Test data" => xtest);

# Using the intensities transformation and a standard BernoulliRBM
xintensities, xtrafo = intensities_encode(x)
rbm = fitrbm(xintensities)
# Transform samples back into the original sample space:
intensities_decode(samples(rbm, 5), xtrafo)

# Example using a GaussianBernoulliRBM
Random.seed!(1);
monitor = Monitor();
rbm = fitrbm(x, rbmtype = GaussianBernoulliRBM,
      nhidden = 3, epochs = 80, learningrate = 0.0005,
      monitoring = (rbm, epoch) ->
            monitorexactloglikelihood!(monitor, rbm, epoch, datadict));

BoltzmannMachinesPlots.plotevaluation(monitor, monitorexactloglikelihood)

# The alternative variant of an RBM with Gaussian visible nodes and binary hidden
# nodes, as described by Cho et. al. in "Improved learning of Gaussian-Bernoulli
# restricted Boltzmann machines", is also available as "GaussianBernoulliRBM2".
rbm = fitrbm(x, rbmtype = GaussianBernoulliRBM2,
      nhidden = 3, epochs = 70, learningrate = 0.001);

# The same monitoring options as for BernoulliRBMs are available
# for GaussianBernoulliRBMs and GaussianBernoulliRBM2s.


# ==============================================================================
# Data with binomially distributed values in {0,1,2}: Binomial2BernoulliRBM
# ------------------------------------------------------------------------------

# Generate data and split it in training and test data set
nsamples = 100;
nvariables = 25;
Random.seed!(12);
x = barsandstripes(nsamples, nvariables) + barsandstripes(nsamples, nvariables);
x, xtest = splitdata(x, 0.3);
datadict = DataDict("Training data" => x, "Test data" => xtest);

# Fit Binomial2BernoulliRBM.
monitor = Monitor();
rbm = fitrbm(x, rbmtype = Binomial2BernoulliRBM,
      nhidden = 10, epochs = 100, learningrates = [0.003*ones(60); 0.001*ones(40)],
      monitoring = (rbm, epoch) ->
            monitorexactloglikelihood!(monitor, rbm, epoch, datadict));

BoltzmannMachinesPlots.plotevaluation(monitor, monitorexactloglikelihood)


# ==============================================================================
# Examples for conditional sampling
# ------------------------------------------------------------------------------
nsamples = 100;
nvariables = 36;

x = barsandstripes(100, 16);
dbm = fitdbm(x);

# Conditional gibbs sampling of particles
particles = initparticles(dbm, 500)
variablestofixto1 = 1:3
particles[1][:, variablestofixto1] .= 1.0
gibbssamplecond!(particles, dbm, variablestofixto1)
particles[1]

# Short alternative, returns only the visible nodes
samples(dbm, 500, conditions = [i => 1.0 for i in 1:3])


# ==============================================================================
# Examples for cross-validation
# ------------------------------------------------------------------------------

nsamples = 500;
nvariables = 16;
x = barsandstripes(nsamples, nvariables);

# Determine the optimal number of training epochs for a RBM
monitor = crossvalidation(x,
      (x, datadict) ->
            begin
               monitor = BoltzmannMachines.Monitor()
               BoltzmannMachines.fitrbm(x, epochs = 50,
                     learningrate = 0.005,
                     monitoring = (rbm, epoch) ->
                           BoltzmannMachines.monitorexactloglikelihood!(
                                 monitor, rbm, epoch, datadict))
               monitor
            end);
BoltzmannMachinesPlots.crossvalidationcurve(monitor)


# Determine the optimal number of pretraining epochs for a DBM
# given the other parameters
using Distributed
@everywhere using BoltzmannMachines
@everywhere function my_pretraining_monitoring(
      x::Matrix{Float64}, datadict::DataDict, epoch::Int)

   monitor = Monitor()
   monitorlogproblowerbound!(monitor,
         stackrbms(x,
               nhiddens = [16; 12],
               learningrate = 0.001,
               predbm = true,
               epochs = epoch),
         epoch,
         datadict,
         parallelized = false)
end

monitor = crossvalidation(x, my_pretraining_monitoring, 10:10:200);
BoltzmannMachinesPlots.crossvalidationcurve(monitor, monitorlogproblowerbound)



# ==============================================================================
# Examples for custom optimization algorithm
# ------------------------------------------------------------------------------

x = barsandstripes(100, 16);

struct MyRegularizedOptimizer{R<:AbstractRBM} <: AbstractOptimizer{R}
   llopt::LoglikelihoodOptimizer{R}
end

function MyRegularizedOptimizer()
   MyRegularizedOptimizer(loglikelihoodoptimizer(learningrate = 0.01))
end

import BoltzmannMachines: initialized, computegradient!, updateparameters!

function initialized(myoptimizer::MyRegularizedOptimizer,
      rbm::R) where {R<: AbstractRBM}
   MyRegularizedOptimizer(initialized(myoptimizer.llopt, rbm))
end

function computegradient!(myoptimizer::MyRegularizedOptimizer,
      v, vmodel, h, hmodel, rbm)
   computegradient!(myoptimizer.llopt, v, vmodel, h, hmodel, rbm)
   # add L2 regularization term
   myoptimizer.llopt.gradient.weights .-= 0.1 .* rbm.weights
   nothing
end

function updateparameters!(rbm::R, myoptimizer::MyRegularizedOptimizer{R}
      ) where {R<: AbstractRBM}
   updateparameters!(rbm, myoptimizer.llopt)
end

# The optimizer can be used for fitting RBMs ...
rbm = fitrbm(x; optimizer = MyRegularizedOptimizer())

# and also for DBMs
dbm = fitdbm(x; optimizer = MyRegularizedOptimizer())