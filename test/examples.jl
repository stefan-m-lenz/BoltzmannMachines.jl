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
BoltzmannMachinesPlots.plotevaluation(monitor1, monitorreconstructionerror)
BoltzmannMachinesPlots.plotevaluation(monitor2, monitorreconstructionerror)
BoltzmannMachinesPlots.plotevaluation(monitor, monitorexactloglikelihood)

# Evaluate final result with exact computation of likelihood
exactloglikelihood(dbm, xtest)


# If the model has more parameters, in this case more hidden nodes,
# the exact calclation is not feasible any more:
# We need to calculate the likelihood using AIS.

Random.seed!(12);
monitor = Monitor();
rbm = fitrbm(x, nhidden = 36, epochs = 100,
      learningrates = [0.008*ones(80); 0.001*ones(20)],
      monitoring = (rbm, epoch) ->
            monitorloglikelihood!(monitor, rbm, epoch, datadict));

BoltzmannMachinesPlots.plotevaluation(monitor, monitorloglikelihood; sdrange = 3.0)


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

# Evaluate final result with AIS-estimated likelihood
loglikelihood(dbm, xtest)


# ==============================================================================
# Real valued data: GaussianBernoulliRBM
# ------------------------------------------------------------------------------

# Use "iris" dataset as example data to train a GaussianBernoulliRBM
using DelimitedFiles
x = convert(Matrix{Float64}, readdlm(
            joinpath(dirname(pathof(BoltzmannMachines)), "..", "test/data/iris.csv"), ',',
            header = true)[1][:,1:4]);
Random.seed!(12);
x, xtest = splitdata(x, 0.3);
datadict = DataDict("Training data" => x, "Test data" => xtest);
monitor = Monitor();
rbm = fitrbm(x, rbmtype = GaussianBernoulliRBM,
      nhidden = 3, epochs = 70, learningrate = 0.001,
      monitoring = (rbm, epoch) ->
            monitorexactloglikelihood!(monitor, rbm, epoch, datadict));

BoltzmannMachinesPlots.plotevaluation(monitor, monitorexactloglikelihood)


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