using BoltzmannMachines

# ==============================================================================
# Binary data: BernoulliRBM and BasicDBM
# ------------------------------------------------------------------------------

# Generate data and split it in training and test data set
nsamples = 100;
nvariables = 36;
srand(1);
x = barsandstripes(nsamples, nvariables);
x, xtest = splitdata(x, 0.3);
datadict = DataDict("Training data" => x, "Test data" => xtest);


# Fit a BernoulliRBM and collect monitoring data during training.
# For small models, the model's loglikelihood can be computed exactly.
monitor = Monitor();
rbm = fitrbm(x; nhidden = 12,
      epochs = 80, learningrate = 0.007,
      monitoring = (rbm, epoch) -> begin
            monitorexactloglikelihood!(monitor, rbm, epoch, datadict)
            monitorreconstructionerror!(monitor, rbm, epoch, datadict)
      end);

# Plot the collected data (requires Gadfly)
BMPlots.plotevaluation(monitor, monitorexactloglikelihood)
BMPlots.plotevaluation(monitor, monitorreconstructionerror)


# DBM-Fitting approach 1 - Pretraining, adding layer by layer.
dbm = BasicDBM();

# With this approach it is possible to also monitor the layerwise pretraining,
# which is basically the same as fitting an RBM.
# (Here only demonstrated for first layer)
monitor1 = Monitor()
addlayer!(dbm, x; nhidden = 6, epochs = 20,
      monitoring = (rbm, epoch) ->
            monitorreconstructionerror!(monitor1, rbm, epoch, datadict));
BMPlots.plotevaluation(monitor1, monitorreconstructionerror)

addlayer!(dbm, x; nhidden = 2, islast = true);

# DBM-Fitting approach 1 - Fine-Tuning
monitor = Monitor();
traindbm!(dbm, x; epochs = 10, learningrate = 0.05,
      monitoring = (dbm, epoch) ->
            monitorexactloglikelihood!(monitor, dbm, epoch, datadict));

BMPlots.plotevaluation(monitor, monitorexactloglikelihood)

# DBM-Fitting approach 2: Pretraining and Fine-Tuning combined in one function
dbm = fitdbm(x, nhiddens = [6;5;2], epochs = 20, epochspretraining = 20,
      learningratepretraining = 0.05, learningrate = 0.05);


# If the model has more parameters, in this case more hidden nodes,
# the exact calclation is not feasible any more:
# We need to calculate the likelihood using AIS.

monitor = Monitor();
rbm = fitrbm(x, nhidden = 36, epochs = 100,
      learningrates = [0.008*ones(80); 0.001*ones(20)],
      monitoring = (rbm, epoch) ->
            monitorloglikelihood!(monitor, rbm, epoch, datadict));

BMPlots.plotevaluation(monitor, monitorloglikelihood; sdrange = 3.0)


# In the DBM, the estimation of the log likelihood is much slower than in the
# RBM because it needs a AIS run for each sample.
# The lower bound of the log probability is faster to estimate:
# (But on the other hand, in some cases it might be too conservative to be 
# useful as it is only a lower bound.)

monitor = Monitor();
dbm = stackrbms(x; nhiddens = [36;10;5], predbm = true, learningrate = 0.05)
traindbm!(dbm, x; epochs = 100, learningrates = [0.008*ones(80); 0.001*ones(20)],
      monitoring = (rbm, epoch) ->
            monitorlogproblowerbound!(monitor, rbm, epoch, datadict));

# Evaluate final result
loglikelihood(dbm, x)

BMPlots.plotevaluation(monitor, monitorlogproblowerbound; sdrange = 3.0)


# ==============================================================================
# Real valued data: GaussianBernoulliRBM
# ------------------------------------------------------------------------------

# Use iris dataset as example data to train a GaussianBernoulliRBM
using RDatasets
x = convert(Matrix{Float64}, dataset("datasets", "iris")[1:4]);
srand(12);
x, xtest = splitdata(x, 0.3);
datadict = DataDict("Training data" => x, "Test data" => xtest);
monitor = Monitor();
rbm = fitrbm(x, rbmtype = GaussianBernoulliRBM,
      nhidden = 3, epochs = 70, learningrate = 0.001,
      monitoring = (rbm, epoch) ->
            monitorexactloglikelihood!(monitor, rbm, epoch, datadict));

BMPlots.plotevaluation(monitor, monitorexactloglikelihood)


# ==============================================================================
# Data with values in {0,1,2}: Binomial2BernoulliRBM
# ------------------------------------------------------------------------------

# Generate data and split it in training and test data set
nsamples = 100;
nvariables = 25;
srand(12);
x = barsandstripes(nsamples, nvariables) + barsandstripes(nsamples, nvariables);
x, xtest = splitdata(x, 0.3);
datadict = DataDict("Training data" => x, "Test data" => xtest);

# Fit Binomial2BernoulliRBM.
monitor = Monitor();
rbm = fitrbm(x, rbmtype = Binomial2BernoulliRBM,
      nhidden = 10, epochs = 90, learningrate = 0.002,
      monitoring = (rbm, epoch) ->
            monitorexactloglikelihood!(monitor, rbm, epoch, datadict));

BMPlots.plotevaluation(monitor, monitorexactloglikelihood)

