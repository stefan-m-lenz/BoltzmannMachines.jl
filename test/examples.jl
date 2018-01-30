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
      epochs = 80, learningrate = 0.006,
      monitoring = (rbm, epoch) -> begin
            monitorexactloglikelihood!(monitor, rbm, epoch, datadict)
            monitorreconstructionerror!(monitor, rbm, epoch, datadict)
      end);

# Plot the collected data (requires Gadfly)
BMPlots.plotevaluation(monitor, monitorexactloglikelihood)
BMPlots.plotevaluation(monitor, monitorreconstructionerror)


# DBM-Fitting approach 1 - Step 1: Pre-training, adding layer by layer.
# With this approach it is possible to also monitor the layerwise pretraining,
# which is basically the same as fitting an RBM.

dbm = BasicDBM();

srand(12);
monitor1 = Monitor()
addlayer!(dbm, x;
      nhidden = 6, epochs = 20, learningrate = 0.05,
      monitoring = (rbm, epoch) ->
            monitorreconstructionerror!(monitor1, rbm, epoch, datadict));
BMPlots.plotevaluation(monitor1, monitorreconstructionerror)

monitor2  = Monitor()
datadict2 = propagateforward(dbm[1], datadict, 2.0);
addlayer!(dbm, x; islast = true,
      nhidden = 2, epochs = 20, learningrate = 0.05,
      monitoring = (rbm, epoch) ->
            monitorreconstructionerror!(monitor2, rbm, epoch, datadict2));
BMPlots.plotevaluation(monitor2, monitorreconstructionerror)

# DBM-Fitting approach 1 - Step 2: Fine-Tuning
monitor = Monitor();
traindbm!(dbm, x; epochs = 50, learningrate = 0.05,
      monitoring = (dbm, epoch) ->
            monitorexactloglikelihood!(monitor, dbm, epoch, datadict));

BMPlots.plotevaluation(monitor, monitorexactloglikelihood)

# DBM-Fitting approach 2: Pretraining and Fine-Tuning combined in one function
dbm = fitdbm(x, nhiddens = [6;2], epochs = 20, epochspretraining = 20,
      learningratepretraining = 0.05, learningrate = 0.05);

# Evaluate final result with exact computation of likelihood
exactloglikelihood(dbm, xtest)


# If the model has more parameters, in this case more hidden nodes,
# the exact calclation is not feasible any more:
# We need to calculate the likelihood using AIS.

srand(12);
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

srand(2);
monitor = Monitor();
dbm = stackrbms(x; nhiddens = [36;10;5], predbm = true, learningrate = 0.05);
traindbm!(dbm, x; epochs = 50, learningrate = 0.008,
      monitoring = (rbm, epoch) ->
            monitorlogproblowerbound!(monitor, rbm, epoch, datadict));

BMPlots.plotevaluation(monitor, monitorlogproblowerbound; sdrange = 3.0)

# Evaluate final result with AIS-estimated likelihood
loglikelihood(dbm, xtest)


# ==============================================================================
# Real valued data: GaussianBernoulliRBM
# ------------------------------------------------------------------------------

# Use "iris" dataset as example data to train a GaussianBernoulliRBM
# (requires package "RDatasets")
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
# Data with binomially distributed values in {0,1,2}: Binomial2BernoulliRBM
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
      nhidden = 10, epochs = 100, learningrates = [0.003*ones(60); 0.001*ones(40)],
      monitoring = (rbm, epoch) ->
            monitorexactloglikelihood!(monitor, rbm, epoch, datadict));

BMPlots.plotevaluation(monitor, monitorexactloglikelihood)


# ==============================================================================
# Training a Multimodal Deep Boltzmann machine on a data set with
# binary and continuously valued variables
# ------------------------------------------------------------------------------


srand(12);
# Generate example data containing binary coded labels and
# continuously valued variables as y-values for curves
# (for more details see help text of function)
ncontinuousvars = 20
x = BMs.curvebundles(nvariables = ncontinuousvars, nbundles = 3,
      nperbundle = 100, noisesd = 0.05, addlabels = false)
# Visualize labels and groups (colored by label)
BMs.BMPlots.plotcurvebundles(x)

# Add some columns without useful information to the data.
nnonsensecols = 4
nonsensecols = rand([0.0 1.0], size(x,1), nnonsensecols)
x = hcat(nonsensecols, x)
# The task for the network is to find the labels in the noise.

nbinaryvars = size(x, 2) - ncontinuousvars

x, xtest = BMs.splitdata(x, 0.3);
datadict = BMs.DataDict("Training data" => x, "Test data" => xtest);

# At first we try to fit a GaussianBernoulliRBM on the continuous variables
# as tuning the parameters for this type of RBM is very difficult.
# The learning rate and the number of epochs probably need to be set to
# different values than for BernoulliRBMs.

c = cov(x)
a = chol(c)
mu = vec(mean(x,1))
nmultisamples = 10000
y = (randn(nmultisamples, size(x, 2)) * a) .+ mu'
BMs.BMPlots.plotcurvebundles(y)

premonitor = BMs.Monitor()
gbrbm = BMs.fitrbm(y;
      nhidden = 20, rbmtype = BMs.GaussianBernoulliRBM,
      learningrate = 0.00005,
      epochs = 6,
      cdsteps = 15,
      pcd = false,
      monitoring = (rbm, epoch) -> begin
         BMs.monitorexactloglikelihood!(premonitor, rbm, epoch, datadict)
      end)
BMs.BMPlots.plotevaluation(premonitor, BMs.monitorexactloglikelihood)

monitor = BMs.Monitor()
gbrbm = BMs.fitrbm(x;
      startrbm = gbrbm,
      learningrate = 0.00005,
      epochs = 200,
      cdsteps = 15,
      #sdgradclipnorm = 0.1,
      pcd = false,
      monitoring = (rbm, epoch) -> begin
         BMs.monitorexactloglikelihood!(monitor, rbm, epoch, datadict)
      end)
BMs.BMPlots.plotevaluation(monitor, BMs.monitorexactloglikelihood)



xgen1 = BMs.samples(gbrbm, 1000; burnin = 200, samplelast = true)
BMs.BMPlots.plotcurvebundles(xgen1)
xgen2 = BMs.samples(gbrbm, 50; burnin = 200, samplelast = false)
BMs.BMPlots.plotcurvebundles(xgen2)

# Then use the parameters for learning a multimodal deep Boltzmann machine
monitorrbm1 = BMs.Monitor();
monitorrbm2 = BMs.Monitor();
monitorgbrbm = BMs.Monitor()
trainlayers = [
      #first layer is partitioned into a BernoulliRBM and a GaussianBernoulliRBM
      # BMs.TrainPartitionedLayer([
      #    BMs.TrainLayer(rbmtype = BMs.BernoulliRBM,
      #          epochs = 3,
      #          nvisible = nbinaryvars, nhidden = 5,
      #          monitoring = (rbm, epoch, datadict) -> begin
      #             BMs.monitorexactloglikelihood!(monitorrbm1, rbm, epoch, datadict)
      #          end),
         BMs.TrainLayer(
            startrbm = gbrbm,
            learningrates = fill(0.00001, 200),
            epochs = 150,
            cdsteps = 15,
            pcd = false,
            monitoring = (rbm, epoch, datadict) -> begin
               BMs.monitorexactloglikelihood!(monitorgbrbm, rbm, epoch, datadict)
            end)
      ])
      ;
      BMs.TrainLayer(
         nhidden = 4, epochs = 30,
         monitoring = (rbm, epoch, datadict) ->
               BMs.monitorloglikelihood!(monitorrbm2, rbm, epoch, datadict)
         )
   ]


monitor = BMs.Monitor();
mdbm = BMs.fitdbm(x, epochs = 30,
      # learning rate and epochs that will be used as default
      # (here for all BernoulliRBMs):
      epochspretraining = 30,
      learningratepretraining = 0.001,
      pretraining = trainlayers,
      learningrates = fill(0.00001, 50),
      monitoring = (dbm, epoch) ->
            BMs.monitorlogproblowerbound!(monitor, dbm, epoch, datadict),
      monitoringdatapretraining = datadict,
      epochs = 20)

BMs.BMPlots.plotevaluation(monitorrbm1, BMs.monitorexactloglikelihood)
BMs.BMPlots.plotevaluation(monitorrbm2, BMs.monitorloglikelihood)
BMs.BMPlots.plotevaluation(monitor, BMs.monitorlogproblowerbound)

      xgen = BMs.samples(mdbm, 100; burnin = 200, samplelast = false)
      BMs.BMPlots.plotcurvebundles(xgen, nlabelvars = 0)
      #BMs.BMPlots.plotcurvebundles(xgen[:,(nnonsensecols+1):end], nlabelvars = 0)

BMs.BMPlots.plotevaluation(monitorrbm1, BMs.monitorexactloglikelihood)
BMs.BMPlots.plotevaluation(monitorgbrbm, BMs.monitorexactloglikelihood)

# Test whether the model can generate correctly labeled curves that look
# like the original ones



# ==============================================================================
# Examples for cross-validation
# ------------------------------------------------------------------------------

nsamples = 500;
nvariables = 16;
x = barsandstripes(nsamples, nvariables);

# Determine the optimal number of training epochs for a RBM
monitor = crossvalidation(x,
      (monitor, datadict, x) ->
            BoltzmannMachines.fitrbm(x, epochs = 50,
                  learningrate = 0.005,
                  monitoring = (rbm, epoch) ->
                        BoltzmannMachines.monitorexactloglikelihood!(
                              monitor, rbm, epoch, datadict)));
BMPlots.crossvalidationcurve(monitor)

