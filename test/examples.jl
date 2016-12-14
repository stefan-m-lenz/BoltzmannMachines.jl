using BoltzmannMachines
srand(100);

# ==============================================================================
# Binary data: BernoulliRBM and BasicDBM
# ------------------------------------------------------------------------------

# Generate data and split it in training and test data set
nsamples = 100;
nvariables = 36;
x = barsandstripes(nsamples, nvariables);
x, xtest = splitdata(x, 0.3);
datadict = DataDict("Training data" => x, "Test data" => xtest);

# Fit a BernoulliRBM and collect monitoring data during training.
# For small models, the model's loglikelihood can be computed exactly.
monitor = Monitor();
rbm = fitrbm(x, nhidden = 7,
      epochs = 50, learningrate = 0.01,
      monitoring = (rbm, epoch) -> begin
            monitorexactloglikelihood!(monitor, rbm, epoch, datadict)
            monitorreconstructionerror!(monitor, rbm, epoch, datadict)
      end);

# Plot the collected data (requires Gadfly)
BMPlots.plotevaluation(monitor, monitorexactloglikelihood)


# Fit a DBM
monitor = Monitor();
dbm = fitdbm(x, nhiddens = [6;5;2], epochs = 20, epochspretraining = 20,
      pretraininglearningrate = 0.05, learningrate = 0.05,
      monitoring = (dbm, epoch) -> begin
            monitorexactloglikelihood!(monitor, dbm, epoch, datadict)
            monitorreconstructionerror!(monitor, dbm, epoch, datadict)
      end)


# If the model has more parameters, in this case more hidden nodes,
# the exact calclation is not feasible any more.

monitor = Monitor();
rbm = fitrbm(x, nhidden = 36,
      epochs = 100, learningrates = [0.008*ones(80); 0.001*ones(20)],
      monitoring = (rbm, epoch) ->
            monitorloglikelihood!(monitor, rbm, epoch, datadict));

BMPlots.plotevaluation(monitor, monitorloglikelihood; sdrange = 3.0)


# In the DBM, the estimation of the log likelihood is much slower than in the
RBM. The lower bound of the log probability is faster to calculate:

monitor = Monitor();
rbm = fitrbm(x, nhidden = 36,
      epochs = 100, learningrates = [0.008*ones(80); 0.001*ones(20)],
      monitoring = (rbm, epoch) ->
            monitorloglikelihood!(monitor, rbm, epoch, datadict));

BMPlots.plotevaluation(monitor, monitorloglikelihood; sdrange = 3.0)



# ==============================================================================
# Real valued data: GaussianBernoulliRBM
# ------------------------------------------------------------------------------
# Generate real-valued data and train GaussianBernoulliRBM
# For a small number of variables, the likelihood of the model can be computed
# fast enough.


# ==============================================================================
# Data with values in {0,1,2}: Binomial2BernoulliRBM
# ------------------------------------------------------------------------------

# Generate data and split it in training and test data set
nsamples = 100;
nvariables = 36;
x = barsandstripes(nsamples, nvariables) + barsandstripes(nsamples, nvariables);
x, xtest = splitdata(x, 0.3);
datadict = DataDict("Training data" => x, "Test data" => xtest);

# Fit Binomial2BernoulliRBM.
# For larger number of variables, we need to calculate the likelihood using AIS.
monitor = Monitor();
rbm = fitrbm(x, rbmtype = Binomial2BernoulliRBM,
      nhidden = 4, nhidden = div(nvariables,2),
      epochs = 50, learningrate = 0.002,
      monitoring = (rbm, epoch) ->
            monitorloglikelihood!(monitor, rbm, epoch, datadict));
BMPlots.plotloglikelihood(monitor, sdrange = 2.0)

