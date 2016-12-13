using BoltzmannMachines
srand(100);

# ==============================================================================

# Generate binary data and split it in training and test data set
nsamples = 100;
nvariables = 16;
x = barsandstripes(nsamples, nvariables);
x, xtest = splitdata(x, 0.3);
datadict = DataDict("Training data" => x, "Test data" => xtest);

# Fit BernoulliRBM and collect monitoring data during training.
monitor = Monitor();
rbm = fitrbm(x, nhidden = 4,
      epochs = 150, learningrates = [0.008*ones(100); 0.001*ones(50)],
      monitoring = (rbm, epoch) -> begin
            monitorloglikelihood!(monitor, rbm, epoch, datadict)
            monitorreconstructionerror!(monitor, rbm, epoch, datadict)
      end);

# Plot the monitoring data (requires Gadfly)
BMPlots.plotloglikelihood(monitor, sdrange = 2.0)
BMPlots.plotevaluation(monitor, monitorreconstructionerror)


#TODO:
# Train a DBM on the same data


#===============================================================================

# Generate real-valued data and train GaussianBernoulliRBM
# For a small number of variables, the likelihood of the model can be computed
# fast enough.


#===============================================================================

# Generate a 0/1/2-valued data and train a Binomial2BernoulliRBM
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

#===============================================================================

