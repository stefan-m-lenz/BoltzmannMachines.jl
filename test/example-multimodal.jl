# TODO finish and integrate in examples.jl



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
pregbrbm = BMs.fitrbm(y;
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
      startrbm = pregbrbm, batchsize = 50,
      learningrate = 0.000005,
      sdlearningrate = 0.0000001,
      epochs = 200,
      #upfactor = 2.0, downfactor = 1.0,
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
            epochs = 1,
            cdsteps = 15,
            pcd = false,
            monitoring = (rbm, epoch, datadict) -> begin
               BMs.monitorexactloglikelihood!(monitorgbrbm, rbm, epoch, datadict)
            end)
      #])
      ;
      BMs.TrainLayer(
         nhidden = 4,
         monitoring = (rbm, epoch, datadict) ->
               BMs.monitorexactloglikelihood!(monitorrbm2, rbm, epoch, datadict)
         )
   ]


monitor = BMs.Monitor();
mdbm = BMs.fitdbm(x, epochs = 200,
      # learning rate and epochs that will be used as default
      # (here for all BernoulliRBMs):
      epochspretraining = 30,
      pretraining = trainlayers,
      learningrate = 0.005,
      monitoring = (dbm, epoch) ->
            BMs.monitorlogproblowerbound!(monitor, dbm, epoch, datadict),
      monitoringdatapretraining = datadict)

BMs.BMPlots.plotevaluation(monitorgbrbm, BMs.monitorexactloglikelihood)
BMs.BMPlots.plotevaluation(monitorrbm2, BMs.monitorexactloglikelihood)
BMs.BMPlots.plotevaluation(monitor, BMs.monitorlogproblowerbound)

      xgen = BMs.samples(mdbm, 100; burnin = 200, samplelast = false)
      BMs.BMPlots.plotcurvebundles(xgen, nlabelvars = 0)
      #BMs.BMPlots.plotcurvebundles(xgen[:,(nnonsensecols+1):end], nlabelvars = 0)

BMs.BMPlots.plotevaluation(monitorrbm1, BMs.monitorexactloglikelihood)
BMs.BMPlots.plotevaluation(monitorgbrbm, BMs.monitorexactloglikelihood)

# Test whether the model can generate correctly labeled curves that look
# like the original ones

