module BoltzmannMachines

using Distributions
using Distributed
using LinearAlgebra
using Random
using Statistics

const BMs = BoltzmannMachines

export
   AbstractBM,
      aislogimpweights, aisprecision, aisstandarddeviation,
      empiricalloglikelihood, energy, exactloglikelihood,
      exactlogpartitionfunction, loglikelihood,
      logpartitionfunction, logpartitionfunctionzeroweights,
      logproblowerbound, reconstructionerror,
      sampleparticles, samples,
      AbstractRBM,
         BernoulliRBM,
         BernoulliGaussianRBM,
         Binomial2BernoulliRBM,
         GaussianBernoulliRBM,
         GaussianBernoulliRBM2,
         Softmax0BernoulliRBM,
         PartitionedRBM,
         fitrbm, freeenergy, initrbm,
         joinrbms, joindbms,
         trainrbm!,
         samplehidden, samplehidden!,
         samplevisible, samplevisible!,
         hiddenpotential, hiddenpotential!,
         hiddeninput, hiddeninput!,
         visiblepotential, visiblepotential!,
         visibleinput, visibleinput!,
      MultimodalDBM,
         BasicDBM,
         AbstractTrainLayer, AbstractTrainLayers,
         TrainLayer, TrainPartitionedLayer,
         stackrbms,
         initparticles, gibbssample!, gibbssamplecond!,
         meanfield, fitdbm, traindbm!,
   Particle, Particles,
   AbstractOptimizer,
      LoglikelihoodOptimizer,
      initialized, computegradient!, updateparameters!,
      loglikelihoodoptimizer,
      beamoptimizer,
   Monitor, MonitoringItem, DataDict,
      monitored_fitrbm, monitored_stackrbms,
      monitored_traindbm!, monitored_fitdbm,
      monitorexactloglikelihood, monitorexactloglikelihood!,
      monitorfreeenergy, monitorfreeenergy!,
      monitorlogproblowerbound, monitorlogproblowerbound!,
      monitorloglikelihood, monitorloglikelihood!,
      monitorreconstructionerror, monitorreconstructionerror!,
      monitorsd, monitorsd!,
      monitorweightsnorm, monitorweightsnorm!,
      propagateforward,
   crossvalidation,
   intensities, intensities_encode, intensities_decode,
   oneornone_encode, oneornone_decode,
   splitdata,
   barsandstripes, logit

include("bmtypes.jl")
include("gibbssampling.jl")
include("samplers.jl")
include("optimizers.jl")
include("rbmtraining.jl")
include("rbmstacking.jl")
include("dbmtraining.jl")
include("weightsjoining.jl")
include("evaluating.jl")
include("monitoring.jl")
include("monitoredfitting.jl")
include("beam.jl")
include("misc.jl")
include("datapreprocessing.jl")


end # of module BoltzmannMachines
