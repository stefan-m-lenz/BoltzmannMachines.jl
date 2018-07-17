module BoltzmannMachines

using Distributions
using Compat.Distributed

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
         addlayer!, stackrbms,
         initparticles, gibbssample!,
         meanfield, fitdbm, traindbm!,
   Particle, Particles,
   AbstractOptimizer,
      computegradient!, updateparameters!,
      LoglikelihoodOptimizer,
      loglikelihoodoptimizer,
      beamoptimizer,
   Monitor, MonitoringItem, DataDict,
      monitorexactloglikelihood, monitorexactloglikelihood!,
      monitorfreeenergy, monitorfreeenergy!,
      monitorlogproblowerbound, monitorlogproblowerbound!,
      monitorloglikelihood, monitorloglikelihood!,
      monitorreconstructionerror, monitorreconstructionerror!,
      monitorweightsnorm, monitorweightsnorm!,
      propagateforward,
   crossvalidation,
   barsandstripes, splitdata

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
include("beam.jl")
include("misc.jl")

end # of module BoltzmannMachines
