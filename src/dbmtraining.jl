function converttopartitionedbernoullidbm(mdbm::MultimodalDBM)
   Vector{Union{BernoulliRBM, PartitionedRBM{BernoulliRBM}}}(mdbm)
end


function defaultfinetuninglearningrates(learningrate, epochs)
   learningrate * 11.0 ./ (10.0 .+ (1:epochs))
end


"""
    fitdbm(x; ...)
Fits a (multimodal) DBM to the data set `x`.
The procedure consists of two parts:
First a stack of RBMs is pretrained in a greedy layerwise manner
(see `stackrbms(x)`). Then the weights of all layers are jointly
trained using the general Boltzmann Machine learning procedure
(see `traindbm!(dbm,x)`).

# Optional keyword arguments (ordered by importance):
* `nhiddens`: vector that defines the number of nodes in the hidden layers of
   the DBM. The default value specifies two hidden layers with the same size
   as the visible layer.
* `epochs`: number of training epochs for joint training, defaults to 10
* `epochspretraining`: number of training epochs for pretraining,
   defaults to `epochs`
* `learningrate`/`learningrates`:
   learning rate(s) for joint training of layers (= fine tuning)
   using the learning algorithm for a general Boltzmann Machine.
   The learning rate for fine tuning is by default decaying with the number of epochs,
   starting with the given value for the `learningrate`.
   (For more details see `traindbm!`).
* `learningratepretraining`: learning rate for pretraining,
   defaults to `learningrate`
* `batchsizepretraining`: batchsize for pretraining, defaults to 1
* `nparticles`: number of particles used for sampling during joint training of
   DBM, default 100
* `pretraining`: The arguments for layerwise pretraining
   can be specified for each layer individually.
   This is done via a vector of `TrainLayer` objects.
   (For a detailed description of the possible parameters,
   see help for `TrainLayer`).
   If the number of training epochs and the learning rate are not specified
   explicitly for a layer, the values of `epochspretraining`,
   `learningratepretraining` and `batchsizepretraining` are used.
* `monitoring`: Monitoring function accepting a `dbm` and the number of epochs,
   returning nothing. Used for the monitoring of fine-tuning.
   See also `monitored_fitdbm` for a more convenient way of monitoring.
* `monitoringdatapretraining`: a `DataDict` that contains data used for
   monitoring the pretraining (see argument `monitoringdata` of `stackrbms`.)
* `optimizer`/`optimizers`: an optimizer or a vector of optimizers for each epoch
   (see `AbstractOptimizer`) used for fine-tuning.
* `optimizerpretraining`: an optimizer used for pre-training.
   Defaults to the `optimizer`.
"""
function fitdbm(x::Matrix{Float64};
      nhiddens::Vector{Int} = Vector{Int}(),
      epochs::Int = 10,
      nparticles::Int = 100,
      learningrate::Float64 = 0.005,
      learningrates::Vector{Float64} =
            defaultfinetuninglearningrates(learningrate, epochs),
      sdlearningrate::Float64 = 0.0,
      sdlearningrates::Vector{Float64} =
            defaultfinetuninglearningrates(sdlearningrate, epochs),
      learningratepretraining::Float64 = learningrate,
      epochspretraining::Int = epochs,
      batchsizepretraining::Int = 1,
      pretraining::AbstractTrainLayers = Vector{TrainLayer}(),
      monitoring::Function = emptyfunc,
      monitoringdatapretraining::DataDict = DataDict(),
      optimizer::AbstractOptimizer = NoOptimizer(),
      optimizers::Vector{<:AbstractOptimizer} = Vector{AbstractOptimizer}(),
      optimizerpretraining::AbstractOptimizer = optimizer)

   # when changing something here, consider also changing monitored_fitdbm

   # Layerwise pre-training
   pretraineddbm = stackrbms(x, nhiddens = nhiddens,
         epochs = epochspretraining, predbm = true,
         batchsize = batchsizepretraining,
         learningrate = learningratepretraining,
         optimizer = optimizer,
         trainlayers = pretraining,
         monitoringdata = monitoringdatapretraining)

   # Fine-tuning using mean-field approximation in algorithm for
   # training a general Boltzmann machine
   traindbm!(pretraineddbm, x, epochs = epochs, nparticles = nparticles,
         learningrate = learningrate, learningrates = learningrates,
         sdlearningrate = sdlearningrate, sdlearningrates = sdlearningrates,
         optimizer = optimizer, optimizers = optimizers,
         monitoring = monitoring)
end


"""
    newparticleslike(particles)
Creates new and uninitialized particles of the same dimensions as the given
`particles`.
"""
function newparticleslike(particles::Particles)
   newparticles = Particles(undef, length(particles))
   for i in eachindex(particles)
      newparticles[i] = Matrix{Float64}(undef, size(particles[i]))
   end
   newparticles
end


"""
    meanfield(dbm, x)
    meanfield(dbm, x, eps)
Computes the mean-field approximation for the data set `x` and
returns a matrix of particles for the DBM.
The number of particles is equal to the number of samples in `x`.
`eps` is the convergence criterion for the fix-point iteration, default 0.001.
It is assumed that all nodes in in-between-layers are Bernoulli distributed.
"""
function meanfield(dbm::MultimodalDBM, x::Array{Float64,2}, eps::Float64 = 0.001)

   nlayers = length(dbm) + 1
   mu = Particles(undef, nlayers)

   # Initialization with single bottom-up pass using twice the weights for all
   # but the topmost layer (see [Salakhutdinov+Hinton, 2012], p. 1985)
   mu[1] = x
   for i=2:(nlayers-1)
      mu[i] = hiddenpotential(dbm[i-1], mu[i-1], 2.0)
   end
   mu[nlayers] = hiddenpotential(dbm[nlayers-1], mu[nlayers-1])

   newmu = newparticleslike(mu)
   newmu[1] = x
   input2 = newparticleslike(mu)

   # mean-field updates until convergence criterion is met
   delta = 1.0
   while delta > eps
      delta = 0.0

      for i = 2:(nlayers-1)
         # total input from layer below
         hiddeninput!(newmu[i], dbm[i-1], mu[i-1])
         # input from layer above
         visibleinput!(input2[i], dbm[i], mu[i+1])
         # combine input
         newmu[i] .+= input2[i]
         # By computing the potential,
         # the assumption is used that all nodes in in-between-layers
         # are Bernoulli-distributed:
         sigm!(newmu[i])

         delta = max(delta, maximum(abs.(mu[i] - newmu[i])))

         # swap new and old mu
         tmp = newmu[i]
         newmu[i] = mu[i]
         mu[i] = tmp
      end

      # last layer
      hiddenpotential!(newmu[end], dbm[end], mu[end-1])
      delta = max(delta, maximum(abs.(mu[end] - newmu[end])))

      # swap new and old mu
      tmp = newmu[end]
      newmu[end] = mu[end]
      mu[end] = tmp
   end

   mu
end


"""
    traindbm!(dbm, x; ...)
Trains the `dbm` (a `BasicDBM` or a more general `MultimodalDBM`) using
the learning procedure for a general Boltzmann Machine with the
training data set `x`.
A learning step consists of mean-field inference (positive phase),
stochastic approximation by Gibbs Sampling (negative phase) and the parameter
updates.

# Optional keyword arguments (ordered by importance):
* `epoch`: number of training epochs
* `learningrate`/`learningrates`: a vector of learning rates for each epoch to
   update the weights and biases. The learning rates should decrease with the
   epochs, e. g. with the factor `a / (b + epoch)`. If only one value is given as
   `learningrate`, `a` and `b` are 11.0 and 10.0, respectively.
* `nparticles`: number of particles used for sampling, default 100
* `monitoring`: A function that is executed after each training epoch.
   It has to accept the trained DBM and the current epoch as arguments.
"""
function traindbm!(dbm::MultimodalDBM, x::Array{Float64,2};
      epochs::Int = 10,
      nparticles::Int = 100,
      learningrate::Float64 = 0.005,
      learningrates::Vector{Float64} =
            defaultfinetuninglearningrates(learningrate, epochs),
      sdlearningrate::Float64 = 0.0,
      sdlearningrates::Vector{Float64} =
            defaultfinetuninglearningrates(sdlearningrate, epochs),
      batchsize::Int = size(x, 2),
      monitoring::Function = emptyfunc,
      optimizer::AbstractOptimizer = NoOptimizer(),
      optimizers::Vector{<:AbstractOptimizer} = Vector{AbstractOptimizer}())

   assert_enoughvaluesforepochs("learningrates", learningrates, epochs)

   optimizer = converttodbmoptimizer(optimizer, dbm)
   map!(opt -> converttodbmoptimizer(opt, dbm), optimizers, optimizers)
   optimizers = assertinitoptimizers(optimizer, optimizers, dbm,
         learningrates, sdlearningrates, epochs)

   particles = initparticles(dbm, nparticles)

   nsamples = size(x, 1)
   batchmasks = randombatchmasks(nsamples, batchsize)
   nbatchmasks = length(batchmasks)
   normalbatchsize = true

   for epoch = 1:epochs
      for batchindex in eachindex(batchmasks)
         traindbm!(dbm, x, particles, optimizers[epoch])
      end

      # monitoring the learning process at the end of epoch
      monitoring(dbm, epoch)
   end

   dbm
end


"""
    traindbm!(dbm, x, particles, learningrate)
Trains the given `dbm` for one epoch.
"""
function traindbm!(dbm::MultimodalDBM, x::Array{Float64,2}, particles::Particles,
      optimizer::AbstractOptimizer)

   gibbssample!(particles, dbm)
   mu = meanfield(dbm, x)

   computegradient!(optimizer, mu, particles, dbm)
   updateparameters!(dbm, optimizer)

   dbm
end
