function converttopartitionedbernoullidbm(mdbm::MultimodalDBM)
   Vector{Union{BernoulliRBM, PartitionedRBM{BernoulliRBM}}}(mdbm)
end


"""
    addlayer!(dbm, x)
Adds a pretrained layer to the BasicDBM `dbm`, given the dataset `x` as input
for the visible layer of the `dbm`.

# Optional keyword arguments:
* `isfirst`, `islast`: indicating that the new RBM should be trained as
   first (assumed if `dbm` is empty) or last layer of the DBM, respectively.
   If those flags are not set, the added layer is trained as intermediate layer.
   This information is needed to determine the factor for the weights during
   pretraining.
* `epochs`, `nhidden`, `learningrate`/`learningrates`, `pcd`, `cdsteps`,
  `monitoring`: used for fitting the weights of the last layer, see `fitrbm(x)`
"""
function addlayer!(dbm::BasicDBM, x::Matrix{Float64};
      isfirst::Bool = isempty(dbm),
      islast::Bool = false,
      nhidden::Int = size(x,2),
      epochs::Int = 10,
      learningrate::Float64 = 0.005,
      learningrates::Vector{Float64} = learningrate * ones(epochs),
      pcd::Bool = true,
      cdsteps::Int = 1,
      monitoring::Function = ((rbm, epoch) -> nothing))

   # propagate input x up to last hidden layer
   hh = x
   for i = 1:length(dbm)
      hh = hiddenpotential(dbm[i], hh, 2.0) # intermediate layer, factor is 2.0
   end

   upfactor = downfactor = 2.0
   if isfirst
      downfactor = 1.0
   end
   if islast
      upfactor = 1.0
   end

   rbm = fitrbm(hh, nhidden = nhidden, epochs = epochs,
         rbmtype = BernoulliRBM,
         learningrate = learningrate,
         learningrates = learningrates,
         pcd = pcd, cdsteps = 1,
         upfactor = upfactor, downfactor = downfactor,
         monitoring = monitoring)

   push!(dbm, rbm)
   dbm
end


function defaultfinetuninglearningrates(learningrate, epochs)
   learningrate * 11.0 ./ (10.0 + (1:epochs))
end


"""
    fitdbm(x; ...)
Fits a BasicDBM model to the data set `x`. The procedure consists of two parts:
First a stack of RBMs is pretrained in a greedy layerwise manner
(see `stackrbms(x)`). Then the weights are jointly trained using the
general Boltzmann Machine learning procedure (see `traindbm!(dbm,x)`).

# Optional keyword arguments (ordered by importance):
* `nhiddens`: vector that defines the number of nodes in the hidden layers of
   the DBM. The default value specifies two hidden layers with the same size
   as the visible layer.
* `epochs`: number of training epochs for joint training
* `epochspretraining`: number of training epochs for pretraining,
   defaults to `epochs`
* `learningrate`: learning rate for joint training of layers (= fine tuning)
   using the learning algorithm for a general Boltzmann Machine.
   The learning rate for fine tuning is by default decaying with the number of epochs,
   starting with the given value. (For more details see `traindbm!`).
* `learningratepretraining`: learning rate for pretraining,
   defaults to `learningrate`
* `nparticles`: number of particles used for sampling during joint training of
   DBM, default 100
* `pretraining`: The arguments for layerwise pretraining
   can be specified for each layer individually.
   This is done via a vector of `TrainLayer` objects.
   (For a detailed description of the possible parameters,
   see help for `TrainLayer`).
   If the number of training epochs and the learning rate are not specified
   explicitly for a layer, the values of `epochspretraining` and
   `learningratepretraining` are used.
"""
function fitdbm(x::Matrix{Float64};
      nhiddens::Vector{Int} = Vector{Int}(),
      epochs::Int = 10,
      nparticles::Int = 100,
      learningrate::Float64 = 0.005,
      learningrates::Vector{Float64} =
            defaultfinetuninglearningrates(learningrate, epochs),
      learningratepretraining::Float64 = learningrate,
      epochspretraining::Int = epochs,
      pretraining::AbstractTrainLayers = Vector{TrainLayer}())

   if isempty(pretraining) && isempty(nhiddens)
      # set default only if there is not any more detailed info
      nvariables = size(x,2)
      nhiddens = [nvariables; nvariables]
   end

   pretraineddbm = stackrbms(x, nhiddens = nhiddens,
         epochs = epochspretraining, predbm = true,
         learningrate = learningratepretraining,
         trainlayers = pretraining)

   traindbm!(pretraineddbm, x, epochs = epochs, nparticles = nparticles,
         learningrate = learningrate, learningrates = learningrates)
end


"""
    newparticleslike(particles)
Creates new and uninitialized particles of the same dimensions as the given
`particles`.
"""
function newparticleslike(particles::Particles)
   newparticles = Particles(length(particles))
   for i in eachindex(particles)
      newparticles[i] = Matrix{Float64}(size(particles[i]))
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
   mu = Particles(nlayers)

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
   epochs, e. g. like `a / (b + epoch)`. If only one value is given as
   `learningrate`, `a` and `b` are 11.0 and 10.0, respectively.
* `nparticles`: number of particles used for sampling, default 100
* `monitoring`: A function that is executed after each training epoch.
   It has to accept the trained DBM and the current epoch as arguments.
"""
function traindbm!(dbm::MultimodalDBM, x::Array{Float64,2};
      epochs::Int = 10,
      nparticles::Int = 100,
      learningrate::Float64 = 0.005,
      learningrates::Array{Float64,1} =
            defaultfinetuninglearningrates(learningrate, epochs),
      monitoring::Function = ((dbm, epoch) -> nothing))

   if length(learningrates) < epochs
      error("Not enough learning rates for training epochs")
   end

   particles = initparticles(dbm, nparticles)

   for epoch=1:epochs
      traindbm!(dbm, x, particles, learningrates[epoch])

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
      learningrate::Float64)

   gibbssample!(particles, dbm)
   mu = meanfield(dbm, x)

   for i in eachindex(dbm)
      updatedbmpart!(dbm[i], learningrate,
            particles[i], particles[i+1], mu[i], mu[i+1])
   end

   dbm
end


function updatedbmpart!(dbmpart::BernoulliRBM,
      learningrate::Float64,
      vgibbs::M, hgibbs::M, vmeanfield::M, hmeanfield::M
      ) where {M<:AbstractArray{Float64,2}}

   updatedbmpartcore!(dbmpart, learningrate,
         vgibbs, hgibbs, vmeanfield, hmeanfield)
end

function updatedbmpart!(dbmpart::Binomial2BernoulliRBM,
      learningrate::Float64,
      vgibbs::M, hgibbs::M, vmeanfield::M, hmeanfield::M
      ) where {M<:AbstractArray{Float64,2}}

   vmeanfield /= 2
   vgibbs /= 2

   updatedbmpartcore!(dbmpart, learningrate,
         vgibbs, hgibbs, vmeanfield, hmeanfield)
end

function updatedbmpart!(dbmpart::GaussianBernoulliRBM,
      learningrate::Float64,
      vgibbs::M, hgibbs::M, vmeanfield::M, hmeanfield::M
      ) where {M<:AbstractArray{Float64,2}}

   # For respecting standard deviation in update rule
   # see [Srivastava+Salakhutdinov, 2014], p. 2962
   vmeanfield = broadcast(/, vmeanfield, dbmpart.sd')
   vgibbs = broadcast(/, vgibbs, dbmpart.sd')

   updatedbmpartcore!(dbmpart, learningrate,
         vgibbs, hgibbs, vmeanfield, hmeanfield)
end

function updatedbmpart!(dbmpart::PartitionedRBM,
      learningrate::Float64,
      vgibbs::M, hgibbs::M, vmeanfield::M, hmeanfield::M
      ) where {M<:AbstractArray{Float64,2}}

   for i in eachindex(dbmpart.rbms)
      visrange = dbmpart.visranges[i]
      hidrange = dbmpart.hidranges[i]
      # TODO does not work with views
      updatedbmpart!(dbmpart.rbms[i], learningrate,
            vgibbs[:, visrange], hgibbs[:, hidrange],
            vmeanfield[:, visrange], hmeanfield[:, hidrange])
   end
end

function updatedbmpartcore!(dbmpart::AbstractRBM,
      learningrate::Float64,
      vgibbs::M, hgibbs::M, vmeanfield::M, hmeanfield::M
      ) where {M<:AbstractArray{Float64,2}}

   nsamples = size(vmeanfield, 1)
   nparticles = size(vgibbs, 1)

   leftw = vmeanfield' * hmeanfield / nsamples
   lefta = mean(vmeanfield, 1)[:]
   leftb = mean(hmeanfield, 1)[:]

   rightw = vgibbs' * hgibbs / nparticles
   righta = mean(vgibbs, 1)[:]
   rightb = mean(hgibbs, 1)[:]

   dbmpart.weights += learningrate*(leftw - rightw)
   dbmpart.visbias += learningrate*(lefta - righta)
   dbmpart.hidbias += learningrate*(leftb - rightb)
   nothing
end
