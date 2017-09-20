const BasicDBM = Vector{BernoulliRBM}

"A DBM with only Bernoulli distributed nodes which may contain partitioned layers."
const PartitionedBernoulliDBM =
      Vector{<:Union{BernoulliRBM, PartitionedRBM{BernoulliRBM}}}

"""
`Particles` are an array of matrices.
The i'th matrix contains in each row the vector of states of the nodes
of the i'th layer of an RBM or a DBM. The set of rows with the same index define
an activation state in a Boltzmann Machine.
Therefore, the size of the i'th matrix is
(number of samples/particles, number of nodes in layer i).
"""
const Particles = Array{Array{Float64,2},1}

const Particle = Array{Array{Float64,1},1}

const MultimodalDBM = Vector{<:AbstractRBM}

function converttopartitionedbernoullidbm(mdbm::MultimodalDBM)
   Vector{Union{BernoulliRBM, PartitionedRBM{BernoulliRBM}}}(mdbm)
end


@compat abstract type AbstractTrainLayer end

type TrainLayer <: AbstractTrainLayer
   epochs::Int
   usedefaultepochs::Bool
   learningrate::Float64
   usedefaultlearningrate::Bool
   sdlearningrate::Float64
   sdlearningrates::Vector{Float64}
   monitoring::Function
   rbmtype::DataType
   nhidden::Int
   nvisible::Int
end


"""
Specify parameters for training one RBM-layer in a DBM.

# Optional keyword arguments:
* `rbmtype`: the type of the RBM that is to be trained.
   This must be a subtype of AbstractRBM and defaults to `BernoulliRBM`.
* `nhidden`: Number of hidden units in the RBM.
* `nvisible`: Number of visible units in the RBM. Only relevant for partitioning.
* `epochs`: number of training epochs.
   A negative value indicates that a default value should be used.
* `learningrate`: learning rate.
   A negative value indicates that a default value should be used.
* `sdlearningrate`/`sdlearningrates`: learning rate / learning rates for each epoch
   for learning the standard deviation. Only used for GaussianBernoulliRBMs.
* `monitoring`:  a function that is executed after each training epoch.
   It takes an RBM and the epoch as arguments.
"""
function TrainLayer(;
      epochs::Int = -1,
      learningrate::Float64 = -Inf,
      sdlearningrate::Float64 = 0.0,
      sdlearningrates::Vector{Float64} = Vector{Float64}(),
      monitoring = ((rbm, epoch) -> nothing),
      rbmtype::DataType = BernoulliRBM,
      nhidden::Int = -1,
      nvisible::Int = -1)

   usedefaultepochs = (epochs < 0)
   usedefaultlearningrate = (learningrate < 0)
   TrainLayer(
         (usedefaultepochs ? 10 : epochs),
         usedefaultepochs,
         (usedefaultlearningrate ? 0.005 : learningrate),
         usedefaultlearningrate,
         sdlearningrate, sdlearningrates,
         monitoring, rbmtype, nhidden, nvisible)
end

type TrainPartitionedLayer <: AbstractTrainLayer
   parts::Vector{TrainLayer}
end

const AbstractTrainLayers = Vector{<:AbstractTrainLayer}


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


"""
    combinedbiases(dbm)
Returns a vector containing in the i'th element the bias vector for the i'th
layer of the `dbm`. For intermediate layers, visible and hidden biases are
combined to a single bias vector.
"""
function combinedbiases(dbm::MultimodalDBM)
   biases = Particle(length(dbm) + 1)
   # Create copy to avoid accidental modification of dbm.
   # Use functions `visiblebias` and `hiddenbias` instead of
   # fields `visbias` and `hidbias` of RBMs to be able to
   # also respect PartitionedRBMs on an abstract level.
   biases[1] = copy(visiblebias(dbm[1]))
   for i = 2:length(dbm)
      biases[i] = visiblebias(dbm[i]) + hiddenbias(dbm[i-1])
   end
   biases[end] = copy(hiddenbias(dbm[end]))
   biases
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
* `learningrate`: learning rate for joint training, see `traindbm!`
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
      nhiddens = size(x,2)*ones(2)
   end

   pretraineddbm = stackrbms(x, nhiddens = nhiddens,
         epochs = epochspretraining, predbm = true,
         learningrate = learningratepretraining,
         trainlayers = pretraining)

   traindbm!(pretraineddbm, x, epochs = epochs, nparticles = nparticles,
         learningrate = learningrate, learningrates = learningrates)
end


"""
    gibbssample!(particles, dbm, nsteps)
Performs Gibbs sampling on the `particles` in the DBM `dbm` for `nsteps` steps.
(See also: `Particles`.)
In-between layers are assumed to contain only Bernoulli-distributed nodes.
"""
function gibbssample!(particles::Particles, dbm::MultimodalDBM,
      nsteps::Int = 5,
      biases::Particle = BMs.combinedbiases(dbm))

   input = deepcopy(particles)
   input2 = deepcopy(particles)

   for step in 1:nsteps
      # first layer gets input only from layer above
      samplevisible!(input[1], dbm[1], particles[2])

      # intermediate layers get input from layers above and below
      for i = 2:(length(particles) - 1)
         visibleinput!(input[i], dbm[i], particles[i+1])
         hiddeninput!(input2[i], dbm[i-1], particles[i-1])
         input[i] .+= input2[i]
         sigm_bernoulli!(input[i]) # Bernoulli-sample from total input
      end

      # last layer gets only input from layer below
      samplehidden!(input[end], dbm[end], particles[end-1])

      # swap input and particles
      tmp = particles
      particles = input
      input = tmp
   end

   particles
end


function hiddenbias(rbm::AbstractRBM)
   rbm.hidbias
end

function hiddenbias(prbm::PartitionedRBM)
   vcat(map(rbm -> rbm.hidbias, prbm.rbms)...)
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
    weightsinput!(input, input2, dbm, particles)
Computes the input that results only from the weights (without biases)
and the previous states in `particles` for all nodes in the DBM
`dbm` and stores it in `input`.
The state of the `particles` and the `dbm` is not altered.
`input2` must have the same size as `input` and `particle`.
For performance reasons, `input2` is used as preallocated space for storing
intermediate results.
"""
function weightsinput!(input::Particles, input2::Particles,
      dbm::PartitionedBernoulliDBM, particles::Particles)

   # first layer gets input only from layer above
   weightsvisibleinput!(input[1], dbm[1], particles[2])

   # intermediate layers get input from layers above and below
   for i = 2:(length(particles) - 1)
      weightsvisibleinput!(input2[i], dbm[i], particles[i+1])
      weightshiddeninput!(input[i], dbm[i-1], particles[i-1])
      input[i] .+= input2[i]
   end

   # last layer gets only input from layer below
   weightshiddeninput!(input[end], dbm[end], particles[end-1])
   input
end

function weightshiddeninput!(hh::M, rbm::BernoulliRBM,
      vv::M) where {M<:AbstractArray{Float64}}

   A_mul_B!(hh, vv, rbm.weights)
end

function weightshiddeninput!(hh::M, prbm::PartitionedRBM{BernoulliRBM},
      vv::M) where {M<:AbstractArray{Float64}}

   for i in eachindex(prbm.rbms)
      visrange = prbm.visranges[i]
      hidrange = prbm.hidranges[i]
      weightshiddeninput!(
            view(hh, :, hidrange), prbm.rbms[i], view(vv,:, visrange))
   end
   hh
end

function weightsvisibleinput!(vv::M, rbm::BernoulliRBM,
      hh::M) where {M<:AbstractArray{Float64}}

   A_mul_Bt!(vv, hh, rbm.weights)
end

function weightsvisibleinput!(vv::M, prbm::PartitionedRBM{BernoulliRBM},
      hh::M) where {M<:AbstractArray{Float64}}

   for i in eachindex(prbm.rbms)
      visrange = prbm.visranges[i]
      hidrange = prbm.hidranges[i]
      weightsvisibleinput!(
            view(vv, :, visrange), prbm.rbms[i], view(hh,:, hidrange))
   end
   vv
end


"""
    initparticles(dbm, nparticles; biased = false)
Creates particles for Gibbs sampling in a DBM. (See also: `Particles`)

The hidden layers of the `dbm` are assumed to be Bernoulli-distributed,
the first `rbm` may have visible nodes of other types.

For Bernoulli distributed layers, the particles are initialized with
Bernoulli(p) distributed values. If `biased == false`, p is 0.5,
otherwise the results of applying the sigmoid function to the bias values
are used as values for the nodes' individual p's.

Gaussian nodes are sampled from a normal distribution if `biased == false`.
If `biased == true` the mean of the Gaussian distribution is shifted by the
bias vector and the standard deviation of the nodes is used.
"""
function initparticles(dbm::MultimodalDBM, nparticles::Int; biased::Bool = false)
   nlayers = length(dbm) + 1
   particles = Particles(nlayers)

   # initialize visible nodes
   particles[1] = Matrix{Float64}(nparticles, length(visiblebias(dbm[1])))
   initvisiblenodes!(particles[1], dbm[1], biased)

   # initialize hidden nodes
   if biased
      biases = combinedbiases(dbm)
      for i in 2:nlayers
         particles[i] = bernoulli!(repmat(sigm(biases[i])', nparticles))
      end
   else
      nunits = BMs.nunits(dbm)
      for i in 2:nlayers
         particles[i] = rand([0.0 1.0], nparticles, nunits[i])
      end
   end
   particles
end

function initvisiblenodes!(v::M, rbm::BernoulliRBM, biased::Bool
      ) where{M <: AbstractArray{Float64}}

   if biased
      for k in size(v,2)
         v[:,k] .= sigm(rbm.visbias[k])
      end
      bernoulli!(v)
   else
      rand!([0.0 1.0], v)
   end
end

function initvisiblenodes!(v::M, b2brbm::Binomial2BernoulliRBM, biased::Bool
   ) where{M <: AbstractArray{Float64}}

   if biased
      for k in size(v,2)
         v[:,k] .= sigm(b2brbm.visbias[k])
      end
      binomial2!(v)
   else
      rand!([0.0 1.0 1.0 2.0], v)
   end
end

function initvisiblenodes!(v::M, rbm::GaussianBernoulliRBM, biased::Bool
      ) where{M <: AbstractArray{Float64}}
   randn!(v)
   if biased
      broadcast!(*, v, v, rbm.sd')
      broadcast!(+, v, v, rbm.visbias')
   end
end

function initvisiblenodes!(v::M, prbm::PartitionedRBM, biased::Bool
      ) where{M <: AbstractArray{Float64}}
   for i in eachindex(prbm.rbms)
      visrange = prbm.visranges[i]
      initvisiblenodes!(view(v, :, visrange), prbm.rbms[i], biased)
   end
   v
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
    stackrbms(x; ...)
Performs greedy layerwise training for Deep Belief Networks or greedy layerwise
pretraining for Deep Boltzmann Machines and returns the trained model.

# Optional keyword arguments (ordered by importance):
* `predbm`: boolean indicating that the greedy layerwise training is
   pre-training for a DBM.
   If its value is false (default), a DBN is trained.
* `nhiddens`: vector containing the number of nodes of the i'th hidden layer in
   the i'th entry
* `epochs`: number of training epochs
* `learningrate`: learningrate, default 0.005
* `trainlayers`: a vector of `TrainLayer` objects. With this argument it is possible
   to specify the training parameters for each layer/RBM individually.
   If the number of training epochs and the learning rate are not specified
   explicitly for a layer, the values of `epochs` and `learningrate` are used.
* `samplehidden`: boolean indicating that consequent layers are to be trained
   with sampled values instead of the deterministic potential,
   which is the default.
"""
function stackrbms(x::Array{Float64,2};
      nhiddens::Vector{Int} = Vector{Int}(),
      epochs::Int = 10,
      predbm::Bool = false,
      samplehidden::Bool = false,
      learningrate::Float64 = 0.005,
      trainlayers::AbstractTrainLayers = Vector{TrainLayer}())

   trainlayers = stackrbms_preparetrainlayers(trainlayers, x, epochs,
         learningrate, nhiddens)

   nrbms = length(trainlayers)
   dbmn = Vector{AbstractRBM}(nrbms)

   upfactor = downfactor = 1.0
   if predbm
      upfactor = 2.0
   end

   dbmn[1] = stackrbms_trainlayer(x, trainlayers[1];
         upfactor = upfactor, downfactor = downfactor)

   hiddenval = x
   for i = 2:nrbms
      hiddenval = hiddenpotential(dbmn[i-1], hiddenval, upfactor)
      if samplehidden
         hiddenval = bernoulli(hiddenval)
      end
      if predbm
         upfactor = downfactor = 2.0
         if i == nrbms
            upfactor = 1.0
         end
      else
         upfactor = downfactor = 1.0
      end
      dbmn[i] = stackrbms_trainlayer(hiddenval, trainlayers[i];
            upfactor = upfactor, downfactor = downfactor)
   end

   dbmn = converttomostspecifictype(dbmn)
   dbmn
end

""" Prepares the layerwise training specifications for `stackrbms` """
function stackrbms_preparetrainlayers(
      trainlayers::AbstractTrainLayers,
      x::Matrix{Float64},
      epochs::Int,
      learningrate::Float64,
      nhiddens::Vector{Int})


   if isempty(trainlayers)
      # construct default "trainlayers"
      if isempty(nhiddens)
         nhiddens = size(x,2)*ones(2) # default value for nhiddens
      end
      trainlayers = map(n -> TrainLayer(nhidden = n,
            learningrate = learningrate, epochs = epochs), nhiddens)
      return trainlayers
   end
   # We are here, Argument "trainlayers" has been specified
   # --> check for correct specification

   if !isempty(nhiddens)
      warn("Argument `nhiddens` not used.")
   end

   function setdefaultsforunspecified(trainlayer::TrainLayer)
      if trainlayer.usedefaultlearningrate
         trainlayer.learningrate = learningrate
      end
      if trainlayer.usedefaultepochs
         trainlayer.epochs = epochs
      end
   end

   function setdefaultsforunspecified(trainpartitionedlayer::TrainPartitionedLayer)
      for trainlayer in trainpartitionedlayer.parts
         setdefaultsforunspecified(trainlayer)
      end
   end

   for trainlayer in trainlayers
      setdefaultsforunspecified(trainlayer)
   end

   function derive_nvisibles!(layer::AbstractTrainLayer,
         prevlayer::AbstractTrainLayer)
      # do nothing
   end

   function derive_nvisibles!(layer::TrainPartitionedLayer,
         prevlayer::TrainPartitionedLayer)

      if length(layer.parts) == length(prevlayer.parts)
         for j in eachindex(layer.parts)
            if layer.parts[j].nvisible < 0
               layer.parts[j].nvisible = prevlayer.parts[j].nhidden
            end
         end
      end
      if any(map(t -> t.nvisible < 0, layer.parts))
         error("Parameter `nvisible` could not be derived.")
      end
   end

   for i = 2:length(trainlayers)
      derive_nvisibles!(trainlayers[i], trainlayers[i-1])
   end

   if isa(trainlayers[1], TrainPartitionedLayer)
      nvisiblestotal = mapreduce(t -> t.nvisible, +, trainlayers[1].parts)
      if nvisiblestotal != size(x,2)
         error("Number of visible nodes for first layer " *
               "($nvisiblestotal) is not of same length as " *
               "number of columns in `x` ($(size(x,2))).")
      end
   end

   trainlayers
end


""" Trains a layer without partitioning for `stackrbms`. """
function stackrbms_trainlayer(x::Matrix{Float64},
      trainlayer::TrainLayer;
      upfactor::Float64 = 1.0, downfactor::Float64 = 1.0)

   BMs.fitrbm(x;
         upfactor = upfactor, downfactor = downfactor, pcd = true,
         nhidden = trainlayer.nhidden,
         epochs = trainlayer.epochs,
         rbmtype = trainlayer.rbmtype,
         learningrate = trainlayer.learningrate,
         sdlearningrate = trainlayer.sdlearningrate,
         sdlearningrates = trainlayer.sdlearningrates,
         monitoring = trainlayer.monitoring)
end

""" Trains a partitioned layer for `stackrbms`. """
function stackrbms_trainlayer(x::Matrix{Float64},
      trainpartitionedlayer::TrainPartitionedLayer;
      upfactor::Float64 = 1.0, downfactor::Float64 = 1.0)

   visranges = ranges(map(t -> t.nvisible, trainpartitionedlayer.parts))

   rbms = Vector{AbstractRBM}(length(trainpartitionedlayer.parts))
   for i in eachindex(trainpartitionedlayer.parts)
      visrange = visranges[i]
      rbms[i] = stackrbms_trainlayer(x[:, visrange],
            trainpartitionedlayer.parts[i];
            upfactor = upfactor, downfactor = downfactor)
   end
   commontype = mostspecifictype(rbms)
   PartitionedRBM{commontype}(Vector{commontype}(rbms))
end


function sigm_bernoulli!(input::Particles)
   for i in eachindex(input)
      sigm_bernoulli!(input[i])
   end
   input
end

# const pgrid = collect(linspace(0.00001,0.99999,99999))
# const etagrid = log.(pgrid./(1.0-pgrid))

function sigm_bernoulli!(input::Matrix{Float64})
   for i in eachindex(input)
      @inbounds input[i] = 1.0*(rand() < 1.0/(1.0 + exp(-input[i])))
      # @inbounds input[i] = 1.0*(etagrid[Int(round(rand()*99998.0+1))] < input[i])
   end
   input
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


function visiblebias(rbm::AbstractRBM)
   rbm.visbias
end

function visiblebias(prbm::PartitionedRBM)
   vcat(map(rbm -> rbm.visbias, prbm.rbms)...)
end
