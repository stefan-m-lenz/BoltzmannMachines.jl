"""
Abstract supertype for layerwise training specification.
May be specifications for a normal RBM layer (see `TrainLayer`) or
multiple combined specifications for a partitioned layer
(see `TrainPartitionedLayer`).
"""
@compat abstract type AbstractTrainLayer end

type TrainLayer <: AbstractTrainLayer
   epochs::Int
   usedefaultepochs::Bool
   learningrate::Float64
   learningrates::Vector{Float64}
   usedefaultlearningrate::Bool
   sdlearningrate::Float64
   sdlearningrates::Vector{Float64}
   monitoring::Function
   rbmtype::DataType
   nhidden::Int
   nvisible::Int
   pcd::Bool
   cdsteps::Int
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
* `learningrates`: a vector of learning rates, containing values for each epoch.
   If this is specified, the argument `learningrate` is ignored.
* `sdlearningrate`/`sdlearningrates`: learning rate / learning rates for each epoch
   for learning the standard deviation. Only used for GaussianBernoulliRBMs.
* `cdsteps`: Number of steps for (persistent) contrastive divergence. Default is 1.
* `pcd`: Boolean indicating whether persistent contrastive divergence is used
   (true, default) or whether contrastive divergence is used (false).
* `monitoring`: a function that is executed after each training epoch.
   It takes an RBM and the epoch as arguments.
"""
function TrainLayer(;
      epochs::Int = -1,
      learningrate::Float64 = -Inf,
      learningrates::Vector{Float64} = Vector{Float64}(),
      sdlearningrate::Float64 = 0.0,
      sdlearningrates::Vector{Float64} = Vector{Float64}(),
      monitoring = ((rbm, epoch) -> nothing),
      rbmtype::DataType = BernoulliRBM,
      nhidden::Int = -1,
      nvisible::Int = -1,
      pcd::Bool = true,
      cdsteps::Int = 1)

   usedefaultepochs = (epochs < 0)
   usedefaultlearningrate = (learningrate < 0 && isempty(learningrates))
   TrainLayer(
         (usedefaultepochs ? 10 : epochs),
         usedefaultepochs,
         (usedefaultlearningrate ? 0.005 : learningrate),
         learningrates,
         usedefaultlearningrate,
         sdlearningrate, sdlearningrates,
         monitoring, rbmtype, nhidden, nvisible, pcd, cdsteps)
end

"""
Encapsulates a vector of `TrainLayer` objects for training a partitioned layer.
"""
type TrainPartitionedLayer <: AbstractTrainLayer
   parts::Vector{TrainLayer}
end


const AbstractTrainLayers = Vector{<:AbstractTrainLayer}


"""
Converts a vector to a vector of the most specific type that all
elements share as common supertype.
"""
function converttomostspecifictype(v::Vector)
   Vector{mostspecifictype(v)}(v)
end


"""
    mostspecifictype(v)
Returns the most specific supertype for all elements in the vector `v`.
"""
function mostspecifictype(v::Vector)
   mapreduce(typeof, typejoin, v)
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
   For more information see help of `TrainLayer`.
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
   # --> check for correct specification and derive/set some parameters

   # avoid modification of given object
   trainlayers = deepcopy(trainlayers)

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
      if isempty(trainlayer.learningrates)
         # learning rates for each epoch have neither been explicitly specified
         # --> set same value for each epoch
         trainlayer.learningrates =
               fill(trainlayer.learningrate, trainlayer.epochs)
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

   # derive number of visible nodes for higher layers
   for i = 2:length(trainlayers)
      derive_nvisibles!(trainlayers[i], trainlayers[i-1])
   end

   # check number of visible nodes for first layer
   if isa(trainlayers[1], TrainPartitionedLayer)
      partswithoutnvisible = find(t -> t.nvisible <= 0, trainlayers[1].parts)
      if length(partswithoutnvisible) > 1
         error("The number of visible nodes is unspecified for more " *
               "than one segment in the first layer. ")
      end

      nvariables = size(x, 2)
      nvisiblestotal = mapreduce(t -> max(t.nvisible, 0), +, trainlayers[1].parts)

      if length(partswithoutnvisible) == 1
         # if only one is unspecified, derive it
         partwithunspecifiednvisible = trainlayers[1].parts[partswithoutnvisible[1]]
         partwithunspecifiednvisible.nvisible = nvariables - nvisiblestotal
      elseif nvisiblestotal != nvariables
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
         upfactor = upfactor, downfactor = downfactor,
         nhidden = trainlayer.nhidden,
         epochs = trainlayer.epochs,
         rbmtype = trainlayer.rbmtype,
         learningrates = trainlayer.learningrates,
         cdsteps = trainlayer.cdsteps,
         pcd = trainlayer.pcd,
         sdlearningrate = trainlayer.sdlearningrate,
         sdlearningrates = trainlayer.sdlearningrates,
         monitoring = trainlayer.monitoring)
end

""" Trains a partitioned layer for `stackrbms`. """
function stackrbms_trainlayer(x::Matrix{Float64},
      trainpartitionedlayer::TrainPartitionedLayer;
      upfactor::Float64 = 1.0, downfactor::Float64 = 1.0)

   visranges = ranges(map(t -> t.nvisible, trainpartitionedlayer.parts))

   # prepare the arguments before the for-loop for each call
   # to avoid unnecessary copying
   trainingargs = map(
         i -> (
               x[:, visranges[i]],
               trainpartitionedlayer.parts[i]
         ),
         eachindex(trainpartitionedlayer.parts))

   rbms = @parallel (vcat) for arg in trainingargs
      stackrbms_trainlayer(arg[1], arg[2];
            upfactor = upfactor, downfactor = downfactor)
   end

   commontype = mostspecifictype(rbms)
   PartitionedRBM{commontype}(Vector{commontype}(rbms))
end
