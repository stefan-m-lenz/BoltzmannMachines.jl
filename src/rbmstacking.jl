"""
Abstract supertype for layerwise training specification.
May be specifications for a normal RBM layer (see `TrainLayer`) or
multiple combined specifications for a partitioned layer
(see `TrainPartitionedLayer`).
"""
abstract type AbstractTrainLayer end

mutable struct TrainLayer <: AbstractTrainLayer
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
   batchsize::Int
   usedefaultbatchsize::Bool
   pcd::Bool
   cdsteps::Int
   startrbm::AbstractRBM
   optimizer::AbstractOptimizer
   optimizers::Vector{AbstractOptimizer}
end


"""
Specify parameters for training one RBM-layer in a DBM.

# Optional keyword arguments:
* The optional keyword arguments `rbmtype`, `nhidden`, `epochs`,
  `learningrate`/`learningrates`, `sdlearningrate`/`sdlearningrates`,
  `batchsize`, `pcd`, `cdsteps`, `startrbm` and `optimizer`/`optimizers`
  are passed to `fitrbm`. For a detailed description, see there.
  If a negative value is specified for `learningrate` or `epochs`, this indicates
  that a corresponding default value should be used
  (parameter defined by call to `stackrbms`).
* `monitoring`: also like in `fitrbm`, but may take a `DataDict` as third argument
   (see function `stackrbms` and its argument `monitoringdata`).
* `nvisible`: Number of visible units in the RBM. Only relevant for partitioning.
   This parameter is derived as much as possible by `stackrbms`.
   For `MultimodalDBM`s with a partitioned first layer, it is necessary to specify
   the number of visible nodes for all but at most one partition in the input layer.
"""
function TrainLayer(;
      epochs::Int = -1,
      learningrate::Float64 = -Inf,
      learningrates::Vector{Float64} = Vector{Float64}(),
      sdlearningrate::Float64 = 0.0,
      sdlearningrates::Vector{Float64} = Vector{Float64}(),
      monitoring = nomonitoring,
      rbmtype::DataType = BernoulliRBM,
      nhidden::Int = -1,
      nvisible::Int = -1,
      batchsize::Int = -1,
      pcd::Bool = true,
      cdsteps::Int = 1,
      startrbm::AbstractRBM = NoRBM(),
      optimizer::AbstractOptimizer = NoOptimizer(),
      optimizers::Vector{<:AbstractOptimizer} = Vector{AbstractOptimizer}())

   usedefaultepochs = (epochs < 0)
   usedefaultlearningrate = (learningrate < 0 && isempty(learningrates))
   usedefaultbatchsize = (batchsize <= 0)
   TrainLayer(
         (usedefaultepochs ? 10 : epochs),
         usedefaultepochs,
         (usedefaultlearningrate ? 0.005 : learningrate),
         learningrates,
         usedefaultlearningrate,
         sdlearningrate, sdlearningrates,
         monitoring, rbmtype, nhidden, nvisible,
         (usedefaultbatchsize ? 1 : batchsize),
         usedefaultbatchsize,
         pcd, cdsteps,
         startrbm,
         optimizer, optimizers)
end

"""
Encapsulates a vector of `TrainLayer` objects for training a partitioned layer.
"""
struct TrainPartitionedLayer <: AbstractTrainLayer
   parts::Vector{TrainLayer}
end


const AbstractTrainLayers = Vector{<:AbstractTrainLayer}


"""
A dictionary containing names of data sets as keys and the data sets (matrices
with samples in rows) as values.
"""
const DataDict = Dict{String,Array{Float64,2}}


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
    propagateforward(rbm, datadict, factor)
Returns a new `DataDict` containing the same labels as the given `datadict` but
as mapped values it contains the hidden potential in the `rbm` of the original
datasets. The factor is applied for calculating the hidden potential and is 1.0
by default.
"""
function propagateforward(rbm::AbstractRBM, datadict::DataDict, factor::Float64 = 1.0)
   DataDict(p[1] => hiddenpotential(rbm, p[2], factor) for p = pairs(datadict))
end

function partitioneddata(datadict::DataDict, visrange::UnitRange{Int})
   DataDict(p[1] => p[2][:, visrange] for p = pairs(datadict))
end


function setdefaultsforunspecified!(trainlayer::TrainLayer,
         learningrate::Float64, epochs::Int, batchsize::Int,
         optimizer::AbstractOptimizer)

   if trainlayer.usedefaultlearningrate
      trainlayer.learningrate = learningrate
   end
   if trainlayer.usedefaultepochs
      trainlayer.epochs = epochs
   end
   if trainlayer.usedefaultbatchsize
      trainlayer.batchsize = batchsize
   end
   if trainlayer.optimizer === NoOptimizer()
      trainlayer.optimizer = optimizer
   end

   if isempty(trainlayer.learningrates)
      # learning rates for each epoch have neither been explicitly specified
      # --> set same value for each epoch
      trainlayer.learningrates =
            fill(trainlayer.learningrate, trainlayer.epochs)
   end
   if isempty(trainlayer.sdlearningrates)
      # same for sdlearningrates
      trainlayer.sdlearningrates =
            fill(trainlayer.sdlearningrate, trainlayer.epochs)
   end
   trainlayer
end

function setdefaultsforunspecified!(trainpartitionedlayer::TrainPartitionedLayer,
      learningrate::Float64, epochs::Int, batchsize::Int,
      optimizer::AbstractOptimizer)

   setdefaultsforunspecified!(trainpartitionedlayer.parts,
         learningrate, epochs, batchsize, optimizer)
end

function setdefaultsforunspecified!(trainlayers::Vector{T},
      learningrate::Float64, epochs::Int, batchsize::Int,
      optimizer::AbstractOptimizer
      ) where {T <: AbstractTrainLayer}

   for trainlayer in trainlayers
      setdefaultsforunspecified!(trainlayer,
            learningrate, epochs, batchsize, optimizer)
   end
   trainlayers
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
* `batchsize`: size of minibatches, defaults to 1
* `trainlayers`: a vector of `TrainLayer` objects. With this argument it is possible
   to specify the training parameters for each layer/RBM individually.
   If the number of training epochs and the learning rate are not specified
   explicitly for a layer, the values of `epochs` and `learningrate` are used.
   For more information see help of `TrainLayer`.
* `monitoringdata`: a data dictionary (see type `DataDict`)
   The data is propagated forward through the
   network to monitor higher levels.
   If a non-empty dictionary is given, the monitoring functions in the
   `trainlayers`-arguments must accept a `DataDict` as third argument.
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
      batchsize::Int = 1,
      optimizer::AbstractOptimizer = NoOptimizer(),
      trainlayers::AbstractTrainLayers = Vector{TrainLayer}(),
      monitoringdata::DataDict = DataDict())

   stackrbms_checkmonitoringdata(x, monitoringdata)

   trainlayers = stackrbms_preparetrainlayers(trainlayers, x, epochs,
         learningrate, nhiddens, batchsize, optimizer)

   nrbms = length(trainlayers)
   dbmn = Vector{AbstractRBM}(undef, nrbms)

   upfactor = downfactor = 1.0
   if predbm
      upfactor = 2.0
   end

   dbmn[1] = stackrbms_trainlayer(x, trainlayers[1];
         monitoringdata = monitoringdata,
         upfactor = upfactor, downfactor = downfactor)

   hiddenval = x
   for i = 2:nrbms
      hiddenval = hiddenpotential(dbmn[i-1], hiddenval, upfactor)
      if samplehidden
         bernoulli!(hiddenval)
      end

      if !isempty(monitoringdata)
         monitoringdata = propagateforward(dbmn[i-1], monitoringdata, upfactor)
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
            monitoringdata = monitoringdata,
            upfactor = upfactor, downfactor = downfactor)
   end

   dbmn = converttomostspecifictype(dbmn)
   dbmn
end

function stackrbms_checkmonitoringdata(x::Matrix{Float64}, monitoringdata::DataDict)
   if !isempty(monitoringdata)
      if any(map(m -> size(m, 2), values(monitoringdata)) .!= size(x, 2))
         error("Matrices in the dictionary `monitoringdata` must have the same number of columns as the input matrix.")
      end
   end
end


""" Prepares the layerwise training specifications for `stackrbms` """
function stackrbms_preparetrainlayers(
      trainlayers::AbstractTrainLayers,
      x::Matrix{Float64},
      epochs::Int,
      learningrate::Float64,
      nhiddens::Vector{Int},
      batchsize::Int,
      optimizer::AbstractOptimizer)

   if isempty(trainlayers)
      # construct default "trainlayers"
      if isempty(nhiddens)
         nhiddens = fill(size(x,2), 2) # default value for nhiddens
      end
      trainlayers = map(n -> TrainLayer(nhidden = n), nhiddens)
      setdefaultsforunspecified!(trainlayers,
            learningrate, epochs, batchsize, optimizer)
      return trainlayers
   end

   # We are here, Argument "trainlayers" has been specified
   # --> check for correct specification and derive/set some parameters

   # avoid modification of given object
   trainlayers = deepcopy(trainlayers)

   if !isempty(nhiddens)
      warn("Argument `nhiddens` not used.")
   end

   setdefaultsforunspecified!(trainlayers,
         learningrate, epochs, batchsize, optimizer)

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
      partswithoutnvisible = findall(t -> t.nvisible <= 0, trainlayers[1].parts)
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
      monitoringdata::DataDict = DataDict(),
      upfactor::Float64 = 1.0, downfactor::Float64 = 1.0)

   if isempty(monitoringdata)
      monitoring = trainlayer.monitoring
   elseif trainlayer.monitoring != nomonitoring
      monitoring = (rbm, epoch) ->
            trainlayer.monitoring(rbm, epoch, monitoringdata)
   end

   BMs.fitrbm(x;
         upfactor = upfactor, downfactor = downfactor,
         nhidden = trainlayer.nhidden,
         epochs = trainlayer.epochs,
         rbmtype = trainlayer.rbmtype,
         learningrates = trainlayer.learningrates,
         batchsize = trainlayer.batchsize,
         cdsteps = trainlayer.cdsteps,
         pcd = trainlayer.pcd,
         sdlearningrate = trainlayer.sdlearningrate,
         sdlearningrates = trainlayer.sdlearningrates,
         monitoring = monitoring,
         startrbm = trainlayer.startrbm,
         optimizer = trainlayer.optimizer,
         optimizers = trainlayer.optimizers)
end

""" Trains a partitioned layer for `stackrbms`. """
function stackrbms_trainlayer(x::Matrix{Float64},
      trainpartitionedlayer::TrainPartitionedLayer;
      monitoringdata::DataDict = DataDict(),
      upfactor::Float64 = 1.0, downfactor::Float64 = 1.0)

   visranges = ranges(map(t -> t.nvisible, trainpartitionedlayer.parts))

   # prepare the arguments before the for-loop for each call
   # to avoid unnecessary copying
   trainingargs = map(
         i -> (
               x[:, visranges[i]],
               trainpartitionedlayer.parts[i],
               partitioneddata(monitoringdata, visranges[i])
         ),
         eachindex(trainpartitionedlayer.parts))

   rbms = @distributed (vcat) for arg in trainingargs
      stackrbms_trainlayer(arg[1], arg[2];
            monitoringdata = arg[3],
            upfactor = upfactor, downfactor = downfactor)
   end

   commontype = mostspecifictype(rbms)
   PartitionedRBM{commontype}(Vector{commontype}(rbms))
end
