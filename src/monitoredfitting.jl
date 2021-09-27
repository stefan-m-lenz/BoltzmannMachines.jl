"""
Return a new function that does all the monitoring using the
`monitoring` function (or functions) and the
`monitoringdata` and stores the result in the given `monitor`.
"""
function combined_monitoring(monitoring::Function,
      monitor::Monitor, monitoringdata::DataDict)

   (bm, epoch) -> monitoring(monitor, bm, epoch, monitoringdata)
end

function combined_monitoring(monitoring::Vector{Function},
      monitor::Monitor, monitoringdata::DataDict)

   (bm, epoch) ->
         for f in monitoring
            f(monitor, bm, epoch, monitoringdata)
         end
end


"""
    monitored_fitrbm(x; ...)
This function performs the same training procedure as `fitrbm`,
but facilitates monitoring:
It fits an RBM model on the data set `x` and collects monitoring results
during the training in one `Monitor` object.
Both the collected monitoring results and the trained RBM are returned.

# Optional keyword arguments:
* `monitoring`: A monitoring function or a vector of monitoring functions
  that accept four arguments:
  1. a `Monitor` object, which is used to collect the result of
     the monitoring function(s)
  2. the RBM
  3. the epoch
  4. the data used for monitoring.
  By default, there is no monitoring.
* `monitoringdata`: a `DataDict`, which contains the data that is used for
   monitoring and passed to the `monitoring` functions(s).
   By default, the training data `x` is used for monitoring.
* Other specified keyword arguments are simply handed to `fitrbm`.
  For more information, please see the documentation there.

# Example:
    using Random; Random.seed!(0)
    xtrain, xtest = splitdata(barsandstripes(100, 4), 0.3)
    monitor, rbm = monitored_fitrbm(xtrain;
        monitoring = [monitorreconstructionerror!, monitorexactloglikelihood!],
        monitoringdata = DataDict("Training data" => xtrain, "Test data" => xtest),
        # some arguments for `fitrbm`:
        nhidden = 10, learningrate = 0.002, epochs = 200)
    using BoltzmannMachinesPlots
    plotevaluation(monitor, monitorreconstructionerror)
    plotevaluation(monitor, monitorexactloglikelihood)
"""
function monitored_fitrbm(x::Matrix{Float64};
      monitoring::Union{Function, Vector{Function}} = emptyfunc,
      monitoringdata::DataDict = DataDict("Training data" => x),
      kwargs...)

   monitor = Monitor()
   monitoringfun = combined_monitoring(monitoring, monitor, monitoringdata)
   rbm = fitrbm(x; monitoring = monitoringfun, kwargs...)
   monitor, rbm
end


"""
Creates monitors and sets the monitoring function in `trainlayer` such that
the monitoring is recorded in the newly created monitors.
Returns the created monitors.
"""
function setmonitorsup!(trainlayer::TrainLayer, monitoring::Function)
   monitor = Monitor()
   trainlayer.monitoring =
         (rbm, epoch, datadict) -> monitoring(monitor, rbm, epoch, datadict)
   monitor
end

function setmonitorsup!(trainlayer::TrainLayer, monitoring::Vector{Function})
   monitor = Monitor()
   trainlayer.monitoring =
         (rbm, epoch, datadict) ->
               for f in monitoring
                  f(monitor, rbm, epoch, datadict)
               end
   monitor
end

function setmonitorsup!(trainlayer::TrainPartitionedLayer,
      monitoring::Union{Function, Vector{Function}})
   setmonitorsup!(trainlayer.parts, monitoring)
end

function setmonitorsup!(trainlayers::Vector{<:AbstractTrainLayer},
      monitoring::Union{Function, Vector{Function}})

   [setmonitorsup!(t, monitoring) for t in trainlayers]
end


"""
    monitored_stackrbms(x; ...)
This function performs the same training procedure as `stackrbms`,
but facilitates monitoring:
It trains a stack of RBMs using the data set `x` as input to the first layer
and collects all the monitoring results during the training in a vector of `Monitor`s,
containing one element for each RBM layer.
(Elements for partitioned layers are again vectors, containing one element for each
partition.)
Both the collected monitoring results and the stack of trained RBMs are returned.

# Optional keyword arguments:
* `monitoring`: A monitoring function or a vector of monitoring functions
  that accept four arguments:
  1. a `Monitor` object, which is used to collect the result of
     the monitoring function(s)
  2. the RBM
  3. the epoch
  4. the data used for monitoring.
  By default, there is no monitoring.
* `monitoringdata`: a `DataDict`, which contains the data that is used for
   monitoring. For the first layer, the data is passed directly to the `monitoring`
   function(s). For monitoring the training of the higher layers,
   the data is propagated through the layers below first.
   By default, the training data `x` is used for monitoring.
* Other specified keyword arguments are simply handed to `stackrbms`.
  For more information, please see the documentation there.

# Example:
    using Random; Random.seed!(0)
    xtrain, xtest = splitdata(barsandstripes(100, 4), 0.5)
    monitors, rbm = monitored_stackrbms(xtrain;
        monitoring = monitorreconstructionerror!,
        monitoringdata = DataDict("Training data" => xtrain, "Test data" => xtest),
        # some arguments for `stackrbms`:
        nhiddens = [4; 3], learningrate = 0.005, epochs = 100)
    using BoltzmannMachinesPlots
    plotevaluation(monitors[1]) # view monitoring of first RBM
    plotevaluation(monitors[2]) # view monitoring of second RBM
"""
function monitored_stackrbms(x::Matrix{Float64};
      monitoring::Union{Function, Vector{Function}} = emptyfunc,
      monitoringdata::DataDict = DataDict("Training data" => x),
      kwargs...)

   trainargs = Dict{Symbol, Any}()
   prepareargs = Dict{Symbol, Any}()
   for (key, value) in kwargs
      if key in [:samplehidden, :predbm]
         trainargs[key] = value
      else
         prepareargs[key] = value
      end
   end

   # First steps like in stackrbms
   stackrbms_checkmonitoringdata(x, monitoringdata)
   trainlayers = stackrbms_preparetrainlayers(x; prepareargs...)

   # Update trainlayers with newly constructed monitoring functions.
   monitors = setmonitorsup!(trainlayers, monitoring)

   # The training is done with the new trainlayers argument that contains
   # the monitoring functions that write into monitors.
   rbmstack = stackrbms_trainlayers(x, trainlayers;
         monitoringdata = monitoringdata, trainargs...)

   monitors, rbmstack
end


"""
    monitored_traindbm!(dbm, x; ...)
This function performs the same training procedure as `traindbm!`,
but facilitates monitoring:
It performs fine-tuning of the given `dbm` on the data set `x`
and collects monitoring results during the training in one `Monitor` object.
Both the collected monitoring results and the trained `dbm` are returned.

# Optional keyword arguments:
* `monitoring`: A monitoring function or a vector of monitoring functions
  that accept four arguments:
  1. a `Monitor` object, which is used to collect the result of
     the monitoring function(s)
  2. the DBM
  3. the epoch
  4. the data used for monitoring.
  By default, there is no monitoring.
* `monitoringdata`: a `DataDict`, which contains the data that is used for
   monitoring and passed to the `monitoring` functions(s).
   By default, the training data `x` is used for monitoring.
* Other specified keyword arguments are simply handed to `traindbm!`.
  For more information, please see the documentation there.

# Example:
    using Random; Random.seed!(0)
    xtrain, xtest = splitdata(barsandstripes(100, 4), 0.1)
    dbm = stackrbms(xtrain; predbm = true, epochs = 20)
    monitor, dbm = monitored_traindbm!(dbm, xtrain;
        monitoring = monitorlogproblowerbound!,
        monitoringdata = DataDict("Training data" => xtrain, "Test data" => xtest),
        # some arguments for `traindbm!`:
        epochs = 100, learningrate = 0.1)
    using BoltzmannMachinesPlots
    plotevaluation(monitor)
"""
function monitored_traindbm!(dbm::MultimodalDBM, x::Matrix{Float64};
      monitoring::Union{Function, Vector{Function}} = emptyfunc,
      monitoringdata::DataDict = DataDict("Training data" => x),
      kwargs...)

   monitor = Monitor()
   monitoringfun = combined_monitoring(monitoring, monitor, monitoringdata)
   traindbm!(dbm, x; monitoring = monitoringfun, kwargs...)
   monitor, dbm
end


"""
    monitored_fitdbm(x; ...)
This function performs the same training procedure as `fitdbm`,
but facilitates monitoring:
It fits an DBM model on the data set `x` using greedy layerwise pre-training and
subsequent fine-tuning and collects all the monitoring results during the training.
The monitoring results are stored in a vector of `Monitor`s,
containing one element for each RBM layer and as last element the monitoring
results for fine-tuning.
(Monitoring elements from the pre-training of partitioned layers are again vectors,
containing one element for each partition.)
Both the collected monitoring results and the trained DBM are returned.

See also: `monitored_stackrbms`, `monitored_traindbm!`

# Optional keyword arguments:
* `monitoring`: Used for fine-tuning.
  A monitoring function or a vector of monitoring functions that accept four arguments:
  1. a `Monitor` object, which is used to collect the result of
     the monitoring function(s)
  2. the DBM
  3. the epoch
  4. the data used for monitoring.
  By default, there is no monitoring of fine-tuning.
* `monitoringdata`: a `DataDict`, which contains the data that is used for the
   monitoring. For the pre-training of the first layer and for fine-tuning,
   the data is passed directly to the `monitoring` function(s).
   For monitoring the pre-training of the higher RBM layers,
   the data is propagated through the layers below first.
   By default, the training data `x` is used for monitoring.
* `monitoringpretraining`: Used for pre-training. A four-argument function like
   `monitoring`, but accepts as second argument an RBM.
   By default there is no monitoring of the pre-training.
* `monitoringdatapretraining`: Monitoring data used only for pre-training.
   Defaults to `monitoringdata`.
* Other specified keyword arguments are simply handed to `fitdbm`.
  For more information, please see the documentation there.

# Example:
    using Random; Random.seed!(1)
    xtrain, xtest = splitdata(barsandstripes(100, 4), 0.5)
    monitors, dbm = monitored_fitdbm(xtrain;
        monitoringpretraining = monitorreconstructionerror!,
        monitoring = monitorlogproblowerbound!,
        monitoringdata = DataDict("Training data" => xtrain, "Test data" => xtest),
        # some arguments for `fitdbm`:
        nhiddens = [4; 3], learningratepretraining = 0.01,
        learningrate = 0.05, epochspretraining = 100, epochs = 50)
    using BoltzmannMachinesPlots
    plotevaluation(monitors[1]) # view monitoring of first RBM
    plotevaluation(monitors[2]) # view monitoring of second RBM
    plotevaluation(monitors[3]) # view monitoring fine-tuning
"""
function monitored_fitdbm(x::Matrix{Float64};
      monitoring::Union{Function, Vector{Function}} = emptyfunc,
      monitoringdata::DataDict = DataDict("Training data" => x),
      monitoringpretraining::Union{Function, Vector{Function}} = emptyfunc,
      monitoringdatapretraining::DataDict = monitoringdata,
      kwargs...)

   stackrbms_kwargs, traindbm_kwargs = monitored_fitdbm_split_kwargs(kwargs)

   monitors, dbm = monitored_stackrbms(x;
         monitoring = monitoringpretraining,
         monitoringdata = monitoringdatapretraining,
         predbm = true,
         stackrbms_kwargs...)

   monitor = Monitor()
   monitoringfun = combined_monitoring(monitoring, monitor, monitoringdata)
   traindbm!(dbm, x; monitoring = monitoringfun, traindbm_kwargs...)

   push!(monitors, monitor)

   monitors, dbm
end


const traindbm_argkeys = [:epochs, :nparticles,
      :learningrate, :learningrates, :sdlearningrate, :sdlearningrates,
      :batchsize, :optimizer, :optimizers]

const fitrbm_argkeys_for_stackrbm = [:nhiddens, :epochspretraining, :batchsizepretraining,
      :learningratepretraining, :optimizer, :optimizerpretraining, :pretraining]

const finetuning_argkeys = [:batchsizefinetuning] # TODO

const monitored_fitdbm_argkeys = union(traindbm_argkeys, fitrbm_argkeys_for_stackrbm)

function monitored_fitdbm_split_kwargs(kwargs)
   argsdict = Dict{Symbol, Any}(kwargs)

   unknownarguments = setdiff(keys(kwargs), monitored_fitdbm_argkeys)
   if !isempty(unknownarguments)
      error("Unknown keyword arguments: " * string(unknownarguments))
   end

   # a function to filter out the arguments with value "nothing":
   # the default values do not need to be touched.
   function kwargs_without_nothings(args)
      Dict{Symbol, Any}(filter(k_v -> k_v[2] !== nothing, args))
   end

   # used to handle the precedence of the default values
   function get_key_or_altkey_else_nothing(dict, key, altkey)
      ret = get(dict, key, nothing)
      if ret === nothing
         ret = get(dict, altkey, nothing)
      end
      ret
   end

   stackrbms_kwargs = kwargs_without_nothings(Pair{Symbol, Any}[
         :nhiddens => get(argsdict, :nhiddens, nothing),
         :epochs => get_key_or_altkey_else_nothing(argsdict, :epochspretraining, :epochs),
         :learningrate => get_key_or_altkey_else_nothing(
               argsdict, :learningratepretraining, :learningrate),
         :batchsize =>  get_key_or_altkey_else_nothing(
               argsdict, :batchsizepretraining, :batchsize),
         :optimizer => get_key_or_altkey_else_nothing(
               argsdict, :optimizerpretraining, :optimizer),
         :trainlayers => get(argsdict, :pretraining, nothing)
   ])

   traindbm_kwargs = kwargs_without_nothings(
         [arg => get(argsdict, arg, nothing) for arg in traindbm_argkeys])

   stackrbms_kwargs, traindbm_kwargs
end
