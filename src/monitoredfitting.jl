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

# TODO document
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
Creates monitors and sets the monitoring function in trainlayers such that
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


# TODO document
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

# TODO document
function monitored_traindbm!(dbm::MultimodalDBM, x::Matrix{Float64};
      monitoring::Union{Function, Vector{Function}} = emptyfunc,
      monitoringdata::DataDict = DataDict("Training data" => x),
      kwargs...)

   monitor = Monitor()
   monitoringfun = combined_monitoring(monitoring, monitor, monitoringdata)
   traindbm!(dbm, x; monitoring = monitoringfun, kwargs...)
   monitor, dbm
end


# TODO document
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


const traindbm_argkeys = [:epochs, :nparticles, :learningrate, :learningrates,
      :sdlearningrate, :sdlearningrates, :optimizer, :optimizers]

const fitrbm_argkeys_for_stackrbm = [:nhiddens, :epochspretraining, :batchsizepretraining,
      :learningratepretraining, :optimizer, :pretraining]

const monitored_fitrbm_argkeys = union(traindbm_argkeys, fitrbm_argkeys_for_stackrbm)

function monitored_fitdbm_split_kwargs(kwargs)
   argsdict = Dict{Symbol, Any}(kwargs)

   unknownarguments = setdiff(keys(kwargs), monitored_fitrbm_argkeys)
   if !isempty(unknownarguments)
      error("Unknown keyword arguments: " * string(unknownarguments))
   end

   # a function to filter out the arguments with value "nothing":
   # the default values do not need to be touched.
   function kwargs_without_nothings(args)
      Dict{Symbol, Any}(filter(k_v -> k_v[2] != nothing, args))
   end

   # used to handle the precedence of the default values
   function get_key_or_altkey_else_nothing(dict, key, altkey)
      ret = get(dict, key, nothing)
      if ret === nothing
         ret = get(dict, altkey, nothing)
      end
      ret
   end

   stackrbms_kwargs = kwargs_without_nothings([
         :nhiddens => get(argsdict, :nhiddens, nothing),
         :epochs => get_key_or_altkey_else_nothing(argsdict, :epochspretraining, :epochs),
         :learningrate => get_key_or_altkey_else_nothing(
               argsdict, :learningratepretraining, :learningrate),
         :batchsize => get(argsdict, :batchsizepretraining, nothing),
         :optimizer => get_key_or_altkey_else_nothing(
               argsdict, :optimizerpretraining, :optimizer),
         :trainlayers => get(argsdict, :pretraining, nothing)
   ])

   traindbm_kwargs = kwargs_without_nothings(
         [arg => get(argsdict, arg, nothing) for arg in traindbm_argkeys])

   stackrbms_kwargs, traindbm_kwargs
end