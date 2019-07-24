"""
Return a new function that does all the monitoring using the
`monitoring` function (or functions) and the
`monitoringdata` and stores the result in the given `monitor`.
"""
function combined_monitoring(monitoring::Function,
      monitor::Monitor, monitoringdata::DataDict)

   (rbm, epoch) -> monitoring(monitor, rbm, epoch, monitoringdata)
end

function combined_monitoring(monitoring::Vector{Function},
      monitor::Monitor, monitoringdata::DataDict)

   (rbm, epoch) ->
         for f in monitoring
            f(monitor, rbm, epoch, monitoringdata)
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

function setmonitorsup!(trainlayer::TrainPartitionedLayer, monitoring::Function)
   setmonitorsup!(trainlayer.parts, monitoring)
end

function setmonitorsup!(trainlayers::Vector{<:AbstractTrainLayer},
      monitoring::Function)

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
   monitors = setmonitorsup!(trainlayers, monitoring) # TODO array of functions

   # The training is done with the new trainlayers argument that contains
   # the monitoring functions that write into monitors.
   rbmstack = stackrbms_trainlayers(x, trainlayers;
         monitoringdata = monitoringdata, trainargs...)

   monitors, rbmstack
end


# TODO fix and document
function monitored_fitdbm(x::Matrix{Float64};
      monitoring::Union{Function, Vector{Function}} = emptyfunc,
      monitoringdata::DataDict = DataDict("Training data" => x),
      monitoringpretraining::Function = emptyfunc,
      monitoringdatapretraining::DataDict = monitoringdata,
      kwargs...)


   monitored_stackrbms(x;
         monitoring = monitoringpretraining,
         monitoringdata = monitoringdatapretraining,
         kwargs...)

   monitor = Monitor()
   rbm = fitdbm(x;
         monitoring = combined_monitoring(monitoring, monitor, monitoringdata),
         monitoringpretraining = (rbm, epoch, datadict) ->
               monitoringpretraining(monitor, rbm, epoch, monitoringdatapretraining),
         monitoringdatapretraining = monitoringdatapretraining,
         kwargs...)
   monitor, rbm
end