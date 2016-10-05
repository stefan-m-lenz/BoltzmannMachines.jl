type MonitoringItem
   evaluation::AbstractString
   epoch::Int
   value::Float64
   datasetname::AbstractString
end

typealias Monitor Vector{MonitoringItem}
typealias DataDict Dict{AbstractString, Array{Float64,2}}

const monitorreconstructionerror = "reconstructionerror"
const monitorloglikelihood = "loglikelihood"
const monitorexactloglikelihood = "exactloglikelihood"
const monitorweightsnorm = "weightsnorm"
const monitorsd = "sd"
const monitorcordiff = "cordiff"
const monitorfreeenergy = "freeenergy"
const monitorlogproblowerbound = "logproblowerbound"
const monitoraisstandarddeviation = "aisstandarddeviation"
const monitoraisr = "aisr"

function monitorlogproblowerbound!(monitor::Monitor, dbm::DBMParam,
      epoch::Int, datadict::DataDict)

   impweights = BMs.aisimportanceweights(dbm) # TODO parameter evtl. anpassen
   r = mean(impweights)
   sd = BMs.aisstandarddeviation(impweights)
   logz = BMs.logpartitionfunction(dbm, r)
   push!(monitor,
         MonitoringItem(BMs.monitoraisstandarddeviation, epoch, sd, ""),
         MonitoringItem(BMs.monitoraisr, epoch, r, ""))
   for (datasetname, x) in datadict
      push!(monitor, MonitoringItem(BMs.monitorlogproblowerbound, epoch,
            BMs.logproblowerbound(dbm, x, logpartitionfunction = logz),
            datasetname))
   end
end

function monitorloglikelihood!(monitor::Monitor, rbm::AbstractRBM,
      epoch::Int, datadict::DataDict;
      # optional arguments for AIS:
      ntemperatures::Int = 100,
      beta::Array{Float64,1} = collect(0:(1/ntemperatures):1),
      nparticles::Int = 100,
      burnin::Int = 10)

   impweights = BMs.aisimportanceweights(rbm;
         ntemperatures = ntemperatures, beta = beta,
         nparticles = nparticles, burnin = burnin)

   r = mean(impweights)
   sd = BMs.aisstandarddeviation(impweights)
   logz = BMs.logpartitionfunction(rbm, r)
   push!(monitor,
         MonitoringItem(BMs.monitoraisstandarddeviation, epoch, sd, ""),
         MonitoringItem(BMs.monitoraisr, epoch, r, ""))
   for (datasetname, x) in datadict
      push!(monitor, MonitoringItem(BMs.monitorloglikelihood, epoch,
            BMs.loglikelihood(rbm, x, logz), datasetname))
   end
end

function monitorexactloglikelihood!(monitor::Monitor, rbm::AbstractRBM,
      epoch::Int, datadict::DataDict)

   logz = BMs.exactlogpartitionfunction(rbm)
   for (datasetname, x) in datadict
      push!(monitor, MonitoringItem(BMs.monitorexactloglikelihood, epoch,
               loglikelihood(rbm, x, logz), datasetname))
   end
end

function monitorreconstructionerror!(monitor::Monitor, rbm::AbstractRBM,
      epoch::Int, datadict::DataDict)

   for (datasetname, x) in datadict
      push!(monitor, MonitoringItem(BMs.monitorreconstructionerror, epoch,
               reconstructionerror(rbm, x), datasetname))
   end
end

function monitorweightsnorm!(monitor::Monitor, rbm::AbstractRBM, epoch::Int)
   push!(monitor,
         MonitoringItem(BMs.monitorweightsnorm, epoch,
               norm(rbm.weights), "Weights"),
         MonitoringItem(BMs.monitorweightsnorm, epoch,
               norm(rbm.a), "Visible bias"),
         MonitoringItem(BMs.monitorweightsnorm, epoch,
               norm(rbm.b,), "Hidden bias"))
end

function monitorsd!(monitor::Monitor, gbrbm::GaussianBernoulliRBM, epoch::Int)
   nvisible = length(gbrbm.sd)
   for i = 1:nvisible
      push!(monitor,
            MonitoringItem(BMs.monitorsd, epoch, gbrbm.sd[i], string(i)))
   end
end

"
Creates and returns a dictionary with the same keys as the given `datadict`.
The values of the returned dictionary are the correlations of the samples in
the datasets given as values in the `datadict`.
"
function correlations(datadict::DataDict)
   DataDict(map(kv -> (kv[1] => cor(kv[2])), datadict))
end

"""
    monitorcordiff!(monitor, rbm, epoch, cordict)
Generates samples and records the distance of their correlation matrix to the
correlation matrices for (original) datasets contained in the `cordict`.
"""
function monitorcordiff!(monitor::Monitor, bm::AbstractBM, epoch::Int,
      cordict::DataDict;
      nparticles::Int = 3000, burnin = 10)

   samplecor = cor(BMs.sampleparticles(bm, nparticles, burnin)[1])
   for (datasetname, datacor) in cordict
      push!(monitor, MonitoringItem(BMs.monitorcordiff, epoch,
               norm(samplecor-datacor), datasetname))
   end
end

function monitorfreeenergy!(monitor::Monitor, rbm::AbstractRBM,
      epoch::Int, datadict::DataDict)

   for (datasetname, x) in datadict
      push!(monitor, MonitoringItem(BMs.monitorfreeenergy, epoch,
            freeenergy(rbm, x), datasetname))
   end
end

