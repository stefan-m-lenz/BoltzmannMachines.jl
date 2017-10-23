"""
Encapsulates the value of an evaluation calculated in one training epoch.
If the evaluation depends on a dataset,
the dataset's name can be specified also.
"""
type MonitoringItem
   evaluation::AbstractString
   epoch::Int
   value::Float64
   datasetname::AbstractString
end


"""
A vector for collecting `MonitoringItem`s during training.
"""
const Monitor = Vector{MonitoringItem}


"""
A dictionary containing names of data sets as keys and the data sets (matrices
with samples in rows) as values.
"""
const DataDict = Dict{AbstractString,Array{Float64,2}}


const monitoraislogr = "aislogr"
const monitoraisstandarddeviation = "aisstandarddeviation"
const monitorcordiff = "cordiff"
const monitorexactloglikelihood = "exactloglikelihood"
const monitorfreeenergy = "freeenergy"
const monitorloglikelihood = "loglikelihood"
const monitorlogproblowerbound = "logproblowerbound"
const monitorreconstructionerror = "reconstructionerror"
const monitorsd = "sd"
const monitorweightsnorm = "weightsnorm"


"""
    correlations(datadict)
Creates and returns a dictionary with the same keys as the given `datadict`.
The values of the returned dictionary are the correlations of the samples in
the datasets given as values in the `datadict`.
"""
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
      nparticles::Int = 3000, burnin::Int = 10)

   samplecor = cor(BMs.sampleparticles(bm, nparticles, burnin)[1])
   for (datasetname, datacor) in cordict
      push!(monitor, MonitoringItem(BMs.monitorcordiff, epoch,
               norm(samplecor-datacor), datasetname))
   end
end


"""
    monitorexactloglikelihood!(monitor, bm, epoch, datadict)
Computes the mean exact log-likelihood in the Boltzmann Machine model `bm`
for the data sets in the DataDict `datadict` and stores this information in
the Monitor `monitor`.
"""
function monitorexactloglikelihood!(monitor::Monitor, bm::AbstractBM,
      epoch::Int, datadict::DataDict)

   logz = BMs.exactlogpartitionfunction(bm)
   for (datasetname, x) in datadict
      push!(monitor, MonitoringItem(BMs.monitorexactloglikelihood, epoch,
               exactloglikelihood(bm, x, logz), datasetname))
   end
end


"""
    monitorfreeenergy!(monitor, rbm, epoch, datadict)
Computes the free energy for the `datadict`'s data sets in the RBM model `rbm`
and stores the information in the `monitor`.
"""
function monitorfreeenergy!(monitor::Monitor, rbm::AbstractRBM,
      epoch::Int, datadict::DataDict)

   for (datasetname, x) in datadict
      push!(monitor, MonitoringItem(BMs.monitorfreeenergy, epoch,
            freeenergy(rbm, x), datasetname))
   end
end


"""
    monitorloglikelihood!(monitor, rbm, epoch, datadict)
Estimates the log-likelihood of the `datadict`'s data sets in the RBM model
`rbm` with AIS and stores the values, together with information about the
variance of the estimator, in the `monitor`.

If there is more than one worker available, the computation is parallelized
by default. Parallelization can be turned on or off with the optional
boolean argument `parallelized`.

For the other optional keyword arguments, see `aislogimportanceweights`.

See also: `loglikelihood`.
"""
function monitorloglikelihood!(monitor::Monitor, rbm::AbstractRBM,
      epoch::Int, datadict::DataDict;
      parallelized::Bool = nworkers() > 1,
      # optional arguments for AIS:
      ntemperatures::Int = 100,
      temperatures::Array{Float64,1} = collect(0:(1/ntemperatures):1),
      nparticles::Int = 100,
      burnin::Int = 5)

   if parallelized
      logimpweights = BMs.batchparallelized(
            n -> BMs.aislogimpweights(rbm;
                  ntemperatures = ntemperatures, temperatures = temperatures,
                  nparticles = n, burnin = burnin),
            nparticles, vcat)
   else
      logimpweights = BMs.aislogimpweights(rbm;
            ntemperatures = ntemperatures, temperatures = temperatures,
            nparticles = nparticles, burnin = burnin)
   end

   logr = logmeanexp(logimpweights)
   sd = BMs.aisstandarddeviation(logimpweights)
   logz = BMs.logpartitionfunction(rbm, logr)
   push!(monitor,
         MonitoringItem(BMs.monitoraisstandarddeviation, epoch, sd, ""),
         MonitoringItem(BMs.monitoraislogr, epoch, logr, ""))
   for (datasetname, x) in datadict
      push!(monitor, MonitoringItem(BMs.monitorloglikelihood, epoch,
            BMs.loglikelihood(rbm, x, logz), datasetname))
   end
end


"""
    monitorlogproblowerbound!(monitor, dbm, epoch, datadict)
Estimates the lower bound of the log probability of the `datadict`'s data sets
in the DBM `dbm` with AIS and stores the values, together with information about
the variance of the estimator, in the `monitor`.

If there is more than one worker available, the computation is parallelized
by default. Parallelization can be turned on or off with the optional
boolean argument `parallelized`.

For the other optional keyword arguments, see `aislogimpweights`.

See also: `logproblowerbound`.
"""
function monitorlogproblowerbound!(monitor::Monitor, dbm::BasicDBM,
      epoch::Int, datadict::DataDict;
      parallelized::Bool = nworkers() > 1,
      # optional arguments for AIS:
      ntemperatures::Int = 100,
      temperatures::Array{Float64,1} = collect(0:(1/ntemperatures):1),
      nparticles::Int = 100,
      burnin::Int = 5)

   if parallelized
      logimpweights = BMs.batchparallelized(
            n -> BMs.aislogimpweights(dbm;
                  ntemperatures = ntemperatures, temperatures = temperatures,
                  nparticles = n, burnin = burnin),
            nparticles, vcat)
   else
      logimpweights = BMs.aislogimpweights(dbm;
            ntemperatures = ntemperatures, temperatures = temperatures,
            nparticles = nparticles, burnin = burnin)
   end

   logr = logmeanexp(logimpweights)
   sd = BMs.aisstandarddeviation(logimpweights)
   logz = BMs.logpartitionfunction(dbm, logr)
   push!(monitor,
         MonitoringItem(BMs.monitoraisstandarddeviation, epoch, sd, ""),
         MonitoringItem(BMs.monitoraislogr, epoch, logr, ""))
   for (datasetname, x) in datadict
      push!(monitor, MonitoringItem(BMs.monitorlogproblowerbound, epoch,
            BMs.logproblowerbound(dbm, x, logpartitionfunction = logz),
            datasetname))
   end
end


"""
    monitorreconstructionerror!(monitor, rbm, epoch, datadict)
Computes the reconstruction error for the data sets in the `datadict`
and the `rbm` and stores the values in the `monitor`.
"""
function monitorreconstructionerror!(monitor::Monitor, rbm::AbstractRBM,
      epoch::Int, datadict::DataDict)

   for (datasetname, x) in datadict
      push!(monitor, MonitoringItem(BMs.monitorreconstructionerror, epoch,
               reconstructionerror(rbm, x), datasetname))
   end
end


function monitorsd!(monitor::Monitor, gbrbm::GaussianBernoulliRBM, epoch::Int)
   nvisible = length(gbrbm.sd)
   for i = 1:nvisible
      push!(monitor,
            MonitoringItem(BMs.monitorsd, epoch, gbrbm.sd[i], string(i)))
   end
end


"""
    monitorweightsnorm!(monitor, rbm, epoch)
Computes the L2-norm of the weights matrix and the bias vectors of the `rbm`
and stores the values in the `monitor`.
These values can give a hint how much the updates are changing the parameters
during learning.
"""
function monitorweightsnorm!(monitor::Monitor, rbm::AbstractRBM, epoch::Int)
   push!(monitor,
         MonitoringItem(BMs.monitorweightsnorm, epoch,
               norm(rbm.weights), "Weights"),
         MonitoringItem(BMs.monitorweightsnorm, epoch,
               norm(rbm.visbias), "Visible bias"),
         MonitoringItem(BMs.monitorweightsnorm, epoch,
               norm(rbm.hidbias), "Hidden bias"))
end


"""
    propagateforward(rbm, datadict, factor)
Returns a new `DataDict` containing the same labels as the given `datadict` but
as mapped values it contains the hidden potential in the `rbm` of the original
datasets. The factor is applied for calculating the hidden potential and is 1.0
by default.
"""
function propagateforward(rbm::AbstractRBM, datadict::DataDict, factor::Float64 = 1.0)
    DataDict(map(kv -> (kv[1] => hiddenpotential(rbm, kv[2], factor)), datadict))
end
