"""
Encapsulates the value of an evaluation calculated in one training epoch.
If the evaluation depends on a dataset,
the dataset's name can be specified also.
"""
struct MonitoringItem
   evaluation::String
   epoch::Int
   value::Float64
   datasetname::String
end


"""
A vector for collecting `MonitoringItem`s during training.
"""
const Monitor = Vector{MonitoringItem}


const monitoraislogr = "aislogr"
const monitoraisstandarddeviation = "aisstandarddeviation"
const monitorcordiff = "cordiff"
const monitorexactloglikelihood = "exactloglikelihood"
const monitorfreeenergy = "freeenergy"
const monitorloglikelihood = "loglikelihood"
const monitorlogproblowerbound = "logproblowerbound"
const monitormeandiff = "monitormeandiff"
const monitormeandiffpervariable = "monitormeandiffpervariable"
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
    means(datadict)
Creates and returns a dictionary with the same keys as the given `datadict`.
The values of the returned dictionary are the samples' means in the `datadict`.
"""
function means(datadict::DataDict)
   DataDict(map(kv -> (kv[1] => mean(kv[2], dims = 1)), datadict))
end


"""
    monitorcordiff!(monitor, rbm, epoch, cordict)
Generates samples and records the distance of their correlation matrix to the
correlation matrices for (original) datasets contained in the `cordict`.
"""
function monitorcordiff!(monitor::Monitor, bm::AbstractBM, epoch::Int,
      cordict::DataDict;
      nparticles::Int = 3000, burnin::Int = 10,
      xgenerated::Matrix{Float64} = sampleparticles(bm, nparticles, burnin)[1])

   samplecor = cor(xgenerated)
   for (datasetname, datacor) in cordict
      push!(monitor, MonitoringItem(monitorcordiff, epoch,
            norm(samplecor - datacor), datasetname))
   end

   monitor
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

   monitor
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

   monitor
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
   for (datasetname, x) in datadict
      push!(monitor, MonitoringItem(BMs.monitorloglikelihood, epoch,
            BMs.loglikelihood(rbm, x, logz), datasetname))
   end
   push!(monitor,
         MonitoringItem(BMs.monitoraisstandarddeviation, epoch, sd, ""),
         MonitoringItem(BMs.monitoraislogr, epoch, logr, ""))

   monitor
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
function monitorlogproblowerbound!(monitor::Monitor, dbm::MultimodalDBM,
      epoch::Int, datadict::DataDict;
      parallelized::Bool = nworkers() > 1,
      # optional arguments for AIS:
      ntemperatures::Int = 100,
      temperatures::Array{Float64,1} = collect(0:(1/ntemperatures):1),
      nparticles::Int = 100,
      burnin::Int = 5)

   if parallelized
      logimpweights = batchparallelized(
            n -> aislogimpweights(dbm;
                  ntemperatures = ntemperatures, temperatures = temperatures,
                  nparticles = n, burnin = burnin),
            nparticles, vcat)
   else
      logimpweights = aislogimpweights(dbm;
            ntemperatures = ntemperatures, temperatures = temperatures,
            nparticles = nparticles, burnin = burnin)
   end

   logr = logmeanexp(logimpweights)
   sd = aisstandarddeviation(logimpweights)
   logz = logpartitionfunction(dbm, logr)

   for (datasetname, x) in datadict
      push!(monitor, MonitoringItem(monitorlogproblowerbound, epoch,
            logproblowerbound(dbm, x, logz),
            datasetname))
   end
   push!(monitor,
         MonitoringItem(monitoraisstandarddeviation, epoch, sd, ""),
         MonitoringItem(monitoraislogr, epoch, logr, ""))

   monitor
end


function monitormeandiff!(monitor::Monitor, bm::AbstractBM, epoch::Int,
      meandict::DataDict;
      nparticles::Int = 3000, burnin::Int = 10,
      xgenerated::Matrix{Float64} = sampleparticles(bm, nparticles, burnin)[1])

   samplemean = mean(xgenerated, dims = 1)
   for (datasetname, datamean) in meandict
      push!(monitor, MonitoringItem(monitormeandiff, epoch,
               norm(samplemean - datamean), datasetname))
   end

   monitor
end


function monitormeandiffpervariable!(monitor::Monitor, bm::AbstractBM, epoch::Int,
      meandict::DataDict;
      nparticles::Int = 3000, burnin::Int = 10,
      xgenerated::Matrix{Float64} = sampleparticles(bm, nparticles, burnin)[1],
      variables::Vector{Int} = collect(1:size(xgenerated, 2)))

   samplemean = mean(xgenerated, dims = 1)
   for (datasetname, datamean) in meandict
      for i in variables
         push!(monitor, MonitoringItem(monitormeandiffpervariable, epoch,
               samplemean[i] - datamean[i], datasetname * "/Var" * string(i)))
      end
   end

   monitor
end


function monitormeanandcordiff!(monitor::Monitor, bm::AbstractBM,
      epoch::Int;
      meandict::DataDict = DataDict(),
      cordict::DataDict = DataDict(),
      nparticles::Int = 3000, burnin::Int = 10,
      xgenerated::Matrix{Float64} = sampleparticles(bm, nparticles, burnin)[1])

   monitormeandiff!(monitor, bm, epoch, meandict; xgenerated = xgenerated)
   monitorcordiff!(monitor, bm, epoch, cordict; xgenerated = xgenerated)
end


"""
    monitorreconstructionerror!(monitor, rbm, epoch, datadict)
Computes the reconstruction error for the data sets in the `datadict`
and the `rbm` and stores the values in the `monitor`.
"""
function monitorreconstructionerror!(monitor::Monitor, rbm::AbstractRBM,
      epoch::Int, datadict::DataDict)

   for (datasetname, x) in datadict
      push!(monitor, MonitoringItem(monitorreconstructionerror, epoch,
               reconstructionerror(rbm, x), datasetname))
   end

   monitor
end


function monitorsd!(monitor::Monitor,
      gbrbm::Union{GaussianBernoulliRBM, GaussianBernoulliRBM2}, epoch::Int)

   nvisible = length(gbrbm.sd)
   for i = 1:nvisible
      push!(monitor,
            MonitoringItem(monitorsd, epoch, gbrbm.sd[i], string(i)))
   end

   monitor
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
         MonitoringItem(monitorweightsnorm, epoch,
               norm(rbm.weights), "Weights"),
         MonitoringItem(monitorweightsnorm, epoch,
               norm(rbm.visbias), "Visible bias"),
         MonitoringItem(monitorweightsnorm, epoch,
               norm(rbm.hidbias), "Hidden bias"))
end
