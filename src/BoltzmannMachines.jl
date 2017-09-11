module BoltzmannMachines

using Compat

const BMs = BoltzmannMachines

export
   AbstractBM,
      aisimportanceweights, aisprecision, aisstandarddeviation,
      empiricalloglikelihood, energy, exactloglikelihood,
      exactlogpartitionfunction, loglikelihood,
      logpartitionfunction, logpartitionfunctionzeroweights,
      logproblowerbound, reconstructionerror,
      sampleparticles,
      AbstractRBM,
         BernoulliRBM,
         BernoulliGaussianRBM,
         Binomial2BernoulliRBM,
         GaussianBernoulliRBM,
         PartitionedRBM,
         fitrbm, freeenergy, initrbm, joinrbms,
         trainrbm!,
         samplehidden, samplehidden!,
         samplevisible, samplevisible!,
         hiddenpotential, hiddenpotential!,
         hiddeninput, hiddeninput!,
         visiblepotential, visiblepotenial!,
         visibleinput, visibleinput!,
      AbstractDBM,
         BasicDBM,
         TrainLayer,
         addlayer!, fitdbm, gibbssample!, meanfield, stackrbms, traindbm!,
   Particle, Particles,
   Monitor, MonitoringItem, DataDict,
      monitorexactloglikelihood, monitorexactloglikelihood!,
      monitorfreeenergy, monitorfreeenergy!,
      monitorlogproblowerbound, monitorlogproblowerbound!,
      monitorloglikelihood, monitorloglikelihood!,
      monitorreconstructionerror, monitorreconstructionerror!,
      monitorweightsnorm, monitorweightsnorm!,
      propagateforward,
   barsandstripes, splitdata,
   BMPlots


include("rbmtraining.jl")
include("dbmtraining.jl")

const AbstractBM = Union{AbstractDBM,AbstractRBM}


"""
    barsandstripes(nsamples, nvariables)
Generates a test data set. To see the structure in the data set, run e. g.
`reshape(barsandstripes(1, 16), 4,4)` a few times.

Example from:
MacKay, D. (2003). Information Theory, Inference, and Learning Algorithms
"""
function barsandstripes(nsamples::Int, nvariables::Int)
   squareside = sqrt(nvariables)
   if squareside != floor(squareside)
      error("Number of variables must be a square number")
   end
   squareside = round(Int, squareside)
   x = zeros(nsamples, nvariables)
   for i in 1:nsamples
      sample = hcat([(rand() < 0.5 ? ones(squareside) : zeros(squareside))
            for j in 1:squareside]...)
      if rand() < 0.5
         sample = transpose(sample)
      end
      x[i,:] .= sample[:]
      fill!(sample, 0.0)
   end
   x
end


function sigm(x::Array{Float64,1})
   1./(1 + exp.(-x))
end

function sigm(x::Array{Float64,2})
   1./(1 + exp.(-x))
end

function sigm!(x::M) where{M <:AbstractArray{Float64}}
   for i in eachindex(x)
      @inbounds x[i] = 1.0/(1.0 + exp(-x[i]))
   end
   x
end


function bernoulli(x::M) where{M <:AbstractArray{Float64}}
   ret = rand(size(x))
   ret .= float.(ret .< x)
   ret
end

function bernoulli!(x::M) where{M <:AbstractArray{Float64}}
   for i in eachindex(x)
      @inbounds x[i] = float(rand() < x[i])
   end
   x
end

function binomial2!(x::M) where{M <:AbstractArray{Float64}}
   for i in eachindex(x)
      @inbounds x[i] = float(rand() < x[i]) + float(rand() < x[i])
   end
   x
end


"""
    splitdata(x, ratio)
Splits the data set `x` randomly in two data sets `x1` and `x2`, such that
the ratio `n2`/`n1` of the numbers of lines/samples in `x1` and `x2` is
approximately equal to the given `ratio`.
"""
function splitdata(x::Matrix{Float64}, ratio::Float64)
   nsamples = size(x,1)
   testidxs = randperm(nsamples)[1:(round(Int, nsamples*ratio))]
   xtest = x[testidxs,:]
   xtraining = x[setdiff(1:nsamples, testidxs),:]
   xtraining, xtest
end


"""
    joindbms(dbms)
    joindbms(dbms, visibleindexes)
Joins the DBMs given by the vector `dbms` by joining each layer of RBMs.

If the vector `visibleindexes` is specified, it is supposed to contain in the
i'th entry an indexing vector that determines the positions in the combined
DBM for the visible nodes of the i'th of the `dbms`.
By default the indexes of the visible nodes are assumed to be consecutive.
"""
function joindbms(dbms::Vector{BasicDBM}, visibleindexes = [])
   jointdbm = BasicDBM(length(dbms[1]))
   jointdbm[1] = joinrbms([dbms[i][1] for i in eachindex(dbms)],
         visibleindexes)
   for j = 2:length(dbms[1])
      jointdbm[j] = joinrbms([dbms[i][j] for i in eachindex(dbms)])
   end
   jointdbm
end


"""
    joinrbms(rbms)
    joinrbms(rbms, visibleindexes)
Joins the given vector of `rbms` of the same type to form one RBM of this type
and returns the joined RBM.
"""
function joinrbms{T<:AbstractRBM}(rbm1::T, rbm2::T)
   joinrbms(T[rbm1, rbm2])
end

function joinrbms(rbms::Vector{BernoulliRBM}, visibleindexes = [])
   jointvisiblebias = joinvecs([rbm.visbias for rbm in rbms], visibleindexes)
   jointhiddenbias = vcat(map(rbm -> rbm.hidbias, rbms)...)
   BernoulliRBM(joinweights(rbms, visibleindexes),
         jointvisiblebias, jointhiddenbias)
end

function joinrbms(rbms::Vector{GaussianBernoulliRBM}, visibleindexes = [])
   jointvisiblebias = joinvecs([rbm.visbias for rbm in rbms], visibleindexes)
   jointhiddenbias = vcat(map(rbm -> rbm.hidbias, rbms)...)
   jointsd = joinvecs([rbm.sd for rbm in rbms], visibleindexes)
   GaussianBernoulliRBM(joinweights(rbms, visibleindexes),
         jointvisiblebias, jointhiddenbias, jointsd)
end


"""
    joinvecs(vecs, indexes)
Combines the Float-vectors in `vecs` into one vector. The `indexes`` vector must
contain in the i'th entry the indexes that the elements of the i'th vector in
`vecs` are supposed to have in the resulting combined vector.
"""
function joinvecs(vecs::Vector{Vector{Float64}}, indexes = [])
   if isempty(indexes)
      jointvec = vcat(vecs...)
   else
      jointlength = mapreduce(v -> length(v), +, 0, vecs)
      jointvec = Vector{Float64}(jointlength)
      for i in eachindex(vecs)
         jointvec[indexes[i]] = vecs[i]
      end
   end
   jointvec
end

"""
    joinweights(rbms)
    joinweights(rbms, visibleindexes)
Combines the weight matrices of the RBMs in the vector `rbms` into one weight
matrix and returns it.

If the vector `visibleindexes` is specified, it is supposed to contain in the
i'th entry an indexing vector that determines the positions in the combined
weight matrix for the visible nodes of the i'th of the `rbms`.
By default the indexes of the visible nodes are assumed to be consecutive.
"""
function joinweights{T<:AbstractRBM}(rbms::Vector{T}, visibleindexes = [])
   jointnhidden = mapreduce(rbm -> length(rbm.hidbias), +, 0, rbms)
   jointnvisible = mapreduce(rbm -> length(rbm.visbias), +, 0, rbms)
   jointweights = zeros(jointnvisible, jointnhidden)
   offset = 0

   # if visibleindexes are not provided, construct them
   if isempty(visibleindexes)
      visibleindexes = Array{UnitRange}(length(rbms))
      for i in eachindex(rbms)
         nvisible = length(rbms[i].visbias)
         visibleindexes[i] = offset + (1:nvisible)
         offset += nvisible
      end
   elseif length(visibleindexes) != length(rbms)
      error("There must be as many indexing vectors as RBMs.")
   end

   offset = 0
   for i = eachindex(rbms)
      nhidden = length(rbms[i].hidbias)
      jointweights[visibleindexes[i], offset + (1:nhidden)] = rbms[i].weights
      offset += nhidden
   end

   jointweights
end


include("evaluating.jl")
include("monitoring.jl")

include("BMPlots.jl")

end # of module BoltzmannMachines
