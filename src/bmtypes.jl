""" Abstract supertype for all RBMs """
abstract type AbstractRBM end

"""
Abstract super type for RBMs with binary and Bernoulli distributed hidden nodes.
"""
abstract type AbstractXBernoulliRBM <: AbstractRBM end

"""
    BernoulliRBM(weights, visbias, hidbias)
Encapsulates the parameters of an RBM with Bernoulli distributed nodes.
* `weights`: matrix of weights with size
  (number of visible nodes, number of hidden nodes)
* `visbias`: bias vector for visible nodes
* `hidbias`: bias vector for hidden nodes
"""
struct BernoulliRBM <: AbstractXBernoulliRBM
   weights::Array{Float64,2}
   visbias::Array{Float64,1}
   hidbias::Array{Float64,1}
end


"""
    GaussianBernoulliRBM(weights, visbias, hidbias, sd)
Encapsulates the parameters of an RBM with Gaussian distributed visible nodes
and Bernoulli distributed hidden nodes.
"""
struct GaussianBernoulliRBM <: AbstractXBernoulliRBM
   weights::Array{Float64,2}
   visbias::Array{Float64,1}
   hidbias::Array{Float64,1}
   sd::Array{Float64,1}
end


"""
    GaussianBernoulliRBM2(weights, visbias, hidbias, sd)
Encapsulates the parameters of an RBM with Gaussian distributed visible nodes
and Bernoulli distributed hidden nodes with the alternative energy formula
proposed by KyungHyun Cho.
"""
struct GaussianBernoulliRBM2 <: AbstractXBernoulliRBM
   weights::Array{Float64,2}
   visbias::Array{Float64,1}
   hidbias::Array{Float64,1}
   sd::Array{Float64,1}
end


"""
    BernoulliGaussianRBM(weights, visbias, hidbias)
Encapsulates the parameters of an RBM with Bernoulli distributed visible nodes
and Gaussian distributed hidden nodes.
The standard deviation of the Gaussian distribution is 1.
"""
struct BernoulliGaussianRBM <: AbstractRBM
   weights::Array{Float64,2}
   visbias::Array{Float64,1}
   hidbias::Array{Float64,1}
end


"""
    Binomial2BernoulliRBM(weights, visbias, hidbias)
Encapsulates the parameters of an RBM with 0/1/2-valued, Binomial (n=2)
distributed visible nodes, and Bernoulli distributed hidden nodes.
This model is equivalent to a BernoulliRBM in which every two visible nodes are
connected with the same weights to each hidden node.
The states (0,0) / (1,0) / (0,1) / (1,1) of the visible nodes connected with
with the same weights translate as states 0 / 1 / 1 / 2 in the
Binomial2BernoulliRBM.
"""
struct Binomial2BernoulliRBM <: AbstractXBernoulliRBM
   weights::Matrix{Float64}
   visbias::Vector{Float64}
   hidbias::Vector{Float64}
end


"""
    ranges(numbers)
Returns a vector of consecutive integer ranges, the first starting with 1.
The i'th such range spans over `numbers[i]` items.
"""
function ranges(numbers::Vector{Int})
   ranges = Vector{UnitRange{Int}}(undef, length(numbers))
   offset = 0
   for i in eachindex(numbers)
      ranges[i] = offset .+ (1:numbers[i])
      offset += numbers[i]
   end
   ranges
end


"""
    PartitionedRBM(rbms)
Encapsulates several (parallel) AbstractRBMs that form one partitioned RBM.
The nodes of the parallel RBMs are not connected between the RBMs.
"""
struct PartitionedRBM{R<:AbstractRBM} <: AbstractRBM
   rbms::Vector{R}
   visranges::Vector{UnitRange{Int}}
   hidranges::Vector{UnitRange{Int}}

   function PartitionedRBM{R}(rbms::Vector{R}) where R
      visranges = ranges([length(rbm.visbias) for rbm in rbms])
      hidranges = ranges([length(rbm.hidbias) for rbm in rbms])
      new(rbms, visranges, hidranges)
   end
end


""" Singleton-Placeholder for `AbstractRBM`s """
struct NoRBM <: AbstractRBM
end


const BasicDBM = Vector{BernoulliRBM}

"A DBM with only Bernoulli distributed nodes which may contain partitioned layers."
const PartitionedBernoulliDBM =
      Vector{<:Union{BernoulliRBM, PartitionedRBM{BernoulliRBM}}}

const MultimodalDBM = Vector{<:AbstractRBM}

const AbstractBM = Union{MultimodalDBM, AbstractRBM}
