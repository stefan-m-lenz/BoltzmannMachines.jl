"""
The `AbstractOptimizer` interface allows to specify optimization procedures.
It consists of three methods:
* `initialized(optimizer, bm)`: May be used for creating an optimizer that is
   specifically initialized for the Boltzmann machine `bm`.
   In particular it may be used to allocate reusable space for the gradient.
   The default implementation simply returns the unmodified `optimizer`.
* `computegradient!(optimizer, v, vmodel, h, hmodel, rbm)` or
  `computegradient!(optimizer, meanfieldparticles, gibbsparticles, dbm)`
   needs to be implemented for computing the gradient given the samples
   from the positive and negative phase.
* `updateparameters!(bm, optimizer)` needs to be specified for taking the
   gradient step. The default implementation for RBMs expects the fields
   `learningrate` and `gradient` and adds `learningrate * gradient` to the
   given RBM.
"""
abstract type AbstractOptimizer{R<:AbstractBM}
end


abstract type
      AbstractLoglikelihoodOptimizer{R<:AbstractRBM} <: AbstractOptimizer{R}
end


struct NoOptimizer <: AbstractOptimizer{AbstractBM}
end


"""
Implements the `AbstractOptimizer` interface for optimizing the loglikelihood
with stochastic gradient descent.
"""
mutable struct LoglikelihoodOptimizer{R<:AbstractRBM} <: AbstractLoglikelihoodOptimizer{R}
   gradient::R
   negupdate::Matrix{Float64}
   learningrate::Float64
   sdlearningrate::Float64
end


function converttodbmoptimizer(optimizer::NoOptimizer, dbm::MultimodalDBM)
   optimizer
end

function converttodbmoptimizer(optimizer::AbstractOptimizer{<:AbstractRBM},
      dbm::MultimodalDBM)
   StackedOptimizer(map(i -> deepcopy(optimizer), eachindex(dbm)))
end

function converttodbmoptimizer(optimizer::AbstractOptimizer{MultimodalDBM},
      dbm::MultimodalDBM)
   optimizer
end


function loglikelihoodoptimizer(;
      learningrate::Float64 = 0.0, sdlearningrate::Float64 = 0.0)

   LoglikelihoodOptimizer(NoRBM(), Matrix{Float64}(undef, 0,0),
         learningrate, sdlearningrate)
end

function loglikelihoodoptimizer(rbm::R;
      learningrate::Float64 = 0.0, sdlearningrate::Float64 = 0.0) where {R<:AbstractRBM}

   LoglikelihoodOptimizer{R}(deepcopy(rbm),
         Matrix{Float64}(undef, size(rbm.weights)),
         learningrate, sdlearningrate)
end

function loglikelihoodoptimizer(prbm::PartitionedRBM;
      learningrate::Float64 = 0.0, sdlearningrate::Float64 = 0.0)

   PartitionedOptimizer(map(
         rbm -> loglikelihoodoptimizer(rbm;
               learningrate = learningrate, sdlearningrate = sdlearningrate),
         prbm.rbms))
end


struct PartitionedOptimizer <: AbstractOptimizer{PartitionedRBM}
   optimizers::Vector{AbstractOptimizer}
end


mutable struct BeamAdversarialOptimizer{R<: AbstractRBM} <: AbstractOptimizer{R}
   gradient::R
   negupdate::Matrix{Float64}
   critic::Vector{Float64}
   learningrate::Float64
   sdlearningrate::Float64
   knearest::Int
end


"""
    StackedOptimizer(optimizers)
Can be used for optimizing a stack of RBMs / a DBM by using the given the vector
of `optimizers` (one for each RBM).
For more information about the concept of optimizers, see `AbstractOptimizer`.
"""
struct StackedOptimizer <: AbstractOptimizer{MultimodalDBM}
   optimizers::Vector{AbstractOptimizer{<:AbstractRBM}}
end


struct CombinedOptimizer{R<: AbstractRBM,
         G1 <: AbstractOptimizer{R},
         G2 <: AbstractOptimizer{R}} <: AbstractOptimizer{R}

   part1::G1
   part2::G2
   gradient::R
   weight1::Float64
   weight2::Float64
   learningrate::Float64
   sdlearningrate::Float64
end


# TODO document
function beamoptimizer(;learningrate::Float64 = 0.05,
      sdlearningrate::Float64 = 0.0,
      adversarialweight::Float64 = 0.1,
      knearest::Int = 5)

   llstep = loglikelihoodoptimizer(
         learningrate = learningrate, sdlearningrate = sdlearningrate)
   advstep = BeamAdversarialOptimizer(
         NoRBM(), Matrix{Float64}(undef, 0, 0), Vector{Float64}(),
         learningrate, sdlearningrate, knearest)

   CombinedOptimizer(advstep, llstep, NoRBM(),
         adversarialweight, 1.0 - adversarialweight,
         learningrate, sdlearningrate)
end


"""
    initialized(optimizer, rbm)
Returns an `AbstractOptimizer` similar to the given `optimizer`
that can be used to optimize the `AbstractRBM` `rbm`.
"""
function initialized(optimizer::AbstractOptimizer, rbm::AbstractRBM)
   optimizer
end

function initialized(optimizer::LoglikelihoodOptimizer, rbm::R
      ) where {R <: AbstractRBM}

   LoglikelihoodOptimizer(deepcopy(rbm),
         Matrix{Float64}(undef, size(rbm.weights)),
         optimizer.learningrate, optimizer.sdlearningrate)
end

function initialized(optimizer::CombinedOptimizer, rbm::R
      ) where {R <: AbstractRBM}

   CombinedOptimizer(initialized(optimizer.part1, rbm),
         initialized(optimizer.part2, rbm),
         deepcopy(rbm),
         optimizer.weight1, optimizer.weight2,
         optimizer.learningrate, optimizer.sdlearningrate)
end

function initialized(optimizer::BeamAdversarialOptimizer{R1}, rbm::R2
      ) where {R1 <: AbstractRBM, R2 <: AbstractRBM}

   BeamAdversarialOptimizer{R2}(deepcopy(rbm),
         Matrix{Float64}(undef, size(rbm.weights)),
         Vector{Float64}(),
         optimizer.learningrate, optimizer.sdlearningrate,
         optimizer.knearest)
end

function initialized(optimizer::PartitionedOptimizer, prbm::PartitionedRBM)
   PartitionedOptimizer(map(
         i -> initialized(optimizer.optimizers[i], prbm.rbms[i]),
         eachindex(prbm)))
end

function initialized(optimizer::AbstractOptimizer{R}, dbm::MultimodalDBM
      ) where {R <: AbstractRBM}
   # transform optimizer for RBM into stacked optimizer for DBM
   StackedOptimizer(map(rbm -> initialized(optimizer, rbm), dbm))
end

function initialized(stackedoptimizer::StackedOptimizer, dbm::MultimodalDBM)
   StackedOptimizer(map(
         i -> initialized(stackedoptimizer.optimizers[i], dbm[i]),
         eachindex(dbm)))
end


"""
    computegradient!(optimizer, v, vmodel, h, hmodel, rbm)
Computes the gradient of the RBM `rbm` given the
the hidden activation `h` induced by the sample `v`
and the vectors `vmodel` and `hmodel` generated by sampling from the model.
The result is stored in the `optimizer` in such a way that it can be applied
by a call to `updateparameters!`. There is no return value.

For RBMs (excluding PartitionedRBMs), this means saving the gradient
in a RBM of the same type in the field `optimizer.gradient`.
"""
function computegradient!(
      optimizer::AbstractLoglikelihoodOptimizer{R},
      v::M1, vmodel::M1, h::M2, hmodel::M2, rbm::R
      ) where {M1 <: AbstractArray{Float64, 2}, M2 <: AbstractArray{Float64, 2},
            R <: AbstractRBM}

   computegradientsweightsandbiases!(optimizer, v, vmodel, h, hmodel, rbm)
end

function computegradient!(
      optimizer::LoglikelihoodOptimizer{GaussianBernoulliRBM},
      v::M1, vmodel::M1, h::M2, hmodel::M2, gbrbm::GaussianBernoulliRBM
      ) where {M1 <: AbstractArray{Float64, 2}, M2 <: AbstractArray{Float64, 2}}

   # See bottom of page 15 in [Krizhevsky, 2009].

   if optimizer.sdlearningrate > 0.0
      optimizer.gradient.sd .=
            sdupdateterm(gbrbm, v, h) - sdupdateterm(gbrbm, vmodel, hmodel)
   end

   v = v ./ gbrbm.sd'
   vmodel = vmodel ./ gbrbm.sd'

   computegradientsweightsandbiases!(optimizer, v, vmodel, h, hmodel, gbrbm)
   optimizer.gradient.visbias ./= gbrbm.sd

   nothing
end

function computegradient!(
      optimizer::AbstractLoglikelihoodOptimizer{GaussianBernoulliRBM2},
      v::M1, vmodel::M1, h::M2, hmodel::M2, gbrbm::GaussianBernoulliRBM2
      ) where {M1 <: AbstractArray{Float64, 2}, M2 <: AbstractArray{Float64, 2}}

   # See Cho,
   # "Improved learning of Gaussian-Bernoulli restricted Boltzmann machines"
   sdsq = gbrbm.sd .^ 2

   if optimizer.sdlearningrate > 0.0
      sdgrads = vmodel .* (hmodel * gbrbm.weights')
      sdgrads .-= v .* (h * gbrbm.weights')
      sdgrads .*= 2.0
      sdgrads .+= (v .- gbrbm.visbias') .^ 2
      sdgrads .-= (vmodel .- gbrbm.visbias') .^ 2
      optimizer.gradient.sd .= vec(mean(sdgrads, dims = 1))
      optimizer.gradient.sd ./= sdsq
      optimizer.gradient.sd ./= gbrbm.sd
   end

   v = v ./ sdsq'
   vmodel = vmodel ./ sdsq'

   computegradientsweightsandbiases!(optimizer, v, vmodel, h, hmodel, gbrbm)
end

function computegradientsweightsandbiases!(
      optimizer::AbstractLoglikelihoodOptimizer,
      v::M1, vmodel::M1, h::M2, hmodel::M2, rbm::R
      ) where {M1 <: AbstractArray{Float64, 2}, M2 <: AbstractArray{Float64, 2},
            R <: AbstractRBM}

   npossamples = size(v, 1)
   nnegsamples = size(vmodel, 1)

   mul!(optimizer.gradient.weights, transpose(v), h)
   if npossamples != 1
      optimizer.gradient.weights ./= npossamples
   end
   mul!(optimizer.negupdate, transpose(vmodel), hmodel)
   if nnegsamples != 1
      optimizer.negupdate ./= nnegsamples
   end
   optimizer.gradient.weights .-= optimizer.negupdate

   optimizer.gradient.hidbias .= vec(mean(h, dims = 1) - mean(hmodel, dims = 1))
   optimizer.gradient.visbias .= vec(mean(v, dims = 1) - mean(vmodel, dims = 1))
   nothing
end

function computegradient!(
      optimizer::BeamAdversarialOptimizer{R},
      v::M1, vmodel::M1, h::M2, hmodel::M2, rbm::AbstractRBM
      ) where {M1 <: AbstractArray{Float64, 2}, M2 <: AbstractArray{Float64, 2},
            R <:AbstractRBM}

   optimizer.critic = nearestneighbourcritic(h, hmodel, optimizer.knearest)

   nvisible = nvisiblenodes(rbm)
   nhidden = nhiddennodes(rbm)

   for i = 1:nvisible
      for j = 1:nhidden
         optimizer.gradient.weights[i, j] =
               cov(optimizer.critic, vmodel[:, i] .* hmodel[:, j])
      end
   end

   for i = 1:nvisible
      optimizer.gradient.visbias[i] = cov(optimizer.critic, vmodel[:, i])
   end

   for j = 1:nhidden
      optimizer.gradient.hidbias[j] = cov(optimizer.critic, hmodel[:, j])
   end

   nothing
end

function computegradient!(
      optimizer::BeamAdversarialOptimizer{GaussianBernoulliRBM2},
      v::M1, vmodel::M1, h::M2, hmodel::M2, rbm::GaussianBernoulliRBM2
      ) where {M1 <: AbstractArray{Float64, 2}, M2 <: AbstractArray{Float64, 2}}

   invoke(computegradient!,
         Tuple{BeamAdversarialOptimizer{GaussianBernoulliRBM2}, M1, M1, M2, M2, AbstractRBM},
         optimizer, v, vmodel, h, hmodel, rbm)

   sdsq = rbm.sd .^ 2

   weights = optimizer.gradient.weights
   visbias = optimizer.gradient.visbias
   nvisible = size(weights, 1)

   weights ./= sdsq
   visbias ./= sdsq

   if optimizer.sdlearningrate > 0
      for i = 1:nvisible
         optimizer.gradient.sd[i] =
               cov(optimizer.critic, (vmodel[:, i] .- visbias[i]) .^ 2 -
                     2.0 .* vmodel[:, i] .* (hmodel * weights[i, :]))
      end
   end

   optimizer.gradient.sd ./= sdsq .* rbm.sd

   nothing
end

function computegradient!(optimizer::CombinedOptimizer{R},
      v::M1, vmodel::M1, h::M2, hmodel::M2, rbm::AbstractRBM
      ) where {M1 <: AbstractArray{Float64, 2}, M2 <: AbstractArray{Float64, 2},
            R <:AbstractRBM}

   computegradient!(optimizer.part1, v, vmodel, h, hmodel, rbm)
   computegradient!(optimizer.part2, v, vmodel, h, hmodel, rbm)

   grad1 = optimizer.part1.gradient
   grad2 = optimizer.part2.gradient

   optimizer.gradient.weights .=
         grad1.weights .* optimizer.weight1 .+
         grad2.weights .* optimizer.weight2

   optimizer.gradient.visbias .=
         grad1.visbias .* optimizer.weight1 .+
         grad2.visbias .* optimizer.weight2

   optimizer.gradient.hidbias .=
         grad1.hidbias .* optimizer.weight1 .+
         grad2.hidbias .* optimizer.weight2

   nothing
end

function computegradient!(optimizer::CombinedOptimizer{R},
      v::M1, vmodel::M1, h::M2, hmodel::M2, rbm::R
      ) where {M1 <: AbstractArray{Float64, 2}, M2 <: AbstractArray{Float64, 2},
            R <:Union{GaussianBernoulliRBM, GaussianBernoulliRBM2}}

   invoke(computegradient!,
         Tuple{CombinedOptimizer{R}, M1, M1, M2, M2, AbstractRBM},
         optimizer, v, vmodel, h, hmodel, rbm)

   if optimizer.sdlearningrate > 0.0
      optimizer.gradient.sd .=
            optimizer.part1.gradient.sd .* optimizer.weight1 .+
            optimizer.part2.gradient.sd .* optimizer.weight2
   end

   nothing
end

function computegradient!(optimizer::PartitionedOptimizer,
      v::M1, vmodel::M1, h::M2, hmodel::M2, prbm::PartitionedRBM
      ) where {M1 <: AbstractArray{Float64, 2}, M2 <: AbstractArray{Float64, 2}}

   for i in eachindex(prbm.rbms)
      computegradient!(optimizer.optimizers[i],
            view(v, :, prbm.visranges[i]),
            view(vmodel, :, prbm.visranges[i]),
            view(h, :, prbm.hidranges[i]),
            view(hmodel, :, prbm.hidranges[i]),
            prbm.rbms[i])
   end
   nothing
end

function computegradient!(stackedoptimizer::StackedOptimizer,
      meanfieldparticles::Particles, gibbsparticles::Particles,
      dbm::MultimodalDBM)

   for i in eachindex(dbm)
      computegradient!(stackedoptimizer.optimizers[i],
            meanfieldparticles[i], gibbsparticles[i],
            meanfieldparticles[i + 1], gibbsparticles[i + 1],
            dbm[i])
   end
   nothing
end


"""
    updateparameters!(rbm, optimizer)
Updates the RBM `rbm` by walking a step in the direction of the gradient that
has been computed by calling `computegradient!` on `optimizer`.
"""
function updateparameters!(rbm::AbstractRBM,
      optimizer::AbstractOptimizer{R}) where {R <: AbstractRBM}

   updateweightsandbiases!(rbm, optimizer)
end

function updateparameters!(rbm::Binomial2BernoulliRBM,
      optimizer::AbstractOptimizer{Binomial2BernoulliRBM})

   # To train a Binomial2BernoulliRBM exactly like
   # training a BernoulliRBM where each two nodes share the weights,
   # use half the learning rate in the visible nodes.
   learningratehidden = optimizer.learningrate
   learningrate = optimizer.learningrate / 2.0

   rbm.weights .+= learningrate .* optimizer.gradient.weights
   rbm.visbias .+= learningrate .* optimizer.gradient.visbias
   rbm.hidbias .+= learningratehidden .* optimizer.gradient.hidbias
   rbm
end

function updateparameters!(rbm::R, optimizer::AbstractOptimizer{R}
      ) where {R <: Union{GaussianBernoulliRBM, GaussianBernoulliRBM2}}

   updateweightsandbiases!(rbm, optimizer)

   if optimizer.sdlearningrate > 0.0
      rbm.sd .+= optimizer.sdlearningrate * optimizer.gradient.sd
   end
   rbm
end

function updateparameters!(prbm::PartitionedRBM, optimizer::PartitionedOptimizer)
   for i in eachindex(prbm.rbms)
      updateparameters!(prbm.rbms[i], optimizer.optimizers[i])
   end
   prbm
end

function updateparameters!(dbm::MultimodalDBM, stackedoptimizer::StackedOptimizer)
   for i in eachindex(dbm)
      updateparameters!(dbm[i], stackedoptimizer.optimizers[i])
   end
   dbm
end


function updateweightsandbiases!(rbm::R,
      optimizer::AbstractOptimizer{R}) where {R <: AbstractRBM}

   rbm.weights .+= optimizer.learningrate .* optimizer.gradient.weights
   rbm.visbias .+= optimizer.learningrate .* optimizer.gradient.visbias
   rbm.hidbias .+= optimizer.learningrate .* optimizer.gradient.hidbias
   rbm
end

