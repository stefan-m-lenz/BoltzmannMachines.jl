abstract type AbstractOptimization{R<:AbstractRBM}
end

abstract type
      AbstractLoglikelihoodGradientStep{R<:AbstractRBM} <: AbstractOptimization{R}
end

struct NoOptimization <: AbstractOptimization{AbstractRBM}
end

mutable struct LoglikelihoodGradientStep{R<:AbstractRBM} <: AbstractLoglikelihoodGradientStep{R}
   gradient::R
   negupdate::Matrix{Float64}
   learningrate::Float64
   sdlearningrate::Float64
end

function LoglikelihoodGradientStep(;
      learningrate::Float64 = 0.0, sdlearningrate::Float64 = 0.0)

   LoglikelihoodGradientStep(NoRBM(), Matrix{Float64}(0,0),
         learningrate, sdlearningrate)
end

function LoglikelihoodGradientStep(rbm::R;
      learningrate::Float64 = 0.0, sdlearningrate::Float64 = 0.0) where {R<:AbstractRBM}

   LoglikelihoodGradientStep{R}(deepcopy(rbm), Matrix{Float64}(size(rbm.weights)),
         learningrate, sdlearningrate)
end


struct BeamAdversarialGradientStep{R<: AbstractRBM} <: AbstractOptimization{R}
   gradient::R
   negupdate::Matrix{Float64}
   learningrate::Float64
   sdlearningrate::Float64
end


struct CombinedGradientStep{R<: AbstractRBM,
         G1 <: AbstractOptimization{R},
         G2 <: AbstractOptimization{R}} <: AbstractOptimization{R}

   part1::G1
   part2::G2
   gradient::R
   weight1::Float64
end


function beamoptimization(learningrate::Float64 = 0.05,
      sdlearningrate::Float64 = 0.0,
      adversarialweight::Float64 = 0.1)

   llstep = LoglikelihoodGradientStep(
         learningrate = learningrate, sdlearningrate = sdlearningrate)
   advstep = BeamAdversarialGradientStep(
         Matrix{Float64}(0,0), learningrate, sdlearningrate)

   CombinedGradientStep(advstep, llstep, NoRBM(), adversarialweight)
end


function initialized(gradientstep::AbstractOptimization, rbm::AbstractRBM)
   # do nothing
end

function initialized(gradientstep::LoglikelihoodGradientStep{R1}, rbm::R2
      ) where {R1 <: AbstractRBM, R2 <: AbstractRBM}

   LoglikelihoodGradientStep{R}(deepcopy(rbm), Matrix{Float64}(size(rbm.weights),
         gradientstep.learningrate, gradientstep.sdlearningrate))
end

function initialized(gradientstep::CombinedGradientStep, rbm::R) where {R <: AbstractRBM}
   CombinedGradientStep{R}(initialized(gradientstep.part1),
         initialized(gradientstep.part2),
         deepcopy(rbm),
         gradientstep.weight1)
end

function initialized(gradientstep::BeamAdversarialGradientStep, rbm::R
      ) where {R <: AbstractRBM}

   BeamAdversarialGradientStep{R}(deepcopy(rbm), Matrix{Float64}(size(rbm.weights)),
         gradientstep.learningrate, gradientstep.sdlearningrate)
end


"""
    computegradient!(gradientstep, v, vmodel, h, hmodel, rbm)
Computes the gradient of the RBM `rbm` given the
the hidden activation `h` induced by the sample `v`
and the vectors `vmodel` and `hmodel` generated by sampling from the model.

!!!  note
      This function may alter all arguments except for `rbm` and `hmodel`.
     `hmodel` must not be changed by implementations of `computegradient!`
     since the persistent chain state is stored there.
"""
function computegradient!(
      gradientstep::AbstractLoglikelihoodGradientStep{R},
      v::M, vmodel::M, h::M, hmodel::M, rbm::R
      ) where {R <: AbstractRBM, M <: AbstractArray{Float64, 2}}

   At_mul_B!(gradientstep.gradient.weights, v, h)
   At_mul_B!(gradientstep.negupdate, vmodel, hmodel)
   gradientstep.gradient.weights .-= gradientstep.negupdate

   gradientstep.gradient.visbias .= vec(mean(v, 1))
   gradientstep.gradient.visbias .-= vec(mean(vmodel, 1))

   gradientstep.gradient.hidbias .= vec(mean(h, 1))
   gradientstep.gradient.hidbias .-= vec(mean(hmodel, 1))
   gradient
end

function computegradient!(
      gradientstep::LoglikelihoodGradientStep{GaussianBernoulliRBM},
      v::M, vmodel::M, h::M, hmodel::M, gbrbm::GaussianBernoulliRBM
      ) where {M<: AbstractArray{Float64, 2}}

   # See bottom of page 15 in [Krizhevsky, 2009].

   if gradientstep.sdlearningrate > 0.0
      gradientstep.gradient.sd .=
            sdupdateterm(gbrbm, v, h) - sdupdateterm(gbrbm, vmodel, hmodel)
   end

   v ./= gbrbm.sd'
   vmodel ./= gbrbm.sd'

   At_mul_B!(gradientstep.gradient.weights, v, h)
   At_mul_B!(gradientstep.negupdate, vmodel, hmodel)
   gradientstep.gradient.weights .-= gradientstep.negupdate

   gradientstep.gradient.hidbias .= vec(mean(h, 1) - mean(hmodel, 1))
   gradientstep.gradient.visbias .= vec(mean(v, 1) - mean(vmodel, 1)) ./ gbrbm.sd

   gradient
end

function computegradient!(
      gradientstep::AbstractLoglikelihoodGradientStep{GaussianBernoulliRBM2},
      v::M, vmodel::M, h::M, hmodel::M, gbrbm::GaussianBernoulliRBM2
      ) where {M<: AbstractArray{Float64, 2}}

   # See Cho,
   # "Improved learning of Gaussian-Bernoulli restricted Boltzmann machines"
   sdsq = gbrbm.sd .^ 2

   if gradientstep.sdlearningrate > 0.0
      sdgrads = vmodel .* (hmodel * gbrbm.weights')
      sdgrads .-= v .* (h * gbrbm.weights')
      sdgrads .*= 2.0
      sdgrads .+= (v .- gbrbm.visbias') .^ 2
      sdgrads .-= (vmodel .- gbrbm.visbias') .^ 2
      gradientstep.gradient.sd .= vec(mean(sdgrads, 1))
      gradientstep.gradient.sd ./= sdsq
      gradientstep.gradient.sd ./= gbrbm.sd
   end

   v ./= sdsq'
   vmodel ./= sdsq'

   At_mul_B!(gradientstep.gradient.weights, v, h)
   At_mul_B!(gradientstep.negupdate, vmodel, hmodel)
   gradientstep.gradient.weights .-= gradientstep.negupdate

   gradientstep.gradient.hidbias .= vec(mean(h, 1) - mean(hmodel, 1))
   gradientstep.gradient.visbias .= vec(mean(v, 1) - mean(vmodel, 1))

   gradient
end

function computegradient!(
      gradientstep::BeamAdversarialGradientStep{R},
      v::M, vmodel::M, h::M, hmodel::M, rbm::AbstractRBM
      ) where {M<: AbstractArray{Float64, 2}, R <:AbstractRBM}

   critic = nearestneighbourcritic(h, hmodel)

   nvisible = nvisiblenodes(rbm)
   nhidden = nhiddennodes(rbm)

   for i = 1:nvisible
      for j = 1:nhidden
         gradientstep.gradient.weights[i, j] =
               cov(critic, vmodel[:, i] .* hmodel[:, j])
      end
   end

   # TODO check if standard deviation needed
   for i = 1:nvisible
      gradientstep.gradient.visbias[i] = cov(critic, vmodel[:, i])
   end

   for j = 1:nhidden
      gradientstep.gradient.hidbias[j] = cov(critic, hmodel[:, j])
   end

   gradient
end

function computegradient!(
      gradientstep::BeamAdversarialGradientStep{GaussianBernoulliRBM2},
      v::M, vmodel::M, h::M, hmodel::M, rbm::GaussianBernoulliRBM2
      ) where {M<: AbstractArray{Float64, 2}}

   invoke(computegradient!,
         Tuple{BeamAdversarialGradientStep{GaussianBernoulliRBM2}, M, M, M, M, AbstractRBM},
         v, vmodel, h, hmodel, rbm)

   sdsq = rbm.sd .^ 2
   gradientstep.gradient.weights ./= sdsq

   gradient
end

function computegradient!(gradientstep::CombinedGradientStep{R},
      v::M, vmodel::M, h::M, hmodel::M, rbm::AbstractRBM
      ) where {M<: AbstractArray{Float64, 2}, R <:AbstractRBM}

   computegradient!(gradientstep.part1, copy(v), copy(vmodel), copy(h), hmodel, rbm)
   computegradient!(gradientstep.part2, v, vmodel, h, hmodel, rbm)

   grad1 = gradientstep.part1.gradient
   grad2 = gradientstep.part2.gradient

   gradientstep.gradient.weights .=
         grad1.weights * gradientstep.weight1 +
         grad2.weights * (1 - gradientstep.weight1)

   gradientstep.gradient.visbias .=
         grad1.visbias * gradientstep.weight1 +
         grad2.visbias * (1 - gradientstep.weight1)

   gradientstep.gradient.hidbias .=
         grad1.hidbias * gradientstep.weight1 +
         grad2.hidbias * (1 - gradientstep.weight1)

   gradient
end

function computegradient!(gradientstep::CombinedGradientStep{R},
      v::M, vmodel::M, h::M, hmodel::M, rbm::R
      ) where {M<: AbstractArray{Float64, 2},
            R <:Union{GaussianBernoulliRBM, GaussianBernoulliRBM2}}

   invoke(computegradient!,
         Tuple{CombinedGradientStep{R}, M, M, M, M, AbstractRBM},
         gradientstep, v, vmodel, h, hmodel, rbm)

   gradientstep.gradient.sd .=
         gradientstep.part1.gradient.sd * gradientstep.weight1 +
         gradientstep.part1.gradient.sd * (1 - gradientstep.weight1)

   gradient
end


"""
    updateparameters!(rbm, gradientstep)
Updates the RBM by walking a step in the direction of the gradient that
has been computed by calling `computegradient!` on `gradientstep`.
"""
function updateparameters!(rbm::AbstractRBM,
      gradientstep::AbstractOptimization{R}) where {R <: AbstractRBM}

   updateweightsandbiases!(rbm, gradientstep)
end

function updateparameters!(rbm::Binomial2BernoulliRBM,
      gradientstep::AbstractOptimization{Binomial2BernoulliRBM})

   # To train a Binomial2BernoulliRBM exactly like
   # training a BernoulliRBM where each two nodes share the weights,
   # use half the learning rate in the visible nodes.
   learningratehidden = gradientstep.learningrate
   learningrate = gradientstep.learningrate / 2.0

   gradientstep.gradient.weights .*= learningrate
   gradientstep.gradient.visbias .*= learningrate
   gradientstep.gradient.hidbias .*= learningratehidden
   rbm.weights .+= gradientstep.gradient.weights
   rbm.visbias .+= gradientstep.gradient.visbias
   rbm.hidbias .+= gradientstep.gradient.hidbias
   rbm
end

function updateparameters!(rbm::R, gradientstep::AbstractOptimization{R}
      ) where {R <: Union{GaussianBernoulliRBM, GaussianBernoulliRBM2}}

   updateweightsandbiases!(rbm, gradientstep)

   if gradientstep.sdlearningrate > 0.0
      gradientstep.gradient.sd .*= gradientstep.sdlearningrate
      rbm.sd .+= gradientstep.gradient.sd
   end
   rbm
end


function updateweightsandbiases!(rbm::R,
      gradientstep::AbstractOptimization{R}) where {R <: AbstractRBM}

   gradientstep.gradient.weights .*= gradientstep.learningrate
   gradientstep.gradient.visbias .*= gradientstep.learningrate
   gradientstep.gradient.hidbias .*= gradientstep.learningrate
   rbm.weights .+= gradientstep.gradient.weights
   rbm.visbias .+= gradientstep.gradient.visbias
   rbm.hidbias .+= gradientstep.gradient.hidbias
   rbm
end