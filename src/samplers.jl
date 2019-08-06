abstract type AbstractSampler
end

struct GibbsSampler <: AbstractSampler
   nsteps::Int
end

struct NoSampler <: AbstractSampler
end

function sample!(vmodel::M1, hmodel::M2, sampler::NoSampler, rbm::AbstractRBM,
      upfactor::Float64 = 1.0, downfactor::Float64 = 1.0
      ) where {M1 <: AbstractArray{Float64}, M2 <: AbstractArray{Float64}}
   # do nothing
end

function sample!(vmodel::M1, hmodel::M2, sampler::GibbsSampler, rbm::AbstractRBM,
      upfactor::Float64 = 1.0, downfactor::Float64 = 1.0
      ) where {M1 <: AbstractArray{Float64}, M2 <: AbstractArray{Float64}}

   for i = 1:sampler.nsteps
      samplevisible!(vmodel, rbm, hmodel, downfactor)
      samplehidden!(hmodel, rbm, vmodel, upfactor)
   end
   nothing
end

function sample!(particles::Particles, sampler::GibbsSampler, rbm::AbstractRBM,
      upfactor::Float64 = 1.0, downfactor::Float64 = 1.0)

   gibbssample!(particles, rbm, sampler.nsteps)
end
