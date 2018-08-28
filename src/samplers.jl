abstract type AbstractSampler
end

struct GibbsSampler <: AbstractSampler
   nsteps::Int
end


struct TemperatureDrivenSampler <: AbstractSampler
   nsteps::Int
   autocorcoeff::Float64
   betasd::Float64
end

function TemperatureDrivenSampler(;
      nsteps::Int = 5, autocorcoeff::Float64 = 0.9,
      betasd::Float64 = 0.9)

   TemperatureDrivenSampler(nsteps, autocorcoeff, betasd)
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

function sample!(vv::M1, hh::M2, sampler::TemperatureDrivenSampler, rbm::AbstractRBM,
      upfactor::Float64 = 1.0, downfactor::Float64 = 1.0
      ) where {M1 <: AbstractArray{Float64}, M2 <: AbstractArray{Float64}}
   # TODO respect upfactor and downfactor
   gibbssample_gamma!(vv, hh, rbm, sampler.nsteps;
         autocorcoeff = sampler.autocorcoeff, betasd = sampler.betasd)
end