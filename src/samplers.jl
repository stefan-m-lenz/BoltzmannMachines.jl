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

function sample!(particles::Particles, sampler::NoSampler, rbm::AbstractRBM,
      upfactor::Float64 = 1.0, downfactor::Float64 = 1.0)
   # do nothing
end

function sample!(vmodel::M, hmodel::M, sampler::NoSampler, rbm::AbstractRBM,
      upfactor::Float64 = 1.0, downfactor::Float64 = 1.0
      ) where {M <:AbstractArray{Float64}}
   # do nothing
end

function sample!(vmodel::M, hmodel::M, sampler::AbstractSampler, rbm::AbstractRBM,
      upfactor::Float64 = 1.0, downfactor::Float64 = 1.0
      ) where {M <:AbstractArray{Float64}}

   particles = Particles(2)
   particles[1] = vmodel
   particles[2] = hmodel
   sample!(particles, sampler, rbm, upfactor, downfactor)
   nothing
end

function sample!(vmodel::M, hmodel::M, sampler::GibbsSampler, rbm::AbstractRBM,
      upfactor::Float64 = 1.0, downfactor::Float64 = 1.0
      ) where {M <:AbstractArray{Float64}}

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

function sample!(particles::Particles, sampler::TemperatureDrivenSampler, rbm::AbstractRBM,
      upfactor::Float64 = 1.0, downfactor::Float64 = 1.0)
   # TODO respect upfactor and downfactor
   gibbssample_gamma!(particles, rbm, sampler.nsteps;
         autocorcoeff = sampler.autocorcoeff, betasd = sampler.betasd)
end