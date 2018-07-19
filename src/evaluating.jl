"""
    aislogimpweights(rbm; ...)
Computes the logarithmised importance weights for estimating the ratio of the
partition functions of the given `rbm` to the RBM with zero weights,
but same visible and hidden bias as the `rbm`.
This function implements the Annealed Importance Sampling algorithm (AIS)
like described in section 4.1.3 of [Salakhutdinov, 2008].

# Optional keyword arguments (for all types of Boltzmann Machines):
* `ntemperatures`: Number of temperatures for annealing from the starting model
  to the target model, defaults to 100
* `temperatures`: Vector of temperatures. By default `ntemperatures` ascending
  numbers, equally spaced from 0.0 to 1.0
* `nparticles`: Number of parallel chains and calculated weights, defaults to
   100
* `burnin`: Number of steps to sample for the Gibbs transition between models
"""
function aislogimpweights(rbm::AbstractXBernoulliRBM;
      ntemperatures::Int = 100,
      temperatures::Array{Float64,1} = collect(0:(1/ntemperatures):1),
      nparticles::Int = 100,
      burnin::Int = 5)

   logimpweights = zeros(nparticles)
   mixrbm = deepcopy(rbm)

   # start with samples from model with zero weights
   hh = repmat(rbm.hidbias', nparticles)
   sigm_bernoulli!(hh)

   vv = Matrix{Float64}(nparticles, length(rbm.visbias))

   for k = 2:length(temperatures)
      logimpweights .+= unnormalizedproblogratios(rbm, hh,
            temperatures[k], temperatures[k-1])

      # Gibbs transition
      copyannealed!(mixrbm, rbm, temperatures[k])
      for burn = 1:burnin
         samplevisible!(vv, mixrbm, hh)
         samplehidden!(hh, mixrbm, vv)
      end
   end

   logimpweights
end


"""
    aislogimpweights(rbm1, rbm2; ...)
Computes the logarithmised importance weights for estimating the log-ratio
log(Z2/Z1) for the partition functions Z1 and Z2 of `rbm1` and `rbm2`, respectively.
Implements the procedure described in section 4.1.2 of [Salakhutdinov, 2008].
This requires that `rbm1` and `rbm2` are of the same type and have the same
number of visible units.
"""
function aislogimpweights(rbm1::R, rbm2::R;
      ntemperatures::Int = 100,
      temperatures::Array{Float64,1} = collect(0:(1/ntemperatures):1),
      nparticles::Int = 100,
      burnin::Int = 5,
      initburnin::Int = 30) where {R <: AbstractRBM}

   if length(rbm2.visbias) != length(rbm1.visbias)
      error("The two RBMs must have the same numer of visible units.")
   end

   logimpweights = zeros(nparticles)

   vv = BMs.sampleparticles(rbm1, nparticles, initburnin)[1]
   nhidden1 = length(rbm1.hidbias)
   nhidden2 = length(rbm2.hidbias)
   hh = Matrix{Float64}(nparticles, nhidden1 + nhidden2)

   prevmixedrbm = BMs.mixedrbm(rbm1, rbm2, temperatures[1])

   for k = 2:length(temperatures)
      curmixedrbm = BMs.mixedrbm(rbm1, rbm2, temperatures[k])

      # log(p*(x)) = -freeenergy(x)
      logimpweights .+= BMs.freeenergydiffs(prevmixedrbm, curmixedrbm, vv)

      # Gibbs transition for the (visible) nodes
      for burn = 1:burnin
         BMs.samplehidden!(hh, curmixedrbm, vv)
         BMs.samplevisible!(vv, curmixedrbm, hh)
      end

      prevmixedrbm = curmixedrbm
   end

   # account for different model size
   logimpweights += log(2) * (nhidden2 - nhidden1)

   logimpweights
end


"""
    aislogimpweights(dbm; ...)
Computes the logarithmised importance weights in the Annealed Importance Sampling
algorithm (AIS) for estimating the ratio of the partition functions of the given
DBM `dbm` to the base-rate DBM with all weights being zero and all biases equal
to the biases of the `dbm`.

Implements algorithm 4 in [Salakhutdinov+Hinton, 2012].
For DBMs with Bernoulli-distributed nodes only
(i. e. here DBMs of type `PartitionedBernoulliDBM`),
it is possible to calculate the importance weights by summing out either
the even layers (h1, h3, ...) or the odd layers (v, h2, h4, ...).
In the first case, the nodes' activations in the odd layers are used to
calculate the probability ratios, in the second case the even layer are used.
If `dbm` is of type `PartitionedBernoulliDBM`, the optional keyword argument
`sumout` can be used to choose by specifying the values `:odd` (default) or
`:even`.
In the case of `MultimodalDBM`s, it is not possible to choose and
the second case applies there.
"""
function aislogimpweights(dbm::PartitionedBernoulliDBM;
      ntemperatures::Int = 100,
      temperatures::Array{Float64,1} = collect(0:(1/ntemperatures):1),
      nparticles::Int = 100,
      burnin::Int = 5,
      sumout::Symbol = :even)

   if sumout == :odd
      # summing out even layers is done in the implementation for
      # MultimodalDBMs
      return invoke(aislogimpweights, Tuple{MultimodalDBM, }, dbm;
            ntemperatures = ntemperatures,
            temperatures = temperatures,
            nparticles = nparticles,
            burnin = burnin)
   elseif sumout != :even
      error("Invalid value for argument `sumout`.")
   end
   # in the following implementation, the even layers are summed out

   logimpweights = zeros(nparticles)
      # Todo: sample from null model, which has changed
   particles = initparticles(dbm, nparticles, biased = true)
   nlayers = length(particles)

   # for performance reasons: preallocate input and combine biases
   input1 = newparticleslike(particles)
   input2 = newparticleslike(particles)
   biases = combinedbiases(dbm)
   mixdbm = deepcopy(dbm)

   for k = 2:length(temperatures)
      # Calculate probability ratios for importance weights
      # according to activation of odd layers (v, h2, h4, ...)
      aisupdatelogimpweights!(logimpweights, input1, input2,
            temperatures[k], temperatures[k-1], dbm, biases, particles)

      # Gibbs transition
      copyannealed!(mixdbm, dbm, temperatures[k])
      gibbssample!(particles, mixdbm, burnin)
   end

   logimpweights
end

function aislogimpweights(mdbm::MultimodalDBM;
      ntemperatures::Int = 100,
      temperatures::Array{Float64,1} = collect(0:(1/ntemperatures):1),
      nparticles::Int = 100,
      burnin::Int = 5)

   if length(mdbm) == 1
      # DBM with only one layer --> RBM
      return aislogimpweights(mdbm[1];
            ntemperatures = ntemperatures,
            temperatures = temperatures,
            nparticles = nparticles,
            burnin = burnin)
   end

   logimpweights = zeros(nparticles)
      # Todo: sample from null model, which has changed
   particles = initparticles(mdbm, nparticles, biased = true)
   nlayers = length(particles)

   # for performance reasons: preallocate input and combine biases
   hiddbminput1 = newparticleslike(particles[2:end])
   hiddbminput2 = newparticleslike(particles[2:end])

   mixdbm = deepcopy(mdbm)
   hiddbm = converttopartitionedbernoullidbm(mdbm[2:end])
   hidbiases = combinedbiases(hiddbm)

   for k = 2:length(temperatures)
      # Calculate probability ratios for importance weights
      # according to activation of odd hidden layers (h1, h3, ...)
      logimpweights += unnormalizedproblogratios(mdbm[1],
            particles[2], temperatures[k], temperatures[k-1])
      aisupdatelogimpweights!(logimpweights, hiddbminput1, hiddbminput2,
            temperatures[k], temperatures[k-1], hiddbm, hidbiases, particles[2:end])

      # Gibbs transition
      copyannealed!(mixdbm, mdbm, temperatures[k])
      gibbssample!(particles, mixdbm, burnin)
   end

   logimpweights
end


"""
    aisprecision(logr, aissd, sdrange)
Returns the differences of the estimated logratio `r` to the lower
and upper bound of the range defined by the multiple `sdrange`
of the standard deviation of the ratio's estimator `aissd`.
"""
function aisprecision(logr::Float64, aissd::Float64, sdrange::Float64 = 1.0)

   t = sdrange * aissd * exp(-logr)

   if 1 - t <= 0 # prevent domainerror
      diffbottom = -Inf
   else
      diffbottom = log(1 - t)
   end

   difftop = log(1 + t)

   diffbottom, difftop
end

"""
    aisprecision(logimpweights, sdrange)
"""
function aisprecision(logimpweights::Array{Float64,1}, sdrange::Float64 = 1.0)
   aisprecision(
         logmeanexp(logimpweights),
         aisstandarddeviation(logimpweights),
         sdrange)
end


"""
Computes the standard deviation of the AIS estimator (not logarithmised)
(eq 4.10 in [Salakhutdinov+Hinton, 2012]) given the logarithmised
importance weights.
"""
function aisstandarddeviation(logimpweights::Array{Float64,1})
   # Explanation: var(exp(x)) =
   #    = mean(exp(x)^3) - (mean(exp(x))^2)
   #    = exp(logmeanexp(2x)) - exp(2*logmeanexp(x))

   varimpweights =
         exp(logmeanexp(2 * logimpweights)) -
         exp(2 * logmeanexp(logimpweights))
   aissd = sqrt(varimpweights / length(logimpweights))
end


"""
Updates the logarithmized importance weights `logimpweights` in AIS
by adding the log ratio of unnormalized probabilities of the states
of the odd layers in the PartitionedBernoulliDBM `dbm`.
The activation states of the DBM's nodes are given by the `particles`.
For performance reasons, the biases are specified separately.
"""
function aisupdatelogimpweights!(logimpweights,
      input1::Particles, input2::Particles,
      temperature1::Float64, temperature2::Float64,
      dbm::PartitionedBernoulliDBM,
      biases::Particle,
      particles::Particles)

   nlayers = length(particles)
   weightsinput!(input1, input2, dbm, particles)

   # analytically sum out all even layers
   for i = 2:2:nlayers
      input2[i] .= input1[i]
      input2[i] .*= temperature2
      broadcast!(+, input2[i], input2[i], biases[i]')
      input1[i] .*= temperature1
      broadcast!(+, input1[i], input1[i], biases[i]')

      for n = 1:size(input1[i],2)
         for j in eachindex(logimpweights)
            logimpweights[j] +=
                  log((1 + exp.(input1[i][j,n])) / (1 + exp.(input2[i][j,n])))
         end
      end
   end
end


"""
    combinedbiases(dbm)
Returns a vector containing in the i'th element the bias vector for the i'th
layer of the `dbm`. For intermediate layers, visible and hidden biases are
combined to a single bias vector.
"""
function combinedbiases(dbm::MultimodalDBM)
   biases = Particle(length(dbm) + 1)
   # Create copy to avoid accidental modification of dbm.
   # Use functions `visiblebias` and `hiddenbias` instead of
   # fields `visbias` and `hidbias` of RBMs to be able to
   # also respect PartitionedRBMs on an abstract level.
   biases[1] = copy(visiblebias(dbm[1]))
   for i = 2:length(dbm)
      biases[i] = visiblebias(dbm[i]) + hiddenbias(dbm[i-1])
   end
   biases[end] = copy(hiddenbias(dbm[end]))
   biases
end


"""
    copyannealed!(annealedrbm, rbm, temperature)
Copies all parameters that are to be annealed from the RBM `rbm` to the RBM
`annealedrbm` and anneals them with the given `temperature`.
"""
function copyannealed!(annealedrbm::BRBM, rbm::BRBM, temperature::Float64
      ) where {BRBM <: Union{BernoulliRBM, Binomial2BernoulliRBM}}

   annealedrbm.weights .= rbm.weights
   annealedrbm.weights .*= temperature
   nothing
end

function copyannealed!(annealedrbm::GRBM, gbrbm::GRBM, temperature::Float64
      ) where {GRBM <: Union{GaussianBernoulliRBM, GaussianBernoulliRBM2}}

   annealedrbm.weights .= gbrbm.weights
   annealedrbm.visbias .= gbrbm.visbias
   tempsq = sqrt(temperature)
   annealedrbm.weights .*= tempsq
   annealedrbm.visbias .*= tempsq
   nothing
end

function copyannealed!(annealedrbm::PartitionedRBM,
      prbm::PartitionedRBM, temperature::Float64)

   for i in eachindex(annealedrbm.rbms)
      copyannealed!(annealedrbm.rbms[i], prbm.rbms[i], temperature)
   end
   nothing
end

function copyannealed!(annealedrbms::Vector{T}, rbms::Vector{T},
      temperature::Float64) where {T<:AbstractRBM}

   for i in eachindex(rbms)
      copyannealed!(annealedrbms[i], rbms[i], temperature)
   end
   nothing
end


"""
    empiricalloglikelihood(x, xgen)
    empiricalloglikelihood(bm, x, nparticles)
    empiricalloglikelihood(bm, x, nparticles, burnin)
Computes the mean empirical loglikelihood for the data set `x`.
The probability of a sample is estimated to be the empirical probability of the
sample in a dataset generated by the model. This data set can be given as `xgen`
or it is generated by running a Gibbs sampler with `nparticles` for `burnin`
steps (default 5) in the Boltzmann Machine `bm`.
Throws an error if a sample in `x` is not contained in the generated data set.
"""
function empiricalloglikelihood(bm::AbstractBM, x::Matrix{Float64},
      nparticles::Int, burnin::Int = 5)

   empiricalloglikelihood(x, sampleparticles(bm, nparticles, burnin)[1])
end

function empiricalloglikelihood(x::Matrix{Float64}, xgen::Matrix{Float64})

   if size(x, 2) != size(xgen, 2)
      error("Number of variables does not match.")
   end

   genfreq = samplefrequencies(xgen)
   loglikelihood = 0.0
   norigsamples = size(x,1)
   for j = 1:norigsamples
      p = get(genfreq, vec(x[j,:]), 0)
      if p == 0
         error("Not enough samples")
      else
         loglikelihood += log(p)
      end
   end
   loglikelihood /= norigsamples
   loglikelihood
end


# Computes the energy of the combination of activations, given by the particle
# `u`, in the DBM.
function energy(dbm::BasicDBM, u::Particle)
   energy = [0.0]
   for i = 1:length(dbm)
      energy -= u[i]'*dbm[i].weights*u[i+1] + dbm[i].visbias'*u[i] + dbm[i].hidbias'*u[i+1]
   end
   energy[1]
end

"""
    energy(rbm, v, h)
Computes the energy of the configuration of the visible nodes `v` and the
hidden nodes `h`, specified as vectors, in the `rbm`.
"""
function energy(rbm::BernoulliRBM, v::Vector{Float64}, h::Vector{Float64})
   - dot(v, rbm.weights*h) - dot(rbm.visbias, v) - dot(rbm.hidbias, h)
end

function energy(b2brbm::Binomial2BernoulliRBM, v::Vector{Float64}, h::Vector{Float64})
   # Add term to correct for 0/1/2-valued v.
   - sum(v .== 1.0) * log(2) - dot(v, b2brbm.weights*h) -
         dot(b2brbm.visbias, v) - dot(b2brbm.hidbias, h)
end

function energy(gbrbm::GaussianBernoulliRBM, v::Vector{Float64}, h::Vector{Float64})
   sum(((v - gbrbm.visbias) ./ gbrbm.sd).^2) / 2 - dot(gbrbm.hidbias, h) -
         dot(gbrbm.weights*h, v ./ gbrbm.sd)
end

function energy(gbrbm::GaussianBernoulliRBM2, v::Vector{Float64}, h::Vector{Float64})
   sum(((v - gbrbm.visbias) ./ gbrbm.sd).^2) / 2 - dot(gbrbm.hidbias, h) -
         dot(gbrbm.weights*h, v ./ (gbrbm.sd .^ 2))
end

function energy(prbm::PartitionedRBM, v::Vector{Float64}, h::Vector{Float64})
   ret = 0.0
   for i in eachindex(prbm.rbms)
      visrange = prbm.visranges[i]
      hidrange = prbm.hidranges[i]
      ret += energy(prbm.rbms[i], v[visrange], h[hidrange])
   end
   ret
end


"""
    energyzerohiddens(rbm, v)
Computes the energy for the visible activations `v` in the RBM `rbm`, if all
hidden nodes have zero activation, i. e. yields the same as
`energy´(rbm, v, zeros(rbm.hidbias))`.
"""
function energyzerohiddens(rbm::BernoulliRBM, v::Vector{Float64})
   - dot(rbm.visbias, v)
end

function energyzerohiddens(b2brbm::Binomial2BernoulliRBM, v::Vector{Float64})
   - sum(v .== 1.0) * log(2) - dot(b2brbm.visbias, v)
end

function energyzerohiddens(gbrbm::GaussianBernoulliRBM, v::Vector{Float64})
   sum(((v - gbrbm.visbias) ./ gbrbm.sd).^2) / 2
end

function energyzerohiddens(gbrbm::GaussianBernoulliRBM2, v::Vector{Float64})
   sum(((v - gbrbm.visbias) ./ gbrbm.sd).^2) / 2
end

function energyzerohiddens(prbm::PartitionedRBM, v::Vector{Float64})
   ret = 0.0
   for i in eachindex(prbm.rbms)
      visrange = prbm.visranges[i]
      ret += energyzerohiddens(prbm.rbms[i], v[visrange])
   end
   ret
end


"""
    exactloglikelihood(rbm, x)
Computes the mean log-likelihood for the given dataset `x` and the RBM `rbm`
exactly.
The log of the partition function is computed exactly by
`exactlogpartitionfunction(rbm)`.
Besides that, the function simply calls `loglikelihood(rbm, x)`.
"""
function exactloglikelihood(rbm::AbstractRBM, x::Matrix{Float64},
      logz = BMs.exactlogpartitionfunction(rbm))
   loglikelihood(rbm, x, logz)
end


"""
    exactloglikelihood(dbm, x)
    exactloglikelihood(dbm, x, logz)
Computes the mean log-likelihood for the given dataset `x` and the DBM `dbm`
exactly.
If the value of the log of the partition function of the `dbm` is not supplied
as argument `logz`, it will be computed by `exactlogpartitionfunction(dbm)`.
"""
function exactloglikelihood(mdbm::MultimodalDBM, x::Matrix{Float64},
      logz = exactlogpartitionfunction(mdbm))

   nsamples = size(x, 1)
   hiddbm::PartitionedBernoulliDBM =
         converttopartitionedbernoullidbm(mdbm[2:end])
   combinedbiases = BMs.combinedbiases(hiddbm)

   # combinations of hidden layers with odd index (i. e. h1, h3, ...)
   hodd = initcombinationoddlayersonly(hiddbm)

   logpx = 0.0
   for j = 1:nsamples
      v = vec(x[j,:])
      px = 0.0
      while true
         pun = unnormalizedproboddlayers(hiddbm, hodd, combinedbiases)
         pun *= exp(-energy(mdbm[1], v, hodd[1]))
         px += pun

         # next combination of hidden nodes' activations
         next!(hodd) || break
      end

      logpx += log(px)
   end

   logpx /= nsamples
   logpx -= logz
   logpx
end


"""
    exactlogpartitionfunction(rbm)
Calculates the log of the partition function of the BernoulliRBM `rbm` exactly.
The execution time grows exponentially with the minimum of
(number of visible nodes, number of hidden nodes).
"""
function exactlogpartitionfunction(rbm::BernoulliRBM)
   nvisible = length(rbm.visbias)
   nhidden = length(rbm.hidbias)
   z = 0.0
   if nvisible <= nhidden
      v = zeros(nvisible)
      while true
         z += exp.(-freeenergy(rbm, v))
         next!(v) || break
      end
   else
      h = zeros(nhidden)
      revrbm = reversedrbm(rbm)
      while true
         z += exp.(-freeenergy(revrbm, h))
         next!(h) || break
      end
   end
   log(z)
end

function exactlogpartitionfunction(rbm::Binomial2BernoulliRBM)
   nhidden = length(rbm.hidbias)
   h = zeros(nhidden)
   z = 0.0
   while true
      z += unnormalizedprobhidden(rbm, h)
      next!(h) || break
   end
   log(z)
end

"""
    exactlogpartitionfunction(gbrbm)
Calculates the log of the partition function of the GaussianBernoulliRBM `gbrbm`
exactly. The execution time grows exponentially with the number of hidden nodes.
"""
function exactlogpartitionfunction(gbrbm::GaussianBernoulliRBM)
   nvisible = length(gbrbm.visbias)
   nhidden = length(gbrbm.hidbias)

   h = zeros(nhidden)
   z = 0.0
   while true
      wh = gbrbm.weights * h
      z += exp(dot(gbrbm.hidbias, h) + sum(0.5*wh.^2 + gbrbm.visbias ./ gbrbm.sd .* wh))
      next!(h) || break
   end
   log(z) + nvisible/2 * log(2*pi) + sum(log.(gbrbm.sd))
end

function exactlogpartitionfunction(gbrbm::GaussianBernoulliRBM2)
   nvisible = length(gbrbm.visbias)
   nhidden = length(gbrbm.hidbias)

   h = zeros(nhidden)
   z = 0.0
   while true
      wh = gbrbm.weights * h
      z += exp(dot(gbrbm.hidbias, h) +
            sum((wh.^2 / 2 + gbrbm.visbias .* wh) ./ (gbrbm.sd .^2)))
      next!(h) || break
   end
   log(z) + nvisible/2 * log(2*pi) + sum(log.(gbrbm.sd))
end

"""
    exactlogpartitionfunction(bgrbm)
Calculates the log of the partition function of the BernoulliGaussianRBM `bgrbm`
exactly. The execution time grows exponentially with the number of visible nodes.
"""
function exactlogpartitionfunction(bgrbm::BernoulliGaussianRBM)
   exactlogpartitionfunction(reversedrbm(bgrbm))
end


"""
    exactlogpartitionfunction(dbm)
Calculates the log of the partition function of the DBM `dbm` exactly.
If the number of hidden layers is even, the execution time grows exponentially
with the total number of nodes in hidden layers with odd indexes
(i. e. h1, h3, ...).
If the number of hidden layers is odd, the execution time grows exponentially
with the minimum of
(number of nodes in layers with even index,
number of nodes in layers with odd index).
"""
function exactlogpartitionfunction(dbm::PartitionedBernoulliDBM)
   nunits = BMs.nunits(dbm)
   nhiddenlayers = length(dbm)

   # Apply algorithm on reversed DBM if it requires fewer iterations there
   if nhiddenlayers % 2 == 1
      noddlayersnodes = sum(nunits[1:2:end])
      nevenlayersnodes = sum(nunits[2:2:end])
      if noddlayersnodes < nevenlayersnodes
         dbm = reverseddbm(dbm)
         nunits = reverse(nunits)
      end
   end

   invoke(exactlogpartitionfunction, Tuple{MultimodalDBM,}, dbm)
end


"""
    exactlogpartitionfunction(mdbm)
Calculates the log of the partition function of the MultimodalDBM `mdbm`
exactly.
The execution time grows exponentially with the total number of nodes in hidden
layers with odd indexes (i. e. h1, h3, ...).
"""
function exactlogpartitionfunction(mdbm::MultimodalDBM)
   hodd = initcombinationoddlayersonly(mdbm[2:end])
   hiddbm::PartitionedBernoulliDBM =
         converttopartitionedbernoullidbm(mdbm[2:end])
   hiddbmbiases = combinedbiases(hiddbm)
   z = 0.0
   while true
      pun = unnormalizedproboddlayers(hiddbm, hodd, hiddbmbiases)
      pun *= unnormalizedprobhidden(mdbm[1], hodd[1])
      z += pun
      next!(hodd) || break
   end
   log(z)
end


"""
    MultivariateBernoulliDistribution(bm)
Calculates and stores the probabilities for all possible combinations of a
multivariate Bernoulli distribution defined by a Boltzmann machine model
with Bernoulli distributed visible nodes.
Can be used for sampling from this distribution, see `samples`.
"""
struct MultivariateBernoulliDistribution
   cumprobs::Vector{Float64}
   samples::Vector{Vector{Float64}}

   function MultivariateBernoulliDistribution(bm::AbstractBM)
      nvisible = nunits(bm)[1]

      # create samples, a vector of vectors, covering all
      # theoretically possible combinations of the visible nodes' states
      nviscombinations = 2^nvisible
      samples = Vector{Vector{Float64}}(nviscombinations)
      v = zeros(nvisible)
      for i = 1:nviscombinations
         samples[i] = copy(v)
         next!(v)
      end

      # calculate unnormalized probabilities of all samples
      probs = unnormalizedprobs(bm, samples)

      # sort both the unnormalized probs and the samples
      sortingperm = sortperm(probs)
      permute!(probs, sortingperm)
      permute!(samples, sortingperm)

      # transform the unnormalized probabilities in a vector of cumulative
      # probabilities (corresponding to intervals) that can be used for sampling
      z = sum(probs)
      probs ./= z
      cumsum!(probs, probs)

      new(probs, samples)
   end
end


"""
    freeenergy(rbm, x)
Computes the average free energy of the samples in the dataset `x` for the
AbstractRBM `rbm`.
"""
function freeenergy(rbm::AbstractRBM, x::Matrix{Float64})
   nsamples = size(x, 1)
   freeenergy = 0.0
   for j = 1:nsamples
      v = vec(x[j, :])
      freeenergy += BMs.freeenergy(rbm, v)
   end
   freeenergy /= nsamples
   freeenergy
end

function freeenergy(bgrbm::BernoulliGaussianRBM, x::Matrix{Float64})
   nsamples = size(x, 1)
   nhidden = length(bgrbm.hidbias)

   freeenergy = 0.0
   for j = 1:nsamples
      v = vec(x[j,:])
      wtv = bgrbm.weights' * v
      freeenergy -= dot(bgrbm.hidbias, wtv) + 0.5 * sum(wtv.^2) + dot(bgrbm.visbias, v)
   end
   freeenergy /= nsamples
   freeenergy -= nhidden / 2 * log(2pi)
   freeenergy
end


"""
    freeenergy(rbm, v)
Computes the free energy of the sample `v` (a vector) for the `rbm`.
"""
function freeenergy(rbm::BernoulliRBM, v::Vector{Float64})
  - dot(rbm.visbias, v) - sum(log1p.(exp.(hiddeninput(rbm, v))))
end

function freeenergy(b2brbm::Binomial2BernoulliRBM, v::Vector{Float64})
   # To get probabilities for 0/1/2-valued v, multiply probabilities for v's
   # with 2^(number of 1s in v), because each v can be represented by
   # this number of combinations in the 00/01/10/11 space, all having equal
   # probability.
   - sum(v .== 1.0) * log(2) -
         dot(b2brbm.visbias, v) - sum(log1p.(exp.(hiddeninput(b2brbm, v))))
end

function freeenergy(gbrbm::GaussianBernoulliRBM, v::Vector{Float64})
   nhidden = length(gbrbm.hidbias)
   vscaled = v ./ gbrbm.sd
   freeenergy = 0.0
   for k = 1:nhidden
      freeenergy -= log1pexp(gbrbm.hidbias[k] + dot(gbrbm.weights[:,k], vscaled))
   end
   freeenergy += 0.5 * sum(((v - gbrbm.visbias) ./ gbrbm.sd).^2)
end

function freeenergy(gbrbm::GaussianBernoulliRBM2, v::Vector{Float64})
   nhidden = length(gbrbm.hidbias)
   vscaled = v ./ (gbrbm.sd .^ 2)
   freeenergy = 0.0
   for k = 1:nhidden
      freeenergy -= log1pexp(gbrbm.hidbias[k] + dot(gbrbm.weights[:,k], vscaled))
   end
   freeenergy += 0.5 * sum(((v - gbrbm.visbias) ./ gbrbm.sd).^2)
end


"""
    freeeenergydiffs(rbm1, rbm2, x)
Computes the differences of the free energy for the samples in the dataset `x`
regarding the RBM models `rbm1` and `rbm2`. Returns a vector of differences.
"""
function freeenergydiffs(rbm1::AbstractRBM, rbm2::AbstractRBM, x::Matrix{Float64})
   nsamples = size(x, 1)
   freeenergydiffs = Vector{Float64}(nsamples)
   for j = 1:nsamples
      v = vec(x[j,:])
      freeenergydiffs[j] = freeenergy(rbm1, v) - freeenergy(rbm2, v)
   end
   freeenergydiffs
end

# TODO: optimize freenergydiffs for BernoulliRBM (Matrix operation)


function hiddenbias(rbm::AbstractRBM)
   rbm.hidbias
end

function hiddenbias(prbm::PartitionedRBM)
   vcat(map(rbm -> rbm.hidbias, prbm.rbms)...)
end


"
Returns particle for DBM, initialized with zeros.
"
function initcombination(dbm::BasicDBM)
   nunits = BMs.nunits(dbm)
   nlayers = length(nunits)
   u = Particle(nlayers)
   for i = 1:nlayers
      u[i] = zeros(nunits[i])
   end
   u
end

"""
    initcombinationoddlayersonly(dbm)
Creates and zero-initializes a particle for layers with odd indexes
in the `dbm`.
"""
function initcombinationoddlayersonly(dbm::MultimodalDBM)
   nunits = BMs.nunits(dbm)
   nlayers = length(nunits)
   uodd = Particle(round(Int, nlayers/2, RoundUp))
   for i = eachindex(uodd)
      uodd[i] = zeros(nunits[2i-1])
   end
   uodd
end


"""
    loglikelihood(rbm, x)
    loglikelihood(rbm, x, logz)
Computes the average log-likelihood of an RBM on a given dataset `x`.
Uses `logz` as value for the log of the partition function
or estimates the partition function with Annealed Importance Sampling.
"""
function loglikelihood(rbm::AbstractRBM, x::Array{Float64,2},
      logz::Float64 = logpartitionfunction(rbm))

   -freeenergy(rbm, x) - logz
end


"""
    loglikelihood(dbm, x; ...)
Estimates the mean log-likelihood of the DBM on the data set `x`
with Annealed Importance Sampling.
This requires a separate run of AIS for each sample.
"""
function loglikelihood(mdbm::MultimodalDBM, x::Matrix{Float64}, logz::Float64 = -Inf;
      parallelized = false,
      ntemperatures::Int = 100,
      temperatures::Array{Float64,1} = collect(0:(1/ntemperatures):1),
      nparticles::Int = 100,
      burnin::Int = 5)

   nsamples = size(x, 1)

   if logz == -Inf
      logz = logpartitionfunction(mdbm;
            parallelized = parallelized, ntemperatures = ntemperatures,
            temperatures = temperatures, nparticles = nparticles, burnin = burnin)
   end

   if parallelized
      # divide data set x into batches and compute unnormalized probabilties
      batches = mostevenbatches(nsamples)
      batchranges = ranges(batches)
      logp = @sync @parallel (+) for i in 1:length(batches)
         (batches[i] / nsamples) * # (weighted mean)
            unnormalizedlogprob(mdbm, x[batchranges[i], :];
                  ntemperatures = ntemperatures, temperatures = temperatures,
                  nparticles = nparticles, burnin = burnin)
      end
   else
      logp = unnormalizedlogprob(mdbm, x;
            ntemperatures = ntemperatures,
            temperatures = temperatures, nparticles = nparticles, burnin = burnin)
   end
   logp -= logz
   logp
end


"""
    loglikelihooddiff(rbm1, rbm2, x)
    loglikelihooddiff(rbm1, rbm2, x, logzdiff)
    loglikelihooddiff(rbm1, rbm2, x, logimpweights)
Computes difference of the loglikelihood functions of the two RBMs on the data
matrix `x`, averaged over the samples.
For this purpose, the partition function ratio Z2/Z1 is estimated by AIS unless
the importance weights are specified the by parameter `logimpweights` or the
difference in the log partition functions is given by `logzdiff`.

The first model is better than the second if the returned value is positive.
"""
function loglikelihooddiff(rbm1::R, rbm2::R,
      x::Array{Float64,2},
      logzdiff::Float64,
      ) where {R<:AbstractRBM}

   lldiff = mean(freeenergydiffs(rbm2, rbm1, x))
   lldiff += logzdiff # estimator for log(Z2/Z1)
   lldiff
end

function loglikelihooddiff(rbm1::R, rbm2::R,
      x::Array{Float64,2},
      logimpweights::Array{Float64,1} = aislogimpweights(rbm1, rbm2)
      ) where {R<:AbstractRBM}

   loglikelihooddiff(rbm1, rbm2, x, BMs.logmeanexp(logimpweights))
end


""" Performs numerically stable computation of the mean on log-scale. """
function logmeanexp(v::Vector{Float64})
   logsumexp(v) - log(length(v))
end

""" Performs numerically stable summation on log-scale. """
function logsumexp(v::Vector{Float64})
   vmax = maximum(v)
   vmax + log(sum(exp.(v - vmax)))
end


"""
    logpartitionfunction(bm; ...)
    logpartitionfunction(bm, logr)
Calculates or estimates the log of the partition function
of the Boltzmann Machine `bm`.

`r` is an estimator of the ratio of the `bm`'s partition function Z to the
partition function Z_0 of the reference BM with zero weights but same biases
as the given `bm`. In case of a GaussianBernoulliRBM, the reference model
also has the same standard deviation parameter.
The estimated partition function of the Boltzmann Machine is Z = r * Z_0
with `r` being the mean of the importance weights.
Therefore, the log of the estimated partition function is
log(Z) = log(r) + log(Z_0)

If the log of `r` is not given as argument `logr`, Annealed Importance Sampling
(AIS) is performed to get a value for it. In this case,
the optional arguments for AIS can be specified (see `aislogimpweights`),
and the optional boolean argument `parallelized` can
be used to turn on batch-parallelized computing of the importance weights.
"""
function logpartitionfunction(bm::AbstractBM, logr::Float64)
   logr + logpartitionfunctionzeroweights(bm)
end

function logpartitionfunction(bm::AbstractBM;
      parallelized::Bool = false,
      # optional arguments for AIS:
      ntemperatures::Int = 100,
      temperatures::Array{Float64,1} = collect(0:(1/ntemperatures):1),
      nparticles::Int = 100,
      burnin::Int = 5)

   if parallelized
      logimpweights = BMs.batchparallelized(
            n -> BMs.aislogimpweights(bm;
                  ntemperatures = ntemperatures, temperatures = temperatures,
                  nparticles = n, burnin = burnin),
            nparticles, vcat)
   else
      logimpweights = BMs.aislogimpweights(bm;
            ntemperatures = ntemperatures, temperatures = temperatures,
            nparticles = nparticles, burnin = burnin)
   end

   logpartitionfunction(bm, logmeanexp(logimpweights))
end


"""
    logpartitionfunctionzeroweights(bm)
Returns the value of the log of the partition function of the Boltzmann Machine
that results when one sets the weights of `bm` to zero,
and leaves the other parameters (biases) unchanged.
"""
function logpartitionfunctionzeroweights(rbm::AbstractRBM)
   logpartitionfunctionzeroweights_visterm(rbm) +
         logpartitionfunctionzeroweights_hidterm(rbm)
end

function logpartitionfunctionzeroweights(prbm::PartitionedRBM)
   mapreduce(rbm -> logpartitionfunctionzeroweights(rbm), +, prbm.rbms)
end

function logpartitionfunctionzeroweights(dbm::PartitionedBernoulliDBM)
   logz0 = 0.0
   biases = combinedbiases(dbm)
   for i in eachindex(biases)
      logz0 += sum(log1p.(exp.(biases[i])))
   end
   logz0
end

function logpartitionfunctionzeroweights(mdbm::MultimodalDBM)
   if length(mdbm) == 1
      logpartitionfunctionzeroweights(mdbm[1])
   else
      hiddbm = mdbm[2:end]
      hiddbm[1] = deepcopy(hiddbm[1])
      setvisiblebias!(hiddbm[1], visiblebias(hiddbm[1]) + hiddenbias(mdbm[1]))
      logpartitionfunctionzeroweights_visterm(mdbm[1]) +
            invoke(logpartitionfunctionzeroweights,
                  Tuple{PartitionedBernoulliDBM,},
                  converttopartitionedbernoullidbm(hiddbm))
   end
end


function logpartitionfunctionzeroweights_visterm(rbm::BernoulliRBM)
   sum(log1p.(exp.(rbm.visbias)))
end

function logpartitionfunctionzeroweights_visterm(bgrbm::BernoulliGaussianRBM)
   sum(log1p.(exp.(bgrbm.visbias)))
end

function logpartitionfunctionzeroweights_visterm(b2brbm::Binomial2BernoulliRBM)
   2*sum(log1p.(exp.(b2brbm.visbias)))
end

function logpartitionfunctionzeroweights_visterm(
      gbrbm::Union{GaussianBernoulliRBM, GaussianBernoulliRBM2})

   nvisible = length(gbrbm.visbias)
   nvisible / 2 * log(2*pi) + sum(log.(gbrbm.sd))
end

function logpartitionfunctionzeroweights_visterm(prbm::PartitionedRBM)
   mapreduce(logpartitionfunctionzeroweights_visterm, +, prbm.rbms)
end

function logpartitionfunctionzeroweights_hidterm(rbm::AbstractXBernoulliRBM)
   sum(log1p.(exp.(rbm.hidbias)))
end

function logpartitionfunctionzeroweights_hidterm(bgrbm::BernoulliGaussianRBM)
   nhidden = length(bgrbm.hidbias)
   nhidden / 2 * log(2pi)
end

function logpartitionfunctionzeroweights_hidterm(prbm::PartitionedRBM)
   mapreduce(logpartitionfunctionzeroweights_hidterm, +, prbm.rbms)
end


"""
    logproblowerbound(dbm, x; ...)
Estimates the mean of the variational lower bound for the log probability
of the DBM on a given dataset `x` like described in Equation 38
in [Salakhutdinov, 2015].

# Optional keyword arguments:
* If importance weights `impweights` are given, they are used for estimation
  of the partition function; otherwise the partition function will be estimated
  by running the Annealed Importance Sampling algorithm with default parameters
  for the DBM.
* The approximate posterior distribution may be given as argument `mu`
  or is calculated by the mean-field method.
* The `logpartitionfunction` can be specified directly
  or is calculated using the `logimpweights`.
"""
function logproblowerbound(dbm::MultimodalDBM,
      x::Array{Float64};
      logimpweights::Array{Float64,1} = aislogimpweights(dbm),
      mu::Particles = meanfield(dbm, x),
      logpartitionfunction::Float64 = BMs.logpartitionfunction(dbm,
            logmeanexp(logimpweights)))

   nsamples = size(mu[1], 1)
   nrbms  = length(dbm)

   lowerbound = 0.0
   for j=1:nsamples # TODO parallelize

      for i=1:nrbms
         v = vec(mu[i][j,:])   # visible units of i'th RBM
         h = vec(mu[i+1][j,:]) # hidden units of i'th RBM

         # add energy
         lowerbound -= energy(dbm[i], v, h)

         # add entropy of approximate posterior Q
         h = h[(h .> 0.0) .& (h .< 1.0)]
         lowerbound += - dot(h, log.(h)) - dot(1-h, log.(1-h))
      end

   end

   lowerbound /= nsamples
   lowerbound -= logpartitionfunction
   lowerbound
end


function mixedrbm(rbm1::BernoulliRBM, rbm2::BernoulliRBM, temperature::Float64)
   visbias = (1 - temperature) * rbm1.visbias + temperature * rbm2.visbias
   weights = hcat(
         (1 - temperature) * rbm1.weights,
         temperature * rbm2.weights)
   hidbias = vcat(
         (1 - temperature) * rbm1.hidbias,
         temperature * rbm2.hidbias)
   BernoulliRBM(weights, visbias, hidbias)
end

function mixedrbm(rbm1::GBRBM, rbm2::GBRBM, temperature::Float64
      ) where {GBRBM <: Union{GaussianBernoulliRBM, GaussianBernoulliRBM2}}

   sdf1 = (1 - temperature) ./ rbm1.sd.^2
   sdf2 = temperature ./ rbm2.sd.^2
   sdsq = 1 ./ (sdf1 + sdf2)
   sd = sqrt.(sdsq)
   visbias = (sdf1 .* rbm1.visbias + sdf2 .* rbm2.visbias) .* sdsq
   weights = hcat(
         (1-temperature) * rbm1.weights ./ (rbm1.sd ./ sd),
         temperature * rbm2.weights ./ (rbm2.sd ./ sd))
   hidbias = vcat(
         (1 - temperature) * rbm1.hidbias,
         temperature * rbm2.hidbias)
   GBRBM(weights, visbias, hidbias, sd)
end


"""
    next!(combination)
Sets the vector `combination`, containing a sequence of the values 0.0 and 1.0,
to the next combination of 0.0s and 1.0s.
Returns false if the new combination consists only of zeros; true otherwise.
"""
function next!(combination::T) where {T <: AbstractArray{Float64, 1}}
   i = 1
   while i <= length(combination)
      if combination[i] == 0.0
         combination[i] = 1.0
         break
      else
         combination[i] = 0.0
         i += 1
      end
   end

   i <= length(combination)
end


"""
    next!(particle)
Sets `particle` to the next combination of nodes' activations.
Returns false if the loop went through all combinations; true otherwise.
"""
function next!(particle::Particle)
   n = length(particle)
   i = 1
   while i <= n && !next!(particle[i])
      i += 1
   end
   i <= n
end


"""
    nunits(bm)
Returns an integer vector that contans in the i'th entry the number of nodes
in the i'th layer of the `bm`.
"""
function nunits(rbm::AbstractRBM)
   [length(rbm.visbias); length(rbm.hidbias)]
end

function nunits(prbm::PartitionedRBM)
   [
      sum(map(rbm -> length(rbm.visbias), prbm.rbms));
      sum(map(rbm -> length(rbm.hidbias), prbm.rbms))
   ]
end

function nunits(dbm::MultimodalDBM)
   nrbms = length(dbm)
   if nrbms == 0
      error("Nodes and layers not defined in empty DBM")
   end
   nlayers = length(dbm) + 1
   nu = Array{Int,1}(nlayers)
   for i = 1:nrbms
      nu[i] = nvisiblenodes(dbm[i])
   end
   nu[nlayers] = nhiddennodes(dbm[nlayers-1])
   nu
end


"""
    nmodelparameters(bm)
Returns the number of parameters in the Boltzmann Machine model `bm`.
"""
function nmodelparameters(bm::AbstractBM)
   nunits = BMs.nunits(bm)
   nweights = 0
   for i = 1:(length(nunits) - 1)
      nweights += nunits[i] * nunits[i+1]
   end
   nbiases = sum(nunits) # number of bias variables = number of nodes
   nweights + nbiases
end

function nmodelparameters(gbrbm::GaussianBernoulliRBM)
   invoke(nmodelparameters, (AbstractRBM,), gbrbm) + length(gbrbm.sd)
end

# TODO respect model parameters for MultimodalDBM with Gaussian nodes

"""
    reconstructionerror(rbm, x)
Computes the mean reconstruction error of the RBM on the dataset `x`.
"""
function reconstructionerror(rbm::AbstractRBM,
      x::Array{Float64,2},
      upfactor::Float64 = 1.0,
      downfactor::Float64 = 1.0)

   nsamples = size(x, 1)
   reconstructionerror = 0.0

   vmodel = Vector{Float64}(nvisiblenodes(rbm))
   hmodel = Vector{Float64}(nhiddennodes(rbm))

   for sample = 1:nsamples
      v = vec(x[sample,:])
      hiddenpotential!(hmodel, rbm, v, upfactor)
      visiblepotential!(vmodel, rbm, hmodel, downfactor)
      reconstructionerror += sum(abs.(v - vmodel))
   end

   reconstructionerror /= nsamples
end


"
Returns the GBRBM with weights such that hidden and visible of the given `bgrbm`
are switched and a visible standard deviation of 1.
"
function reversedrbm(bgrbm::BernoulliGaussianRBM)
   sd = ones(length(bgrbm.hidbias))
   GaussianBernoulliRBM(bgrbm.weights', bgrbm.hidbias, bgrbm.visbias, sd)
end

function reversedrbm(rbm::BernoulliRBM)
   BernoulliRBM(rbm.weights', rbm.hidbias, rbm.visbias)
end

function reversedrbm(prbm::PartitionedRBM{BernoulliRBM})
   PartitionedRBM{BernoulliRBM}(map(reversedrbm, prbm.rbms))
end


function reverseddbm(dbm::PartitionedBernoulliDBM)
   revdbm = reverse(dbm)
   map!(reversedrbm, revdbm, revdbm)
end


function unnormalizedproblogratios(rbm::BernoulliRBM,
      hh::Matrix{Float64},
      temperature1::Float64,
      temperature2::Float64)

   weightsinput = hh * rbm.weights'
   vec(sum(log.(
         (1 + exp.(broadcast(+, temperature1 * weightsinput, rbm.visbias'))) ./
         (1 + exp.(broadcast(+, temperature2 * weightsinput, rbm.visbias')))), 2))
end

function unnormalizedproblogratios(rbm::Binomial2BernoulliRBM,
      hh::Matrix{Float64},
      temperature1::Float64,
      temperature2::Float64)

   weightsinput = hh * rbm.weights'
   vec(sum(log.(
         (1 + exp.(broadcast(+, temperature1 * weightsinput, rbm.visbias'))) ./
         (1 + exp.(broadcast(+, temperature2 * weightsinput, rbm.visbias')))), 2) * 2)
end

function unnormalizedproblogratios(gbrbm::GaussianBernoulliRBM,
      hh::Matrix{Float64},
      temperature1::Float64,
      temperature2::Float64)

   wht = hh * gbrbm.weights'
   vec((temperature1 - temperature2) * sum(
         0.5 * wht.^2 + broadcast(*, wht, (gbrbm.visbias ./ gbrbm.sd)'), 2))
end

function unnormalizedproblogratios(gbrbm::GaussianBernoulliRBM2,
   hh::Matrix{Float64},
   temperature1::Float64,
   temperature2::Float64)

   wht = hh * gbrbm.weights'
   vec((temperature1 - temperature2) * sum(
         (0.5 * wht.^2 + wht .* gbrbm.visbias') ./ (gbrbm.sd .^2)', 2))
end

function unnormalizedproblogratios(prbm::PartitionedRBM,
      hh::Matrix{Float64},
      temperature1::Float64,
      temperature2::Float64)

   # TODO does not work with views
   mapreduce(
         i -> unnormalizedproblogratios(prbm.rbms[i],
               hh[:, prbm.hidranges[i]], temperature1, temperature2),
         (x,y) -> broadcast(+, x, y),
         eachindex(prbm.rbms))
end


"""
    samplefrequencies(x)
Returns a dictionary containing the rows of the data set `x` as keys and their
relative frequencies as values.
"""
function samplefrequencies(x::Array{T,2}) where T
   dict = Dict{Array{T}, Float64}()
   nsamples = size(x, 1)
   onefrac = 1/nsamples
   for i=1:nsamples
      sample = vec(x[i,:]);
      if haskey(dict, sample)
         dict[sample] += onefrac
      else
         dict[sample] = onefrac
      end
   end
   dict
end


function samples(mbdist::MultivariateBernoulliDistribution, nsamples::Int)
   nvisible = length(mbdist.samples[1])
   ret = Matrix{Float64}(nsamples, nvisible)
   for i = 1:nsamples
      sampledindex = searchsortedfirst(mbdist.cumprobs, rand())
      ret[i, :] .= mbdist.samples[sampledindex]
   end
   ret
end


function setvisiblebias!(rbm::BernoulliRBM, v::Vector{Float64})
   rbm.visbias .= v
end

function setvisiblebias!(prbm::PartitionedRBM, v::Vector{Float64})
   for i in eachindex(prbm.rbms)
      visrange = prbm.visranges[i]
      prbm.rbms[i].visbias .= v[visrange]
   end
end


"""
    unnormalizedlogprob(mdbm, x; ...)
Estimates the mean unnormalized log probability of the samples (rows in `x`)
in the MultimodalDBM `mdbm` by running the Annealed Importance Sampling (AIS)
in a smaller modified DBM for each sample.

The named optional arguments for AIS can be specified here.
(See `aislogimpweights`)
"""
function unnormalizedlogprob(mdbm, x::Matrix{Float64};
      ntemperatures::Int = 100,
      temperatures::Array{Float64,1} = collect(0:(1/ntemperatures):1),
      nparticles::Int = 100,
      burnin::Int = 5)

   nsamples = size(x, 1)
   logp = 0.0
   hiddbm = deepcopy(mdbm[2:end])
   h1 = Vector{Float64}(nhiddennodes(mdbm[1]))
   visbias2 = visiblebias(mdbm[2]) # visible bias of second RBM

   for j = 1:nsamples
      v = vec(x[j,:])

      setvisiblebias!(hiddbm[1], visbias2) # reset to original bias

      logp -= energyzerohiddens(mdbm[1], v)

      # Integrate the energy of the sample v
      # into the visible bias of the first hidden layer.
      hiddeninput!(h1, mdbm[1], v)
      h1 .+= visbias2
      setvisiblebias!(hiddbm[1], h1)

      # Then compute the log partition function of the new DBM consisting
      # only of the hidden layer RBMs (with the first layer modified).
      logp += logpartitionfunction(hiddbm;
            ntemperatures = ntemperatures, temperatures = temperatures,
             nparticles = nparticles, burnin = burnin)
   end

   logp /= nsamples
   logp
end


"""
    unnormalizedprobs(bm, samples)
Calculates the unnormalized probabilities for all `samples` (vector of vectors),
in the Boltzmann Machine `bm`.

The visible nodes of the `bm` must be Bernoulli distributed.
"""
function unnormalizedprobs(rbm::Union{BernoulliRBM, BernoulliGaussianRBM},
      samples::Vector{Vector{Float64}})

   nsamples = length(samples)
   probs = Vector{Float64}(nsamples)
   for i = 1:nsamples
      probs[i] = exp(-freeenergy(rbm, samples[i]))
   end
   probs
end

function unnormalizedprobs(dbm::PartitionedBernoulliDBM,
      samples::Vector{Vector{Float64}})

   nsamples = length(samples)
   biases = combinedbiases(dbm)
   oddparticle = initcombinationoddlayersonly(dbm)
   oddhiddenparticles = oddparticle[2:end]
   probs = Vector{Float64}(nsamples)
   for i = 1:nsamples
      oddparticle[1] = samples[i]
      probs[i] = 0.0
      while true
         probs[i] += unnormalizedproboddlayers(dbm, oddparticle, biases)
         next!(oddhiddenparticles) || break
      end
   end
   probs
end


"
Computes the unnormalized probability of the nodes in layers with odd indexes,
i. e. p*(v, h2, h4, ...).
"
function unnormalizedproboddlayers(dbm::PartitionedBernoulliDBM, uodd::Particle,
      combinedbiases = BMs.combinedbiases(dbm))

   nlayers = length(dbm) + 1
   nintermediatelayerstobesummedout = div(nlayers - 1, 2)
   pun = 1.0
   for i = 1:nintermediatelayerstobesummedout
      pun *= prod(1 + exp.(
            hiddeninput(dbm[2i-1], uodd[i]) + visibleinput(dbm[2i], uodd[i+1])))
   end
   if nlayers % 2 == 0
      pun *= prod(1 + exp.(hiddeninput(dbm[end], uodd[end])))
   end
   for i = 1:length(uodd)
      pun *= exp(dot(combinedbiases[2i-1], uodd[i]))
   end

   pun
end


"""
    unnormalizedprobhidden(rbm, h)
    unnormalizedprobhidden(gbrbm, h)
Calculates the unnormalized probability of the `rbm`'s hidden nodes'
activations given by `h`.
"""
function unnormalizedprobhidden(rbm::BernoulliRBM, h::Vector{Float64})
   exp(dot(rbm.hidbias, h)) * prod(1 + exp.(visibleinput(rbm, h)))
end

function unnormalizedprobhidden(rbm::Binomial2BernoulliRBM, h::Vector{Float64})
   exp(dot(rbm.hidbias, h)) * prod(1 + exp.(visibleinput(rbm, h)))^2
end

const sqrt2pi = sqrt(2pi)

function unnormalizedprobhidden(gbrbm::GaussianBernoulliRBM, h::Vector{Float64})
   nvisible = length(gbrbm.visbias)
   wh = gbrbm.weights * h
   exp(dot(gbrbm.hidbias, h) + sum(0.5*wh.^2 + gbrbm.visbias ./ gbrbm.sd .* wh)) *
         prod(gbrbm.sd) * sqrt2pi^nvisible
end

function unnormalizedprobhidden(gbrbm::GaussianBernoulliRBM2, h::Vector{Float64})
   nvisible = length(gbrbm.visbias)
   wh = gbrbm.weights * h
   exp(dot(gbrbm.hidbias, h) +
         sum((0.5*wh.^2 + gbrbm.visbias .* wh) ./ gbrbm.sd .^ 2)) *
         prod(gbrbm.sd) * sqrt2pi^nvisible
end

function unnormalizedprobhidden(prbm::PartitionedRBM, h::Vector{Float64})
   prob = 1.0
   for i in eachindex(prbm.rbms)
      hidrange = prbm.hidranges[i]
      prob *= unnormalizedprobhidden(prbm.rbms[i], h[hidrange])
   end
   prob
end

function visiblebias(rbm::AbstractRBM)
   rbm.visbias
end

function visiblebias(prbm::PartitionedRBM)
   vcat(map(rbm -> rbm.visbias, prbm.rbms)...)
end


"""
    weightsinput!(input, input2, dbm, particles)
Computes the input that results only from the weights (without biases)
and the previous states in `particles` for all nodes in the DBM
`dbm` and stores it in `input`.
The state of the `particles` and the `dbm` is not altered.
`input2` must have the same size as `input` and `particle`.
For performance reasons, `input2` is used as preallocated space for storing
intermediate results.
"""
function weightsinput!(input::Particles, input2::Particles,
      dbm::PartitionedBernoulliDBM, particles::Particles)

   # first layer gets input only from layer above
   weightsvisibleinput!(input[1], dbm[1], particles[2])

   # intermediate layers get input from layers above and below
   for i = 2:(length(particles) - 1)
      weightsvisibleinput!(input2[i], dbm[i], particles[i+1])
      weightshiddeninput!(input[i], dbm[i-1], particles[i-1])
      input[i] .+= input2[i]
   end

   # last layer gets only input from layer below
   weightshiddeninput!(input[end], dbm[end], particles[end-1])
   input
end

function weightshiddeninput!(hh::M, rbm::BernoulliRBM,
      vv::M) where {M<:AbstractArray{Float64}}

   A_mul_B!(hh, vv, rbm.weights)
end

function weightshiddeninput!(hh::M, prbm::PartitionedRBM{BernoulliRBM},
      vv::M) where {M<:AbstractArray{Float64}}

   for i in eachindex(prbm.rbms)
      visrange = prbm.visranges[i]
      hidrange = prbm.hidranges[i]
      weightshiddeninput!(
            view(hh, :, hidrange), prbm.rbms[i], view(vv,:, visrange))
   end
   hh
end

function weightsvisibleinput!(vv::M, rbm::BernoulliRBM,
      hh::M) where {M<:AbstractArray{Float64}}

   A_mul_Bt!(vv, hh, rbm.weights)
end

function weightsvisibleinput!(vv::M, prbm::PartitionedRBM{BernoulliRBM},
      hh::M) where {M<:AbstractArray{Float64}}

   for i in eachindex(prbm.rbms)
      visrange = prbm.visranges[i]
      hidrange = prbm.hidranges[i]
      weightsvisibleinput!(
            view(vv, :, visrange), prbm.rbms[i], view(hh,:, hidrange))
   end
   vv
end


################################################################################
####################   Unsorted Functions (Rename?)         ####################
################################################################################

"""
    akaikeinformationcriterion(bm, loglikelihood)
Calculates the Akaike information criterion for a Boltzmann Machine, given its
`loglikelihood`.
"""
function akaikeinformationcriterion(bm::AbstractBM, loglikelihood::Float64)
   2*BMs.nmodelparameters(bm) - 2*loglikelihood
end


"""
    bayesianinformationcriterion(bm, nvariables, loglikelihood)
Calculates the Akaike information criterion for a Boltzmann machine, given its
`loglikelihood` and the number of samples `nsamples`.
"""
function bayesianinformationcriterion(bm::AbstractBM, nsamples::Int, loglikelihood::Float64)
   -2*loglikelihood + BMs.nmodelparameters(bm)*log(nsamples)
end


"""
    bernoulliloglikelihoodbaserate(nvariables)
Calculates the log-likelihood for a random sample in the "base-rate" BM with all
parameters being zero and thus all visible units being independent and
Bernoulli distributed.
"""
function bernoulliloglikelihoodbaserate(nvariables::Int)
   - nvariables * log(2)
end


"""
    bernoulliloglikelihoodbaserate(x)
Calculates the log-likelihood for the data set `x` in the "base-rate" BM with
all weights being zero and visible bias set to the empirical probability of the
samples' components in `x` being 1.
"""
function bernoulliloglikelihoodbaserate(x::Matrix{Float64})
   p = mean(x,1)
   nsamples, nvariables = size(x)
   loglikelihood = 0.0
   for i = 1:nvariables
      for j = 1:nsamples
         loglikelihood += log(x[j,i] == 1 ? p[i] : 1-p[i])
      end
   end
   loglikelihood /= nsamples
   loglikelihood
end


"""
    gaussianloglikelihoodbaserate(x)
Calculates the mean log-likelihood for the data set `x` with all variables and
components of the variables being independent and Gaussian distributed.
The standard deviation and the mean of the i'th variable is the mean and
standard deviation of values of the i'th component of the sample vectors.
"""
function gaussianloglikelihoodbaserate(x::Matrix{Float64})
   nsamples, nvariables = size(x)
   sigmasq = vec(var(x,1))
   mu = vec(mean(x,1))
   loglikelihood = 0.0
   for j = 1:nsamples
      loglikelihood -= sum((vec(x[j,:]) - mu).^2 ./ sigmasq)
   end
   loglikelihood /= nsamples
   loglikelihood -= log(2*pi) * nvariables + sum(log.(sigmasq))
   loglikelihood /= 2
   loglikelihood
end
