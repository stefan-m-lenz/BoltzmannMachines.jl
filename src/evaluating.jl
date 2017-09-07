"""
    aisimportanceweights(rbm; ...)
Computes the importance weights for estimating the ratio of the partition
functions of the given `rbm` to the RBM with zero weights,
but same visible and hidden bias as the `rbm`.
This function implements the Annealed Importance Sampling algorithm (AIS)
like described in section 4.1.3 of [Salakhutdinov, 2008].

# Optional keyword arguments (for all types of Boltzmann Machines):
* `ntemperatures`: Number of temperatures for annealing from the starting model
  to the target model, defaults to 100
* `beta`: Vector of temperatures. By default `ntemperatures` ascending
  numbers, equally spaced from 0.0 to 1.0
* `nparticles`: Number of parallel chains and calculated weights, defaults to
   100
* `burnin`: Number of steps to sample for the Gibbs transition between models
"""
function aisimportanceweights(rbm::BernoulliRBM;
      ntemperatures::Int = 100,
      beta::Array{Float64,1} = collect(0:(1/ntemperatures):1),
      nparticles::Int = 100,
      burnin::Int = 5)

   impweights = ones(nparticles)
   mixrbm = deepcopy(rbm)

   # start with samples from model with zero weights
   hh = repmat(rbm.hidbias', nparticles)
   sigm_bernoulli!(hh)

   for k = 2:length(beta)
      impweights .*= aisunnormalizedprobratios(rbm, hh, beta[k], beta[k-1])

      # Gibbs transition
      mixrbm.weights = rbm.weights * beta[k]
      for burn = 1:burnin
         vv = samplevisible(mixrbm, hh)
         hh = samplehidden(mixrbm, vv)
      end
   end

   impweights
end


"""
    aisimportanceweights(rbm1, rbm2; ...)
Computes the importance weights for estimating the ratio Z2/Z1 of the partition
functions of the two given RBMs.
Implements the procedure described in section 4.1.2 of [Salakhutdinov, 2008].
"""
function aisimportanceweights(rbm1::BernoulliRBM, rbm2::BernoulliRBM;
      ntemperatures::Int = 100,
      beta::Array{Float64,1} = collect(0:(1/ntemperatures):1),
      nparticles::Int = 100,
      burnin::Int = 5)

   nvisible = length(rbm1.visbias)

   if length(rbm2.visbias) != nvisible
      error("The two RBMs must have the same numer of visible units.")
   end

   impweights = ones(nparticles)

   visbiasdiff = rbm2.visbias - rbm1.visbias

   for j=1:nparticles # TODO parallelize
      v = rand(nvisible)

      for k=2:length(beta)

         # Sample next value for v using Gibbs Sampling in the RBM
         for burn=1:burnin
            h1 = bernoulli!(hiddenpotential(rbm1, v, 1 - beta[k]));
            h2 = bernoulli!(hiddenpotential(rbm2, v,     beta[k]));

            v = bernoulli!(sigm(
                (1-beta[k]) * visibleinput(rbm1, h1) +
                   beta[k]  * visibleinput(rbm2, h2)))
         end

         # Multiply ratios of unnormalized probabilities
         hinput1 = hiddeninput(rbm1, v)
         hinput2 = hiddeninput(rbm2, v)
         impweights[j] *= exp.((beta[k]-beta[k-1]) * dot(visbiasdiff, v)) *
               prod((1 + exp.( (1-beta[k])  * hinput1 )) ./
                    (1 + exp.( (1-beta[k-1])* hinput1 ))) *
               prod((1 + exp.(  beta[k]     * hinput2 )) ./
                    (1 + exp.(  beta[k-1]   * hinput2 )))
      end
   end

   impweights
end


"""
    aisimportanceweights(bgrbm; ...)
Computes the importance weights in the Annealed Importance Sampling algorithm
for estimating the ratio of the partition functions of the given
BernoulliGaussianRBM `bgrbm` to the BernoulliGaussianRBM with the same visible
bias and hidden bias but zero weights.
"""
function aisimportanceweights(bgrbm::BernoulliGaussianRBM;
      ntemperatures::Int = 100,
      beta::Array{Float64,1} = collect(0:(1/ntemperatures):1),
      nparticles::Int = 100,
      burnin::Int = 5)

   # reversed RBM has the same partition function
   aisimportanceweights(reversedrbm(bgrbm), ntemperatures = ntemperatures,
         beta = beta, nparticles = nparticles, burnin = burnin)
end


"""
    aisimportanceweights(b2brbm; ...)
Computes the importance weights in the Annealed Importance Sampling algorithm
for estimating the ratio of the partition functions of the given
Binomial2BernoulliRBM `b2brbm` to the Binomial2BernoulliRBM with same visible
and hidden bias.
"""
function aisimportanceweights(b2brbm::Binomial2BernoulliRBM;
      ntemperatures::Int = 100,
      beta::Array{Float64,1} = collect(0:(1/ntemperatures):1),
      nparticles::Int = 100,
      burnin::Int = 5)

   # compute importance weights for BernoulliRBM with duplicated weights and
   # visible bias
   rbm = BernoulliRBM(vcat(b2brbm.weights, b2brbm.weights),
         vcat(b2brbm.visbias, b2brbm.visbias), b2brbm.hidbias)

   aisimportanceweights(rbm;
         ntemperatures = ntemperatures, beta = beta,
         nparticles = nparticles, burnin = burnin)
end


"""
    aisimportanceweights(gbrbm; ...)
Computes the importance weights in the Annealed Importance Sampling algorithm
for estimating the ratio of the partition functions of the given
GaussianBernoulliRBM `gbrbm` to the GaussianBernoulliRBM with same hidden and
visible biases and same standard deviation but with zero weights.
"""
function aisimportanceweights(gbrbm::GaussianBernoulliRBM;
      ntemperatures::Int = 100,
      beta::Array{Float64,1} = collect(0:(1/ntemperatures):1),
      nparticles::Int = 100,
      burnin::Int = 5)

   logimpweights = zeros(nparticles)

   mixrbm = deepcopy(gbrbm)

   for j=1:nparticles
      h = bernoulli!(sigm(gbrbm.hidbias))

      for k=2:(ntemperatures + 1)

         wh = gbrbm.weights*h
         logimpweights[j] += (beta[k] - beta[k-1]) *
               sum(0.5 * wh.^2 + gbrbm.visbias ./gbrbm.sd .* wh)

         # Gibbs transition
         mixrbm.weights = gbrbm.weights * sqrt(beta[k])
         mixrbm.sd = gbrbm.sd / sqrt(beta[k])
         for burn = 1:burnin
            v = samplevisible(mixrbm, h)
            h = samplehidden(mixrbm, v)
         end

      end
   end

   exp.(logimpweights)
end


"""
    aisimportanceweights(dbm; ...)
Computes the importance weights in the Annealed Importance Sampling algorithm
for estimating the ratio of the partition functions of the given `dbm` to the
base-rate DBM with all weights being zero and all biases equal to the biases of
the `dbm`.
Implements algorithm 4 in [Salakhutdinov+Hinton, 2012].
"""
function aisimportanceweights(dbm::BasicDBM;
      ntemperatures::Int = 100,
      beta::Array{Float64,1} = collect(0:(1/ntemperatures):1),
      nparticles::Int = 100,
      burnin::Int = 5)

   impweights = ones(nparticles)
   # Todo: sample from null model, which has changed
   particles = initparticles(dbm, nparticles, biased = true)
   nlayers = length(particles)

   # for performance reasons: preallocate input and combine biases
   input1 = deepcopy(particles)
   input2 = deepcopy(particles)
   biases = combinedbiases(dbm)
   mixdbm = deepcopy(dbm)

   for k = 2:length(beta)
      # calculate importance weights
      aisupdateimportanceweights!(impweights, input1, input2,
            beta[k], beta[k-1], dbm, biases, particles)

      # Gibbs transition
      copyannealed!(mixdbm, dbm, beta[k])
      gibbssample!(particles, mixdbm, burnin)
   end

   impweights
end


"""
    aisprecision(r, aissd, sdrange)
Returns the differences of the estimated logratio `r` to the lower
and upper bound of the range defined by the multiple `sdrange`
of the standard deviation of the ratio's estimator `aissd`.
"""
function aisprecision(r::Float64, aissd::Float64, sdrange::Float64 = 1.0)

   if r - sdrange*aissd <= 0 # prevent domainerror
      diffbottom = -Inf
   else
      diffbottom = log(r - sdrange*aissd) - log(r)
   end

   difftop = log(r + sdrange*aissd) - log(r)

   diffbottom, difftop
end


"""
    aisprecision(impweights, sdrange)
"""
function aisprecision(impweights::Array{Float64,1}, sdrange::Float64 = 1.0)
   aisprecision(mean(impweights), aisstandarddeviation(impweights), sdrange)
end

"
Computes the standard deviation of the AIS estimator
(eq 4.10 in [Salakhutdinov+Hinton, 2012]) given the importance weights.
"
function aisstandarddeviation(impweights::Array{Float64,1})
   aissd = sqrt(var(impweights) / length(impweights))
end


function aisunnormalizedprobratios(rbm::BernoulliRBM,
      hh::Matrix{Float64},
      temperature1::Float64,
      temperature2::Float64)

   weightsinput = hh * rbm.weights'
   vec(prod(
         (1 + exp.(broadcast(+, temperature1 * weightsinput, rbm.visbias'))) ./
         (1 + exp.(broadcast(+, temperature2 * weightsinput, rbm.visbias'))), 2))
end

function aisunnormalizedprobratios(rbm::Binomial2BernoulliRBM,
      hh::Matrix{Float64},
      temperature1::Float64,
      temperature2::Float64)

   weightsinput = hh * rbm.weights'
   vec(prod(
         (1 + exp.(broadcast(+, temperature1 * weightsinput, rbm.visbias'))) ./
         (1 + exp.(broadcast(+, temperature2 * weightsinput, rbm.visbias'))), 2).^2)
end

function aisunnormalizedprobratios(gbrbm::GaussianBernoulliRBM,
      hh::Matrix{Float64},
      temperature1::Float64,
      temperature2::Float64)

   wht = hh * gbrbm.weights'
   vec(exp.((temperature1 - temperature2) * sum(
         0.5 * wht.^2 + broadcast(.*, wht, (gbrbm.visbias ./ gbrbm.sd)'), 2)))
end


"""
Updates the importance weights `impweights` in AIS by multiplying the ratio of
unnormalized probabilities of the states of the odd layers in the BasicDBM
`dbm`. The activation states of the DBM's nodes are given by the `particles`.
For performance reasons, the biases are specified separately
"""
function aisupdateimportanceweights!(impweights,
      input1::Particles, input2::Particles,
      temperature1::Float64, temperature2::Float64,
      dbm::BasicDBM,
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
         for j in eachindex(impweights)
            impweights[j] *=
                  (1 + exp.(input1[i][j,n])) / (1 + exp.(input2[i][j,n]))
         end
      end
   end
end


"""
    copyannealed!(annealedrbm, rbm, temperature)
Copies all parameters that are to be annealed from the RBM `rbm` to the RBM
`annealedrbm` and anneals them with the given `temperature`.
"""
function copyannealed!(annealedrbm::AbstractRBM,
      rbm::AbstractRBM, temperature::Float64)

   annealedrbm.weights .= rbm.weights
   annealedrbm.weights .*= temperature
end

function copyannealed!(annealedrbm::GaussianBernoulliRBM,
         gbrbm::GaussianBernoulliRBM, temperature::Float64)

   annealedrbm.weights .= gbrbm.weights
   annealedrbm.sd .= gbrbm.sd
   annealedrbm.weights .*= sqrt(temperature)
   annealedrbm.sd ./= sqrt(temperature)
end

function copyannealed!{T<:AbstractRBM}(annealedrbms::Vector{T}, rbms::Vector{T},
      temperature::Float64)

   for i in eachindex(rbms)
      copyannealed!(annealedrbms[i], rbms[i], temperature)
   end
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


"
Computes the energy of the combination of activations, given by the particle
`u`, in the DBM.
"
function energy(dbm::BasicDBM, u::Particle)
   energy = [0.0]
   for i = 1:length(dbm)
      energy -= u[i]'*dbm[i].weights*u[i+1] + dbm[i].visbias'*u[i] + dbm[i].hidbias'*u[i+1]
   end
   energy[1]
end

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
function exactloglikelihood(dbm::BasicDBM, x::Matrix{Float64},
      logz = exactlogpartitionfunction(dbm))

   nsamples = size(x, 1)
   combinedbiases = BMs.combinedbiases(dbm)
   uodd = initcombinationoddlayersonly(dbm)

   logp = 0.0
   for j = 1:nsamples
      uodd[1] = vec(x[j,:])

      p = 0.0
      while true
         p += unnormalizedproboddlayers(dbm, uodd, combinedbiases)

         # next combination of hidden nodes' activations
         next!(uodd[2:end]) || break
      end

      logp += log(p)
   end

   logp /= nsamples
   logp -= logz
   logp
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
function exactlogpartitionfunction(dbm::BasicDBM)
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

   # Initialize particles for hidden layers with odd indexes
   hodd = Particle(round(Int, nhiddenlayers / 2, RoundUp))
   for i = eachindex(hodd)
      hodd[i] = zeros(nunits[2i])
   end

   # Calculate the unnormalized probabilities for all combinations of nodes
   # in the hidden layers with odd indexes
   z = 0.0
   biases = combinedbiases(dbm)
   nintermediatelayerstobesummedout = div(nhiddenlayers - 1, 2)
   while true
      pun = prod(1 + exp.(visibleinput(dbm[1], hodd[1])))
      for i = 1:nintermediatelayerstobesummedout
         pun *= prod(1 + exp.(
               hiddeninput(dbm[2i], hodd[i]) + visibleinput(dbm[2i+1], hodd[i+1])))
      end
      if nhiddenlayers % 2 == 0
         pun *= prod(1 + exp.(hiddeninput(dbm[end], hodd[end])))
      end
      for i = 1:length(hodd)
         pun *= exp.(dot(biases[2i], hodd[i]))
      end

      z += pun

      # next combination of odd hidden layers' nodes
      next!(hodd) || break
   end
   log(z)
end


"""
    freeenergy(rbm, x)
Computes the average free energy of the samples in the dataset `x` for the
AbstractRBM `rbm`.
"""
function freeenergy(rbm::BernoulliRBM, x::Matrix{Float64})
   nsamples = size(x,1)
   freeenergy = 0.0
   for j = 1:nsamples
      v = vec(x[j,:])
      freeenergy -= dot(rbm.visbias, v) + sum(log.(1 + exp.(hiddeninput(rbm, v))))
   end
   freeenergy /= nsamples
   freeenergy
end

function freeenergy(rbm::BernoulliRBM, v::Vector{Float64})
   - dot(rbm.visbias, v) - sum(log.(1 + exp.(hiddeninput(rbm, v))))
end

function freeenergy(b2brbm::Binomial2BernoulliRBM, x::Matrix{Float64})
   nsamples = size(x,1)
   freeenergy = 0.0
   for j = 1:nsamples
      v = vec(x[j,:])
      # To get probabilities for 0/1/2-valued v, multiply probabilities for v's
      # with 2^(number of 1s in v), because each v can be represented by
      # this number of combinations in the 00/01/10/11 space, all having equal
      # probability.
      freeenergy -= sum(v .== 1.0) * log(2) +
            dot(b2brbm.visbias, v) + sum(log.(1 + exp.(hiddeninput(b2brbm, v))))
   end
   freeenergy /= nsamples
   freeenergy
end

function freeenergy(gbrbm::GaussianBernoulliRBM, x::Matrix{Float64})
   # For derivation of formula for free energy of Gaussian-Bernoulli-RBMs,
   # see [Krizhevsky, 2009], page 15.
   nsamples = size(x,1)
   nhidden = length(gbrbm.hidbias)

   freeenergy = 0.0
   for j = 1:nsamples
      v = vec(x[j,:])
      for k = 1:nhidden
         freeenergy -=
               log(1 + exp(gbrbm.hidbias[k] + dot(gbrbm.weights[:,k], v ./ gbrbm.sd)))
      end
      freeenergy += 0.5 * sum(((v - gbrbm.visbias) ./ gbrbm.sd).^2)
   end
   freeenergy /= nsamples
   freeenergy
end

function freeenergy(bgrbm::BernoulliGaussianRBM, x::Matrix{Float64})
   nsamples = size(x,1)
   nhidden = length(bgrbm.hidbias)

   freeenergy = 0.0
   for j = 1:nsamples
      v = vec(x[j,:])
      wtv = bgrbm.weights'*v
      freeenergy -= dot(bgrbm.hidbias, wtv) + 0.5 * sum(wtv.^2) + dot(bgrbm.visbias, v)
   end
   freeenergy /= nsamples
   freeenergy -= nhidden / 2 * log(2pi)
   freeenergy
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

"
Creates and zero-initializes a particle for layers with odd indexes
in the `dbm`.
"
function initcombinationoddlayersonly(dbm)
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
function loglikelihood(dbm::BasicDBM, x::Matrix{Float64};
      ntemperatures::Int = 100,
      beta::Array{Float64,1} = collect(0:(1/ntemperatures):1),
      nparticles::Int = 100,
      burnin::Int = 5)

   nsamples = size(x,1)

   r = mean(aisimportanceweights(dbm; ntemperatures = ntemperatures,
         beta = beta, nparticles = nparticles, burnin = burnin))
   logz = logpartitionfunction(dbm, r)

   hiddbm = deepcopy(dbm[2:end])

   logp = 0.0
   for j = 1:nsamples
      v = vec(x[j,:])
      logp += dot(dbm[1].visbias, v)
      hiddbm[1].visbias = hiddeninput(dbm[1], v) + dbm[2].visbias
      r = mean(aisimportanceweights(hiddbm; ntemperatures = ntemperatures,
            beta = beta, nparticles = nparticles, burnin = burnin))
      logp += logpartitionfunction(hiddbm, r)
   end

   logp /= nsamples
   logp -= logz
   logp
end


"""
    loglikelihooddiff(rbm1, rbm2, x)
    loglikelihooddiff(rbm1, rbm2, x, impweights)
Computes difference of the loglikelihood functions of the two RBMs on the data
matrix `x`, averaged over the samples.
For this purpose, the partition function ratio Z2/Z1 is estimated by AIS, if
the importance weights are not given by parameter `impweights`.
The first model is better than the second if the returned value is positive.
"""
function loglikelihooddiff(rbm1::BernoulliRBM, rbm2::BernoulliRBM,
      x::Array{Float64,2},
      impweights::Array{Float64,1} = aisimportanceweights(rbm1, rbm2))

   nsamples = size(x,1)

   lldiff = 0.0

   visbiasdiff = rbm1.visbias - rbm2.visbias

   # calculate difference of sum of unnormalized log probabilities
   for j=1:nsamples
      v = vec(x[j,:])
      lldiff += dot(visbiasdiff, v) +
            sum(log.(1 + exp.(hiddeninput(rbm1, v)))) -
            sum(log.(1 + exp.(hiddeninput(rbm2, v))))
   end

   # average over samples
   lldiff /= nsamples
   r = mean(impweights) # estimator for Z2/Z1
   lldiff += log(r)
   lldiff
end


"""
    logpartitionfunction(bm)
    logpartitionfunction(bm, r)
Calculates the log of the partition function of the Boltzmann Machine `bm`
from the estimator `r`.
`r` is an estimator of the ratio of the `bm`'s partition function Z to the
partition function Z_0 of the reference BM with zero weights but same biases
as the given `bm`. In case of a GaussianBernoulliRBM, the reference model
also has the same standard deviation parameter.
If the estimator `r` is not given as argument, Annealed Importance Sampling
is performed (with default parameters) to get a value for it.
The estimated partition function of the Boltzmann Machine is Z = r * Z_0
with `r` being the mean of the importance weights.
Therefore, the log of the estimated partition function is
log(Z) = log(r) + log(Z_0)
"""
function logpartitionfunction(bm::AbstractBM,
      r::Float64 = mean(aisimportanceweights(bm)))

   logz = log(r) + logpartitionfunctionzeroweights(bm)
end


"""
    logpartitionfunctionzeroweights(bm)
Returns the value of the log of the partition function of the Boltzmann Machine
that results when one sets the weights of `bm` to zero,
and leaves the other parameters (biases) unchanged.
"""
function logpartitionfunctionzeroweights(rbm::BernoulliRBM)
   sum(log.(1 + exp.(rbm.visbias))) + sum(log.(1 + exp.(rbm.hidbias)))
end

function logpartitionfunctionzeroweights(bgrbm::BernoulliGaussianRBM)
   nhidden = length(bgrbm.hidbias)
   nhidden / 2 * log(2pi) + sum(log.(1 + exp.(bgrbm.visbias)))
end

function logpartitionfunctionzeroweights(b2brbm::Binomial2BernoulliRBM)
   2*sum(log.(1 + exp.(b2brbm.visbias))) + sum(log.(1 + exp.(b2brbm.hidbias)))
end

function logpartitionfunctionzeroweights(gbrbm::GaussianBernoulliRBM)
   nvisible = length(gbrbm.visbias)
   logz0 = nvisible / 2 * log(2*pi) + sum(log.(gbrbm.sd)) + sum(log.(1 + exp(gbrbm.hidbias)))
end

function logpartitionfunctionzeroweights(dbm::BasicDBM)
   logz0 = 0.0
   biases = combinedbiases(dbm)
   for i in eachindex(biases)
      logz0 += sum(log.(1 + exp.(biases[i])))
   end
   logz0
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
  or is calculated using the `impweights`.
"""
function logproblowerbound(dbm::BasicDBM,
      x::Array{Float64};
      impweights::Array{Float64,1} = aisimportanceweights(dbm),
      mu::Particles = meanfield(dbm, x),
      logpartitionfunction::Float64 = BMs.logpartitionfunction(dbm, mean(impweights)))

   nsamples = size(mu[1], 1)
   nrbms  = length(dbm)

   lowerbound = 0.0
   for j=1:nsamples # TODO parallelize

      for i=1:nrbms
         v = vec(mu[i][j,:])   # visible units of i'th RBM
         h = vec(mu[i+1][j,:]) # hidden units of i'th RBM

         # add energy
         lowerbound +=
               dot(v, dbm[i].visbias) + dot(h, dbm[i].hidbias) + dot(v, dbm[i].weights * h)

         # add entropy of approximate posterior Q
         h = h[(h .> 0.0) .& (h .< 1.0)]
         lowerbound += - dot(h, log.(h)) - dot(1-h, log.(1-h))
      end

   end

   lowerbound /= nsamples
   lowerbound -= logpartitionfunction
   lowerbound
end


"""
    next!(combination)
Sets the vector `combination`, containing a sequence of the values 0.0 and 1.0,
to the next combination of 0.0s and 1.0s.
Returns false if the new combination consists only of zeros; true otherwise.
"""
function next!(combination::Vector{Float64})
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

function nunits(dbm::AbstractDBM)
   nrbms = length(dbm)
   if nrbms == 0
      error("Nodes and layers not defined in empty DBM")
   end
   nlayers = length(dbm) + 1
   nu = Array{Int,1}(nlayers)
   for i = 1:nrbms
      nu[i] = length(dbm[i].visbias)
   end
   nu[nlayers] = length(dbm[nlayers-1].hidbias)
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


"""
    reconstructionerror(rbm, x)
Computes the mean reconstruction error of the RBM on the dataset `x`.
"""
function reconstructionerror(rbm::AbstractRBM,
      x::Array{Float64,2},
      upfactor::Float64 = 1.0,
      downfactor::Float64 = 1.0)

   nsamples = size(x,1)
   reconstructionerror = 0.0

   for sample = 1:nsamples
      v = vec(x[sample,:])
      hmodel = hiddenpotential(rbm, v, upfactor)
      vmodel = visiblepotential(rbm, hmodel, downfactor)
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

function reverseddbm(dbm::BasicDBM)
   revdbm = reverse(dbm)
   map!(reversedrbm, revdbm, revdbm)
end


"""
    samplefrequencies(x)
Returns a dictionary containing the rows of the data set `x` as keys and their
relative frequencies as values.
"""
function samplefrequencies{T}(x::Array{T,2})
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


"""
    sampleparticles(bm, nparticles, burnin)
Samples in the Boltzmann Machine model `bm` by running `nparticles` parallel,
randomly initialized Gibbs chains for `burnin` steps.
Returns particles containing `nparticles` generated samples.
See also: `Particles`.
"""
function sampleparticles(dbm::AbstractDBM, nparticles::Int, burnin::Int = 10)
   particles = initparticles(dbm, nparticles)
   gibbssample!(particles, dbm, burnin)
   particles
end

function sampleparticles(rbm::AbstractRBM, nparticles::Int, burnin::Int = 10)
   particles = Particles(2)
   particles[2] = rand([0.0 1.0], nparticles, length(rbm.hidbias))

   for i=1:burnin
      particles[1] = samplevisible(rbm, particles[2])
      particles[2] = samplehidden(rbm, particles[1])
   end
   particles
end

function sampleparticles(gbrbm::GaussianBernoulliRBM, nparticles::Int, burnin::Int = 10)
   particles = invoke(sampleparticles, (AbstractRBM,Int,Int), gbrbm, nparticles, burnin-1)
   # do not sample in last step to avoid that the noise dominates the data
   particles[1] = visiblepotential(gbrbm, particles[2])
   particles
end


"
Computes the unnormalized probability of the nodes in layers with odd indexes,
i. e. p*(v, h2, h4, ...).
"
function unnormalizedproboddlayers(dbm::BasicDBM, uodd::Particle,
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
