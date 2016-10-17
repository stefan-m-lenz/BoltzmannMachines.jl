"""
    aisimportanceweights(rbm; ...)
Computes the importance weights for estimating the ratio of the partition
functions of the given `rbm` to the RBM with zeros weights, zero hidden bias
and visible bias `visbias` using the Annealed Importance Sampling algorithm,
like described in section 4.1.3 of [Salakhutdinov, 2008].
`visbias` can be given as optional keyword argument and is by default the
visible bias of the given `rbm`.

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
      visbias::Array{Float64,1} = rbm.visbias,
      ntemperatures::Int = 100,
      beta::Array{Float64,1} = collect(0:(1/ntemperatures):1),
      nparticles::Int = 100,
      burnin::Int = 10)

   impweights = ones(nparticles)

   visbiasdiff = rbm.visbias - visbias

   for j=1:nparticles
      v = bernoulli!(sigm(visbias))

      for k=2:(ntemperatures + 1)

         hinput = hiddeninput(rbm, v)
         impweights[j] *= exp((beta[k] - beta[k-1]) * dot(visbiasdiff, v)) *
               prod((1 + exp(beta[k]   * hinput)) ./
                    (1 + exp(beta[k-1] * hinput)))

         # Gibbs transition
         for burn=1:burnin
            h = bernoulli!(hiddenpotential(rbm, v, beta[k]))
            vinput = beta[k] * visibleinput(rbm, h) + (1 - beta[k]) * visbias
            v = bernoulli!(sigm(vinput))
         end
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
      burnin::Int = 10)

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
         impweights[j] *= exp((beta[k]-beta[k-1]) * dot(visbiasdiff, v)) *
               prod((1 + exp( (1-beta[k])  * hinput1 )) ./
                    (1 + exp( (1-beta[k-1])* hinput1 ))) *
               prod((1 + exp(  beta[k]     * hinput2 )) ./
                    (1 + exp(  beta[k-1]   * hinput2 )))
      end
   end

   impweights
end


"""
    aisimportanceweights(gbrbm; ...)
Computes the importance weights using the Annealed Importance Sampling algorithm
for estimating the ratio of the partition functions of the given
GaussianBernoulliRBM `gbrbm` to the GaussianBernoulliRBM with same hidden and
visible biases and same standard deviation but with zero weights.
"""
function aisimportanceweights(gbrbm::GaussianBernoulliRBM;
      ntemperatures::Int = 100,
      beta::Array{Float64,1} = collect(0:(1/ntemperatures):1),
      nparticles::Int = 100,
      burnin::Int = 10)

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

   exp(logimpweights)
end

# TODO document or remove
function aisimportanceweights(bgrbm::BernoulliGaussianRBM;
      ntemperatures::Int = 100,
      beta::Array{Float64,1} = collect(0:(1/ntemperatures):1),
      nparticles::Int = 100,
      burnin::Int = 10)

   aisimportanceweights(reversedrbm(bgrbm), ntemperatures = ntemperatures,
         beta = beta, nparticles = nparticles, burnin = burnin)
end


"""
    aisimportanceweights(dbm; ...)
Computes the importance weights using the Annealed Importance Sampling algorithm
for estimating the ratio of the partition functions of the given `dbm` to the
base-rate DBM with all weights being zero.
Implements algorithm 4 in [Salakhutdinov+Hinton, 2012].
"""
function aisimportanceweights(dbm::BasicDBM;
      ntemperatures::Int = 100,
      beta::Array{Float64,1} = collect(0:(1/ntemperatures):1),
      nparticles::Int = 100,
      burnin::Int = 10)

   nrbms = length(dbm)
   nlayers = nrbms + 1

   impweights = ones(nparticles)
   particles = initparticles(dbm, nparticles)

   for k=2:length(beta)

      for j=1:nparticles # TODO parallelize

         oddbiasenergy = dot(dbm[1].visbias, vec(particles[1][j,:]))
         for i = 3:2:nrbms
            oddbiasenergy += dot(dbm[i].visbias + dbm[i-1].hidbias, vec(particles[i][j,:]))
         end
         if nlayers % 2 != 0
            oddbiasenergy += dot(dbm[nrbms].hidbias, vec(particles[nlayers][j,:]))
         end
         impweights[j] *= exp((beta[k] - beta[k-1]) * oddbiasenergy)

         # Analytically sum out all uneven hidden layers.
         for i = 1:2:nrbms
            if i == nrbms
               # If the number of layers is even,
               # the last layer has also to be summed out.
               # It gets its input only from the layer below.
               hinput = BMs.hiddeninput(dbm[i], vec(particles[i][j,:]))
            else
               # All other hidden layers get their input from the
               # layer below and the one above.
               hinput = BMs.hiddeninput(dbm[i], vec(particles[i][j,:])) +
                  BMs.visibleinput(dbm[i+1], vec(particles[i+2][j,:]))
            end
            impweights[j] *= prod(
                     (1 + exp(beta[k]   * hinput)) ./
                     (1 + exp(beta[k-1] * hinput)))
         end

      end

      gibbssample!(particles, dbm, burnin, beta[k])

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

function exactloglikelihood(mvdbm::MultivisionDBM, x::Matrix{Float64},
      logz = exactlogpartitionfunction(mvdbm))

   nsamples = size(x, 1)
   combinedbiases = BMs.combinedbiases(mvdbm.hiddbm)

   # combinations of hidden layers with odd index (i. e. h1, h3, ...)
   hodd = initcombinationoddlayersonly(mvdbm.hiddbm)

   logpx = 0.0
   for j = 1:nsamples
      v = vec(x[j,:])
      px = 0.0
      while true
         pun = unnormalizedproboddlayers(mvdbm.hiddbm, hodd, combinedbiases)

         firstlayerenergy = 0.0
         for i = eachindex(mvdbm.visrbms)
            firstlayerenergy += energy(mvdbm.visrbms[i],
                  v[mvdbm.visrbmsvisranges[i]],
                  hodd[1][mvdbm.visrbmshidranges[i]])
         end
         pun *= exp(-firstlayerenergy)

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
         z += exp(-freeenergy(rbm, v))
         next!(v) || break
      end
   else
      h = zeros(nhidden)
      revrbm = reversedrbm(rbm)
      while true
         z += exp(-freeenergy(revrbm, h))
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
   log(z) + nvisible/2 * log(2*pi) + sum(log(gbrbm.sd))
end

# TODO document or remove
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
      pun = prod(1 + exp(visibleinput(dbm[1], hodd[1])))
      for i = 1:nintermediatelayerstobesummedout
         pun *= prod(1 + exp(
               hiddeninput(dbm[2i], hodd[i]) + visibleinput(dbm[2i+1], hodd[i+1])))
      end
      if nhiddenlayers % 2 == 0
         pun *= prod(1 + exp(hiddeninput(dbm[end], hodd[end])))
      end
      for i = 1:length(hodd)
         pun *= exp(dot(biases[2i], hodd[i]))
      end

      z += pun

      # next combination of odd hidden layers' nodes
      next!(hodd) || break
   end
   log(z)
end


"""
    exactlogpartitionfunction(mvdbm)
Calculates the log of the partition function of the MultivisionDBM `mvdbm`
exactly.
The execution time grows exponentially with the total number of nodes in hidden
layers with odd indexes (i. e. h1, h3, ...).
"""
function exactlogpartitionfunction(mvdbm::MultivisionDBM)
   hodd = initcombinationoddlayersonly(mvdbm.hiddbm)
   combinedbiases = BMs.combinedbiases(mvdbm.hiddbm)
   z = 0.0
   while true
      pun = unnormalizedproboddlayers(mvdbm.hiddbm, hodd, combinedbiases)

      for i = 1:length(mvdbm.visrbms)
         pun *= unnormalizedprobhidden(mvdbm.visrbms[i],
               hodd[1][mvdbm.visrbmshidranges[i]])
      end

      z += pun
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
      freeenergy -= dot(rbm.visbias, v) + sum(log(1 + exp(hiddeninput(rbm, v))))
   end
   freeenergy /= nsamples
   freeenergy
end

function freeenergy(rbm::BernoulliRBM, v::Vector{Float64})
   - dot(rbm.visbias, v) - sum(log(1 + exp(hiddeninput(rbm, v))))
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
            dot(b2brbm.visbias, v) + sum(log(1 + exp(hiddeninput(b2brbm, v))))
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
function loglikelihood(dbm::BasicDBM, x::Array{Float64,2};
      ntemperatures::Int = 100,
      beta::Array{Float64,1} = collect(0:(1/ntemperatures):1),
      nparticles::Int = 100,
      burnin::Int = 10)

   nsamples = size(x,1)

   r = mean(aisimportanceweights(dbm; ntemperatures = ntemperatures,
         beta = beta, nparticles = nparticles, burnin = burnin))
   logz = logpartitionfunction(dbm, r)

   # DBM consisting only of hidden layers, with visible bias incorporating v
   hdbm = deepcopy(dbm[2:end])

   logp = 0.0
   for j = 1:nsamples
      v = vec(x[j,:])
      logp += dot(dbm[1].visbias, v)
      hdbm[1].visbias = hiddeninput(dbm[1], v) + dbm[2].visbias
      r = mean(aisimportanceweights(hdbm; ntemperatures = ntemperatures,
            beta = beta, nparticles = nparticles, burnin = burnin))
      logp += log(r)
   end

   logp /= nsamples
   logp += log(2)*sum(nunits(hdbm)) # log(Z_0) for hdbm
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
            sum(log(1 + exp(hiddeninput(rbm1, v)))) -
            sum(log(1 + exp(hiddeninput(rbm2, v))))
   end

   # average over samples
   lldiff /= nsamples
   r = mean(impweights) # estimator for Z2/Z1
   lldiff += log(r)
   lldiff
end


"""
    logpartitionfunction(rbm)
    logpartitionfunction(rbm, r)
    logpartitionfunction(rbm, visbias, r)
Calculates the log of the partition function of the RBM from the estimator `r`.
`r` is an estimator of the ratio of the RBM's partition function to the
partition function of the RBM with zero weights and visible bias `visbias`, Z_0.
If the estimator `r` is not given as argument, Annealed Importance Sampling
is performed to get a value for it.
By default, `visbias` is the visible bias of the `rbm`.
The estimated partition function of the Boltzmann Machine is Z = r * Z_0
with `r` being the mean of the importance weights.
Therefore, the log of the estimated partition function is
log(Z) = log(r) + log(Z_0)
"""
function logpartitionfunction(rbm::BernoulliRBM,
      r::Float64 = mean(aisimportanceweights(rbm)))

   logpartitionfunction(rbm, rbm.visbias, r)
end

function logpartitionfunction(rbm::BernoulliRBM,
      visbias::Vector{Float64},
      r::Float64 = mean(aisimportanceweights(rbm; visbias = visbias)))

   nhidden = length(rbm.hidbias)
   # Uses equation 43 in [Salakhutdinov, 2008]
   logz = log(r) + log(2)*nhidden + sum(log(1 + exp(visbias)))
end


"""
    logpartitionfunction(gbrbm)
    logpartitionfunction(gbrbm, r)
Calculates the log of the partition function of the GaussianBernoulliRBM `gbrbm`
from the estimator `r`.
`r` is an estimator of the ratio of the `gbrbm`'s partition function to the
partition function of the GBRBM with zero weights, and same
standard deviation and same visible and hidden bias as the given `gbrbm`.
"""
function logpartitionfunction(gbrbm::GaussianBernoulliRBM,
      r::Float64 = mean(aisimportanceweights(gbrbm)))

   nvisible = length(gbrbm.visbias)
   logz0 = nvisible / 2 * log(2*pi) + sum(log(gbrbm.sd)) + sum(log(1 + exp(gbrbm.hidbias)))
   logz = log(r) + logz0
end


"""
    logpartitionfunction(bgrbm)
    logpartitionfunction(bgrbm, r)
Calculates the log of the partition function of the BernoulliGaussianRBM `bgrbm`
from the estimator `r`.
`r` is an estimator of the ratio of the `bgrbm`'s partition function to the
partition function of the BGRBM with zero weights
and same visible and hidden bias as the given `bgrbm`.
"""
function logpartitionfunction(bgrbm::BernoulliGaussianRBM,
      r::Float64 = mean(aisimportanceweights(bgrbm)))

   nhidden = length(bgrbm.hidbias)
   logz0 = nhidden / 2 * log(2pi) + sum(log(1 + exp(bgrbm.visbias)))
   logz = log(r) + logz0
end


"""
    logpartitionfunction(dbm)
    logpartitionfunction(dbm, r)
Calculates the log of the partition function of the DBM from the estimator `r`.
`r` is an estimator of the ratio of the DBM's partition function to the
partition function of the DBM with zero weights and zero biases.
"""
function logpartitionfunction(dbm::BasicDBM,
      r::Float64 = mean(aisimportanceweights(dbm)))

   logz = log(r) + log(2)*sum(nunits(dbm))
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
         h = h[(h .> 0.0) & (h .< 1.0)]
         lowerbound += - dot(h, log(h)) - dot(1-h, log(1-h))
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

function nunits(dbm::BasicDBM)
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
Returns the number of parameters in the Boltzmann Machine.
"""
function nmodelparameters(bm::AbstractBM)
   nunits = BMs.nunits(bm)
   prod(nunits) + sum(nunits)
end

# TODO add nmodelparameters for GaussianBernoulli

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
      reconstructionerror += sum(abs(v - vmodel))
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
   map!(reversedrbm, revdbm)
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
      pun *= prod(1 + exp(
            hiddeninput(dbm[2i-1], uodd[i]) + visibleinput(dbm[2i], uodd[i+1])))
   end
   if nlayers % 2 == 0
      pun *= prod(1 + exp(hiddeninput(dbm[end], uodd[end])))
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
   exp(dot(rbm.hidbias, h)) * prod(1 + exp(visibleinput(rbm, h)))
end

function unnormalizedprobhidden(rbm::Binomial2BernoulliRBM, h::Vector{Float64})
   exp(dot(rbm.hidbias, h)) * prod(1 + exp(visibleinput(rbm, h)))^2
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
   sigmasq = var(x,1)
   mu = vec(mean(x,1))
   loglikelihood = 0.0
   for j = 1:nsamples
      loglikelihood -= sum((x[j,:] - mu).^2 ./ sigmasq)
   end
   loglikelihood /= nsamples
   loglikelihood -= log(2*pi) * nvariables + sum(log(sigmasq))
   loglikelihood /= 2
   loglikelihood
end
