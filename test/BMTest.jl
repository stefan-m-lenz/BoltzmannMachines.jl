module BMTest

import BoltzmannMachines
const BMs = BoltzmannMachines

function randrbm(nvisible, nhidden, factorw = 1.0, factora = 1.0, factorb = 1.0)
   w = factorw*randn(nvisible, nhidden)
   a = factora*rand(nvisible)
   b = factorb*ones(nhidden)
   BMs.BernoulliRBM(w, a, b)
end

function randgbrbm(nvisible, nhidden, factorw = 1.0, factora = 1.0, factorb = 1.0, factorsd = 1.0)
   w = factorw*randn(nvisible, nhidden)
   a = factora*rand(nvisible)
   b = factorb*ones(nhidden)
   sd = factorsd*ones(nvisible)
   BMs.GaussianBernoulliRBM(w, a, b, sd)
end

function randbgrbm(nvisible, nhidden, factorw = 1.0, factora = 1.0, factorb = 1.0)
   w = factorw*randn(nvisible, nhidden)
   a = factora*rand(nvisible)
   b = factorb*ones(nhidden)
   BMs.BernoulliGaussianRBM(w, a, b)
end

function logit(p::Array{Float64})
   log(p./(1-p))
end

function rbmexactloglikelihoodvsbaserate(x::Matrix{Float64}, nhidden::Int)
   a = logit(vec(mean(x,1)))
   nvisible = length(a)
   rbm = BMs.BernoulliRBM(zeros(nvisible, nhidden), a, ones(nhidden))
   baserate = BMs.bernoulliloglikelihoodbaserate(x)
   exactloglik = BMs.loglikelihood(rbm, x, BMs.exactlogpartitionfunction(rbm))
   baserate - exactloglik
end

function bgrbmexactloglikelihoodvsbaserate(x::Matrix{Float64}, nhidden::Int)
   a = logit(vec(mean(x,1)))
   nvisible = length(a)
   bgrbm = BMs.BernoulliGaussianRBM(zeros(nvisible, nhidden), a, ones(nhidden))
   baserate = BMs.bernoulliloglikelihoodbaserate(x)
   exactloglik = BMs.loglikelihood(bgrbm, x, BMs.exactlogpartitionfunction(bgrbm))
   baserate - exactloglik
end

function gbrbmexactloglikelihoodvsbaserate(x::Matrix{Float64}, nhidden::Int)
   a = vec(mean(x,1))
   nvisible = length(a)
   sd = vec(std(x,1))
   gbrbm = BMs.GaussianBernoulliRBM(zeros(nvisible, nhidden), a, ones(nhidden), sd)
   baserate = BMs.gaussianloglikelihoodbaserate(x)
   exactloglik = BMs.loglikelihood(gbrbm, x, BMs.exactlogpartitionfunction(gbrbm))
   baserate - exactloglik
end

function randdbm(nunits)
   nrbms = length(nunits) - 1
   dbm = BMs.DBMParam(nrbms)
   for i = 1:nrbms
      dbm[i] = randrbm(nunits[i], nunits[i+1])
   end
   dbm
end

"
Calculates the exact value for the partition function and an estimate with AIS
and return the difference between the logs of the two values
for a given RBM.
"
function aisvsexact(rbm::BMs.AbstractRBM, ntemperatures::Int = 100,
      nparticles::Int = 100)

   # Test log partition funtion estimation vs exact calculation
   exact = BMs.exactlogpartitionfunction(rbm)
   impweights = BMs.aisimportanceweights(rbm,
         ntemperatures = ntemperatures, nparticles = nparticles)
   r = mean(impweights)
   estimated = BMs.logpartitionfunction(rbm, r)
   println("Range of 2 * sd aroung AIS-estimated log partition function")
   println(BMs.aisprecision(impweights, 2.0))
   println("Difference between exact log partition function and AIS-estimated one")
   println("in percent of log of exact value")
   println((exact - estimated)/exact*100)
end

function aisvsexact(dbm::BMs.DBMParam, ntemperatures = 100, nparticles = 100)
   nrbms = length(dbm)

   impweights = BMs.aisimportanceweights(dbm, ntemperatures = ntemperatures,
      nparticles = nparticles)

   r = mean(impweights)
   println("Range of 2 * sd aroung AIS-estimated log partition function")
   println(BMs.aisprecision(impweights, 2.0))
   exact = BMs.exactlogpartitionfunction(dbm)
   estimated = BMs.logpartitionfunction(dbm, r)
   println("Difference between exact log partition function and AIS-estimated one")
   println("in percent of log of exact value")
   println((exact - estimated)/exact*100)
   # TODO loglikelihood base-rate vs loglikelihood dbm
end

function exactlogpartitionfunctioninefficient(dbm::BMs.DBMParam)
   nlayers = length(dbm) + 1
   u = BMs.initcombination(dbm)
   z = 0.0
   while true
      z += exp(-BMs.energy(dbm, u))
      BMs.next!(u) || break
   end
   log(z)
end

end