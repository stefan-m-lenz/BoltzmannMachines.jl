module BMTest

using Base.Test
import BoltzmannMachines
const BMs = BoltzmannMachines

function createsamples(nsamples::Int, nvariables::Int, samerate=0.7)
   x = round(rand(nsamples,nvariables))
   samerange = 1:round(Int, samerate*nsamples)
   x[samerange,3] = x[samerange,2] = x[samerange,1]
   x = x[randperm(nsamples),:] # shuffle lines
   x
end

function randrbm(nvisible, nhidden, factorw = 1.0, factora = 1.0, factorb = 1.0)
   w = factorw*randn(nvisible, nhidden)
   a = factora*rand(nvisible)
   b = factorb*(0.5 - rand(nhidden))
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

function exactlogpartitionfunctionwithoutsummingout(dbm::BMs.DBMParam)
   nlayers = length(dbm) + 1
   u = BMs.initcombination(dbm)
   z = 0.0
   while true
      z += exp(-BMs.energy(dbm, u))
      BMs.next!(u) || break
   end
   log(z)
end

function testsummingoutforexactloglikelihood(nunits::Vector{Int})
   x = BMTest.createsamples(1000, nunits[1]);
   dbm = BMs.stackrbms(x, nhiddens = nunits[2:end],
         epochs = 50, predbm = true, learningrate = 0.001);
   dbm = BMs.fitbm(x, dbm,
         learningrates = [0.02*ones(10); 0.01*ones(10); 0.001*ones(10)],
         epochs = 30);
   logz = BMs.exactlogpartitionfunction(dbm)
   @test_approx_eq(BMs.exactloglikelihood(dbm, x, logz),
         exactloglikelihoodwithoutsummingout(dbm, x, logz))
end

function exactloglikelihoodwithoutsummingout(dbm::BMs.DBMParam, x::Array{Float64,2},
      logz = BMs.exactlogpartitionfunction(dbm))

   nsamples = size(x,1)
   nlayers = length(dbm) + 1

   u = BMs.initcombination(dbm)
   logp = 0.0
   for j = 1:nsamples
      u[1] = vec(x[j,:])

      p = 0.0
      while true
         p += exp(-BMs.energy(dbm, u))

         # next combination of hidden nodes' activations
         BMs.next!(u[2:end]) || break
      end

      logp += log(p)
   end

   logp /= nsamples
   logp -= logz
   logp
end

"
Tests whether the exact loglikelihood of a MultivisionDBM with two visible
input layers of Bernoulli units is equal to the loglikelihood of the DBM
where the two visible RBMs are joined to one RBM.
"
function testexactloglikelihood_bernoullimvdbm(nunits::Vector{Int})

   nvisible1 = floor(Int, nunits[1]/2)
   nvisible2 = ceil(Int, nunits[1]/2)
   nhidden1 = floor(Int, nunits[2]/2)
   nhidden2 = ceil(Int, nunits[2]/2)

   rbm1 = randrbm(nvisible1, nhidden1)
   rbm2 = randrbm(nvisible2, nhidden2)

   hiddbm = randdbm(nunits[2:end])
   mvdbm = BMs.MultivisionDBM([rbm1;rbm2])
   mvdbm.hiddbm = hiddbm

   jointrbm = BMs.joinrbms(rbm1, rbm2)
   dbm = BMs.BernoulliRBM[jointrbm, hiddbm...]

   nsamples = 25
   x = hcat(createsamples(nsamples, nvisible1),
      createsamples(nsamples, nvisible2))

   @test_approx_eq(BMs.exactloglikelihood(dbm, x),
         BMs.exactloglikelihood(mvdbm, x))
end

"
Tests whether the log-likelihood of Binomial2BernoulliRBMs, and of
MultivisionDBMs with Binomial2BernoulliRBMs in the first layer, is approximately
equal to the empirical loglikelihood of data generated by the models.
"
function testexactloglikelihood_b2brbm()
   x1 = createsamples(100, 4) + createsamples(100, 4)
   x2 = createsamples(100, 4)
   x = hcat(x1, x2)
   b2brbm = BMs.fitrbm(x1, rbmtype = BMs.Binomial2BernoulliRBM, epochs = 30,
         nhidden = 4, learningrate = 0.001)
   rbm = BMs.fitrbm(x2, rbmtype = BMs.BernoulliRBM, epochs = 30,
         nhidden = 3, learningrate = 0.001)

   emploglik = BMs.empiricalloglikelihood(b2brbm, x1, 1000000)
   exactloglik = BMs.exactloglikelihood(b2brbm, x1)
   @test abs((exactloglik - emploglik)/exactloglik) < 0.01

   mvdbm = BMs.MultivisionDBM([b2brbm, rbm]);
   BMs.addlayer!(mvdbm, x, nhidden = 5);
   mvdbm = BMs.fitbm(x, mvdbm; epochs = 20);
   exactloglik = BMs.exactloglikelihood(mvdbm, x)
   emploglik = BMs.empiricalloglikelihood(mvdbm, x, 1000000)
   @test abs((exactloglik - emploglik)/exactloglik) < 0.01
end

end