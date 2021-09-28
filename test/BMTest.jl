module BMTest

using DelimitedFiles
using LinearAlgebra
using Random
using Test
using Statistics

import BoltzmannMachines
const BMs = BoltzmannMachines


macro check(expr)
   msg = string(expr) # TODO: print vales
   quote
      if !($(esc(expr)))
         @warn "Check failed:\n\t" * $msg
         return false
      else
         true # no return
      end
   end
end


function createsamples(nsamples::Int, nvariables::Int, samerate::Float64 = 0.7)

   x = round.(rand(nsamples, nvariables))
   samerows = ((x[:,1] .== x[:,2]) .& (x[:,2] .== x[:,3]))
   sameratestart = count(samerows) / nsamples

   sameratediff = samerate - sameratestart

   if sameratediff > 0
      addedsamerange = findall(.!samerows)
      shuffle!(addedsamerange)
      nsameadded = round(Int, sameratediff * nsamples)
      addedsamerange = addedsamerange[1:nsameadded]
      x[addedsamerange, 3] = x[addedsamerange, 2] = x[addedsamerange, 1]
   else
      subtractedsamerange = findall(samerows)
      shuffle!(subtractedsamerange)
      nsamesubtracted = round(Int, - sameratediff * nsamples)
      subtractedsamerange = subtractedsamerange[1:nsamesubtracted]
      for row in subtractedsamerange
         while ((x[row,1] == x[row,2]) & (x[row,2] == x[row,3]))
            xrow = view(x, row, 1:3)
            rand!(xrow)
            map!(round, xrow, xrow)
         end
      end
   end
   x
end


function createsamples_categorical(nsamples::Int, nvariables::Int,
      samerates::Vector{Float64} = [0.2; 0.2; 0.4])

   x = zeros(nsamples, nvariables)
   nsames = Int.(floor.(nsamples .* samerates))
   sampleranges = BMs.ranges(nsames)
   for i in eachindex(samerates)
      samplerange = sampleranges[i]
      x[samplerange, 2] .= x[samplerange, 3] .= x[samplerange, 1] .= i - 1
   end

   notsamerange = (sampleranges[end][end] + 1):nsamples
   ncategories = length(samerates) - 1
   for i in 1:ncategories
      x[notsamerange, 1:3] .+= round.(rand(length(notsamerange), 3))
      x[:, 4:end] .+= round.(rand(nsamples, nvariables - 3))
   end
   x = x[shuffle!(collect(1:size(x,1))), :]
   x
end


function randrbm(nvisible, nhidden, factorw = 1.0, factora = 1.0, factorb = 1.0)
   w = factorw*randn(nvisible, nhidden)
   a = factora*rand(nvisible)
   b = factorb*(0.5 .- rand(nhidden))
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

function randdbm(nunits)
   nrbms = length(nunits) - 1
   dbm = BMs.BasicDBM(undef, nrbms)
   for i = 1:nrbms
      dbm[i] = randrbm(nunits[i], nunits[i+1])
   end
   dbm
end


"""
Tests the functions for computing the activation potentials
"""
function test_potentials()
   nvisible = 3
   nhidden = 2
   nsamples = 5
   vv = rand(nsamples, nvisible)

   # Test Test activation potential of hidden nodes for BernoulliRBM
   hh = rand(nsamples, nhidden)
   rbm = BMTest.randrbm(nvisible, nhidden)
   BMs.hiddenpotential!(hh, rbm, vv)
   @test isapprox(sum(abs.(hh - BMs.hiddenpotential(rbm, vv))), 0)

   # Test that activation potential of hidden nodes of a
   # Binomial2BernoulliRBM is the same as that of a BernoulliRBM
   b2brbm = BMs.Binomial2BernoulliRBM(rbm.weights, rbm.visbias, rbm.hidbias)
   hh2 = BMs.hiddenpotential(b2brbm, vv)
   @test sum(abs.(hh - hh2)) == 0
   BMs.hiddenpotential!(hh, b2brbm, vv)
   @test isapprox(sum(abs.(hh - hh2)), 0)

   # Test activation potential of visible nodes for BernoulliRBM
   BMs.visiblepotential!(vv, rbm, hh)
   @test isapprox(sum(abs.(vv - BMs.visiblepotential(rbm, hh))), 0)

   # Test activation potential of the visible nodes of a Binomial2BernoulliRBM
   BMs.visiblepotential!(vv, b2brbm, hh)
   @test isapprox(sum(abs.(vv- BMs.visiblepotential(b2brbm, hh))), 0)

   # Test activation potential of hidden nodes for GBRBM
   hh = rand(nsamples, nhidden)
   gbrbm = BMTest.randgbrbm(nvisible, nhidden)
   BMs.hiddenpotential!(hh, gbrbm, vv)
   @test isapprox(sum(abs.(hh - BMs.hiddenpotential(gbrbm, vv))), 0)

   # Test activation potential of visible nodes for GBRBM
   BMs.visiblepotential!(vv, gbrbm, hh)
   @test isapprox(sum(abs.(vv - BMs.visiblepotential(gbrbm, hh))), 0)

   # Test activation potential of hidden nodes for PartitionedRBM.
   # For a PartitionedRBM consisting of BernoulliRBMs, the
   # activation potential has to be the same as the activation potential
   # of the BernoulliRBM resulting from joining the RBMs.
   nvisible2 = 7
   nhidden2 = 4
   vv = rand(nsamples, nvisible + nvisible2)
   hhpartitioned = rand(nsamples, nhidden + nhidden2)
   rbm2 = BMTest.randrbm(nvisible2, nhidden2)
   prbm = BMs.PartitionedRBM{BMs.BernoulliRBM}([rbm; rbm2])
   joinedrbm = BMs.joinrbms([rbm; rbm2])
   BMs.hiddenpotential!(hhpartitioned, joinedrbm, vv)
   @test all(isapprox.(hhpartitioned, BMs.hiddenpotential(joinedrbm, vv)))

   # Test activation potential of visible nodes for PartitionedRBM
   BMs.visiblepotential!(vv, prbm, hhpartitioned)
   @test all(isapprox.(vv, BMs.visiblepotential(joinedrbm, hhpartitioned)))
end


function rbmexactloglikelihoodvsbaserate(x::Matrix{Float64}, nhidden::Int)
   a = BMs.logit.(vec(mean(x, dims = 1)))
   nvisible = length(a)
   rbm = BMs.BernoulliRBM(zeros(nvisible, nhidden), a, ones(nhidden))
   baserate = BMs.bernoulliloglikelihoodbaserate(x)
   exactloglik = BMs.loglikelihood(rbm, x, BMs.exactlogpartitionfunction(rbm))
   baserate - exactloglik
end

function bgrbmexactloglikelihoodvsbaserate(x::Matrix{Float64}, nhidden::Int)
   a = BMs.logit.(vec(mean(x, dims = 1)))
   nvisible = length(a)
   bgrbm = BMs.BernoulliGaussianRBM(zeros(nvisible, nhidden), a, ones(nhidden))
   baserate = BMs.bernoulliloglikelihoodbaserate(x)
   exactloglik = BMs.loglikelihood(bgrbm, x, BMs.exactlogpartitionfunction(bgrbm))
   baserate - exactloglik
end

function gbrbmexactloglikelihoodvsbaserate(x::Matrix{Float64}, nhidden::Int)
   a = vec(mean(x, dims = 1))
   nvisible = length(a)
   sd = vec(std(x, dims = 1))
   gbrbm = BMs.GaussianBernoulliRBM(zeros(nvisible, nhidden), a, ones(nhidden), sd)
   baserate = BMs.gaussianloglikelihoodbaserate(x)
   exactloglik = BMs.loglikelihood(gbrbm, x, BMs.exactlogpartitionfunction(gbrbm))
   baserate - exactloglik
end


function test_likelihoodbaserates()
   @test isapprox(BMs.bernoulliloglikelihoodbaserate(10),
         BMs.bernoulliloglikelihoodbaserate(rand([0.0 1.0], 10000, 10)),
         atol = 0.002)

   # Test exact loglikelihood of a baserate RBM
   # vs the calculated loglikelihoodbaserate
   x = rand([0.0 0.0 0.0 1.0 1.0], 100, 10);
   @test isapprox(BMTest.bgrbmexactloglikelihoodvsbaserate(x, 10), 0, atol = 1e-10)
   @test isapprox(BMTest.rbmexactloglikelihoodvsbaserate(x, 10), 0, atol = 1e-10)
   x = rand(100, 10) + randn(100, 10);
   @test isapprox(BMTest.gbrbmexactloglikelihoodvsbaserate(x, 10), 0, atol = 1e-10)


   @test isapprox(BMs.bernoulliloglikelihoodbaserate(10),
         BMs.bernoulliloglikelihoodbaserate(rand([0.0 1.0], 10000, 10)),
         atol = 0.002)

   # Test exact loglikelihood of a baserate RBM
   # vs the calculated loglikelihoodbaserate
   x = rand([0.0 0.0 0.0 1.0 1.0], 100, 10);
   @test isapprox(BMTest.bgrbmexactloglikelihoodvsbaserate(x, 10), 0, atol = 1e-10)
   @test isapprox(BMTest.rbmexactloglikelihoodvsbaserate(x, 10), 0, atol = 1e-10)
   x = rand(100, 10) + randn(100, 10);
   @test isapprox(BMTest.gbrbmexactloglikelihoodvsbaserate(x, 10), 0, atol = 1e-10)


   nvisible = 10
   nhidden = 10
   bgrbm = BMs.BernoulliGaussianRBM(zeros(nvisible, nhidden), rand(nvisible), ones(nhidden));
   @test isapprox(BMs.logpartitionfunction(bgrbm, 0.0),
         BMs.exactlogpartitionfunction(bgrbm))
end


function test_stackrbms_preparetrainlayers()
   x = Matrix{Float64}(undef, 22, 10);
   epochs = 17
   learningrate = 0.007
   nhiddens = [7;6;5]
   learningrate1 = 0.006
   learningrate2 = 0.017
   epochs1 = 27
   epochs2 = 17
   batchsize = 5
   batchsize1 = 2
   batchsize2 = 4
   optimizer = BMs.loglikelihoodoptimizer(learningrate = learningrate)
   optimizer1 = BMs.loglikelihoodoptimizer(learningrate = learningrate1)
   optimizer2 = BMs.loglikelihoodoptimizer(learningrate = learningrate2)

   # no layer-specific instructions
   trainlayers = Vector{BMs.TrainLayer}()
   trainlayers = BMs.stackrbms_preparetrainlayers(x;
         trainlayers = trainlayers, epochs = epochs,
         learningrate = learningrate, nhiddens = nhiddens,
         batchsize = batchsize, optimizer = optimizer)
   @test length(trainlayers) == length(nhiddens)
   @test all(map(t -> t.learningrate == learningrate, trainlayers))
   @test all(map(t -> t.epochs == epochs, trainlayers))
   @test map(t-> t.nhidden, trainlayers) == nhiddens
   @test all(map(t-> t.batchsize == batchsize, trainlayers))
   @test all(map(t-> t.optimizer.learningrate == learningrate, trainlayers))

   # layer-specific instructions without partitioning
   trainlayers = [
         BMs.TrainLayer(learningrate = learningrate1, epochs = epochs1,
               batchsize = batchsize1, nhidden = nhiddens[1],
               optimizer = optimizer1);
         BMs.TrainLayer(learningrate = learningrate2, epochs = epochs2,
               batchsize = batchsize2, nhidden = nhiddens[2],
               optimizer = optimizer2);
         BMs.TrainLayer(nhidden = nhiddens[3])
   ]
   trainlayers = BMs.stackrbms_preparetrainlayers(x;
         trainlayers = trainlayers, epochs = epochs,
         learningrate = learningrate, nhiddens = Vector{Int}(),
         batchsize = batchsize, optimizer = optimizer)
   @test length(trainlayers) == length(nhiddens)
   @test trainlayers[1].learningrate == learningrate1
   @test trainlayers[2].learningrate == learningrate2
   @test trainlayers[3].learningrate == learningrate
   @test trainlayers[1].epochs == epochs1
   @test trainlayers[2].epochs == epochs2
   @test trainlayers[3].epochs == epochs
   @test map(t-> t.nhidden, trainlayers) == nhiddens
   @test trainlayers[1].batchsize == batchsize1
   @test trainlayers[2].batchsize == batchsize2
   @test trainlayers[3].batchsize == batchsize
   @test trainlayers[1].optimizer.learningrate == learningrate1
   @test trainlayers[2].optimizer.learningrate == learningrate2
   @test trainlayers[3].optimizer.learningrate == learningrate

   # wrong/misleading specification of number of visible nodes
   badtrainlayers = deepcopy(trainlayers)
   badtrainlayers[1].nvisible = 100
   @test_throws ErrorException BMs.stackrbms_preparetrainlayers(x;
         trainlayers = badtrainlayers)

   # partitioning in first layer
   trainlayers = [
         BMs.TrainPartitionedLayer([
            BMs.TrainLayer(learningrate = learningrate1, epochs = epochs1,
                  rbmtype = BMs.GaussianBernoulliRBM, nhidden = 3,
                  nvisible = 3, batchsize = batchsize2);
            BMs.TrainLayer(nhidden = 4, nvisible = 7)
         ]);
         BMs.TrainLayer(nhidden = 6);
         BMs.TrainLayer(nhidden = 5)
   ]
   trainlayers = BMs.stackrbms_preparetrainlayers(x;
         trainlayers = trainlayers,
         epochs = epochs, learningrate = learningrate,
         batchsize = batchsize1)
   @test length(trainlayers) == length(nhiddens)
   @test trainlayers[1].parts[1].learningrate == learningrate1
   @test trainlayers[1].parts[2].learningrate == learningrate
   @test trainlayers[3].learningrate == learningrate
   @test trainlayers[1].parts[1].epochs == epochs1
   @test trainlayers[1].parts[2].epochs == epochs
   @test trainlayers[2].epochs == epochs
   @test trainlayers[3].epochs == epochs
   @test [trainlayers[1].parts[1].nhidden + trainlayers[1].parts[2].nhidden;
         trainlayers[2].nhidden; trainlayers[3].nhidden] == nhiddens
   @test trainlayers[1].parts[1].batchsize == batchsize2
   @test trainlayers[1].parts[2].batchsize == batchsize1

   # incorrect partitioning must not be allowed
   badtrainlayers = deepcopy(trainlayers)
   badtrainlayers[1].parts[1].nvisible = 20
   @test_throws ErrorException BMs.stackrbms_preparetrainlayers(x;
         trainlayers = badtrainlayers)

   # too many visible nodes in layer 1
   badtrainlayers = deepcopy(trainlayers)
   badtrainlayers[1].parts[1].nvisible = 100
   @test_throws ErrorException BMs.stackrbms_preparetrainlayers(x;
         trainlayers = badtrainlayers)

   nothing
end

function test_summing_out()
   # Test exact computation of log partition function of DBM:
   # Compare inefficient simple implementation with more efficient one that
   # sums out layers analytically.
   nhiddensvecs = Array[[5;4;6;3], [5;6;7], [4;4;2;4;3], [4;3;2;2;3;2]];
   for i = 1:length(nhiddensvecs)
      dbm = BMTest.randdbm(nhiddensvecs[i]);
      @test isapprox(BMs.exactlogpartitionfunction(dbm),
            BMTest.exactlogpartitionfunctionwithoutsummingout(dbm))
   end
end


function test_oneornone_encoding()
   x = [1 2 3; 0 1 0; 3 0 1]
   ncategories = 4
   @test all(x .== BMs.oneornone_decode(BMs.oneornone_encode(x, ncategories), ncategories))
   nscategories = [4;3;4]
   @test all(x .== BMs.oneornone_decode(BMs.oneornone_encode(x, nscategories), nscategories))
end


function test_softmaxvssigm()
   nsamples = 5
   nvariables = 6
   x1 = 6. .* rand(nsamples, nvariables) .- 3.
   x2 = copy(x1)
   # softmax transfomation to each of the columns
   for i in 1:nvariables
      BMs.softmax0!(view(x1, :, i:i))
   end
   @test all(isapprox.(x1, BMs.sigm!(x2)))
end


"""
    testaisvsexact(bm, percentalloweddiff)
Tests whether the exact log partition function is approximated by the value
estimated by AIS for the given Boltzmann Machine.
The test is successful, if the difference in percent of the log of the exact value
between the exact log partition function and AIS-estimated one is less than
`percentalloweddiff`.
"""
function checkaisvsexact(bm::BMs.AbstractBM, percentalloweddiff::Float64;
      ntemperatures = 100, nparticles = 100)

   exact = BMs.exactlogpartitionfunction(bm)
   estimated = BMs.logpartitionfunction(bm;
         ntemperatures = ntemperatures, nparticles = nparticles)

   @check abs((exact - estimated)/exact) < percentalloweddiff / 100
end

function exactlogpartitionfunctionwithoutsummingout(dbm::BMs.BasicDBM)
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
   dbm = BMs.stackrbms(x, nhiddens = nunits[2:end], batchsize = 5,
         epochs = 50, predbm = true, learningrate = 0.001);
   dbm = BMs.traindbm!(dbm, x,
         learningrates = [0.02*ones(10); 0.01*ones(10); 0.001*ones(10)],
         epochs = 30);
   logz = BMs.exactlogpartitionfunction(dbm)
   @test isapprox(BMs.exactloglikelihood(dbm, x, logz),
         BMTest.exactloglikelihoodwithoutsummingout(dbm, x, logz))
end

function exactloglikelihoodwithoutsummingout(dbm::BMs.BasicDBM, x::Array{Float64,2},
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


function test_dbmjoining()
   dbm1 = BMTest.randdbm([5;4;3])
   dbm2 = BMTest.randdbm([4;5;2])
   dbm3 = BMTest.randdbm([6;7;8])

   exactlogpartitionfunction1 = BMs.exactlogpartitionfunction(dbm1)
   exactlogpartitionfunction2 = BMs.exactlogpartitionfunction(dbm2)
   exactlogpartitionfunction3 = BMs.exactlogpartitionfunction(dbm3)

   jointdbm1 = BMs.joindbms(BMs.BasicDBM[dbm1, dbm2])
   # Test use of visibleindexes
   indexes = randperm(15)

   jointdbm2 = BMs.joindbms(BMs.BasicDBM[dbm1, dbm2, dbm3],
            [indexes[1:5], indexes[6:9], indexes[10:15]])

   @test isapprox(
               exactlogpartitionfunction1 + exactlogpartitionfunction2,
               BMs.exactlogpartitionfunction(jointdbm1))

   @test isapprox(
            exactlogpartitionfunction1 + exactlogpartitionfunction2 +
                  exactlogpartitionfunction3,
            BMs.exactlogpartitionfunction(jointdbm2))
end


"""
Tests whether the empirical loglikelihood from samples that are generated from
the exact distribution of Boltzmann machines is equal to the exact loglikelihood.
"""
function check_exactsampling()
   nsamples = 200
   x = BMs.barsandstripes(nsamples, 9)[:, 1:6]

   dbm1 = BMs.fitdbm(x; epochspretraining = 20, epochs = 15, nhiddens = [8; 5])
   BMTest.check_exactsampling(dbm1, x)

   dbm1 = BMs.fitdbm(x; epochspretraining = 20, epochs = 15, nhiddens = [6; 5; 4])
   BMTest.check_exactsampling(dbm1, x)

   rbm = BMs.fitrbm(x; epochs = 100, nhidden = 20)
   BMTest.check_exactsampling(rbm, x)
end

function check_exactsampling(bm::BMs.AbstractBM, x::Matrix{Float64};
      percentalloweddiff = 2.0, nsamples::Int = 35000)

   mbdist = BMs.MultivariateBernoulliDistribution(bm)
   exactsamples = BMs.samples(mbdist, nsamples)

   emploglikexactsamples = BMs.empiricalloglikelihood(x, exactsamples)
   exactloglik = BMs.exactloglikelihood(bm, x)
   @check abs((emploglikexactsamples - exactloglik) / exactloglik) <
         percentalloweddiff / 100
end


"""
Fits a larger RBM and a DBM and tests whether the AIS estimation
yields approximately the same results, if performed parallel or not.
"""
function check_likelihoodconsistency()
   nsamples = 1000
   nvariables = 64
   x = BMs.barsandstripes(nsamples, nvariables)
   dbm = BMs.fitdbm(x; epochspretraining = 20, epochs = 15,
         nhiddens = [50; 25; 20])

   logp1 = BMs.loglikelihood(dbm, x[1:10,:];
         parallelized = false, nparticles = 300, ntemperatures = 200)
   logp2 = BMs.loglikelihood(dbm, x[1:10,:];
         parallelized = true, nparticles = 300, ntemperatures = 200)
   @check isapprox(logp1, logp2, rtol = 0.055)

   rbm1 = BMs.fitrbm(x; epochs = 20)
   logz1 = BMs.logpartitionfunction(rbm1; parallelized = false)
   logz1_2 = BMs.logpartitionfunction(rbm1; parallelized = true)
   @check isapprox(logz1, logz1_2, rtol = 0.02)

   logp1 = BMs.loglikelihood(rbm1, x, logz1)
   rbm2 = BMs.fitrbm(x; epochs = 20, nhidden = 30)
   logz2 = BMs.logpartitionfunction(rbm2)
   logp2 = BMs.loglikelihood(rbm2, x, logz2)

   samplemeanrbm =
         BMs.BernoulliRBM(fill(0.0, nvariables, 1), vec(mean(x, dims = 1)), fill(0.0, 1))
   samplemeanrbm2 =
         BMs.BernoulliRBM(fill(0.0, size(rbm1.weights)), vec(mean(x, dims = 1)),
               fill(0.0, length(rbm1.hidbias)))

   # Annealing between two RBMs with zero weights
   @check isapprox(
         BMs.logmeanexp(BMs.aislogimpweights(samplemeanrbm, samplemeanrbm2)),
         (BMs.logpartitionfunctionzeroweights(samplemeanrbm2) -
               BMs.logpartitionfunctionzeroweights(samplemeanrbm)))

   # Annealing from RBM to RBM with sample mean and zero weights
   @check abs(logz1 - BMs.logpartitionfunctionzeroweights(samplemeanrbm) -
         BMs.logmeanexp(BMs.aislogimpweights(samplemeanrbm, rbm1))) <
               logz1 * 0.013

   # Annealing from one RBM to another RBM: Compare partition functions ...
   logimpweights = BMs.aislogimpweights(rbm1, rbm2;
         ntemperatures = 1000, nparticles = 200, burnin = 10)
   @check abs(BMs.logmeanexp(logimpweights) - (logz2 - logz1)) <
         max(logz2, logz1) * 0.03
   # ... and loglikelihood
   lldiff2 = BMs.loglikelihooddiff(rbm1, rbm2, x, logimpweights)
   lldiff1 = logp1 - logp2
   # very weak test, TODO: check and improve
   @check sign(lldiff1) == sign(lldiff2)
end


"""
Tests the likelihood estimation approaches for GaussianBernoulliRBMs
and alternative GaussianBernoulliRBMs.
"""
function check_likelihoodconsistency_gaussian(gbrbmtype::Type{GBRBM}
      ) where GBRBM <: Union{BMs.GaussianBernoulliRBM, BMs.GaussianBernoulliRBM2}

   x = BMs.curvebundles(nperbundle = 300, nvariables = 10,
         nbundles = 4)

   gbrbm1 = BMs.fitrbm(x; nhidden = 30, rbmtype = gbrbmtype,
         cdsteps = 10,
         pcd = false,
         learningrate = 0.001, epochs = 50,
         sdlearningrate = 0.00001)

   datainitrbm = GBRBM(zeros(size(x,2), 1),
         vec(mean(x, dims = 1)), [0.0], vec(std(x, dims = 1)))


   # Compare estimated difference of log partition functions to difference
   # between estimation and exact log partition function of GBRBM with zero weights
   logz1 = BMs.logpartitionfunction(gbrbm1)
   @check abs(logz1 - BMs.logpartitionfunctionzeroweights(datainitrbm) -
         BMs.logmeanexp(BMs.aislogimpweights(datainitrbm, gbrbm1,
               nparticles = 200, ntemperatures = 200))) < 0.03 * logz1

   gbrbm2 = BMs.fitrbm(x; nhidden = 15, rbmtype = gbrbmtype,
         cdsteps = 10,
         learningrate = 0.001, epochs = 100,
         sdlearningrates = [fill(0.0, 30); fill(0.00001, 300)])

   logz2 = BMs.logpartitionfunction(gbrbm2)

   logdiffimpweights = BMs.aislogimpweights(gbrbm1, gbrbm2)
   logzdiff = BMs.logmeanexp(logdiffimpweights)

   # Annealing from one RBM to another RBM: Compare partition functions ...
   @check abs(logzdiff - (logz2 - logz1)) < max(logz2, logz1) * 0.03
   # ... and loglikelihood
   logp1 = BMs.loglikelihood(gbrbm1, x, logz1)
   logp2 = BMs.loglikelihood(gbrbm2, x, logz2)
   lldiff2 = BMs.loglikelihooddiff(gbrbm1, gbrbm2, x, logdiffimpweights)
   lldiff1 = logp1 - logp2
   # very weak test, TODO: check and improve
   @check sign(lldiff1) == sign(lldiff2)
end


"
Tests whether the log-likelihood of Binomial2BernoulliRBMs is approximately
equal to the empirical loglikelihood of data generated by the models, and whether
the partition function estimated by AIS is near the exact value.
"
function check_b2brbm()
   x = BMTest.createsamples(100, 4) + BMTest.createsamples(100, 4)
   b2brbm = BMs.fitrbm(x, rbmtype = BMs.Binomial2BernoulliRBM, epochs = 150,
         nhidden = 4, learningrate = 0.001)
   @check checklikelihoodempirically(b2brbm, x; percentalloweddiff = 1.0)
end


function check_rbm()
   x = BMTest.createsamples(100, 4)
   rbm = BMs.fitrbm(x, epochs = 30,
         nhidden = 4, learningrate = 0.001)
   @check checklikelihoodempirically(rbm, x)
end

function check_sampling()
   realsamerate = 0.8
   x = BMTest.createsamples(100, 4, realsamerate)
   rbm = BMs.fitrbm(x, epochs = 400, nhidden = 5, learningrate = 0.002)

   samerate(x) = sum((x[:,1] .== x[:,2]) .& (x[:,2] .== x[:,3])) ./ size(x, 1)
   ngensamples = 5000
   @check abs(samerate(BMs.samples(rbm, ngensamples)) - realsamerate) < 0.1
   @test samerate(BMs.samples(rbm, 5, conditions = [1 => 1.0, 2 => 1.0, 3 => 1.0])) == 1.0
   @check samerate(BMs.samples(rbm, 5, samplelast = false)) == 0.0
   @test samerate(BMs.samples(rbm, 5, conditions = [1 => 0.0, 2 => 0.0, 3 => 0.0],
         samplelast = true)) == 1.0
   true
end

function check_softmax0rbm()
   samerates = [0.2; 0.3; 0.4]
   categories = length(samerates)
   x = BMs.oneornone_encode(BMTest.createsamples_categorical(100, 5, samerates), categories)
   datadict = BMs.DataDict("x" => x)
   monitor = BMs.Monitor()
   rbm = BMs.fitrbm(x, epochs = 150, rbmtype = BMs.Softmax0BernoulliRBM,
         categories = 3, nhidden = 4, learningrate = 0.01,
         monitoring = (rbm, epoch) -> begin
            if epoch % 50 == 0
               BMs.monitorreconstructionerror!(monitor, rbm, epoch, datadict)
               BMs.monitorexactloglikelihood!(monitor, rbm, epoch, datadict)
               BMs.monitorloglikelihood!(monitor, rbm, epoch, datadict)
            end
         end)

   xtest = x[1:5, :]

   # Softmax0BernoulliRBM with two categories must be equivalent to BernoulliRBM
   rbm2_softmax = BMs.Softmax0BernoulliRBM(rbm.weights, rbm.visbias, rbm.hidbias, 2)
   rbm2_bernoulli = BMs.BernoulliRBM(rbm.weights, rbm.visbias, rbm.hidbias)
   @test isapprox(BMs.exactloglikelihood(rbm2_softmax, xtest),
         BMs.exactloglikelihood(rbm2_bernoulli, xtest))


   zeroweightsrbm = BMs.Softmax0BernoulliRBM(
         zeros(size(rbm.weights)), rbm.visbias, rbm.hidbias, 3)
   @test isapprox(BMs.logpartitionfunctionzeroweights(zeroweightsrbm),
         BMs.exactlogpartitionfunction(zeroweightsrbm))


   @check BMTest.check_exactsampling(rbm, xtest; nsamples = 200000)

   # TODO evaluate quality of sample generation
   # sampled = BMs.oneornone_decode(BMs.samples(rbm, 1500), categories)
   # sampled[:, 1] .== sampled[:, 2] .== sampled[:, 3]

   # Note: Needs a little bit more burnin than the default. Problem?
   @check isapprox(BMs.exactloglikelihood(rbm, x),
         BMs.empiricalloglikelihood(rbm, x, 500000, 100), rtol = 0.03)

   # Test AIS for estimation of partition function
   logz_exact = BMs.exactlogpartitionfunction(rbm)
   logz_ais = BMs.logpartitionfunction(rbm)
   @check isapprox(logz_exact, logz_ais, rtol = 0.02)
end


function check_softmax0dbm()
   samerates = [0.2; 0.3; 0.4]
   categories = length(samerates)
   x = BMs.oneornone_encode(BMTest.createsamples_categorical(100, 5, samerates), categories)
   dbm = BMs.fitdbm(x, epochs = 150,
         pretraining = [
               BMs.TrainLayer(rbmtype = BMs.Softmax0BernoulliRBM, categories = 3,
                     nhidden = 5);
               BMs.TrainLayer(nhidden = 4)])

   xtest = x[1:5, :]

   @check BMTest.check_exactsampling(dbm, xtest; nsamples = 200000)

   # Test exact vs empirical loglikelihood
   @check isapprox(BMs.exactloglikelihood(dbm, x),
         BMs.empiricalloglikelihood(dbm, x, 50000, 150), rtol = 0.03)

   # Test AIS for estimation of partition function
   logz_exact = BMs.exactlogpartitionfunction(dbm)
   logz_ais = BMs.logpartitionfunction(dbm)
   @check isapprox(logz_exact, logz_ais, rtol = 0.02)

   # Test AIS for likelihood estimation vs exact calculation
   @check isapprox(BMs.loglikelihood(dbm, xtest; nparticles = 200, ntemperatures = 200),
         BMs.exactloglikelihood(dbm, xtest), rtol = 0.05)
end


function check_gbrbm(gbrbmtype::Type{GBRBM}
      ) where GBRBM <:Union{BMs.GaussianBernoulliRBM, BMs.GaussianBernoulliRBM2}

   x = BMs.curvebundles(nperbundle = 300, nvariables = 10,
         nbundles = 4)
   gbrbm = BMs.fitrbm(x; nhidden = 15, rbmtype = gbrbmtype,
         cdsteps = 10,
         pcd = false,
         learningrate = 0.001, epochs = 50,
         sdlearningrate = 0.00001)

   logz = BMs.logpartitionfunction(gbrbm; parallelized = true,
         ntemperatures = 500, nparticles = 500)
   estloglik = BMs.loglikelihood(gbrbm, x, logz)
   exactloglik = BMs.exactloglikelihood(gbrbm, x)
   @check abs((exactloglik - estloglik) / exactloglik) < 2.5 / 100
end


function test_intensities()
   x = randn(5, 3) .* [1 2 3]
   transformed, transformation = BMs.intensities_encode(x)
   @test isapprox(BMs.intensities_decode(transformed, transformation), x)
end


function test_rbm_monitoring(gbrbmtype::Type{GBRBM}
      ) where GBRBM <:Union{BMs.GaussianBernoulliRBM, BMs.GaussianBernoulliRBM2}

   x = BMs.curvebundles(nperbundle = 20, nvariables = 5,
      nbundles = 4)

   # Test some more options and monitoring
   monitor = BMs.Monitor()
   datadict = BMs.DataDict("Small subset" => x[1:3, :])
   gbrbm = BMs.fitrbm(x; nhidden = 5, rbmtype = gbrbmtype,
         learningrate = 0.000001, epochs = 20,
         batchsize = 10,
         pcd = false,
         sdlearningrate = 0.000000001,
         monitoring = (rbm, epoch) -> begin
            if epoch == 10 || epoch == 20
               BoltzmannMachines.monitorexactloglikelihood!(monitor, rbm, epoch, datadict)
               BoltzmannMachines.monitorreconstructionerror!(monitor, rbm, epoch, datadict)
            end
         end)
   nothing
end


function test_mdbm_monitoring()
   x1 = BMTest.createsamples(80, 4)
   x2 = BMs.curvebundles(nperbundle = 20, nvariables = 5, nbundles = 4)
   data = hcat(x1, x2)
   traindata, testdata = BMs.splitdata(data, 0.2)
   monitor = BMs.Monitor(); monitor1 = BMs.Monitor()
   datadict = BMs.DataDict("Training data" => traindata, "Test data" => testdata)
   dbm = BMs.fitdbm(traindata,
      epochs = 2,
      monitoringdatapretraining = datadict,
      batchsizepretraining = 10,
      pretraining = [
         BMs.TrainPartitionedLayer([
            BMs.TrainLayer(nvisible = 4, nhidden = 5,
                  monitoring = (rbm, epoch, datadict) ->
                        BMs.monitorreconstructionerror!(monitor1, rbm, epoch, datadict))
            BMs.TrainLayer(nhidden = 6, rbmtype = BMs.GaussianBernoulliRBM2)
         ]),
         BMs.TrainLayer(nhidden = 5)
      ],
      monitoring = (dbm, epoch) -> BMs.monitorlogproblowerbound!(monitor, dbm, epoch, datadict))
   nothing
end


function test_monitored_fitrbm()
   nvisible1 = 4
   x = BMTest.createsamples(5, nvisible1)

   # without monitoring:
   # Test whether monitored_fitrbm and fitrbm do the same thing
   randomseed = abs(rand(Int))
   Random.seed!(randomseed)
   monitor, rbm1 = BMs.monitored_fitrbm(x;
         pcd = false, cdsteps = 3, nhidden = 5, epochs = 2)
   @test isempty(monitor)
   Random.seed!(randomseed)
   rbm2 = BMs.fitrbm(x; pcd = false, cdsteps = 3, nhidden = 5, epochs = 2)
   @test all(isapprox.(rbm1.weights, rbm2.weights))

   # one monitoring function
   monitor, rbm1 = BMs.monitored_fitrbm(x;
         monitoring = BMs.monitorreconstructionerror!, batchsize = 4, epochs = 3)
   @test length(monitor) == 3

   # two monitoring functions
   # (also test uneven batch size)
   monitor, rbm1 = BMs.monitored_fitrbm(x; epochs = 3, batchsize = 3,
         monitoring = [BMs.monitorfreeenergy!;
               BMs.monitorexactloglikelihood!])
   @test length(monitor) == 6

   nothing
end


function test_monitored_stackrbms()
   nvisible1 = 4
   nvisible2 = 3
   x = BMTest.createsamples(20, nvisible1)
   xpart = hcat(x, rand(20, nvisible2))
   nhiddens = [4; 3]

   # Test whether monitored_stackrbm and stackrbm do the same thing
   randomseed = abs(rand(Int))
   Random.seed!(randomseed)
   monitors1, dbm1 = BMs.monitored_stackrbms(x; predbm = true, nhiddens = nhiddens)
   @test length(monitors1) == 2
   @test all(isempty.(monitors1))
   Random.seed!(randomseed)
   dbm2 = BMs.stackrbms(x; predbm = true, nhiddens = nhiddens)
   @test all([all(isapprox.(dbm1[i].weights, dbm2[i].weights)) for i in eachindex(dbm1)])

   # partitioned
   monitor1 = BMs.Monitor()
   trainlayers = [
      BMs.TrainPartitionedLayer([
         BMs.TrainLayer(nvisible = nvisible1, nhidden = 2)
         BMs.TrainLayer(nhidden = 2, rbmtype = BMs.GaussianBernoulliRBM2)
      ]),
      BMs.TrainLayer(nhidden = 4)
   ]
   epochs = 5
   monitors1, dbm1 = BMs.monitored_stackrbms(xpart;
         trainlayers = trainlayers,
         monitoring = BMs.monitorexactloglikelihood!,
         predbm = true, epochs = epochs)
   @test length(monitors1) == 2
   @test length(monitors1[1]) == 2 # layers with two partitions
   @test length(monitors1[1][1]) == epochs
   @test length(monitors1[1][2]) == epochs
   @test length(monitors1[2]) == epochs

   # Vector of monitoring functions
   monitors, dbn = BMs.monitored_stackrbms(x;
         nhiddens = nhiddens,
         monitoring = [BMs.monitorexactloglikelihood!, BMs.monitorreconstructionerror!],
         predbm = false, epochs = epochs)
   @test length(monitors) == length(nhiddens) == length(dbn)
   @test all(map(length, monitors) .== 2*epochs)

   nothing
end


function dbms_are_approx_equal(dbm1, dbm2)
   all([all(isapprox.(dbm1[i].weights, dbm2[i].weights)) for i in eachindex(dbm1)])
end


function test_monitored_fitdbm()
   nvisible1 = 4
   nvisible2 = 3
   x = BMTest.createsamples(20, nvisible1)
   xpart = hcat(x, rand(20, nvisible2))

   kwargs = Dict{Symbol, Any}(:epochs=> 5, :epochspretraining => 10,
         :nhiddens => [4; 3], :batchsizepretraining => 5,
         :learningrate => 0.001, :nparticles => 17)
   # Test whether monitored_fitdbm and fitdbm do the same thing
   randomseed = abs(rand(Int))
   Random.seed!(randomseed)
   monitors1, dbm1 = BMs.monitored_fitdbm(x; kwargs...)
   @test length(monitors1) == 3
   @test all(isempty.(monitors1))
   Random.seed!(randomseed)
   dbm2 = BMs.fitdbm(x; kwargs...)
   @test dbms_are_approx_equal(dbm1, dbm2)

   # ensure real types are not unified
   BMs.monitored_fitdbm(x; epochs = 1, learningrate = 0.05)

   # partitioned
   monitoringdata = BMs.DataDict("Only part" => BMs.splitdata(xpart, 0.5)[1])
   epochspretraining = 4
   epochs = 5
   trainlayers = [
      BMs.TrainPartitionedLayer([
         BMs.TrainLayer(nvisible = nvisible1, nhidden = 2)
         BMs.TrainLayer(nhidden = 2, rbmtype = BMs.GaussianBernoulliRBM2)
      ]),
      BMs.TrainLayer(nhidden = 4)
   ]
   monitors, dbm = BMs.monitored_fitdbm(xpart;
         pretraining = trainlayers,
         monitoringpretraining = BMs.monitorfreeenergy!,
         monitoring = [BMs.monitorexactloglikelihood!,
               BMs.monitorlogproblowerbound!],
         epochs = epochs,
         epochspretraining = epochspretraining)
   @test length(monitors) == 3
   @test length(monitors[1]) == 2 # layers with two partitions
   @test length(monitors[1][1]) == epochspretraining
   @test length(monitors[1][2]) == epochspretraining
   @test length(monitors[2]) == epochspretraining
   lowerbounditems = filter(m -> m.evaluation == BMs.monitorlogproblowerbound, monitors[3])
   exactloglikitems = filter(m -> m.evaluation == BMs.monitorexactloglikelihood, monitors[3])
   @test length(lowerbounditems) == epochs
   @test length(exactloglikitems) == epochs

   # passing an unknown argument must throw an exception
   @test_throws ErrorException BMs.monitored_fitdbm(x; bla = 10, epochs = 5)

   nothing
end


function test_batchsize_dbm_training()
   x = BMTest.createsamples(20, 4)

   randomseed = abs(rand(Int))
   # Test correct defaults for batchsize
   Random.seed!(randomseed)
   dbm1 = BMs.fitdbm(x; epochs = 2)
   Random.seed!(randomseed)
   _, dbm2 = BMs.monitored_fitdbm(x; epochs = 2, batchsize = 1, batchsizefinetuning = size(x,1))
   @test BMTest.dbms_are_approx_equal(dbm1, dbm2)

   # Test specifying different batchsizefinetuning
   Random.seed!(randomseed)
   dbm2 = BMs.fitdbm(x; epochs = 2, batchsizefinetuning = 2)
   Random.seed!(randomseed)
   _, dbm3 = BMs.monitored_fitdbm(x; epochs = 2, batchsize = 1, batchsizefinetuning = 2)
   @test BMTest.dbms_are_approx_equal(dbm2, dbm3)
   @test !BMTest.dbms_are_approx_equal(dbm1, dbm3)

   # Test that batchsize argument determines both fine-tuning and pretraining
   Random.seed!(randomseed)
   dbm4 = BMs.fitdbm(x; epochs = 2, batchsize = 2)
   Random.seed!(randomseed)
   _, dbm5 = BMs.monitored_fitdbm(x; epochs = 2, batchsizepretraining = 2,  batchsizefinetuning = 2)
   @test BMTest.dbms_are_approx_equal(dbm4, dbm5)
   @test !BMTest.dbms_are_approx_equal(dbm3, dbm5)

   nothing
end


function test_epochs_dbm_training()
   x = BMTest.createsamples(20, 4)

   randomseed = abs(rand(Int))
   # Test correct defaults for epochs
   Random.seed!(randomseed)
   dbm1 = BMs.fitdbm(x; epochs = 2)
   Random.seed!(randomseed)
   _, dbm2 = BMs.monitored_fitdbm(x; epochspretraining = 2, epochsfinetuning = 2)
   @test BMTest.dbms_are_approx_equal(dbm1, dbm2)

   # Test specifying different epochs for finetuning
   Random.seed!(randomseed)
   dbm2 = BMs.fitdbm(x; epochs = 1, epochsfinetuning = 2)
   Random.seed!(randomseed)
   _, dbm3 = BMs.monitored_fitdbm(x; epochspretraining = 1, epochsfinetuning = 2)
   @test BMTest.dbms_are_approx_equal(dbm2, dbm3)
   @test !BMTest.dbms_are_approx_equal(dbm1, dbm3)

   nothing
end


function check_mdbm_rbm_b2brbm()
   nsamples = 100
   nvariables = 4

   x1 = BMTest.createsamples(nsamples, nvariables) +
         BMTest.createsamples(nsamples, nvariables);
   x2 = BMTest.createsamples(nsamples, nvariables);
   x = hcat(x1, x2);
   trainlayers1 = [
         BMs.TrainPartitionedLayer([
            BMs.TrainLayer(rbmtype = BMs.Binomial2BernoulliRBM,
                  nvisible = nvariables, nhidden = nvariables - 1,
                  learningrate = 0.0015);
            BMs.TrainLayer(rbmtype = BMs.BernoulliRBM,
                  nvisible = nvariables, nhidden = nvariables)
         ]);
         BMs.TrainLayer(nhidden = nvariables);
         BMs.TrainLayer(nhidden = 3)]

   dbm1 = BMs.fitdbm(x, epochs = 15,
         epochspretraining = 15,
         learningratepretraining = 0.001,
         pretraining = trainlayers1)

   @check BMTest.checklikelihoodempirically(dbm1, x; percentalloweddiff = 5.5,
         ntemperatures = 300, nparticles = 300)

   # partitioned second layer
   trainlayers2 = [
         trainlayers1[1],
         BMs.TrainPartitionedLayer([
            BMs.TrainLayer(nhidden = 3);
            BMs.TrainLayer(nhidden = 2)
         ]),
         BMs.TrainLayer(nhidden = 4),
         BMs.TrainLayer(nhidden = 3)
   ]

   dbm2 = BMs.fitdbm(x, pretraining = trainlayers2)
   @check BMTest.checklikelihoodempirically(dbm2, x; percentalloweddiff = 6.3,
         ntemperatures = 300, nparticles = 300)
end


function checklikelihoodempirically(rbm::BMs.AbstractRBM, x::Matrix{Float64};
      percentalloweddiff = 0.5, ntemperatures::Int = 100, nparticles::Int = 100)

   logz = BMs.logpartitionfunction(rbm; parallelized = true,
      nparticles = nparticles, ntemperatures = ntemperatures)
   estloglik = BMs.loglikelihood(rbm, x, logz)
   exactloglik = BMs.exactloglikelihood(rbm, x)
   @check abs((exactloglik - estloglik) / exactloglik) < percentalloweddiff / 100

   emploglik = BMs.empiricalloglikelihood(rbm, x, 1000000)
   @check abs((exactloglik - emploglik) / exactloglik) < percentalloweddiff / 100
end

function checklikelihoodempirically(dbm::BMs.MultimodalDBM, x::Matrix{Float64};
      percentalloweddiff = 0.5, ntemperatures::Int = 100, nparticles::Int = 100)

   emploglik = BMs.empiricalloglikelihood(dbm, x, 1000000)
   estloglik = BMs.loglikelihood(dbm, x; parallelized = true,
         ntemperatures = ntemperatures, nparticles = nparticles)
   exactloglik = BMs.exactloglikelihood(dbm, x)
   @check abs((exactloglik - emploglik) / exactloglik) < percentalloweddiff / 100
   @check abs((exactloglik - estloglik) / exactloglik) < percentalloweddiff / 100
end


"""
Test DBMs with Gaussian visible nodes.
"""
function check_mdbm_gaussianvisibles()

   x = convert(Matrix{Float64}, readdlm(
         joinpath(dirname(pathof(BoltzmannMachines)), "..", "test/data/iris.csv"), ',',
         header = true)[1][:,1:4]);

   datadict = BMs.DataDict("x" => x)

   monitor1 = BMs.Monitor()
   trainlayers = [
         BMs.TrainLayer(rbmtype = BMs.GaussianBernoulliRBM,
               nhidden = 4,
               sdlearningrate = 0.000001,
               monitoring = (rbm, epoch) -> begin
                  BMs.monitorexactloglikelihood!(monitor1, rbm, epoch, datadict)
               end);
         BMs.TrainLayer(nhidden = 4);
         BMs.TrainLayer(nhidden = 4)]

   learningrates = [0.02*ones(10); 0.01*ones(10); 0.001*ones(10)]

   seed = round(Int, rand()*typemax(Int), RoundDown)
   Random.seed!(seed)
   dbm1 = BMs.stackrbms(x, epochs = 20, predbm = true, learningrate = 0.001,
         trainlayers = trainlayers)
   # BoltzmannMachinesPlots.plotevaluation(monitor1, BMs.monitorexactloglikelihood)
   dbm1 = BMs.traindbm!(dbm1, x,
         learningrates = learningrates,
         epochs = 30);

   Random.seed!(seed)
   dbm2 = BMs.fitdbm(x, epochs = 30,
         epochspretraining = 20,
         learningratepretraining = 0.001,
         pretraining = trainlayers,
         learningrates = learningrates)

   # first and second dbm must be equal
   @test length(dbm1) == length(dbm2)
   @test isapprox(dbm1[1].sd, dbm2[1].sd)
   for i in 1:length(dbm1)
      @test isapprox(dbm1[i].weights, dbm2[i].weights)
      @test isapprox(dbm1[i].visbias, dbm2[i].visbias)
      @test isapprox(dbm1[i].hidbias, dbm2[i].hidbias)
   end

   # Test AIS
   @check BMTest.checkaisvsexact(dbm1, 0.6)

   # Test exact likelihood vs estimated likelihood
   estloglik = BMs.loglikelihood(dbm1, x; nparticles = 400, ntemperatures = 200)
   exactloglik = BMs.exactloglikelihood(dbm1, x)
   @check abs((exactloglik - estloglik) / exactloglik) < 3.5 / 100
end


function check_softmaxsampling()
   probs = [0.5 0.3 0.7 0.2; 0.2 0.25 0.1 0.15]
   sampled = zeros(size(probs))
   s0brbm = BMs.Softmax0BernoulliRBM(zeros(4,4), zeros(4), zeros(2), 3)
   nsampled = 1000
   for i = 1:nsampled
      newsamples = BMs.samplevisiblepotential!(copy(probs), s0brbm)
      @test all(map(s -> s in [1.0 0.0], newsamples))
      sampled .+= newsamples
   end
   @check all(1.12 .> (sampled ./ nsampled ./ probs) .> 0.88)
end


function test_mdbm_architectures()
   nsamples = 5
   data = hcat(float.(rand(nsamples, 3) .< 0.5), randn(nsamples, 3))

   BMs.fitdbm(data; epochs = 2, pretraining = [
         BMs.TrainPartitionedLayer([
               BMs.TrainLayer(nvisible = 3, nhidden = 3);
               BMs.TrainLayer(nvisible = 3, nhidden = 3,
                     rbmtype = BMs.GaussianBernoulliRBM)]);
               BMs.TrainLayer(nhidden = 4);
               BMs.TrainLayer(nhidden = 3)])

   BMs.fitdbm(data; epochs = 2, pretraining = [
         BMs.TrainPartitionedLayer([
               BMs.TrainLayer(nvisible = 3, nhidden = 3);
               BMs.TrainLayer(nvisible = 3, nhidden = 3,
                     rbmtype = BMs.GaussianBernoulliRBM2)]);
         BMs.TrainPartitionedLayer([
               BMs.TrainLayer(nhidden = 2)
               BMs.TrainLayer(nhidden = 2)]);
         BMs.TrainLayer(nhidden = 4)])

   data2 = hcat(data,
         float.(rand(nsamples, 3) .< 0.5) .+ float.(rand(nsamples, 3) .< 0.5))
   BMs.fitdbm(data2; epochs = 2, pretraining = [
         BMs.TrainPartitionedLayer([
               BMs.TrainLayer(nvisible = 3, nhidden = 3);
               BMs.TrainLayer(nvisible = 3, nhidden = 3,
                     rbmtype = BMs.GaussianBernoulliRBM2);
               BMs.TrainLayer(nvisible = 3, nhidden = 3,
                     rbmtype = BMs.Binomial2BernoulliRBM)]);
         BMs.TrainPartitionedLayer([
               BMs.TrainLayer(nvisible = 3, nhidden = 3)
               BMs.TrainLayer(nvisible = 6, nhidden = 3)]);
         BMs.TrainLayer(nhidden = 3)])
   nothing
end


function test_unevenbatches()
   x = BMs.barsandstripes(20, 4)
   rbm = BMs.fitrbm(x, batchsize = 3, epochs = 2) # 20 % 3 !== 0
   @test_throws ErrorException BMs.fitrbm(x, batchsize = 21, epochs = 2)
   nothing
end


function test_beam()
   x = randn(100, 10)
   rbm = BMs.fitrbm(x, rbmtype = BMs.GaussianBernoulliRBM2,
         batchsize = 20,
         cdsteps = 50,
         nhidden = 20, epochs = 30,
         sdinitfactor = 0.1,
         optimizers = [
               fill(BMs.loglikelihoodoptimizer(learningrate = 0.0001,
                     sdlearningrate = 0.000001), 10);
               fill(BMs.beamoptimizer(learningrate = 0.0001,
                     sdlearningrate = 0.000001,
                     adversarialweight = 0.5), 20)
         ])
   nothing
end


end # of module BMTest
