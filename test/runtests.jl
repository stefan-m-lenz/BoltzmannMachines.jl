using Distributed
if nprocs() == 1
   # test with multiple processes, but use only 2
   # as travis testing has two processes
   addprocs(2)
end

# This assumes we are in the /src or /test folder.
@everywhere push!(LOAD_PATH, "../src")
import BoltzmannMachines
BMs = BoltzmannMachines

@everywhere push!(LOAD_PATH, "../test/")
import BMTest


#######################
# Deterministic tests #
#######################

BMTest.test_potentials()
BMTest.test_likelihoodbaserates()
BMTest.test_stackrbms_preparetrainlayers()
BMTest.test_dbmjoining()
BMTest.test_rbm_monitoring(BMs.GaussianBernoulliRBM)
BMTest.test_rbm_monitoring(BMs.GaussianBernoulliRBM2)
BMTest.test_mdbm_architectures()
BMTest.test_mdbm_monitoring()
BMTest.test_summing_out()
BMTest.test_unevenbatches()
BMTest.test_beam()
BMTest.test_softmaxvssigm()
BMTest.test_oneornone_encoding()
BMTest.test_intensities()
BMTest.test_monitored_fitrbm()
BMTest.test_monitored_stackrbms()
BMTest.test_monitored_fitdbm()

println("Finished deterministic tests")

#########################################################################
# Tests which may fail sometimes due to stochastic nature of algorithms #
#########################################################################

nondeterministictestidx = 0

# Perform non-deterministic tests, allow failures up to n times
function softly(fun, n::Int = 7)
   for i = 1:n
      if fun()
         global nondeterministictestidx += 1
         # print some output to keep Travis going
         println("Non-deterministic test #$nondeterministictestidx passed" *
               " after $i trie(s).")
         return true
      end
   end
   false
end

using Test

@test softly(BMTest.check_rbm)
@test softly(BMTest.check_b2brbm)
@test softly(BMTest.check_likelihoodconsistency)
@test softly(() -> BMTest.check_gbrbm(BMs.GaussianBernoulliRBM))
@test softly(() -> BMTest.check_gbrbm(BMs.GaussianBernoulliRBM2))
@test softly(() ->
      BMTest.check_likelihoodconsistency_gaussian(BMs.GaussianBernoulliRBM))
@test softly(() ->
      BMTest.check_likelihoodconsistency_gaussian(BMs.GaussianBernoulliRBM2))
@test softly(BMTest.check_mdbm_rbm_b2brbm)
@test softly(BMTest.check_softmax0rbm)
@test softly(BMTest.check_softmax0dbm)
@test softly(BMTest.check_mdbm_gaussianvisibles)
@test softly(BMTest.check_exactsampling)
@test softly(BMTest.check_softmaxsampling)
@test softly(BMTest.check_sampling)


##########################################
# Run examples (without actual plotting) #
##########################################
@everywhere push!(LOAD_PATH, "../test/mock")
using BoltzmannMachinesPlots
include("../test/examples.jl")

@everywhere pop!(LOAD_PATH)
@everywhere pop!(LOAD_PATH)
@everywhere pop!(LOAD_PATH)