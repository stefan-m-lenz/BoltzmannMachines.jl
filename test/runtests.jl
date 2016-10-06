using Base.Test

using BoltzmannMachines
const BMs = BoltzmannMachines

include("BMTest.jl")

@test_approx_eq_eps(BMs.bernoulliloglikelihoodbaserate(10),
      BMs.bernoulliloglikelihoodbaserate(rand([0.0 1.0], 10000, 10)), 0.002)

x = rand([0.0 0.0 0.0 1.0 1.0], 1000, 10);

@test_approx_eq_eps(BMTest.bgrbmexactloglikelihoodvsbaserate(x, 10), 0, 1e-10)
@test_approx_eq_eps(BMTest.rbmexactloglikelihoodvsbaserate(x, 10), 0, 1e-10)

nvisible = 10
nhidden = 10
bgrbm = BMs.BernoulliGaussianRBM(zeros(nvisible, nhidden), rand(nvisible), ones(nhidden));

@test_approx_eq(BMs.logpartitionfunction(bgrbm, 1.0),
      BMs.exactlogpartitionfunction(bgrbm))

# Test exact computation of log partition function of DBM:
# Compare inefficient simple implementation with more efficient one that
# sums out layers analytically.
nhiddensvecs = Array[[5;4;6;3], [5;6;7], [4;4;2;4;3], [4;3;2;2;3;2]];
for i = 1:length(nhiddensvecs)
   dbm = BMTest.randdbm(nhiddensvecs[i]);
   @test_approx_eq(BMs.exactlogpartitionfunction(dbm),
         BMTest.exactlogpartitionfunctionwithoutsummingout(dbm))
end

for nunits in Array[[10;5;4], [20;5;4;2], [4;2;2;3;2]]
   BMTest.testsummingoutforexactloglikelihood(nunits)
end