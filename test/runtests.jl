using Base.Test

import BoltzmannMachines
const BMs = BoltzmannMachines

include("BMTest.jl")

@test_approx_eq_eps(BMs.bernoulliloglikelihoodbaserate(10),
      BMs.bernoulliloglikelihoodbaserate(rand([0.0 1.0], 10000, 10)), 0.002)

# Test exact loglikelihood of a baserate RBM
# vs the calculated loglikelihoodbaserate
x = rand([0.0 0.0 0.0 1.0 1.0], 100, 10);
@test_approx_eq_eps(BMTest.bgrbmexactloglikelihoodvsbaserate(x, 10), 0, 1e-10)
@test_approx_eq_eps(BMTest.rbmexactloglikelihoodvsbaserate(x, 10), 0, 1e-10)
x = rand(100, 10) + randn(100, 10);
@test_approx_eq_eps(BMTest.gbrbmexactloglikelihoodvsbaserate(x, 10), 0, 1e-10)


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

for nunits in Array[[10;5;4;3], [18;6;3], [11;6;7;4]]
   BMTest.testexactloglikelihood_bernoullimvdbm(nunits)
end

for nunits in Array[[50;10;90;20], [18;6;79], [100;6;50;15]]
   BMTest.testlogpartitionfunction_bernoullimvdbm(nunits)
end

BMTest.testloglikelihood_b2brbm()

BMTest.testgaussianmvdbm()