module BoltzmannMachinesPlots
# This module allows to run the examples without plotting anything,
# such that the testing does not have the BoltzmannMachinesPlots package
# as a requirement.

function plotevaluation(args1...; kwargs...) end
function crossvalidationcurve(args1...; kwargs...) end

end