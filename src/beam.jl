# This file implements methods from the paper
# "Boltzmann Encoded Adversarial Machines" of Fisher et al. (2018)

function nearestneighbourcritic(xdata::M, xmodel::M, k::Int,
      distanceweighted::Bool = true) where {M <: AbstractArray{Float64,2}}

   ndatasamples = size(xdata, 1)
   if ndatasamples != size(xmodel, 1)
      error("Number of samples in `xdata` and `xmodel` must be equal.")
   end

   x = vcat(xdata, xmodel)

   nsamples = 2 * ndatasamples

   # Distancematrix is filled with the distances
   # between all combinations of samples in x.
   distancematrix = Matrix{Float64}(undef, nsamples, nsamples)
   for i = 1:nsamples
      for j = (i+1):nsamples
         distancematrix[i, j] = distancematrix[j, i] = norm(x[j, :] - x[i, :])
      end
      distancematrix[i, i] = 0.0
   end

   ret = Vector{Float64}(undef, ndatasamples)
   eps = 0.1
   if distanceweighted
      for i = 1:ndatasamples
         knearestindices = partialsortperm(distancematrix[i, :], (1:k) .+ 1)
         nearestdatasamplesindices = knearestindices[knearestindices .<= ndatasamples]
         datasamplesdistances = distancematrix[i, nearestdatasamplesindices]
         ret[i] = 2.0 * ( sum(1.0 ./ (datasamplesdistances .+ eps)) /
               sum(1.0 ./ (distancematrix[i, knearestindices] .+ eps))) - 1
      end
   else
      for i = 1:ndatasamples
         knearestindices = partialsortperm(distancematrix[i, :], (1:k) .+ 1)
         nnearestdatasamples = sum(knearestindices .<= ndatasamples)
         ret[i] = 2 * nnearestdatasamples / k - 1
      end
   end

   ret
end
