type CrossValidationItem
   param
   model
   score
end


"""
Encapsulates the full results of a cross-validation.
"""
const CrossValidationResult = Vector{CrossValidationItem}


"""
    crossvalidation(...)
Performs k-fold cross-validation. Returns a `CrossValidationResult` that
ist best viewed by creating a cross-validation plot via
`BMPlots.crossvalidationplot`.

The `k` in "`k`-fold" can be specified with the named argument `kfold`
(defaults to 10).

# Required named arguments:
* `x`: the complete data set (Float64-Matrix with samples as rows)
* `fit`: a function accepting a data set and a parameter, returning a model
* `eval`: a function accepting a model and a dataset, returning a score
* `params`: a vector of parameters that are to be cross-validated
"""
function crossvalidation(;
      fit::Function = emptyfunc,
      eval::Function = emptyfunc,
      x::Matrix{Float64} = Matrix{Float64}(0, 0),
      params::AbstractArray = Vector(),
      kfold::Int = 10)

   if isempty(x) || isempty(params) || fit == emptyfunc || eval == emptyfunc
      error("Arguments `x`, `params`, `fit`, and `eval` are required.")
   end

   nsamples = size(x, 1)

   batchranges = ranges(mostevenbatches(nsamples, kfold))
   validationmasks = map(rng -> [i in rng for i in 1:nsamples], batchranges)

   ret = CrossValidationResult()

   @parallel (vcat) for mask in validationmasks
      trainingdata = x[.!mask, :]
      evaluationdata = x[mask, :]
      cvres = CrossValidationResult(length(params))
      for i in 1:length(params)
         model = fit(trainingdata, params[i])
         score = eval(model, evaluationdata)
         cvres[i] = CrossValidationItem(params[i], model, score)
      end
      cvres
   end

end
