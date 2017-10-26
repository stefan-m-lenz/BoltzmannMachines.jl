struct CrossValidationItem{P}
   param::P
   score::Float64
end


"""
Encapsulates the results of a cross-validation.
"""
const CrossValidationResult = Vector{CrossValidationItem{P}} where P


"""
    crossvalidation(fit, params; ...)
Performs k-fold cross-validation, given either
* `fit`: a function that accepts a data set and a parameter
   and returns a trained model and
* `params`: a vector of parameters for cross-validation.

Returns a `CrossValidationResult` that is best viewed by creating
a cross-validation plot via `BMPlots.crossvalidationplot`.

# Named arguments:
* `data` (required): the complete data set (Float64-Matrix with samples as rows)
* `eval` (required): a function accepting a model and a dataset, returning a score
* `kfold`: specifies the `k` in "`k`-fold" (defaults to 10).
"""
function crossvalidation(fit::Function, params::PV;
      eval::Function = emptyfunc,
      data::Matrix{Float64} = Matrix{Float64}(0, 0),
      kfold::Int = 10) where {P, PV<:AbstractArray{P,1}}

   if isempty(data) || eval == emptyfunc
      error("Named arguments `data` and `eval` are required.")
   end

   @parallel (vcat) for mask in validationmasks(data, kfold)
      trainingdata = data[.!mask, :]
      evaluationdata = data[mask, :]
      cvres = CrossValidationResult{P}(length(params))
      for i in eachindex(params)
         model = fit(trainingdata, params[i])
         score = eval(model, evaluationdata)
         cvres[i] = CrossValidationItem{P}(params[i], score)
      end
      cvres
   end
end


"""
    crossvalidation_epochs(startmodel, train, epochs; ...)
Performs k-fold cross-validation, given
* `initmodel`: a function accepting a data set and returning an
   initialized or pre-trained model
* `train`: a function that accepts a model and a data set
   and returns a further trained model
* `epochs`: an integer specifying the number of training epochs (defaults to 20).

For return value and other named arguments, see `crossvalidation`.
"""
function crossvalidation_epochs(initmodel::Function, train::Function,
      epochs::Int = 20;
      eval::Function = emptyfunc,
      data::Matrix{Float64} = Matrix{Float64}(0, 0),
      kfold::Int = 10)

   if isempty(data) || eval == emptyfunc
      error("Named arguments `data` and `eval` are required.")
   end

   @parallel (vcat) for mask in validationmasks(data, kfold)
      trainingdata = data[.!mask, :]
      evaluationdata = data[mask, :]
      cvres = CrossValidationResult{Int}(epochs)
      model = initmodel(trainingdata)
      for epoch = 1:epochs
         model = train(model, trainingdata)
         score = eval(model, evaluationdata)
         cvres[epoch] = CrossValidationItem{Int}(epoch, score)
      end
      cvres
   end
end


"""
Returns an array of BitArrays to index the validation data sets´
for k-folg cross validation.
"""
function validationmasks(data::Matrix{Float64}, kfold::Int)
   nsamples = size(data, 1)
   batchranges = ranges(mostevenbatches(nsamples, kfold))
   map(rng -> [i in rng for i in 1:nsamples], batchranges)
end