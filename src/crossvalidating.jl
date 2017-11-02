
"""
    crossvalidation(x, monitoredfit; ...)
Performs k-fold cross-validation, given
* the data set `x` and
* `monitoredfit`: a function that fits and evaluates a model.
  As arguments it must accept:
  - a `Monitor` object in which the evaluations are stored
  - a `DataDict` containing the evaluation data
  - a training data data set.

`crossvalidation` returns a combined `Monitor` object that is best viewed
by creating a cross-validation plot via `BMPlots.crossvalidationplot`.

# Optional named argument:
* `kfold`: specifies the `k` in "`k`-fold" (defaults to 10).
"""
function crossvalidation(x::Matrix{Float64}, monitoredfit::Function;
      kfold::Int = 10)

   masks = crossvalidationmasks(x, kfold)

   @parallel (vcat) for i in eachindex(masks)
      mask = masks[i]
      trainingdata = x[.!mask, :]
      evaluationdata = x[mask, :]
      monitor = Monitor()
      datadict = DataDict(string(i) => evaluationdata)
      monitoredfit(monitor, datadict, x)
      monitor
   end
end


"""
Returns an array of BitArrays to index the validation data sets
for k-fold cross validation.
"""
function crossvalidationmasks(x::Matrix{Float64}, kfold::Int)
   nsamples = size(x, 1)
   batchranges = ranges(mostevenbatches(nsamples, kfold))
   map(rng -> [i in rng for i in 1:nsamples], batchranges)
end