"""
    barsandstripes(nsamples, nvariables)
Generates a test data set. To see the structure in the data set, run e. g.
`reshape(barsandstripes(1, 16), 4, 4)` a few times.

Example from:
MacKay, D. (2003). Information Theory, Inference, and Learning Algorithms
"""
function barsandstripes(nsamples::Int, nvariables::Int)
   squareside = sqrt(nvariables)
   if squareside != floor(squareside)
      error("Number of variables must be a square number")
   end
   squareside = round(Int, squareside)
   x = zeros(nsamples, nvariables)
   for i in 1:nsamples
      sample = hcat([(rand() < 0.5 ? ones(squareside) : zeros(squareside))
            for j in 1:squareside]...)
      if rand() < 0.5
         sample = transpose(sample)
      end
      x[i,:] .= sample[:]
      fill!(sample, 0.0)
   end
   x
end


"""
    batchparallelized(f, n, op)
Distributes the work for executing the function `f` `n` times
on all the available workers and reduces the results with the operator `op`.
`f` is a function that gets a number (of tasks) to execute the tasks.

# Example:
    batchparallelized(n -> aislogimpweights(dbm; nparticles = n), 100, vcat)
"""
function batchparallelized(f::Function, n::Int, op::Function)
   batches = mostevenbatches(n)
   if length(batches) > 1
      return @sync @distributed (op) for batch in batches
         f(batch)
      end
   else
      return f(n)
   end
end


"""
    crossvalidation(x, monitoredfit; ...)
Performs k-fold cross-validation, given
* the data set `x` and
* `monitoredfit`: a function that fits and evaluates a model.
  As arguments it must accept:
  - a training data data set
  - a `DataDict` containing the evaluation data.


The return values of the calls to the `monitoredfit` function
are concatenated with `vcat`.
If the monitoredfit function returns `Monitor` objects,
`crossvalidation` returns a combined `Monitor` object that can be displayed
by creating a cross-validation plot via
`BoltzmannMachinesPlots.crossvalidationplot`.

# Optional named argument:
* `kfold`: specifies the `k` in "`k`-fold" (defaults to 10).

    crossvalidation(x, monitoredfit, pars; ...)
If additionaly a vector of parameters `pars` is given, `monitoredfit`
also expects an additional parameter from the parameter set.
"""
function crossvalidation(x::Matrix{Float64}, monitoredfit::Function, pars...;
      kfold::Int = 10)

   vcat(pmap(monitoredfit, crossvalidationargs(x, pars...)...)...)
end


"""
    crossvalidationargs(x, pars...; )

Returns a tuple of argument vectors containing the parameters for a function
such as the `monitoredfit` argument in `crossvalidation`.

Usage example:
    map(monitoredfit, crossvalidationargs(x)...)

# Optional named argument:
* `kfold`: see `crossvalidation`.
"""
function crossvalidationargs(x::Matrix{Float64}, pars ...; kfold::Int = 10)

   nsamples = size(x, 1)
   batchranges = BMs.ranges(BMs.mostevenbatches(nsamples, kfold))

   args_data = Vector{Tuple{Matrix{Float64}, BMs.DataDict}}(undef, kfold)
   for i in eachindex(batchranges)
      rng = batchranges[i]
      trainingdata = x[[!(j in rng) for j in 1:nsamples], :]
      evaluationdata = x[rng, :]
      datadict = BMs.DataDict(string(i) => evaluationdata)
      args_data[i] = (trainingdata, datadict)
   end

   # this implementation is inefficient and overly complicated

   if isempty(pars)
      args = map(arg -> (arg,), args_data)
   else
      args = (args_data, map(p -> collect(p), pars)...)
      args = vec(collect(Iterators.product(args...)))
   end

   # vector of tuples to tuples of vectors
   (
      map(arg -> arg[1][1], args), # x
      map(arg -> arg[1][2], args), # datadict
      [map(arg -> arg[i], args) for i in 2:length(args[1])]...
   )
end


"""
A function with no arguments doing nothing.
Usable as default argument for functions as arguments.W
"""
function emptyfunc
end


"""
    log1pexp(x)
Calculates log(1+exp(x)). For sufficiently large values of x, the approximation
log(1+exp(x)) ≈ x is used. This is useful to prevent overflow.
"""
function log1pexp(x::Float64)
   if x > 34
      return x
   else
      return log1p(exp(x))
   end
end

function logit(p::Float64)
   -log.(1.0 ./ p .- 1.0)
end


"""
    intensities(x)
    intensities(x, q1)
    intesities(x, q1, q2)
Performs a linear and monotonous transformation on the data set `x`
to fit it into the interval [0.0, 1.0] and returns the transformed data set.

If `q1` is specified, all values below or equal to the quantile specified  by `q1`
are mapped to 0.0.
All values above or equal to the quantile specified by `q2` are mapped to 1.0.
`q2` defaults to `1 - q1`.

The quantiles are calculated per column/variable.
"""
function intensities(data::Matrix{Float64}, q1::Float64 = 0.0, q2::Float64 = 1.0 - q1)
   mapslices(col -> begin
         bottomval = quantile(col, q1)
         topval = quantile(col, q2)
         col[col .< bottomval] .= bottomval
         col[col .> topval] .= topval
         col .-= bottomval
         col ./= topval - bottomval
         col
      end, data, dims = 1)
end


"""
    mostevenbatches(ntasks)
    mostevenbatches(ntasks, nbatches)
Splits a number of tasks `ntasks` into a number of batches `nbatches`.
The number of batches is by default `min(nworkers(), ntasks)`.
The returned result is a vector containing the numbers of tasks for each batch.
"""
function mostevenbatches(ntasks::Int, nbatches::Int = min(nworkers(), ntasks))

   minparticlesperbatch, nbatcheswithplus1 = divrem(ntasks, nbatches)
   batches = Vector{Int}(undef, nbatches)
   for i = 1:nbatches
      if i <= nbatcheswithplus1
         batches[i] = minparticlesperbatch + 1
      else
         batches[i] = minparticlesperbatch
      end
   end
   batches
end


"""
    curvebundles(...)
Generates an example dataset that can be visualized as bundles of trend curves
with added noise. Additional binary columns with labels may be added.

## Optional named arguments:
* `nbundles`: number of bundles
* `nperbundle`: number of sequences per bundle
* `nvariables`: number of variables in the sequences
* `noisesd`: standard deviation of the noise added on all sequences
* `addlabels`: add leading columns to the resulting dataset, specifying the
   membership to a bundle
* `pbreak`: probability that an intermediate point in a sequence is a
   breakpoint, defaults to 0.2.
* `breakval`: a function that expects no input and generates a single
  (random) value for a defining point of a piecewise linear sequence.
  Defaults to `rand`.

# Example:
To quickly grasp the idea, plot generated samples against the variable index:

    x = BMs.curvebundles(nvariables = 10, nbundles = 3,
                       nperbundle = 4, noisesd = 0.03,
                       addlabels = true)
    BoltzmannMachinesPlots.plotcurvebundles(x)
"""
function curvebundles(;
      nbundles::Int = 3,
      nperbundle::Int = 50,
      nvariables::Int = 10,
      noisesd::Float64 = 0.05,
      addlabels::Bool = false,
      pbreak::Float64 = 0.2,
      breakval::Function = rand)

   x = piecewiselinearsequences(nbundles, nvariables;
         pbreak = pbreak, breakval = breakval)
   x = repeat(x, nperbundle)
   nsamples = size(x, 1)
   x .+= noisesd * randn(nsamples, nvariables)

   if addlabels
      nlabelvars = Int(ceil(log(nbundles)/log(2)))
      labels = Matrix{Float64}(undef, nbundles, nlabelvars)
      labels[1,:] .= 0.0
      for j = 2:nbundles
         labels[j,:] .= labels[j-1,:]
         next!(view(labels, j, :))
      end
      labels = repeat(labels, nperbundle)
      x = hcat(labels, x)
   end

   x = x[randperm(nsamples), :]
   x
end


"""
    piecewiselinearsequences(nsequences, nvariables; ...)
Generates a dataset consisting of samples with values that
are piecewise linear functions of the variable index.

Optional named arguments: `pbreak`, `breakval`,
see `piecewiselinearsequencebundles`.
"""
function piecewiselinearsequences(nsequences::Int, nvariables::Int;
      pbreak::Float64 = 0.2, breakval::Function = rand)

   inbetweenvariables = 2:(nvariables-1)
   x = Matrix{Float64}(undef, nsequences, nvariables)
   for i = 1:nsequences
      breakpointindexes = [1; randsubseq(inbetweenvariables, pbreak); nvariables]
      breakpointvalues = [breakval() for i = 1:length(breakpointindexes)]
      lastbreakpoint = 1
      x[i, breakpointindexes] .= breakpointvalues
      for breakpoint in breakpointindexes[2:end]
         valdiff = (x[i, breakpoint] - x[i, lastbreakpoint]) /
               (breakpoint - lastbreakpoint)
         for j = (lastbreakpoint + 1):(breakpoint - 1)
            x[i, j] = x[i, j - 1] + valdiff
         end
         lastbreakpoint = breakpoint
      end
   end
   x
end


"""
    oneornone_decode(x, categories)
Returns a dataset such that
`x .== oneornone_decode(oneornone_encode(x, categories), categories)`.

For more, see `oneornone_encode`.
"""
function oneornone_decode(x::Matrix{Float64}, ncategories::Int)
   ncategoricalvariables, r = divrem(size(x, 2), ncategories - 1)
   if r != 0
      error("Wrong number of categories specified.")
   end
   oneornone_decode(x, fill(ncategories, ncategoricalvariables))
end

function oneornone_decode(x::Matrix{Float64}, nscategories::Vector{Int})
   varranges = ranges(nscategories .- 1)
   nsamples = size(x, 1)
   ncategoricalvariables = length(nscategories)
   xcategorical = zeros(nsamples, ncategoricalvariables)
   for k in 1:ncategoricalvariables
      for i in 1:nsamples
         category = findfirst(x[i, varranges[k]] .== 1.0)
         if category == nothing
            xcategorical[i, k] = 0.0
         else
            xcategorical[i, k] = category
         end
      end
   end
   xcategorical
end


"""
    oneornone_encode(x, categories)
Expects a data set `x` containing values 0.0, 1.0, 2.0 ... encoding the categories.
Returns a data set that encodes the variables/columns in `x` in multiple columns
with only values 0.0 and 1.0, similiar to the one-hot encoding with the deviation that
a zero is encoded as all-zeros.

The `categories` can be specified as
 * integer number if all variables have the same number of categories or as
 * integer vector, containing for each variable the number of categories encoded.

See also `oneornone_decode` for the reverse transformation.
"""
function oneornone_encode(x::M, nscategories::Vector{Int}
      ) where {N <: Number, M <:AbstractArray{N,2}}

   nsamples, norigvariables = size(x)
   n10variables = sum(nscategories .- 1)
   xout = zeros(nsamples, n10variables)
   varoffset = 0
   for k in 1:norigvariables
      for i in 1:nsamples
         if x[i, k] > 0.0
            xout[i, varoffset + Int(x[i,k])] = 1.0
         end
      end
      varoffset += nscategories[k] - 1
   end
   xout
end

function oneornone_encode(x, ncategories::Int
      ) where {N <: Number, M <:AbstractArray{N,2}}

   oneornone_encode(x, fill(ncategories, size(x, 2)))
end


"""
    splitdata(x, ratio)
Splits the data set `x` randomly in two data sets `x1` and `x2`, such that
the ratio `n2`/`n1` of the numbers of lines/samples in `x1` and `x2` is
approximately equal to the given `ratio`.
"""
function splitdata(x::Matrix{Float64}, ratio::Float64)
   nsamples = size(x,1)
   testidxs = randperm(nsamples)[1:(round(Int, nsamples*ratio))]
   xtest = x[testidxs,:]
   xtraining = x[setdiff(1:nsamples, testidxs),:]
   xtraining, xtest
end