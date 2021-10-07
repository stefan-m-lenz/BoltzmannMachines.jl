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
    blocksinnoise(nsamples, nvariables; ...)
Produces an artificial data set where there are sequences of consecutive 1s
in half of the binary samples.
The samples are labeled whether they belong one of the `nblocks`.
The samples are otherwise randomly generated with Bernoulli distributed noise.
The first return value is a matrix that has `nsamples` rows and `nvariables` columns
containing zeros and ones as values.
The second return value is a vector of labels.

## Optional named arguments:
* `noise`: Bernoulli distributed noise (probability for 1s)
* `blocklen`: length of sequences of consecutive variables set to 1 in subgroups
   of samples
* `nblocks`: number of different locations for sequences, i.e. the number of
   different subgroups with sequences of 1s
"""
function blocksinnoise(nsamples, nvariables; noise = 0.1, blocklen = 5, nblocks = 5)
   ret = float.(rand(nsamples, nvariables) .< noise)
   labels = fill(0, nsamples)
   randindexes = Random.randperm(nsamples)
   nsamplesperblock = div(nsamples, 2*nblocks)
   for block in 1:nblocks
      for i in randindexes[((block-1)*nsamplesperblock + 1):(block*nsamplesperblock)]
         ret[i, ((block-1)*blocklen+1):(block*blocklen)] .= 1.0
         labels[i] = block
      end
   end
   ret, labels
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
A function accepting everything, doing nothing.
Usable as default argument for functions as arguments.W
"""
function emptyfunc(x...; kwargs...)
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
    top2latentdims(dbm, x)
Get a two-dimensional representation for all the samples/rows
in the data set `x`, employing a given `dbm` for the dimension reduction.
For achieving the dimension reduction,
at first the mean-field activation induced by the samples
in the hidden nodes of the last/top hidden layer of the `dbm` is calculated.
The mean-field activation of the top hidden nodes is logit-transformed
to get a better separation.
If the number of hidden nodes in the last hidden layer is greater than 2, a
principal component analysis (PCA) is used
on these logit-transformed mean-field values to obtain a two-dimensional representation.
The result is a matrix with 2 columns, each row belonging to a sample/row in `x`.
"""
function top2latentdims(dbm::MultimodalDBM, x::AbstractArray)
   h = meanfield(dbm, x)[end]

   # logit transform
   h .= -log.(1 ./ h .- 1)

   function standardize(x)
      (x.- mean(x, dims = 1)) ./ std(x, dims = 1)
   end

   ntophidden = size(h, 2)
   if ntophidden > 2
      # reduce top activations further with PCA
      u, s, v = svd(standardize(h))
      return u[:, 1:2]
   elseif ntophidden == 2
      return h
   else
      # if there is only one hidden node, add zeros as 2nd dimension
      return hcat(h, zeros(eltype(h), size(h)...))
   end
end
