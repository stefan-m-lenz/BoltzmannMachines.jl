module BoltzmannMachines

using Compat

const BMs = BoltzmannMachines

export
   AbstractBM,
      aislogimpweights, aisprecision, aisstandarddeviation,
      empiricalloglikelihood, energy, exactloglikelihood,
      exactlogpartitionfunction, loglikelihood,
      logpartitionfunction, logpartitionfunctionzeroweights,
      logproblowerbound, reconstructionerror,
      sampleparticles,
      AbstractRBM,
         BernoulliRBM,
         BernoulliGaussianRBM,
         Binomial2BernoulliRBM,
         GaussianBernoulliRBM,
         PartitionedRBM,
         fitrbm, freeenergy, initrbm,
         joinrbms, joindbms,
         trainrbm!,
         samplehidden, samplehidden!,
         samplevisible, samplevisible!,
         hiddenpotential, hiddenpotential!,
         hiddeninput, hiddeninput!,
         visiblepotential, visiblepotential!,
         visibleinput, visibleinput!,
      MultimodalDBM,
         BasicDBM,
         TrainLayer, TrainPartitionedLayer,
         addlayer!, fitdbm, gibbssample!, meanfield, stackrbms, traindbm!,
   Particle, Particles,
   Monitor, MonitoringItem, DataDict,
      monitorexactloglikelihood, monitorexactloglikelihood!,
      monitorfreeenergy, monitorfreeenergy!,
      monitorlogproblowerbound, monitorlogproblowerbound!,
      monitorloglikelihood, monitorloglikelihood!,
      monitorreconstructionerror, monitorreconstructionerror!,
      monitorweightsnorm, monitorweightsnorm!,
      propagateforward,
   crossvalidation,
   barsandstripes, splitdata,
   BMPlots


include("rbmtraining.jl")
include("rbmstacking.jl")
include("dbmtraining.jl")

const AbstractBM = Union{MultimodalDBM, AbstractRBM}


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
      return @sync @parallel (op) for batch in batches
         f(batch)
      end
   else
      return f(n)
   end
end


"""
    mostevenbatches(ntasks)
    mostevenbatches(ntasks, nbatches)
Splits a number of tasks `ntasks` into a number of batches `nbatches`.
The number of batches is by default equal to the number of workers.
The returned result is a vector containing the numbers of tasks for each batch.
"""
function mostevenbatches(ntasks::Int, nbatches::Int = nworkers())

   minparticlesperbatch, nbatcheswithplus1 = divrem(ntasks, nbatches)
   batches = Vector{Int}(nbatches)
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
A function with no arguments doing nothing.
Usable as default argument for functions as arguments.W
"""
function emptyfunc
end


function sigm(x::Float64)
   1./(1 + exp(-x))
end

function sigm(x::Array{Float64,1})
   1./(1 + exp.(-x))
end

function sigm(x::Array{Float64,2})
   1./(1 + exp.(-x))
end

function sigm!(x::M) where{M <:AbstractArray{Float64}}
   for i in eachindex(x)
      @inbounds x[i] = 1.0/(1.0 + exp(-x[i]))
   end
   x
end


function bernoulli(x::M) where{M <:AbstractArray{Float64}}
   ret = rand(size(x))
   ret .= float.(ret .< x)
   ret
end

function bernoulli!(x::M) where{M <:AbstractArray{Float64}}
   for i in eachindex(x)
      @inbounds x[i] = float(rand() < x[i])
   end
   x
end

function binomial2!(x::M) where{M <:AbstractArray{Float64}}
   for i in eachindex(x)
      @inbounds x[i] = float(rand() < x[i]) + float(rand() < x[i])
   end
   x
end


"""
    piecewiselinearsequences(nsamples, nvariables)
Generates a dataset consisting of samples with values that
are piecewise linear functions of the variable index.

## Optional named arguments:
 * `pbreak`: probability that an intermediate point is a breakpoint,
    defaults to 0.2.
 * `breakval`: a function that expects no input and generates a single
   (random) value. Defaults to `rand`.

# Example:
To quickly grasp the idea, plot generated samples against the variable index, e. g.:

    using Gadfly
    plot(piecewiselinearsequences(3, 10),
         x = Col.index, y = Col.value, color = Row.index, Geom.line,
         Guide.colorkey("Sample"), Guide.xlabel("Variable index"),
         Guide.ylabel("Value"), Scale.x_discrete, Scale.color_discrete)
"""
function piecewiselinearsequences(nsamples::Int, nvariables::Int,
      pbreak::Float64 = 0.2, breakval::Function = rand)

   inbetweenvariables = 2:(nvariables-1)
   x = Matrix{Float64}(nsamples, nvariables)
   for i = 1:nsamples
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

include("weightsjoining.jl")

include("evaluating.jl")
include("monitoring.jl")

include("crossvalidating.jl")

include("BMPlots.jl")

end # of module BoltzmannMachines
