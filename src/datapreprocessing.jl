"""
Encapsulates all data needed to transform a vector of values into the interval [0.0, 1.0]
by a linear and monotonous transformation.

See `intensities_encode`, `intensities_decode` for the usage.
"""
struct IntensityTransformation
   bottomval::Float64
   topval::Float64

   function IntensityTransformation(v::AbstractVector{Float64},
         q1::Float64 = 0.0, q2::Float64 = 1.0 - q1)

      bottomval = quantile(v, q1)
      topval = quantile(v, q2)
      new(bottomval, topval)
   end
end


"""
    intensities(x)
    intensities(x, q1)
    intensities(x, q1, q2)
Performs a linear and monotonous transformation on the data set `x`
to fit it the values into the interval [0.0, 1.0].
For more information see `intensities_encode`, `intensities_decode`.
"""
function intensities(x::Matrix{Float64}, q1::Float64 = 0.0, q2::Float64 = 1.0 - q1)
   intensities_encode(x, q1, q2)[1]
end


"""
    intensities_decode(x, its)
Backtransforms the intensity values in the data set `x`
(values in the interval [0.0, 1.0])` to the range of the original values
and returns the new data set or vector.
The `its` argument contains the information about the transformation, as it is returned by
`intensities_encode`.

Note that the range is truncated if the original transformation used other quantiles
than 0.0 or 1.0 (minimum and maximum).

# Example:
    x = randn(5, 4)
    xint, its = intensities_encode(x, 0.05)
    dbm = fitdbm(xint)
    xgen = samples(dbm, 5)
    intensities_decode(xgen, its)
"""
function intensities_decode(v::AbstractVector{Float64}, it::IntensityTransformation)
   v .*= it.topval - it.bottomval
   v .+= it.bottomval
end

function intensities_decode(x::AbstractMatrix{Float64},
      its::AbstractVector{IntensityTransformation})

   if length(its) != size(x, 2)
      error("Number of transformations is not equal to the number of columns " *
            "that are to be transformed.")
   end
   mapreduce(i -> intensities_decode(x[:, i], its[i]), hcat, 1:length(its))
end


"""
    intensities_encode(x)
    intensities_encode(x, q1)
    intensities_encode(x, q1, q2)
Performs a linear and monotonous transformation on the data set `x`
to fit it into the interval [0.0, 1.0].
It returns the transformed data set as a first result and
the information to reverse the tranformation as a second result.
If you are only interested in the transformed values, you can use the function `intensities`.

If `q1` is specified, all values below or equal to the quantile specified  by `q1`
are mapped to 0.0.
All values above or equal to the quantile specified by `q2` are mapped to 1.0.
`q2` defaults to `1 - q1`.

The quantiles are calculated per column/variable.

See also `intensities_decode` for the reverse transformation.
"""
function intensities_encode(x::AbstractMatrix{Float64},
      q1::Float64 = 0.0, q2::Float64 = 1.0 - q1)

   intensitytransformations = mapreduce(
         i -> IntensityTransformation(x[:, i], q1, q2),
         vcat, 1:size(x, 2))

   function transform(it::IntensityTransformation, v::AbstractVector{Float64})
      v[v .< it.bottomval] .= it.bottomval
      v[v .> it.topval] .= it.topval
      v .-= it.bottomval
      v ./= it.topval - it.bottomval
   end

   transformed = mapreduce(i -> transform(intensitytransformations[i], x[:, i]),
         hcat, 1:length(intensitytransformations))

   transformed, intensitytransformations
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