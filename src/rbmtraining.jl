@compat abstract type AbstractRBM end

"""
Abstract super type for RBMs with binary and Bernoulli distributed hidden nodes.
"""
@compat abstract type AbstractXBernoulliRBM <: AbstractRBM end

"""
    BernoulliRBM(weights, visbias, hidbias)
Encapsulates the parameters of an RBM with Bernoulli distributed nodes.
* `weights`: matrix of weights with size
  (number of visible nodes, number of hidden nodes)
* `visbias`: bias vector for visible nodes
* `hidbias`: bias vector for hidden nodes
"""
type BernoulliRBM <: AbstractXBernoulliRBM
   weights::Array{Float64,2}
   visbias::Array{Float64,1}
   hidbias::Array{Float64,1}
end


"""
    GaussianBernoulliRBM(weights, visbias, hidbias, sd)
Encapsulates the parameters of an RBM with Gaussian distributed visible nodes
and Bernoulli distributed hidden nodes.
"""
type GaussianBernoulliRBM <: AbstractXBernoulliRBM
   weights::Array{Float64,2}
   visbias::Array{Float64,1}
   hidbias::Array{Float64,1}
   sd::Array{Float64,1}
end


"""
    BernoulliGaussianRBM(weights, visbias, hidbias)
Encapsulates the parameters of an RBM with Bernoulli distributed visible nodes
and Gaussian distributed hidden nodes.
The standard deviation of the Gaussian distribution is 1.
"""
type BernoulliGaussianRBM <: AbstractRBM
   weights::Array{Float64,2}
   visbias::Array{Float64,1}
   hidbias::Array{Float64,1}
end


"""
    Binomial2BernoulliRBM(weights, visbias, hidbias)
Encapsulates the parameters of an RBM with 0/1/2-valued, Binomial (n=2)
distributed visible nodes, and Bernoulli distributed hidden nodes.
This model is equivalent to a BernoulliRBM in which every two visible nodes are
connected with the same weights to each hidden node.
The states (0,0) / (1,0) / (0,1) / (1,1) of the visible nodes connected with
with the same weights translate as states 0 / 1 / 1 / 2 in the
Binomial2BernoulliRBM.
"""
type Binomial2BernoulliRBM <: AbstractXBernoulliRBM
   weights::Matrix{Float64}
   visbias::Vector{Float64}
   hidbias::Vector{Float64}
end


"""
    ranges(numbers)
Returns a vector of consecutive integer ranges, the first starting with 1.
The i'th such range spans over `numbers[i]` items.
"""
function ranges(numbers::Vector{Int})
   ranges = Vector{UnitRange{Int}}(length(numbers))
   offset = 0
   for i in eachindex(numbers)
      ranges[i] = offset + (1:numbers[i])
      offset += numbers[i]
   end
   ranges
end


"""
    PartitionedRBM(rbms)
Encapsulates several (parallel) AbstractRBMs that form one partitioned RBM.
The nodes of the parallel RBMs are not connected between the RBMs.
"""
type PartitionedRBM{R<:AbstractRBM} <: AbstractRBM
   rbms::Vector{R}
   visranges::Vector{UnitRange{Int}}
   hidranges::Vector{UnitRange{Int}}

   function PartitionedRBM{R}(rbms::Vector{R}) where R
      visranges = ranges([length(rbm.visbias) for rbm in rbms])
      hidranges = ranges([length(rbm.hidbias) for rbm in rbms])
      new(rbms, visranges, hidranges)
   end
end


"""
    fitrbm(x; ...)
Fits an RBM model to the data set `x`, using Stochastic Gradient Descent (SGD)
with Contrastive Divergence (CD), and returns it.

# Optional keyword arguments (ordered by importance):
* `rbmtype`: the type of the desired RBM. This must be a subtype of AbstractRBM
   and defaults to `BernoulliRBM`.
* `nhidden`: number of hidden units for returned RBM
* `epoch`: number of training epochs
* `learningrate`/`learningrates`: The learning rate for the weights and biases
   can be specified as single value, used throughout all epochs, or as a vector
   of `learningrates` that contains a value for each epoch. Defaults to 0.005.
* `pcd`: indicating whether Persistent Contrastive Divergence (PCD) is to
   be used (true, default) or simple CD that initializes the Gibbs Chain with
   the training sample (false)
* `cdsteps`: number of Gibbs sampling steps in CD/PCD, defaults to 1
* `monitoring`: a function that is executed after each training epoch.
   It takes an RBM and the epoch as arguments.
* `upfactor`, `downfactor`: If this function is used for pretraining a part of
   a DBM, it is necessary to multiply the weights of the RBM with factors.
* `sdlearningrate`/`sdlearningrates`: learning rate(s) for the
   standard deviation if training a `GaussianBernoulliRBM`. Ignored for other
   types of RBMs. It usually must be much smaller than the learning rates for
   the weights. By default, it is 0.0 which means that the standard deviation
   is not learned.
"""
function fitrbm(x::Matrix{Float64};
      nhidden::Int = size(x,2),
      epochs::Int = 10,
      upfactor::Float64 = 1.0,
      downfactor::Float64 = 1.0,
      learningrate::Float64 = 0.005,
      learningrates::Vector{Float64} = learningrate * ones(epochs),
      pcd::Bool = true,
      cdsteps::Int = 1,
      rbmtype::DataType = BernoulliRBM,
      monitoring::Function = ((rbm, epoch) -> nothing),

      # these arguments are only relevant for GaussianBernoulliRBMs:
      sdlearningrate::Float64 = 0.0,
      sdlearningrates::Vector{Float64} = Vector{Float64}(),
      sdgradclipnorm::Float64 = 0.0,
      sdinitfactor::Float64 = 0.0)

   rbm = initrbm(x, nhidden, rbmtype)

   if isempty(sdlearningrates)
      sdlearningrates = sdlearningrate * ones(epochs)
   end
   if sdinitfactor > 0 && rbmtype == GaussianBernoulliRBM
      rbm.sd .*= sdinitfactor
   end

   if length(learningrates) < epochs ||
         (rbmtype == GaussianBernoulliRBM && length(sdlearningrates) < epochs)
      error("Not enough learning rates for training epochs")
   end

   if pcd
      chainstate = rand(nhidden)
   else
      chainstate = Array{Float64,1}()
   end

   # allocate space for trainrbm!
   nvisible = size(x, 2)
   h = Vector{Float64}(nhidden)
   hmodel = Vector{Float64}(nhidden)
   vmodel = Vector{Float64}(nvisible)
   posupdate = Matrix{Float64}(nvisible, nhidden)
   negupdate = Matrix{Float64}(nvisible, nhidden)

   for epoch = 1:epochs

      # Train RBM on data set
      trainrbm!(rbm, x, cdsteps = cdsteps, chainstate = chainstate,
            upfactor = upfactor, downfactor = downfactor,
            learningrate = learningrates[epoch],
            sdlearningrate = sdlearningrates[epoch],
            sdgradclipnorm = sdgradclipnorm,
            h = h, hmodel = hmodel, vmodel = vmodel,
            posupdate = posupdate, negupdate = negupdate)

      # Evaluation of learning after each training epoch
      monitoring(rbm, epoch)
   end

   rbm
end


"""
    hiddeninput(rbm, v)
Computes the total input of the hidden units in the AbstractRBM `rbm`,
given the activations of the visible units `v`.
`v` may be a vector or a matrix that contains the samples in its rows.
"""
function hiddeninput(rbm::BernoulliRBM, v::Array{Float64,1})
   rbm.weights'*v + rbm.hidbias
end

function hiddeninput(rbm::BernoulliRBM, vv::Array{Float64,2})
   input = vv*rbm.weights
   broadcast!(+, input, input, rbm.hidbias')
end

function hiddeninput(b2brbm::Binomial2BernoulliRBM, v::Vector{Float64})
   # Hidden input is implicitly doubled
   # because the visible units range from 0 to 2.
   b2brbm.weights' * v + b2brbm.hidbias
end

function hiddeninput(b2brbm::Binomial2BernoulliRBM, vv::Array{Float64,2})
   input = vv * b2brbm.weights
   broadcast!(+, input, input, b2brbm.hidbias')
end

function hiddeninput(gbrbm::GaussianBernoulliRBM, v::Array{Float64,1})
   gbrbm.weights'* (v ./ gbrbm.sd) + gbrbm.hidbias
end

function hiddeninput(gbrbm::GaussianBernoulliRBM, vv::Array{Float64,2})
   hh = broadcast(/, vv, gbrbm.sd')
   hh = hh * gbrbm.weights
   broadcast!(+, hh, hh, gbrbm.hidbias')
   hh
end

function hiddeninput(prbm::PartitionedRBM, v::Vector{Float64})
   nhidden = prbm.hidranges[end][end]
   h = Vector{Float64}(nhidden)
   hiddeninput!(h, prbm, v)
end


"""
    hiddeninput!(h, rbm, v)
Like `hiddeninput`, but stores the returned result in `h`.
"""
function hiddeninput!(h::M, rbm::BernoulliRBM, v::M
      ) where{M <: AbstractArray{Float64,1}}

   At_mul_B!(h, rbm.weights, v)
   h .+= rbm.hidbias
end

function hiddeninput!(hh::M, rbm::BernoulliRBM, vv::M
      ) where{M <: AbstractArray{Float64,2}}

   A_mul_B!(hh, vv, rbm.weights)
   broadcast!(+, hh, hh, rbm.hidbias')
end

function hiddeninput!(h::M, rbm::Binomial2BernoulliRBM, v::M,
      ) where{M <: AbstractArray{Float64,1}}

   # again same code for Binomial2BernoulliRBM as for BernoulliRBM
   At_mul_B!(h, rbm.weights, v)
   h .+= rbm.hidbias
end

function hiddeninput!(hh::M, rbm::Binomial2BernoulliRBM, vv::M,
      ) where{M <: AbstractArray{Float64,2}}

   # again same code for Binomial2BernoulliRBM as for BernoulliRBM
   A_mul_B!(hh, vv, rbm.weights)
   broadcast!(+, hh, hh, rbm.hidbias')
end

function hiddeninput!(h::M, gbrbm::GaussianBernoulliRBM, v::M,
   ) where{M <: AbstractArray{Float64,1}}

   scaledweights = broadcast(/, gbrbm.weights, gbrbm.sd)
   At_mul_B!(h, scaledweights, v)
   h .+= gbrbm.hidbias
end

function hiddeninput!(hh::M, gbrbm::GaussianBernoulliRBM, vv::M,
      ) where{M <: AbstractArray{Float64,2}}

   scaledweights = broadcast(/, gbrbm.weights, gbrbm.sd)
   A_mul_B!(hh, vv, scaledweights)
   broadcast!(+, hh, hh, gbrbm.hidbias')
end

function hiddeninput!(h::M, prbm::PartitionedRBM, v::M,
      ) where{M <: AbstractArray{Float64,1}}

   for i in eachindex(prbm.rbms)
      visrange = prbm.visranges[i]
      hidrange = prbm.hidranges[i]
      hiddeninput!(view(h, hidrange), prbm.rbms[i], view(v, visrange))
   end
   h
end

function hiddeninput!(hh::M, prbm::PartitionedRBM, vv::M,
      ) where{M <: AbstractArray{Float64,2}}

   for i in eachindex(prbm.rbms)
      visrange = prbm.visranges[i]
      hidrange = prbm.hidranges[i]
      hiddeninput!(view(hh, :, hidrange), prbm.rbms[i], view(vv, :, visrange))
   end
   hh
end


"""
    hiddenpotential(rbm, v)
    hiddenpotential(rbm, v, factor)
Returns the potential for activations of the hidden nodes in the AbstractRBM
`rbm`, given the activations `v` of the visible nodes.
`v` may be a vector or a matrix that contains the samples in its rows.
The potential is a deterministic value to which sampling can be applied to get
the activations.
In RBMs with Bernoulli distributed hidden units, the potential of the hidden
nodes is the vector of probabilities for them to be turned on.

The total input can be scaled with the `factor`. This is needed when pretraining
the `rbm` as part of a DBM.
"""
function hiddenpotential(rbm::AbstractXBernoulliRBM, v::Array{Float64}, factor::Float64 = 1.0)
   sigm(factor*(hiddeninput(rbm, v)))
end

function hiddenpotential(bgrbm::BernoulliGaussianRBM, v::Array{Float64,1}, factor::Float64 = 1.0)
   factor * (bgrbm.hidbias + bgrbm.weights' * v)
end

function hiddenpotential(bgrbm::BernoulliGaussianRBM, vv::Array{Float64,2}, factor::Float64 = 1.0)
   factor * broadcast(+, vv*bgrbm.weights, bgrbm.hidbias')
end

function hiddenpotential(prbm::PartitionedRBM, vv::Array{Float64,2}, factor::Float64)
   nsamples = size(vv, 1)
   nhidden = prbm.hidranges[end][end]
   hh = Matrix{Float64}(nsamples, nhidden)
   hiddenpotential!(hh, prbm, vv, factor)
end

"""
    hiddenpotential!(hh, rbm, vv)
    hiddenpotential!(hh, rbm, vv, factor)
Like `hiddenpotential`, but stores the returned result in `hh`.
"""
function hiddenpotential!(hh::M, rbm::AbstractXBernoulliRBM, vv::M,
      factor::Float64 = 1.0) where{M <: AbstractArray{Float64}}

   hiddeninput!(hh, rbm, vv)
   if factor != 1.0
      hh .*= factor
   end
   sigm!(hh)
end

function hiddenpotential!(h::M, bgrbm::BernoulliGaussianRBM, v::M,
      factor::Float64 = 1.0) where{M <: AbstractArray{Float64,1}}

   At_mul_B!(h, bgrbm.weights, v)
   h .+= bgrbm.hidbias
   if factor != 1.0
      h .*= factor
   end
   h
end

function hiddenpotential!(hh::M, bgrbm::BernoulliGaussianRBM, vv::M,
      factor::Float64 = 1.0) where{M <: AbstractArray{Float64,2}}

   A_mul_B!(hh, vv, bgrbm.weights)
   broadcast!(+, hh, hh, bgrbm.hidbias')
   if factor != 1.0
      hh .*= factor
   end
   hh
end

function hiddenpotential!(hh::M, prbm::PartitionedRBM, vv::M,
      factor::Float64 = 1.0) where{M <: AbstractArray{Float64,2}}

   for i in eachindex(prbm.rbms)
      visrange = prbm.visranges[i]
      hidrange = prbm.hidranges[i]
      hiddenpotential!(view(hh, :, hidrange), prbm.rbms[i], view(vv, :, visrange),
         factor)
   end
   hh
end


"""
    initrbm(x, nhidden)
    initrbm(x, nhidden, rbmtype)
Creates a RBM with `nhidden` hidden units and initalizes its weights for
training on dataset `x`.
`rbmtype` can be a subtype of `AbstractRBM`, default is `BernoulliRBM`.
"""
function initrbm(x::Array{Float64,2}, nhidden::Int,
      rbmtype::DataType = BernoulliRBM)

   nsamples, nvisible = size(x)
   weights = randn(nvisible, nhidden)/sqrt(nvisible)
   hidbias = zeros(nhidden)

   if rbmtype == BernoulliRBM
      visbias = initvisiblebias(x)
      return BernoulliRBM(weights, visbias, hidbias)

   elseif rbmtype == GaussianBernoulliRBM
      visbias = vec(mean(x, 1))
      sd = vec(std(x, 1))
      #hidbias = randn(nhidden)/sqrt(nhidden)
      # weights = rand(nvisible, nhidden)
      # bengioglorotfactor = sqrt(6 / (nvisible + nhidden))
      # weights .*= 2 * bengioglorotfactor
      # weights .-= bengioglorotfactor
      # visbiassqnorm = sum(visbias.^2)
      #hidbias = randn(nhidden)*0.01
      # hidbias = [-(sum((weights .+ visbias)[:,j].^2) - visbiassqnorm)/2*0.1^2 + log(0.01) for j = 1:nhidden]
      return GaussianBernoulliRBM(weights, visbias, hidbias, sd)

   elseif rbmtype == BernoulliGaussianRBM
      visbias = initvisiblebias(x)
      return BernoulliGaussianRBM(weights, visbias, hidbias)

   elseif rbmtype == Binomial2BernoulliRBM
      visbias = initvisiblebias(x/2)
      return Binomial2BernoulliRBM(weights, visbias, hidbias)

   else
      error(string("Datatype for RBM is unsupported: ", rbmtype))
   end
end


"""
    initvisiblebias(x)
Returns sensible initial values for the visible bias for training an RBM
on the data set `x`.
"""
function initvisiblebias(x::Array{Float64,2})
   nvisible = size(x,2)
   initbias = zeros(nvisible)
   for j=1:nvisible
      empprob = mean(x[:,j])
      if empprob > 0
         initbias[j] = log(empprob/(1-empprob))
      end
   end
   initbias
end


"""
    samplehidden(rbm, v)
    samplehidden(rbm, v, factor)
Returns activations of the hidden nodes in the AbstractRBM `rbm`, sampled
from the state `v` of the visible nodes.
`v` may be a vector or a matrix that contains the samples in its rows.
For the `factor`, see `hiddenpotential(rbm, v, factor)`.
"""
function samplehidden(rbm::AbstractXBernoulliRBM, v::M,
      factor::Float64 = 1.0) where{M <: AbstractArray{Float64}}
   bernoulli!(hiddenpotential(rbm, v, factor))
end

function samplehidden(bgrbm::BernoulliGaussianRBM, v::M,
      factor::Float64 = 1.0) where{M <: AbstractArray{Float64}}
   h = hiddenpotential(bgrbm, v, factor)
   h .+ randn(size(h))
end


"""
    samplehidden!(h, rbm, v)
    samplehidden!(h, rbm, v, factor)
Like `samplehidden`, but stores the returned result in `h`.
"""
function samplehidden!(h, rbm::AbstractRBM, v::M,
      factor::Float64 = 1.0) where{M <: AbstractArray{Float64}}

   samplehiddenpotential!(hiddenpotential!(h, rbm, v, factor), rbm)
end


"""
    samplehiddenpotential!(h, rbm)
Samples the activation of the hidden nodes from the potential `h`
and stores the returned result in `h`.
"""
function samplehiddenpotential!(h::M, rbm::AbstractXBernoulliRBM
      ) where{M <: AbstractArray{Float64}}

   bernoulli!(h)
end

function samplehiddenpotential!(h::M, rbm::BernoulliGaussianRBM
      ) where{M <: AbstractArray{Float64}}

   h .+= randn(size(h))
end

function samplehiddenpotential!(h::M, prbm::PartitionedRBM
      ) where{M <: AbstractArray{Float64,1}}

   for i in eachindex(prbm.rbms)
      hidrange = prbm.hidranges[i]
      samplehiddenpotential!(view(h, hidrange), prbm.rbms[i])
   end
   h
end

function samplehiddenpotential!(hh::M, prbm::PartitionedRBM
      ) where{M <: AbstractArray{Float64,2}}

   for i in eachindex(prbm.rbms)
      hidrange = prbm.hidranges[i]
      samplehiddenpotential!(view(hh, :, hidrange), prbm.rbms[i])
   end
   hh
end


"""
    samplevisible(rbm, h)
    samplevisible(rbm, h, factor)
Returns activations of the visible nodes in the AbstractRBM `rbm`, sampled
from the state `h` of the hidden nodes.
`h` may be a vector or a matrix that contains the samples in its rows.
For the `factor`, see `visiblepotential(rbm, h, factor)`.
"""
function samplevisible(rbm::AbstractXBernoulliRBM, hh::M,
      factor::Float64 = 1.0) where{M <: AbstractArray{Float64}}
   bernoulli!(visiblepotential(rbm, hh, factor))
end

function samplevisible(b2brbm::Binomial2BernoulliRBM, hh::M,
      factor::Float64 = 1.0) where{M <: AbstractArray{Float64}}

   vv = visibleinput(b2brbm, hh)
   vv .*= factor
   sigm!(vv)
   binomial2!(vv)
end

function samplevisible(gbrbm::GaussianBernoulliRBM, h::Array{Float64,1}, factor::Float64 = 1.0)
   visiblepotential(gbrbm, h, factor) + gbrbm.sd .* randn(length(gbrbm.visbias))
end

function samplevisible(gbrbm::GaussianBernoulliRBM, hh::Array{Float64,2}, factor::Float64 = 1.0)
   hh = visiblepotential(gbrbm, hh, factor)
   hh += broadcast(.*, randn(size(hh)), gbrbm.sd')
   hh
end

"""
    samplevisible!(v, rbm, h)
    samplevisible!(v, rbm, h, factor)
Like `samplevisible`, but stores the returned result in `v`.
"""
function samplevisible!(v::M, rbm::AbstractRBM, h::M,
  factor::Float64 = 1.0) where{M <: AbstractArray{Float64}}

      samplevisiblepotential!(visiblepotential!(v, rbm, h, factor), rbm)
end

# TODO specialize for Binomial2BernoulliRBM to avoid multiplication and division by 2

"""
    samplehiddenpotential!(v, rbm)
Samples the activation of the visible nodes from the potential `v`
and stores the returned result in `v`.
"""
function samplevisiblepotential!(v::M, rbm::BernoulliRBM
      ) where{M <: AbstractArray{Float64}}
   bernoulli!(v)
end

function samplevisiblepotential!(v::M, bgrbm::BernoulliGaussianRBM
      ) where{M <: AbstractArray{Float64}}
   bernoulli!(v)
end

function samplevisiblepotential!(v::M, b2brbm::Binomial2BernoulliRBM
      ) where{M <: AbstractArray{Float64}}
   v ./= 2
   binomial2!(v)
end

function samplevisiblepotential!(v::M, gbrbm::GaussianBernoulliRBM
      ) where{M <: AbstractArray{Float64, 1}}

   gaussiannoise = randn(length(v))
   gaussiannoise .*= gbrbm.sd
   v .+= gaussiannoise
end

function samplevisiblepotential!(v::M, gbrbm::GaussianBernoulliRBM
      ) where{M <: AbstractArray{Float64, 2}}

   gaussiannoise = randn(size(v))
   gaussiannoise .*= gbrbm.sd'
   v .+= gaussiannoise
end

function samplevisiblepotential!(vv::M, prbm::PartitionedRBM
      ) where{M <: AbstractArray{Float64,2}}

   for i in eachindex(prbm.rbms)
      visrange = prbm.visranges[i]
      samplevisiblepotential!(view(vv, :, visrange), prbm.rbms[i])
   end
end


function sdupdateterm(gbrbm::GaussianBernoulliRBM, v::Array{Float64,1}, h::Array{Float64,1})
   (v - gbrbm.visbias).^2 ./ (gbrbm.sd .^3) - (v ./ (gbrbm.sd .^ 2)) .* (gbrbm.weights * h)
end


"""
    trainrbm!(rbm, x)
Trains the given `rbm` for one epoch. (See also function `fitrbm`.)

# Optional keyword arguments:
* `learningrate`, `cdsteps`, `sdlearningrate`, `upfactor`, `downfactor`:
   See documentation of function `fitrbm`.
* `chainstate`: a vector for holding the state of the RBM's hidden nodes. If
   it is specified, PCD is used.
"""
function trainrbm!(rbm::AbstractRBM, x::Array{Float64,2};
      chainstate::Array{Float64,1} = Array{Float64,1}(),
      upfactor::Float64 = 1.0,
      downfactor::Float64 = 1.0,
      learningrate::Float64 = 0.005,
      cdsteps::Int = 1,
      sdlearningrate::Float64 = 0.0,
      sdgradclipnorm::Float64 = 0.0,

      # write-only arguments for reusing allocated space:
      h::Vector{Float64} = Vector{Float64}(length(rbm.hidbias)),
      hmodel::Vector{Float64} = Vector{Float64}(length(rbm.hidbias)),
      vmodel::Vector{Float64} = Vector{Float64}(length(rbm.visbias)),
      posupdate::Matrix{Float64} = Matrix{Float64}(size(rbm.weights)),
      negupdate::Matrix{Float64} = Matrix{Float64}(size(rbm.weights)))

   nsamples = size(x,1)

   # perform PCD if a chain state is provided as parameter
   pcd = !isempty(chainstate)

   for j = 1:nsamples
      v = vec(x[j,:])

      # Calculate potential induced by visible nodes, used for update
      hiddenpotential!(h, rbm, v, upfactor)

      # In case of CD, start Gibbs chain with the hidden state induced by the
      # sample. In case of PCD, start Gibbs chain with
      # previous state of the Gibbs chain.
      if pcd
         hmodel = chainstate # note: state of chain will be visible by the caller
      else
         copy!(hmodel, h)
      end
      samplehiddenpotential!(hmodel, rbm)

      for step = 2:cdsteps
         samplevisible!(vmodel, rbm, hmodel, downfactor)
         samplehidden!(hmodel, rbm, vmodel, upfactor)
      end

      # Do not sample in last step to avoid unnecessary sampling noise
      visiblepotential!(vmodel, rbm, hmodel, downfactor)
      hiddenpotential!(hmodel, rbm, vmodel, upfactor)

      updateparameters!(rbm, v, vmodel, h, hmodel,
            learningrate, sdlearningrate, sdgradclipnorm,
            posupdate, negupdate)
   end

   rbm
end


"""
    visibleinput(rbm, h)
Returns activations of the visible nodes in the AbstractXBernoulliRBM `rbm`,
sampled from the state `h` of the hidden nodes.
`h` may be a vector or a matrix that contains the samples in its rows.
"""
function visibleinput(rbm::BernoulliRBM, h::Array{Float64,1})
   rbm.weights*h + rbm.visbias
end

function visibleinput(rbm::BernoulliRBM, hh::Array{Float64,2})
   input = hh*rbm.weights'
   broadcast!(+, input, input, rbm.visbias')
end

function visibleinput(rbm::BernoulliGaussianRBM, h::Array{Float64,1})
   rbm.weights*h + rbm.visbias
end

function visibleinput(rbm::BernoulliGaussianRBM, hh::Array{Float64,2})
   input = hh*rbm.weights'
   broadcast!(+, input, input, rbm.visbias')
end

function visibleinput(b2brbm::Binomial2BernoulliRBM, h::Vector{Float64})
   b2brbm.weights * h + b2brbm.visbias
end

function visibleinput(b2brbm::Binomial2BernoulliRBM, hh::Matrix{Float64})
   input = hh * b2brbm.weights'
   broadcast!(+, input, input, b2brbm.visbias')
end

function visibleinput(prbm::PartitionedRBM, h::Vector{Float64})
   nvisible = prbm.visranges[end][end]
   v = Matrix{Float64}(1, nvisible)
   vec(visibleinput!(v, pbrbm, h))
end


"""
    visibleinput!(v, rbm, h)
Like `visibleinput` but stores the returned result in `v`.
"""
function visibleinput!(v::M,
      rbm::Union{BernoulliRBM, BernoulliGaussianRBM, Binomial2BernoulliRBM},
      h::M) where {M <:AbstractArray{Float64,1}}

   A_mul_B!(v, rbm.weights, h)
   v .+= rbm.visbias
   v
end

function visibleinput!(vv::M,
      rbm::Union{BernoulliRBM, BernoulliGaussianRBM, Binomial2BernoulliRBM},
      hh::M) where {M <:AbstractArray{Float64,2}}

   A_mul_Bt!(vv, hh, rbm.weights)
   broadcast!(+, vv, vv, rbm.visbias')
end

function visibleinput!(v::M, prbm::PartitionedRBM, h::M
      ) where{M <: AbstractArray{Float64,1}}

   for i in eachindex(pbrbm.rbms)
      visrange = prbm.visranges[i]
      hidrange = prbm.hidranges[i]
      visibleinput!(view(v, visrange), prbm.rbms[i], view(h, hidrange))
   end
   v
end

function visibleinput!(v::M, prbm::PartitionedRBM, h::M
      ) where{M <: AbstractArray{Float64,2}}

   for i in eachindex(prbm.rbms)
      visrange = prbm.visranges[i]
      hidrange = prbm.hidranges[i]
      visibleinput!(view(v, :, visrange), prbm.rbms[i], view(h, :, hidrange))
   end
   v
end


"""
    visiblepotential(rbm, h)
    visiblepotential(rbm, h, factor)
Returns the potential for activations of the visible nodes in the AbstractRBM
`rbm`, given the activations `h` of the hidden nodes.
`h` may be a vector or a matrix that contains the samples in its rows.
The potential is a deterministic value to which sampling can be applied to get
the activations.

The total input can be scaled with the `factor`. This is needed when pretraining
the `rbm` as part of a DBM.

In RBMs with Bernoulli distributed visible units, the potential of the visible
nodes is the vector of probabilities for them to be turned on.
"""
function visiblepotential{N}(rbm::BernoulliRBM, h::Array{Float64,N}, factor::Float64 = 1.0)
   sigm(factor*visibleinput(rbm, h))
end

function visiblepotential{N}(bgrbm::BernoulliGaussianRBM, h::Array{Float64,N}, factor::Float64 = 1.0)
   sigm(factor*visibleinput(bgrbm, h))
end

"""
For a Binomial2BernoulliRBM, the visible units are sampled from a
Binomial(2,p) distribution in the Gibbs steps. In this case, the potential is
the vector of values for 2p.
(The value is doubled to get a value in the same range as the sampled one.)
"""
function visiblepotential{N}(b2brbm::Binomial2BernoulliRBM, h::Array{Float64,N}, factor::Float64 = 1.0)
   2*sigm(factor * visibleinput(b2brbm, h))
end

"""
For GaussianBernoulliRBMs, the potential of the visible nodes is the vector of
means of the Gaussian distributions for each node.
"""
function visiblepotential(gbrbm::GaussianBernoulliRBM, h::Array{Float64,1}, factor::Float64 = 1.0)
   factor*(gbrbm.visbias + gbrbm.sd .* (gbrbm.weights * h))
end

function visiblepotential(gbrbm::GaussianBernoulliRBM, hh::Array{Float64,2}, factor::Float64 = 1.0)
   # factor is ignored, GaussianBernoulliRBMs should only be used in bottom layer of DBM
   mu = hh*gbrbm.weights'
   broadcast!(*, mu, mu, gbrbm.sd')
   broadcast!(+, mu, mu, gbrbm.visbias')
   mu
end


"""
    visiblepotential!(v, rbm, h)
Like `visiblepotential` but stores the returned result in `v`.
"""
function visiblepotential!(v::M, rbm::BernoulliRBM, h::M,
      factor::Float64 = 1.0) where {M <: AbstractArray{Float64}}

   visibleinput!(v, rbm, h)
   if factor != 1.0
      v .*= factor
   end
   sigm!(v)
end

function visiblepotential!(v::M, rbm::BernoulliGaussianRBM, h::M,
      factor::Float64 = 1.0) where {M <: AbstractArray{Float64}}

   visibleinput!(v, rbm, h)
   if factor != 1.0
      v .*= factor
   end
   sigm!(v)
end

function visiblepotential!(v::M, rbm::Binomial2BernoulliRBM, h::M,
      factor::Float64 = 1.0) where{M <: AbstractArray{Float64}}

   visibleinput!(v, rbm, h)
   if factor != 1.0
      v .*= factor
   end
   sigm!(v)
   v .*= 2.0
end

function visiblepotential!(v::Vector{Float64}, gbrbm::GaussianBernoulliRBM,
      h::Vector{Float64}, factor::Float64 = 1.0)

   A_mul_B!(v, gbrbm.weights, h)
   v .*= gbrbm.sd
   v .+= gbrbm.visbias
   v
end

function visiblepotential!(v::M, gbrbm::GaussianBernoulliRBM, h::M,
      factor::Float64 = 1.0) where{M <: AbstractArray{Float64,2}}

   A_mul_Bt!(v, h, gbrbm.weights)
   broadcast!(*, v, v, gbrbm.sd')
   broadcast!(+, v, v, gbrbm.visbias')
end

function visiblepotential!(v::M, prbm::PartitionedRBM, h::M,
      factor::Float64 = 1.0) where{M <: AbstractArray{Float64,2}}

   for i in eachindex(prbm.rbms)
      visrange = prbm.visranges[i]
      hidrange = prbm.hidranges[i]
      visiblepotential!(view(v, :, visrange), prbm.rbms[i], view(h, :, hidrange))
   end
   v
end


"""
    updateparameters!(rbm, v, vmodel, h, hmodel, learningrate, sdlearningrate,
            posupdate, negupdate)
Updates the RBM `rbm` given the sample `v`,
the hidden activation `h` induced by the sample
the vectors `vmodel` and `hmodel` generated by Gibbs sampling, the `learningrate`,
the learningrate for the standard deviation `learningratesd` (only relevant for
GaussianBernoulliRBMs) and allocated space for the weights update
as in form of the write-only arguments `posupdate` and `negupdate`.

!!!  note
     `hmodel` must not be changed by implementations of `updateparameters!`
     since the persistent chain state is stored there.
"""
function updateparameters!(rbm::AbstractRBM,
      v::Vector{Float64}, vmodel::Vector{Float64},
      h::Vector{Float64}, hmodel::Vector{Float64},
      learningrate::Float64,
      sdlearningrate::Float64,
      sdgradclipnorm::Float64,
      posupdate::Matrix{Float64}, negupdate::Matrix{Float64})

   A_mul_Bt!(posupdate, v, h)
   A_mul_Bt!(negupdate, vmodel, hmodel)
   posupdate .-= negupdate
   posupdate .*= learningrate
   rbm.weights .+= posupdate
   rbm.visbias .+= (v - vmodel) * learningrate
   rbm.hidbias .+= (h - hmodel) * learningrate
   nothing
end


function updateparameters!(rbm::Binomial2BernoulliRBM,
      v::Vector{Float64}, vmodel::Vector{Float64},
      h::Vector{Float64}, hmodel::Vector{Float64},
      learningrate::Float64,
      sdlearningrate::Float64,
      sdgradclipnorm::Float64,
      posupdate::Matrix{Float64}, negupdate::Matrix{Float64})

   # To train a Binomial2BernoulliRBM exactly like
   # training a BernoulliRBM where each two nodes share the weights,
   # use half the learning rate in the visible nodes.
   learningratehidden = learningrate
   learningrate /= 2.0

   A_mul_Bt!(posupdate, v, h)
   A_mul_Bt!(negupdate, vmodel, hmodel)
   posupdate .-= negupdate
   posupdate .*= learningrate
   rbm.weights .+= posupdate
   rbm.visbias .+= (v - vmodel) * learningrate
   rbm.hidbias .+= (h - hmodel) * learningratehidden
   nothing
end

function updateparameters!(gbrbm::GaussianBernoulliRBM,
      v::Vector{Float64}, vmodel::Vector{Float64},
      h::Vector{Float64}, hmodel::Vector{Float64},
      learningrate::Float64,
      sdlearningrate::Float64,
      sdgradclipnorm::Float64,
      posupdate::Matrix{Float64}, negupdate::Matrix{Float64})

   # See bottom of page 15 in [Krizhevsky, 2009].

   if sdlearningrate > 0.0
      sdgrad = sdupdateterm(gbrbm, v, h) - sdupdateterm(gbrbm, vmodel, hmodel)

      v ./= gbrbm.sd
      vmodel ./= gbrbm.sd

      if sdgradclipnorm > 0.0
         sdgradnorm = norm(sdgrad)
         if sdgradnorm > sdgradclipnorm
            rescaling = sdgradclipnorm / sdgradnorm
            sdlearningrate *= rescaling
            learningrate *= rescaling
         end
      end
      sdgrad .*= sdlearningrate
      gbrbm.sd .+= sdgrad
      if any(gbrbm.sd .< 0.0)
         warn("SD-Update leading to negative standard deviation not performed")
         gbrbm.sd .-= sdgrad
      end
   else
      v ./= gbrbm.sd
      vmodel ./= gbrbm.sd
   end

   A_mul_Bt!(posupdate, v, h)
   A_mul_Bt!(negupdate, vmodel, hmodel)
   posupdate .-= negupdate
   posupdate .*= learningrate
   gbrbm.weights .+= posupdate
   gbrbm.hidbias .+= (h - hmodel) * learningrate

   gbrbm.visbias .+= (v - vmodel) ./ gbrbm.sd * learningrate
   nothing
end