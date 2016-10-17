abstract AbstractRBM

type BernoulliRBM <: AbstractRBM
   weights::Array{Float64,2}
   a::Array{Float64,1}
   b::Array{Float64,1}
end


"""
Parameters of a RBM with real valued, Gaussian distributed visible nodes and
binary, Bernoulli distributed hidden nodes.
"""
type GaussianBernoulliRBM <: AbstractRBM
   weights::Array{Float64,2}
   a::Array{Float64,1}
   b::Array{Float64,1}
   sd::Array{Float64,1}
end


type BernoulliGaussianRBM <: AbstractRBM
   weights::Array{Float64,2}
   a::Array{Float64,1}
   b::Array{Float64,1}
end


"""
Contains the parameters of an RBM with 0/1/2-valued, Binomial (n=2) distributed
visible nodes, and binary, Bernoulli distributed hidden nodes.
This model is equivalent to a BernoulliRBM in which every two visible nodes are
connected with the same weights to one hidden node.
The states (0,0) / (1,0) / (0,1) / (1,1) of the visible nodes connected with
with the same weights translate as states 0 / 1 / 1 / 2 in the
Binomial2BernoulliRBM.
"""
type Binomial2BernoulliRBM <: AbstractRBM
   weights::Matrix{Float64}
   a::Vector{Float64}
   b::Vector{Float64}
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
   of `learningrates` that contains a value for each epoch.
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
   the weights. By default, it is 0.0 and the standard deviation is not learned.
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
      sdlearningrates::Vector{Float64} = sdlearningrate * ones(epochs),
      sdinitfactor::Float64 = 0.0)

   rbm = initrbm(x, nhidden, rbmtype)
   if sdinitfactor > 0 && rbmtype == GaussianBernoulliRBM
      rbm.sd *= sdinitfactor
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

   for epoch = 1:epochs

      # Train RBM on data set
      trainrbm!(rbm, x, cdsteps = cdsteps, chainstate = chainstate,
            upfactor = upfactor, downfactor = downfactor,
            learningrate = learningrates[epoch],
            sdlearningrate = sdlearningrates[epoch])

      # Evaluation of learning after each training epoch
      monitoring(rbm, epoch)
   end

   rbm
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


function sdupdateterm(gbrbm::GaussianBernoulliRBM, v::Array{Float64,1}, h::Array{Float64,1})
   (v - gbrbm.a).^2 ./ (gbrbm.sd .^3) - (v ./ (gbrbm.sd .^ 2)) .* (gbrbm.weights * h)
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
      sdlearningrate::Float64 = 0.0)

   nsamples = size(x,1)

   # perform PCD if a chain state is provided as parameter
   pcd = !isempty(chainstate)

   for j = 1:nsamples
      v = vec(x[j,:])

      h = samplehidden(rbm, v, upfactor)

      # In case of CD, start Gibbs chain with the hidden state induced by the
      # states of the visible units. In case of PCD, start Gibbs chain with
      # previous state of the Gibbs chain.
      hmodel = pcd ? chainstate : h

      for step = 2:cdsteps
         vmodel = samplevisible(rbm, hmodel, downfactor)
         hmodel = samplehidden(rbm, vmodel, upfactor)
      end

      # Do not sample in last step to avoid unnecessary sampling noise
      vmodel = visiblepotential(rbm, hmodel, downfactor)
      hmodel = hiddenpotential(rbm, vmodel, upfactor)

      if pcd
         # Preserve state of chain in a way that changes are visible to the caller.
         copy!(chainstate, hmodel)
         bernoulli!(chainstate) # TODO change to support other types of hidden nodes
      end

      updateparameters!(rbm, v, vmodel, h, hmodel, learningrate, sdlearningrate)
   end

   rbm
end

function updateparameters!(rbm::AbstractRBM,
      v::Vector{Float64}, vmodel::Vector{Float64},
      h::Vector{Float64}, hmodel::Vector{Float64},
      learningrate::Float64,
      sdlearningrate::Float64)

   deltaw = v*h' - vmodel*hmodel'
   rbm.weights += deltaw * learningrate
   rbm.a += (v - vmodel) * learningrate
   rbm.b += (h - hmodel) * learningrate
   nothing
end

# See bottom of page 15 in [Krizhevsky, 2009].
function updateparameters!(gbrbm::GaussianBernoulliRBM,
      v::Vector{Float64}, vmodel::Vector{Float64},
      h::Vector{Float64}, hmodel::Vector{Float64},
      learningrate::Float64,
      sdlearningrate::Float64)

   sd = gbrbm.sd

   if sdlearningrate > 0
      gbrbm.sd -= (sdupdateterm(gbrbm, v, h) -
            sdupdateterm(gbrbm, vmodel, hmodel)) * sdlearningrate
   end

   v = v ./ sd
   vmodel = vmodel ./ sd

   deltaw = v*h' - vmodel*hmodel'
   gbrbm.weights += deltaw * learningrate
   gbrbm.a += (v - vmodel) * learningrate
   gbrbm.b += (h - hmodel) * learningrate
   nothing
end