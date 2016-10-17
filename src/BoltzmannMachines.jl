module BoltzmannMachines

const BMs = BoltzmannMachines

export
   AbstractBM,
      AbstractRBM,
         BernoulliRBM,
         BernoulliGaussianRBM,
         Binomial2BernoulliRBM,
         GaussianBernoulliRBM,
         fitrbm, trainrbm!, samplevisible, samplehidden, sampleparticles,
         hprob, vprob, samplerbm,
      AbstractDBM,
         DBMParam,
         MultivisionDBM,
         fitrbm, stackrbms, meanfield, gibbssample!, fitdbm, sampledbm


abstract AbstractRBM

type BernoulliRBM <: AbstractRBM
   weights::Array{Float64,2}
   a::Array{Float64,1}
   b::Array{Float64,1}
end

typealias DBMParam Array{BernoulliRBM,1}
typealias Particles Array{Array{Float64,2},1}
typealias Particle Array{Array{Float64,1},1}

"
Parameters of a RBM with real valued, Gaussian distributed visible nodes and
binary, Bernoulli distributed hidden nodes.
"
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
A MultivisionDBM consists of several visible layers (may have different
input types) and binary hidden layers.
Nodes of different visible layers are connected to non-overlapping parts
of the first hidden layer.
"""
type MultivisionDBM

   visrbms::Vector{AbstractRBM}
   visrbmsvisranges::Array{UnitRange{Int}}
   visrbmshidranges::Array{UnitRange{Int}}

   hiddbm::DBMParam

   function MultivisionDBM(visrbms, hiddbm)
      # initialize ranges of hidden units for different RBMs
      nvisrbms = length(visrbms)

      # calculate visible ranges
      visrbmsvisranges = Array{UnitRange{Int}}(nvisrbms)
      offset = 0
      for i = 1:nvisrbms
         nvisibleofcurrentvisrbm = length(visrbms[i].a)
         visrbmsvisranges[i] = offset + (1:nvisibleofcurrentvisrbm)
         offset += nvisibleofcurrentvisrbm
      end

      # calculate hidden ranges
      visrbmshidranges = Array{UnitRange{Int}}(nvisrbms)
      offset = 0
      for i = 1:nvisrbms
         nhiddenofcurrentvisrbm = length(visrbms[i].b)
         visrbmshidranges[i] = offset + (1:nhiddenofcurrentvisrbm)
         offset += nhiddenofcurrentvisrbm
      end

      new(visrbms, visrbmsvisranges, visrbmshidranges, hiddbm)
   end
end

typealias AbstractDBM Union{DBMParam, MultivisionDBM}
typealias AbstractBM Union{AbstractDBM, AbstractRBM}

function MultivisionDBM{T<:AbstractRBM}(visrbms::Vector{T})
   MultivisionDBM(visrbms, DBMParam())
end

function MultivisionDBM(visrbm::AbstractRBM)
   MultivisionDBM([visrbm], DBMParam())
end


"""
    addlayer!(mvdbm, x)
Adds a pretrained layer to the MultivisionDBM, given the dataset `x` as input
for the visible layer.
The variables/columns of `x` are divided among the visible RBMs.
"""
function addlayer!(mvdbm::MultivisionDBM, x::Matrix{Float64};
      islast::Bool = false,
      nhidden::Int = size(x,2),
      epochs::Int = 10,
      learningrate::Float64 = 0.005,
      learningrates::Vector{Float64} = learningrate * ones(epochs),
      pcd::Bool = true,
      cdsteps::Int = 1,
      monitoring::Function = ((rbm, epoch) -> nothing))

   # propagate input x up to last hidden layer
   hh = visiblestofirsthidden(mvdbm, x)
   for i = 1:length(mvdbm.hiddbm)
      hh = hprob(mvdbm.hiddbm[i], hh, 2.0) # intermediate layer, factor is 2.0
   end

   upfactor = downfactor = 2.0
   if islast
      upfactor = 1.0
   end

   rbm = fitrbm(hh, nhidden = nhidden, epochs = epochs,
         rbmtype = BernoulliRBM,
         learningrate = learningrate,
         learningrates = learningrates,
         pcd = pcd, cdsteps = 1,
         upfactor = upfactor, downfactor = downfactor,
         monitoring = monitoring)

   push!(mvdbm.hiddbm, rbm)
   mvdbm
end

"""
    visiblestofirsthidden(mvdbm, x)
Returns the activations induced by the forward pass of the dataset `x`
as inputs for the visible layer.
The variables/columns of `x` are divided among the visible RBMs.
"""
function visiblestofirsthidden(mvdbm::MultivisionDBM, x::Matrix{Float64})
   nsamples = size(x, 1)
   nvisrbms = length(mvdbm.visrbms)

   probs = Vector{Matrix{Float64}}(nvisrbms)

   for i = 1:nvisrbms
      visiblerange = mvdbm.visrbmsvisranges[i]
      input = hiddeninput(mvdbm.visrbms[i], x[:, visiblerange])
      probs[i] = bernoulli(sigm(2.0 * input))
   end

   hcat(probs...)
end

"
Computes the activation probability of the visible units in an RBM, given the
values `h` of the hidden units.
"
function vprob(gbrbm::GaussianBernoulliRBM, h::Array{Float64,1}, factor::Float64 = 1.0)
   factor*(gbrbm.a + gbrbm.sd .* (gbrbm.weights * h))
end

# factor is ignored, GaussianBernoulliRBMs should only be used in bottom layer of DBM
function vprob(gbrbm::GaussianBernoulliRBM, hh::Array{Float64,2}, factor::Float64 = 1.0)
   mu = hh*gbrbm.weights'
   broadcast!(.*, mu, mu, gbrbm.sd')
   broadcast!(+, mu, mu, gbrbm.a')
   mu
end

function vprob(bgrbm::BernoulliGaussianRBM, h::Array{Float64,1}, factor::Float64 = 1.0)
   sigm(factor*(bgrbm.weights*h + bgrbm.a))
end

function hprob(bgrbm::BernoulliGaussianRBM, v::Array{Float64,1}, factor::Float64 = 1.0)
   factor*(bgrbm.b + bgrbm.weights' * v)
end

function hprob(bgrbm::BernoulliGaussianRBM, vv::Array{Float64,2}, factor::Float64 = 1.0)
   # factor ignored
   broadcast(+, vv*bgrbm.weights, bgrbm.b')
end

function hiddeninput(gbrbm::GaussianBernoulliRBM, v::Array{Float64,1})
   gbrbm.weights'* (v ./ gbrbm.sd) + gbrbm.b
end

function hiddeninput(gbrbm::GaussianBernoulliRBM, vv::Array{Float64,2})
   mu = broadcast(./, vv, gbrbm.sd')
   mu = mu*gbrbm.weights
   broadcast!(+, mu, mu, gbrbm.b')
   mu
end

function hprob(gbrbm::GaussianBernoulliRBM, v::Array{Float64,1}, factor::Float64 = 1.0)
   sigm(factor*(hiddeninput(gbrbm, v)))
end

function hprob(gbrbm::GaussianBernoulliRBM, vv::Array{Float64,2}, factor::Float64 = 1.0)
   sigm(factor*hiddeninput(gbrbm, vv))
end

function hprob(b2brbm::Binomial2BernoulliRBM, v::Array{Float64}, factor::Float64 = 1.0)
   sigm(factor*(hiddeninput(b2brbm, v)))
end

function samplevisible(rbm::BernoulliRBM, h::Array{Float64,1}, factor::Float64 = 1.0)
   bernoulli!(vprob(rbm, h, factor))
end

function samplevisible(rbm::BernoulliRBM, hh::Array{Float64,2}, factor::Float64 = 1.0)
   bernoulli(vprob(rbm, hh, factor))
end

function samplevisible(gbrbm::GaussianBernoulliRBM, h::Array{Float64,1}, factor::Float64 = 1.0)
   vprob(gbrbm, h, factor) + gbrbm.sd .* randn(length(gbrbm.a))
end

function samplevisible(gbrbm::GaussianBernoulliRBM, hh::Array{Float64,2}, factor::Float64 = 1.0)
   hh = vprob(gbrbm, hh, factor)
   hh += broadcast(.*, randn(size(hh)), gbrbm.sd')
   hh
end

function samplevisible(bgrbm::BernoulliGaussianRBM, h::Array{Float64,1}, factor::Float64 = 1.0)
   bernoulli!(vprob(bgrbm, h, factor))
end

function samplevisible(bgrbm::BernoulliGaussianRBM, hh::Array{Float64,2}, factor::Float64 = 1.0)
   bernoulli(vprob(bgrbm, hh, factor))
end

function samplevisible{N}(b2brbm::Binomial2BernoulliRBM, h::Array{Float64,N}, factor::Float64 = 1.0)
   v = sigm(factor * visibleinput(b2brbm, h))
   bernoulli(v) + bernoulli(v)
end

function samplehidden(rbm::BernoulliRBM, v::Array{Float64,1}, factor::Float64 = 1.0)
   bernoulli!(hprob(rbm, v, factor))
end

function samplehidden(rbm::BernoulliRBM, vv::Array{Float64,2}, factor::Float64 = 1.0)
   bernoulli(hprob(rbm, vv, factor))
end

function samplehidden(gbrbm::GaussianBernoulliRBM, h::Array{Float64,1}, factor::Float64 = 1.0)
   bernoulli!(hprob(gbrbm, h, factor))
end

function samplehidden(gbrbm::GaussianBernoulliRBM, hh::Array{Float64,2}, factor::Float64 = 1.0)
   bernoulli(hprob(gbrbm, hh, factor))
end

function samplehidden(b2brbm::Binomial2BernoulliRBM, h::Array{Float64,1}, factor::Float64 = 1.0)
   bernoulli!(hprob(b2brbm, h, factor))
end

function samplehidden(b2brbm::Binomial2BernoulliRBM, hh::Array{Float64,2}, factor::Float64 = 1.0)
   bernoulli(hprob(b2brbm, hh, factor))
end

function samplehidden(bgrbm::BernoulliGaussianRBM, h::Array{Float64,1}, factor::Float64 = 1.0)
   hprob(bgrbm, h, factor) + randn(length(bgrbm.b))
end

function samplehidden(bgrbm::BernoulliGaussianRBM, hh::Array{Float64,2}, factor::Float64 = 1.0)
   hh = hprob(bgrbm, hh, factor)
   hh + randn(size(hh))
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

function sigm(x::Array{Float64,1})
   1./(1 + exp(-x))
end

function sigm(x::Array{Float64,2})
   1./(1 + exp(-x))
end

function bernoulli(x::Float64)
   float(rand() < x)
end

function bernoulli!(x::Array{Float64,1})
   map!(bernoulli, x)
end

function bernoulli(x::Array{Float64,1})
   float(rand(length(x)) .< x)
end

function bernoulli(x::Array{Float64,2})
   float(rand(size(x)) .< x)
end

"
Computes the activation probability of the hidden units in an RBM, given the
values `v` of the visible units.
"
function hprob(rbm::BernoulliRBM, v::Array{Float64,1}, factor::Float64 = 1.0)
   sigm(factor*(rbm.weights'*v + rbm.b))
end

"
Computes the activation probability of the visible units in an RBM, given the
values `h` of the hidden units.
"
function vprob(rbm::BernoulliRBM, h::Array{Float64,1}, factor::Float64 = 1.0)
   sigm(factor*(rbm.weights*h + rbm.a))
end

function vprob(rbm::BernoulliRBM, hh::Array{Float64,2}, factor::Float64 = 1.0)
   sigm(factor*visibleinput(rbm,hh))
end

function vprob(bgrbm::BernoulliGaussianRBM, hh::Array{Float64,2}, factor::Float64 = 1.0)
   sigm(factor*visibleinput(bgrbm,hh))
end

"""
    vprob(b2brbm, h)
Each visible node in the Binomial2BernoulliRBM is sampled from a Binomial(2,p)
distribution in the Gibbs steps. This functions returns the vector of values for
2*p. (The value is doubled to get a value in the same range as the sampled one.)
"""
function vprob{N}(b2brbm::Binomial2BernoulliRBM, h::Array{Float64,N}, factor::Float64 = 1.0)
   2*sigm(factor * visibleinput(b2brbm, h))
end

"
    hiddeninput(rbm, v)
Computes the total input of the hidden units in an RBM, given the activations
of the visible units `v`.
"
function hiddeninput(rbm::BernoulliRBM, v::Array{Float64,1})
   rbm.weights'*v + rbm.b
end

function hiddeninput(rbm::BernoulliRBM, vv::Array{Float64,2})
   input = vv*rbm.weights
   broadcast!(+, input, input, rbm.b')
end

function hiddeninput(b2brbm::Binomial2BernoulliRBM, v::Vector{Float64})
   # Hidden input is implicitly doubled
   # because the visible units range from 0 to 2.
   b2brbm.weights' * v + b2brbm.b
end

function hiddeninput(b2brbm::Binomial2BernoulliRBM, vv::Array{Float64,2})
   input = vv * b2brbm.weights
   broadcast!(+, input, input, b2brbm.b')
end

function hprob(rbm::BernoulliRBM, vv::Array{Float64,2}, factor::Float64 = 1.0)
   sigm(factor*hiddeninput(rbm,vv))
end

"
Computes the total input of the visible units in an RBM, given the activations
of the hidden units.
"
function visibleinput(rbm::BernoulliRBM, h::Array{Float64,1})
   rbm.weights*h + rbm.a
end

function visibleinput(rbm::BernoulliRBM, hh::Array{Float64,2})
   input = hh*rbm.weights'
   broadcast!(+, input, input, rbm.a')
end

function visibleinput(rbm::BernoulliGaussianRBM, h::Array{Float64,1})
   rbm.weights*h + rbm.a
end

function visibleinput(rbm::BernoulliGaussianRBM, hh::Array{Float64,2})
   input = hh*rbm.weights'
   broadcast!(+, input, input, rbm.a')
end

function visibleinput(b2brbm::Binomial2BernoulliRBM, h::Vector{Float64})
   b2brbm.weights * h + b2brbm.a
end

function visibleinput(b2brbm::Binomial2BernoulliRBM, hh::Matrix{Float64})
   input = hh * b2brbm.weights'
   broadcast!(+, input, input, b2brbm.a')
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

function combinedbiases(dbm::DBMParam)
   biases = Particle(length(dbm) + 1)
   biases[1] = dbm[1].a
   for i = 2:length(dbm)
      biases[i] = dbm[i].a + dbm[i-1].b
   end
   biases[end] = dbm[end].b
   biases
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
      vmodel = vprob(rbm, hmodel, downfactor)
      hmodel = hprob(rbm, vmodel, upfactor)

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

function sdupdateterm(gbrbm::GaussianBernoulliRBM, v::Array{Float64,1}, h::Array{Float64,1})
   (v - gbrbm.a).^2 ./ (gbrbm.sd .^3) - (v ./ (gbrbm.sd .^ 2)) .* (gbrbm.weights * h)
end

"
    samplerbm(rbm, n)
    samplerbm(rbm, n, burnin)
Generates `n` samples from a given `rbm` by running a single Gibbs chain with
`burnin`.
"
function samplerbm(rbm::BernoulliRBM, n::Int, burnin::Int = 10)

   nvisible = length(rbm.a)
   nhidden = length(rbm.b)

   x = zeros(n, nvisible)

   h = round(rand(nhidden))
   for i=1:(n+burnin)
      v = bernoulli!(vprob(rbm, h))
      h = bernoulli!(hprob(rbm, v))

      if i > burnin
         x[i-burnin,:] = v
      end
   end

   x
end

"""
Performs greedy layerwise training for Deep Belief Networks or greedy layerwise
pretraining for Deep Boltzmann Machines.
"""
function stackrbms(x::Array{Float64,2};
      nhiddens::Array{Int,1} = round(Int64, size(x,2) * ones(3)), # TODO besserer default-Wert für Anzahl hidden units
      epochs::Int = 10,
      predbm::Bool = false,
      samplehidden::Bool = true,
      learningrate::Float64 = 0.005,
      layerwisemonitoring::Function = (rbmindex, rbm, epoch) -> nothing)

   nrbms = length(nhiddens)
   dbmn = Array{BernoulliRBM,1}(nrbms)

   upfactor = downfactor = 1.0
   if predbm
      upfactor = 2.0
   end

   dbmn[1] = BMs.fitrbm(x, nhidden = nhiddens[1], epochs = epochs,
         upfactor = upfactor, downfactor = downfactor, pcd = true,
         learningrate = learningrate,
         monitoring = (rbm, epoch) -> layerwisemonitoring(1, rbm, epoch))

   hiddenval = x
   for i=2:nrbms
      hiddenval = BMs.hprob(dbmn[i-1], hiddenval, upfactor)
      if samplehidden
         hidden = bernoulli(hiddenval)
      end
      if predbm
         upfactor = downfactor = 2.0
         if i == nrbms
            upfactor = 1.0
         end
      else
         upfactor = downfactor = 1.0
      end
      dbmn[i] = BMs.fitrbm(hiddenval, nhidden = nhiddens[i], epochs = epochs,
            upfactor = upfactor, downfactor = downfactor, pcd = true,
            learningrate = learningrate,
            monitoring = (rbm, epoch) -> layerwisemonitoring(i, rbm, epoch))
   end

   dbmn
end

"""
    meanfield(dbm, x)
    meanfield(dbm, x, eps)
Computes the mean-field approximation for the data set `x` and
returns a matrix of particles for the DBM.
The number of particles is equal to the number of samples in `x`.
`eps` is the convergence criterion for the fix-point iteration, default 0.001.
"""
function meanfield(dbm::DBMParam, x::Array{Float64,2}, eps::Float64 = 0.001)

   nlayers = length(dbm) + 1
   mu = Particles(nlayers)

   # Initialization with single bottom-up pass using twice the weights for all
   # but the topmost layer (see [Salakhutdinov+Hinton, 2012], p. 1985)
   mu[1] = x
   for i=2:(nlayers-1)
      mu[i] = hprob(dbm[i-1], mu[i-1], 2.0)
   end
   mu[nlayers] = hprob(dbm[nlayers-1], mu[nlayers-1])

   # mean-field updates until convergence criterion is met
   delta = 1.0
   while delta > eps
      delta = 0.0

      for i=2:(nlayers-1)
         newmu = mu[i+1]*dbm[i].weights' + mu[i-1]*dbm[i-1].weights
         broadcast!(+, newmu, newmu, (dbm[i-1].b + dbm[i].a)')
         newmu = sigm(newmu)
         newdelta = maximum(abs(mu[i] - newmu))
         if newdelta > delta
            delta = newdelta
         end
         mu[i] = newmu
      end

      # last layer
      newmu = hprob(dbm[nlayers-1], mu[nlayers-1])
      newdelta = maximum(abs(mu[nlayers] - newmu))
      if newdelta > delta
         delta = newdelta
      end
      mu[nlayers] = newmu
   end

   mu
end

function meanfield(mvdbm::MultivisionDBM, x::Array{Float64}, eps::Float64 = 0.001)

   nlayers = length(mvdbm.hiddbm) + 2
   nsamples = size(x,1)
   nfirsthidden = length(mvdbm.hiddbm[1].a)
   mu = Particles(nlayers)

   # Initialization with single bottom-up pass
   mu[1] = x
   mu[2] = visiblestofirsthidden(mvdbm, x)
   for i = 3:(nlayers-1) # intermediate hidden layers after second
      mu[i] = hprob(mvdbm.hiddbm[i-2], mu[i-1], 2.0)
   end
   mu[nlayers] = hprob(mvdbm.hiddbm[nlayers-2], mu[nlayers-1])

   # mean-field updates until convergence criterion is met
   delta = Inf
   while delta > eps
      delta = 0.0

      # input of first hidden layer from second hidden layer
      hinputtop = mu[3] * mvdbm.hiddbm[1].weights'
      newmu = Matrix{Float64}(nsamples, nfirsthidden)
      for i = eachindex(mvdbm.visrbms)
         hiddenrange = mvdbm.visrbmshidranges[i]
         visiblerange = mvdbm.visrbmsvisranges[i]

         newmu[:,hiddenrange] = sigm(hinputtop[:,hiddenrange] +
               hiddeninput(mvdbm.visrbms[i], mu[1][:,visiblerange]))
      end
      delta = max(delta, maximum(abs(mu[2] - newmu)))
      mu[2] = newmu

      for i = 3:nlayers
         input = mu[i-1] * mvdbm.hiddbm[i-2].weights
         broadcast!(+, input, input, mvdbm.hiddbm[i-2].b')
         if i < nlayers
            input += mu[i+1] * mvdbm.hiddbm[i-1].weights'
            broadcast!(+, input, input, mvdbm.hiddbm[i-1].a')
         end
         newmu = sigm(input)
         delta = max(delta, maximum(abs(mu[i] - newmu)))
         mu[i] = newmu
      end
   end

   mu
end

function gibbssample!(particles::Particles,
      mvdbm::MultivisionDBM,
      nsteps::Int = 5, beta::Float64 = 1.0)

   nhiddenlayers = length(mvdbm.hiddbm) + 1

   for step in 1:nsteps

      # save state of first hidden layer
      oldstate = copy(particles[2])

      # input of first hidden layer from second hidden layer
      hinputtop = particles[3] * mvdbm.hiddbm[1].weights'
      broadcast!(+, hinputtop, hinputtop, mvdbm.hiddbm[1].a')

      for i = eachindex(mvdbm.visrbms)
         hiddenrange = mvdbm.visrbmshidranges[i]
         visiblerange = mvdbm.visrbmsvisranges[i]

         # sample first hidden from visible layers of visrbms
         # and second hidden layer
         input = hinputtop[:,hiddenrange] +
               hiddeninput(mvdbm.visrbms[i], particles[1][:,visiblerange])
         particles[2][:,hiddenrange] =
               bernoulli(sigm(beta * input))

         # sample visible from old first hidden
         particles[1][:,visiblerange] =
               samplevisible(mvdbm.visrbms[i], oldstate[:,hiddenrange], beta)
      end

      # sample other hidden layers
      for i = 2:nhiddenlayers
         input = oldstate * mvdbm.hiddbm[i-1].weights
         broadcast!(+, input, input, mvdbm.hiddbm[i-1].b')
         if i < nhiddenlayers
            input += particles[i+2] * mvdbm.hiddbm[i].weights'
            broadcast!(+, input, input, mvdbm.hiddbm[i].a')
         end
         oldstate = copy(particles[i+1])
         particles[i+1] = bernoulli(sigm(beta * input))
      end
   end

   particles
end

function gibbssample!(particles::Particles, dbm::DBMParam,
      steps::Int = 5, beta::Float64 = 1.0)

   for step in 1:steps
      oldstate = copy(particles[1])
      for i = 1:length(particles)
         input = zeros(particles[i])
         if i < length(particles)
            input += particles[i+1] * dbm[i].weights'
            broadcast!(+, input, input, dbm[i].a')
         end
         if i > 1
            input += oldstate * dbm[i-1].weights
            broadcast!(+, input, input, dbm[i-1].b')
         end
         oldstate = copy(particles[i])
         particles[i] = bernoulli(sigm(beta * input))
      end
   end

   particles
end

"""
    initparticles(dbm, nparticles)
Creates particles for Gibbs sampling in a DBM and initializes them with random
Bernoulli distributed (p=0.5) values.
Returns an array containing in the i'th entry a matrix of size
(`nparticles`, number of nodes in layer i) such that
the particles are contained in the rows of these matrices.
"""
function initparticles(dbm::DBMParam, nparticles::Int)
   nlayers = length(dbm) + 1
   particles = Particles(nlayers)
   particles[1] = rand([0.0 1.0], nparticles, length(dbm[1].a))
   for i in 2:nlayers
      particles[i] = rand([0.0 1.0], nparticles, length(dbm[i-1].b))
   end
   particles
end

function initparticles(mvdbm::MultivisionDBM, nparticles::Int)
   nhidrbms = length(mvdbm.hiddbm)
   if nhidrbms == 0
      error("Add layers to MultivisionDBM to be able to call `initparticles`")
   end

   particles = Particles(nhidrbms + 2)
   particles[1] = Matrix{Float64}(nparticles, mvdbm.visrbmsvisranges[end][end])
   for i = 1:length(mvdbm.visrbms)
      visiblerange = mvdbm.visrbmsvisranges[i]
      # TODO treat Gaussian visible differently
      particles[1][:,visiblerange] = rand([0.0 1.0], nparticles, length(mvdbm.visrbms[i].a))
   end
   particles[2:end] = initparticles(mvdbm.hiddbm, nparticles)
   particles
end


function fitpartdbmcore(x::Array{Float64,2},
      nhiddens::Array{Int,1},
      visibleindex,
      epochs::Int = 10,
      nparticles::Int = 100;
      jointepochs::Int = epochs,
      learningrate::Float64 = 0.005,
      jointlearningrate::Float64 = learningrate,
      jointinitscale::Float64 = 1.0,
      topn::Int=0)

   nparts = length(visibleindex)
   p = size(x)[2]

   nhiddensmat = zeros(Int,nparts,length(nhiddens))
   for i=1:(nparts-1)
      nhiddensmat[i,:] = floor(nhiddens ./ (p/length(visibleindex[i])))
   end
   nhiddensmat[nparts,:] = nhiddens .- vec(sum(nhiddensmat,1))

   partparams = Array{Array{BMs.BernoulliRBM,1},1}(nparts)

   for i=1:nparts
      partparams[i] = fitdbm(x[:,visibleindex[i]],vec(nhiddensmat[i,:]),epochs,nparticles,learningrate=learningrate)
   end

   params = Array{BMs.BernoulliRBM,1}(length(nhiddens))

   for i=1:length(nhiddens)
      curin = (i == 1 ? p : nhiddens[i-1])
      curout = nhiddens[i]
      if jointinitscale == 0.0
         weights = zeros(curin,curout)
      else
         weights = randn(curin,curout) / curin * jointinitscale
      end
      a = zeros(curin)
      b = zeros(curout)

      startposin = 1
      startposout = 1
      for j=1:nparts
         if i == 1
            inindex = visibleindex[j]
         else
            inindex = collect(startposin:(startposin+nhiddensmat[j,i-1]-1))
            startposin += nhiddensmat[j,i-1]
         end
         outindex = collect(startposout:(startposout+nhiddensmat[j,i]-1))
         startposout += nhiddensmat[j,i]

         weights[inindex,outindex] = partparams[j][i].weights
         a[inindex] = partparams[j][i].a
         b[outindex] = partparams[j][i].b

         params[i] = BernoulliRBM(weights,a,b)
      end
   end

   if topn > 0
      bottomparams = params
      params = Array{BMs.BernoulliRBM,1}(length(nhiddens)+1)
      params[1:length(bottomparams)] = bottomparams
      topweights = randn(nhiddens[end],topn) / nhiddens[end] * jointinitscale
      topa = zeros(nhiddens[end])
      topb = zeros(topn)
      params[end] = BernoulliRBM(topweights,topa,topb)
   end

   if jointepochs == 0
      return params
   end

   fitbm(x, params, epochs = jointepochs, nparticles = nparticles,learningrate=jointlearningrate)
end

function fitpartdbm(x::Array{Float64,2},
      nhiddens::Array{Int,1},
      nparts::Int = 2,
      epochs::Int = 10,
      nparticles::Int = 100;
      jointepochs::Int = epochs,
      learningrate::Float64 = 0.005,
      jointlearningrate::Float64 = learningrate,
      topn::Int=0)

   if (nparts < 2)
      return fitdbm(x,nhiddens,epochs,nparticles)
   end

   partitions = Int(ceil(log(2,nparts)))
   nparts = 2^partitions
   p = size(x)[2]

   visibleindex = BMs.vispartcore(BMs.vistabs(x),collect(1:size(x)[2]),partitions)

   fitpartdbmcore(x,nhiddens,visibleindex,epochs,nparticles,jointepochs=jointepochs,learningrate=learningrate,jointlearningrate=jointlearningrate)
end

"
Convenience Wrapper to pretrain an RBM-Stack and to jointly adjust the weights using fitbm
"
function fitdbm(x, nhiddens::Array{Int,1}, epochs=10, nparticles=100; learningrate::Float64 = 0.005)
   pretraineddbm = BMs.stackrbms(x, nhiddens = nhiddens, epochs = epochs, predbm = true, learningrate=learningrate)
   fitbm(x, pretraineddbm, epochs = epochs, nparticles = nparticles)
end

"""
    fitbm(x, dbm; ...)
Performs fitting of a Deep Boltzmann Machine. Expects a training data set `x`
and a pre-trained Deep Boltzmann machine `dbm` as arguments.

# Optional keyword arguments:
* `epoch`: number of training epochs
* `nparticles`: number of particles used for sampling, default 100
* The learning rate for each epoch is given as vector `learningrates`.
* `monitoring`: A function that is executed after each training epoch.
   It takes a DBM and the epoch as arguments.
"""
function fitbm(x::Array{Float64,2}, dbm::AbstractDBM;
      epochs::Int = 10,
      nparticles::Int = 100,
      learningrate::Float64 = 0.005,
      learningrates::Array{Float64,1} = learningrate*11.0 ./ (10.0 + (1:epochs)),
      monitoring::Function = ((dbm, epoch) -> nothing))

   if length(learningrates) < epochs
      error("Not enough learning rates for training epochs")
   end

   particles = initparticles(dbm, nparticles)

   for epoch=1:epochs
      traindbm!(dbm, x, particles, learningrates[epoch])

      # monitoring the learning process at the end of epoch
      monitoring(dbm, epoch)
   end

   dbm
end

function traindbm!(dbm::DBMParam, x::Array{Float64,2}, particles::Particles,
      learningrate::Float64)

   gibbssample!(particles, dbm)
   mu = meanfield(dbm, x)

   for i = eachindex(dbm)
      updatedbmpart!(dbm[i], learningrate,
            particles[i], particles[i+1], mu[i], mu[i+1])
   end

   dbm
end

function updatedbmpart!(dbmpart::BernoulliRBM,
      learningrate::Float64,
      vgibbs::Matrix{Float64},
      hgibbs::Matrix{Float64},
      vmeanfield::Matrix{Float64},
      hmeanfield::Matrix{Float64})

   updatedbmpartcore!(dbmpart, learningrate,
         vgibbs, hgibbs, vmeanfield, hmeanfield)
end

function updatedbmpart!(dbmpart::Binomial2BernoulliRBM,
      learningrate::Float64,
      vgibbs::Matrix{Float64},
      hgibbs::Matrix{Float64},
      vmeanfield::Matrix{Float64},
      hmeanfield::Matrix{Float64})

   vmeanfield /= 2
   vgibbs /= 2

   updatedbmpartcore!(dbmpart, learningrate,
         vgibbs, hgibbs, vmeanfield, hmeanfield)
end

function updatedbmpart!(dbmpart::GaussianBernoulliRBM,
      learningrate::Float64,
      vgibbs::Matrix{Float64},
      hgibbs::Matrix{Float64},
      vmeanfield::Matrix{Float64},
      hmeanfield::Matrix{Float64})

   # For respecting standard deviation in update rule
   # see [Srivastava+Salakhutdinov, 2014], p. 2962
   vmeanfield = broadcast(./, vmeanfield, dbmpart.sd')
   vgibbs = broadcast(./, vgibbs, dbmpart.sd')

   updatedbmpartcore!(dbmpart, learningrate,
         vgibbs, hgibbs, vmeanfield, hmeanfield)
end

function updatedbmpartcore!(dbmpart::AbstractRBM,
      learningrate::Float64,
      vgibbs::Matrix{Float64},
      hgibbs::Matrix{Float64},
      vmeanfield::Matrix{Float64},
      hmeanfield::Matrix{Float64})

   nsamples = size(vmeanfield, 1)
   nparticles = size(vgibbs, 1)

   leftw = vmeanfield' * hmeanfield / nsamples
   lefta = mean(vmeanfield, 1)[:]
   leftb = mean(hmeanfield, 1)[:]

   rightw = vgibbs' * hgibbs / nparticles
   righta = mean(vgibbs, 1)[:]
   rightb = mean(hgibbs, 1)[:]

   dbmpart.weights += learningrate*(leftw - rightw)
   dbmpart.a += learningrate*(lefta - righta)
   dbmpart.b += learningrate*(leftb - rightb)
   nothing
end

function traindbm!(mvdbm::MultivisionDBM, x::Array{Float64,2},
      particles::Particles, learningrate::Float64)

   gibbssample!(particles, mvdbm)
   mu = meanfield(mvdbm, x)

   # update parameters of each visible RBM
   for i = eachindex(mvdbm.visrbms)
      visiblerange = mvdbm.visrbmsvisranges[i]
      hiddenrange = mvdbm.visrbmshidranges[i]

      updatedbmpart!(mvdbm.visrbms[i], learningrate,
            particles[1][:,visiblerange], particles[2][:,hiddenrange],
            mu[1][:,visiblerange], mu[2][:,hiddenrange])
   end

   # update parameters of each RBM in hiddbm
   for i = eachindex(mvdbm.hiddbm)
      updatedbmpart!(mvdbm.hiddbm[i], learningrate,
            particles[i+1], particles[i+2], mu[i+1], mu[i+2])
   end

   mvdbm
end

function sampledbm(dbm::DBMParam, n::Int, burnin::Int=10, returnall=false)

   particles = initparticles(dbm, n)

   gibbssample!(particles, dbm, burnin)

   if returnall
      return particles
   else
      return particles[1]
   end
end

function sampleparticles(dbm::AbstractDBM, nparticles::Int, burnin::Int = 10)
   particles = initparticles(dbm, nparticles)
   gibbssample!(particles, dbm, burnin)
   particles
end

function sampleparticles(rbm::AbstractRBM, nparticles::Int, burnin::Int = 10)
   particles = Particles(2)
   particles[2] = rand([0.0 1.0], nparticles, length(rbm.b))

   for i=1:burnin
      particles[1] = samplevisible(rbm, particles[2])
      particles[2] = samplehidden(rbm, particles[1])
   end
   particles
end

function sampleparticles(gbrbm::GaussianBernoulliRBM, nparticles::Int, burnin::Int = 10)
   particles = invoke(sampleparticles, (AbstractRBM,Int,Int), gbrbm, nparticles, burnin-1)
   # do not sample in last step to avoid that the noise dominates the data
   particles[1] = vprob(gbrbm, particles[2])
   particles
end

function joinrbms{T<:AbstractRBM}(rbm1::T, rbm2::T)
   joinrbms(T[rbm1, rbm2])
end

function joinweights{T<:AbstractRBM}(rbms::Vector{T})
   jointnhidden = mapreduce(rbm -> length(rbm.b), +, 0, rbms)
   jointnvisible = mapreduce(rbm -> length(rbm.a), +, 0, rbms)
   jointweights = zeros(jointnvisible, jointnhidden)
   offset1 = 0
   offset2 = 0
   for i = eachindex(rbms)
      nvisible = length(rbms[i].a)
      nhidden = length(rbms[i].b)
      jointweights[offset1 + (1:nvisible), offset2 + (1:nhidden)] =
            rbms[i].weights
      offset1 += nvisible
      offset2 += nhidden
   end
   jointweights
end

function joinrbms(rbms::Vector{BernoulliRBM})
   jointvisiblebias = cat(1, map(rbm -> rbm.a, rbms)...)
   jointhiddenbias = cat(1, map(rbm -> rbm.b, rbms)...)
   BernoulliRBM(joinweights(rbms), jointvisiblebias, jointhiddenbias)
end

function joinrbms(rbms::Vector{GaussianBernoulliRBM})
   jointvisiblebias = cat(1, map(rbm -> rbm.a, rbms)...)
   jointhiddenbias = cat(1, map(rbm -> rbm.b, rbms)...)
   jointsd = cat(1, map(rbm -> rbm.sd, rbms)...)
   GaussianBernoulliRBM(joinweights(rbms), jointvisiblebias, jointhiddenbias,
         jointsd)
end

"
Returns a matrix, containing in the entry [i,j]
the associations between the variables contained in the columns i and j
in the matrix `x` by calculating chi-squared-statistics.
"
function vistabs(x::Array{Float64,2})
   nsamples, nvariables = size(x)

   # freq[i] is number of times when unit i is on
   freq = sum(x,1)

   chisq = zeros(nvariables, nvariables)

   for i=1:(nvariables-1)
      for j=(i+1):nvariables

         # Contingency table:
         #
         # i\j | 1 0
         #  --------
         #   1 | a b
         #   0 | c d

         a = sum(x[:,i] .* x[:,j])
         b = freq[i] - a
         c = freq[j] - a
         d = nsamples - (a+b+c)

         chisq[i,j] = chisq[j,i] = (a*d - b*c)^2*nsamples/
               (freq[i]*freq[j]*(nsamples-freq[i])*(nsamples-freq[j]))
      end
   end

   chisq
end

function vispartcore(chisq,refindex,npart)
   p = size(chisq)[2]
   halfp = div(p,2)

   ifirst = ind2sub(chisq,indmax(chisq)) # coordinates of maximum element
   ifirstvec = chisq[ifirst[1],:] # all associations with maximum element
   ifirstvec[ifirst[1]] = maximum(chisq) + 1.0 # Association of maximum elements with itself is made highest

   diffind = sortperm(vec(ifirstvec);rev=true)

   firstindex = diffind[1:halfp]
   secondindex = diffind[(halfp+1):p]

   firstoriindex = refindex[firstindex]
   secondoriindex = refindex[secondindex]

   if npart == 1
      return Array[firstoriindex, secondoriindex]
   else
      return append!(vispartcore(chisq[firstindex,firstindex],firstoriindex,npart-1),vispartcore(chisq[secondindex,secondindex],secondoriindex,npart-1))
   end
end

function parttoindex(partvecs,index)
   for i=1:length(partvecs)
      if index in partvecs[i]
         return i
      end
   end
   0
end

function vispartition(x,npart=1)
   chisq = BMs.vistabs(x)
   partvecs = BMs.vispartcore(chisq,collect(1:size(x)[2]),npart)
   map(_ -> parttoindex(partvecs,_),1:size(x)[2])
end

"
Returns a matrix containing in the entry [i,j] the Chi-squared test statistic
for testing that the i'th visible unit
and the j'th hidden unit of the last hidden layer are independent.
"
function vishidtabs(particles::Particles)
   hidindex = length(particles)
   n = size(particles[1])[1] # number of samples
   visfreq = sum(particles[1],1)
   hidfreq = sum(particles[hidindex],1)

   chisq = zeros(length(visfreq),length(hidfreq))
   for i=eachindex(visfreq)
      for j=eachindex(hidfreq)
         a = sum(particles[1][:,i] .* particles[hidindex][:,j])
         b = visfreq[i] - a
         c = hidfreq[j] - a
         d = n - (a+b+c)
         chisq[i,j] = (a*d - b*c)^2*n/(visfreq[i]*hidfreq[j]*(n-visfreq[i])*(n-hidfreq[j]))
      end
   end

   chisq
end

function tstatistics(particles::Particles)
   # for formula see "Methodik klinischer Studien", p.40+41
   hidindex = length(particles)
   nsamples, nvisible = size(particles[1])
   nhidden = size(particles[hidindex], 2)
   t = zeros(nvisible, nhidden)
   for i = 1:nvisible
      for j = 1:nhidden
         zeromask = (particles[hidindex][:,j] .== 0)
         n1 = sum(zeromask)
         n2 = nsamples - n1
         x1 = particles[1][zeromask,i]    # group 1: hidden unit j is 0
         x2 = particles[1][!zeromask,i]   # group 2: hidden unit j is 1
         m1 = mean(x1)
         m2 = mean(x2)
         s1 = 1 / (n1 - 1) * norm(x1 - m1)
         s2 = 1 / (n2 - 1) * norm(x2 - m2)
         s12 = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
         t[i,j] = sqrt(n1 * n2 / (n1 + n2)) * abs(m1 - m2) / sqrt(s12)
      end
   end
   t
end

function comparecrosstab(x::Array{Float64,2}, z::Array{Float64,2})

   xfreq = samplefrequencies(x)
   zfreq = samplefrequencies(z)

   sumsqdiff = 0.0

   for key in keys(zfreq)
      sumsqdiff += (get(xfreq, key, 0) - zfreq[key])^2 / zfreq[key]
   end

   sumsqdiff
end

function sumlogprob(x,z)

   zfreq = samplefrequencies(x)

   sumprob = 0.0
   nsamples = size(x, 1)
   for i=1:nsamples
      freq = get(zfreq, vec(x[i,:]), 0)
      if freq != 0
         sumprob += log(freq)
      end
   end

   sumprob
end

function partprob(x,z,partition)
   partidxs = sort(unique(partition))

   probs = ones(size(x)[1])
   for index in partidxs
      curmask = (partition .== index)
      zfreq = BMs.samplefrequencies(z[:,curmask])

      for i=1:size(x)[1]
         probs[i] *= get(zfreq, vec(x[i,curmask]), 0)
      end
      # println(probs')
   end

   probs
end

"
Compares DBMs based on generated data (in dictionary 'modelx')
with respect to the probability of test data ('xtest'),
based on partitions of the variables obtained from
reference visible and hidden states in 'refparticles'
"
function comparedbms(modelx,refparticles::Particles,xtest)
   chires = BMs.vishidtabs(refparticles)
   maxchi = map(_ -> indmax(chires[_,:]),1:size(xtest)[2])
   comparedbms(modelx,maxchi,xtest)
end

function comparedbms(modelx,partition::Array{Int,1},xtest)
   loglik = Dict()
   for key in keys(modelx)
      loglik[key] = sum(log(BMs.partprob(xtest,modelx[key],partition)))
   end
   loglik
end

# TODO Funktionen comparedbms und partprob mit folgender Funktion ersetzen, wenn ausreichend getestet:
"
Estimates the log-likelihood of each of the given DBMs by partitioning the
visible units in blocks that are independent from each other and estimating the
probabilites for these blocks by their frequency in generated samples.
"
function haraldsloglikelihoods(dbms, x::Array{Float64,2};
      ntogenerate::Int = 100000, burnin::Int = 10)

   # Use the first DBM as reference model, which is used to identify groups of
   # input units that will be assumed to be independent when calculating the
   # probability of the training samples.
   refparticles = BMs.sampledbm(dbms[1], ntogenerate, burnin, true)

   # Matrix containing assocations of visible to hidden units in the reference model
   chisq = BMs.vishidtabs(refparticles)

   # Array containing in the j'th entry the index of the hidden unit that is
   # highest associated with the j'th visible unit/variable:
   nvariables = size(x, 2)
   assochiddenindexes = map(_ -> indmax(chisq[_,:]), 1:nvariables)

   # Set of indexes of hidden units that are highly associated with
   # a visible unit in the reference model.
   assochiddengroup = sort(unique(assochiddenindexes))

   ndbms = length(dbms)
   loglikelihoods = Array{Float64,1}(ndbms)

   ntrainingsamples = size(x, 1)

   for j = 1:ndbms
      # generate samples from j'th DBM
      generated = BMs.sampledbm(dbms[j], ntogenerate, burnin)

      pxmodel = ones(ntrainingsamples)
      for hiddenindex in assochiddengroup
         correlatedvars = (assochiddenindexes .== hiddenindex)
         genfreq = BMs.samplefrequencies(generated[:, correlatedvars])

         for i = 1:ntrainingsamples
            # The frequency of (the part of) a training sample
            # occurring in the generated samples is an estimator for the
            # probability that the model assigns to the (part of the)
            # training sample.
            pxmodelcorrelatedvars = get(genfreq, vec(x[i, correlatedvars]), 0)

            # The probabilities of independent parts are
            # multiplied to get the probability of the sample.
            pxmodel[i] *= pxmodelcorrelatedvars
         end
      end

      loglikelihoods[j] = sum(log(pxmodel))

   end

   loglikelihoods
end

function findcliques(rbm::Union{BernoulliRBM, DBMParam},
      nparticles::Int = 10000, burnin::Int = 10)

   findcliques(vishidtabs(sampleparticles(rbm, nparticles, burnin)))
end

function findcliques(gbrbm::GaussianBernoulliRBM,
      nparticles::Int = 10000, burnin::Int = 10)

   findcliques(tstatistics(sampleparticles(gbrbm, nparticles, burnin)))
end

function findcliques(vishidstatistics::Matrix{Float64})
   nvariables = size(vishidstatistics, 1)

   # compute distances of visible units based on similarity of chi-square-statistics
   chidists = zeros(nvariables, nvariables)
   for i=1:(nvariables-1)
      for j=(i+1):nvariables
         chidists[i,j] = chidists[j,i] =
               sum((vishidstatistics[i,:] - vishidstatistics[j,:]).^2)
      end
   end

   # fill set of arrays to contain correlated variables
   cliques = Dict{Array{Int,1},Void}()
   for i = 1:nvariables

      # sort distances, first element is always zero after sorting
      # (distance to itself)
      sorteddistances = sort(vec(chidists[i,:]))

      # jumps are differences of distances
      jumps = sorteddistances[3:end] - sorteddistances[2:(end-1)]

      # maximum distance that will still be considered a neighbor is the
      # distance before the biggest jump
      maxdist = sorteddistances[indmax(jumps) + 1]

      neighmask = vec(chidists[i,:] .<= maxdist)
      if sum(neighmask) <= nvariables/2 # exclude complements of variable sets
         neighbors = (1:nvariables)[neighmask]
         cliques[neighbors] = nothing
      end
   end

   collect(keys(cliques))
end

"
Finds the best result generated in `n` tries by function `gen`
with respect to the comparison of a score , determined by function `score`.
Returns the result with the best score together with the list of all scores.
The function `gen` must be callable with no arguments and return one result,
the function `score` must take this result and return a number of type Float64.
"
function findbest(gen::Function, score::Function, n::Int;
      parallelized::Bool = false)

   if parallelized
      highestscoring = @parallel (reducehighscore) for i=1:n
         result = gen()
         resultscore = score(result)
         result, resultscore, [resultscore]
      end
      bestresult = highestscoring[1]
      scores = highestscoring[3]
   else
      scores = Array{Float64,1}(n)
      bestresult = gen()
      bestscore = score(bestresult)
      scores[1] = bestscore

      for j = 2:n
         nextresult = gen()
         nextscore = score(nextresult)
         if nextscore > bestscore
            bestresult = nextresult
            bestscore = nextscore
         end
         scores[j] = nextscore
      end
   end

   bestresult, scores
end

"
Gets two tuples of form (object, score of object, scores of all objects)
and compares the two scores contained in the second element of the tuple.
Returns the tuple
(object with highest score,
 score of object with highest score,
 array containing all scores).
"
function reducehighscore(t1::Tuple{Any, Float64, Array{Float64,1}},
      t2::Tuple{Any, Float64, Array{Float64,1}})

   if t1[2] > t2[2]
      bestobject = t1[1]
      bestscore = t1[2]
   else
      bestobject = t2[1]
      bestscore = t2[2]
   end
   bestobject, bestscore, vcat(t1[3], t2[3])
end

include("evaluating.jl")
include("monitoring.jl")

include("BMPlots.jl")

end # of module BoltzmannMachines


# References:
# [Salakhutdinov, 2015]: Learning Deep Generative Models
# [Salakhutdinov+Hinton, 2012]: An Efficient Learning Procedure for Deep Boltzmann Machines
# [Salakhutdinov, 2008]: Learning and Evaluating Boltzmann Machines
# [Krizhevsky, 2009] : Learning Multiple Layers of Features from Tiny Images
# [Srivastava+Salakhutdinov, 2014]: Multimodal Learning with Deep Boltzmann Machines
