const BasicDBM = Array{BernoulliRBM,1}

"""
`Particles` are an array of matrices.
The i'th matrix contains in each row the vector of states of the nodes
of the i'th layer of an RBM or a DBM. The set of rows with the same index define
an activation state in a Boltzmann Machine.
Therefore, the size of the i'th matrix is
(number of samples/particles, number of nodes in layer i).
"""
const Particles = Array{Array{Float64,2},1}

const Particle = Array{Array{Float64,1},1}

"""
Not implemented yet. Planned for future release.
"""
type MultimodalDBM
end

const AbstractDBM = Vector{<:AbstractRBM}


"""
    addlayer!(dbm, x)
Adds a pretrained layer to the BasicDBM `dbm`, given the dataset `x` as input
for the visible layer of the `dbm`.

# Optional keyword arguments:
* `isfirst`, `islast`: indicating that the new RBM should be trained as
   first (assumed if `dbm` is empty) or last layer of the DBM, respectively.
   If those flags are not set, the added layer is trained as intermediate layer.
   This information is needed to determine the factor for the weights during
   pretraining.
* `epochs`, `nhidden`, `learningrate`/`learningrates`, `pcd`, `cdsteps`,
  `monitoring`: used for fitting the weights of the last layer, see `fitrbm(x)`
"""
function addlayer!(dbm::BasicDBM, x::Matrix{Float64};
      isfirst::Bool = isempty(dbm),
      islast::Bool = false,
      nhidden::Int = size(x,2),
      epochs::Int = 10,
      learningrate::Float64 = 0.005,
      learningrates::Vector{Float64} = learningrate * ones(epochs),
      pcd::Bool = true,
      cdsteps::Int = 1,
      monitoring::Function = ((rbm, epoch) -> nothing))

   # propagate input x up to last hidden layer
   hh = x
   for i = 1:length(dbm)
      hh = hiddenpotential(dbm[i], hh, 2.0) # intermediate layer, factor is 2.0
   end

   upfactor = downfactor = 2.0
   if isfirst
      downfactor = 1.0
   end
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

   push!(dbm, rbm)
   dbm
end


"""
    combinedbiases(dbm)
Returns a vector containing in the i'th element the bias vector for the i'th
layer of the `dbm`. For intermediate layers, visible and hidden biases are
combined to a single bias vector.
"""
function combinedbiases(dbm::BasicDBM)
   biases = Particle(length(dbm) + 1)
   # create copy to avoid accidental modification of dbm
   biases[1] = copy(dbm[1].visbias)
   for i = 2:length(dbm)
      biases[i] = dbm[i].visbias + dbm[i-1].hidbias
   end
   biases[end] = copy(dbm[end].hidbias)
   biases
end


"""
    fitdbm(x; ...)
Fits a BasicDBM model to the data set `x`. The procedure consists of two parts:
First a stack of RBMs is pretrained in a greedy layerwise manner
(see `stackrbms(x)`). Then the weights are jointly trained using the
general Boltzmann Machine learning procedure (see `traindbm!(dbm,x)`).

# Optional keyword arguments (ordered by importance):
* `nhiddens`: vector that defines the number of nodes in the hidden layers of
   the DBM. The default value specifies two hidden layers with the same size
   as the visible layer.
* `epochs`: number of training epochs for joint training
* `epochspretraining`: number of training epochs for pretraining,
   defaults to `epochs`
* `learningrate`: learning rate for joint training, see `traindbm!`
* `learningratepretraining`: learning rate for pretraining,
   defaults to `learningrate`
* `nparticles`: number of particles used for sampling during joint training of
   DBM, default 100
"""
function fitdbm(x::Matrix{Float64};
      nhiddens::Vector{Int} = size(x,2)*ones(2),
      epochs::Int = 10,
      nparticles::Int = 100,
      learningrate::Float64 = 0.005,
      learningratepretraining::Float64 = learningrate,
      epochspretraining::Int = epochs)

   pretraineddbm = BMs.stackrbms(x, nhiddens = nhiddens,
         epochs = epochspretraining, predbm = true,
         learningrate = learningratepretraining)

   traindbm!(pretraineddbm, x, epochs = epochs, nparticles = nparticles,
      learningrate = learningrate)
end


"""
    gibbssample!(particles, dbm, nsteps)
Performs Gibbs sampling on the `particles` in the DBM `dbm` for `nsteps` steps.
(See also: `Particles`.)
In-between layers are assumed to contain only Bernoulli-distributed nodes.
"""
function gibbssample!(particles::Particles, dbm::AbstractDBM,
      nsteps::Int = 5,
      biases::Particle = BMs.combinedbiases(dbm))

   input = deepcopy(particles)
   input2 = deepcopy(particles)

   for step in 1:nsteps
      # first layer gets input only from layer above
      samplevisible!(input[1], dbm[1], particles[2])
   
      # intermediate layers get input from layers above and below
      for i = 2:(length(particles) - 1)
         visibleinput!(input[i], dbm[i], particles[i+1])
         hiddeninput!(input2[i], dbm[i-1], particles[i-1])
         input[i] .+= input2[i]
         sigm_bernoulli!(input[i]) # Bernoulli-sample from total input
      end
   
      # last layer gets only input from layer below
      samplehidden!(input[end], dbm[end], particles[end-1])

      # swap input and particles
      tmp = particles
      particles = input
      input = tmp
   end

   particles
end


"""
    newparticleslike(particles)
Creates new and uninitialized particles of the same dimensions as the given
`particles`.
"""
function newparticleslike(particles::Particles)
   newparticles = Particles(length(particles))
   for i in eachindex(particles)
      newparticles[i] = Matrix{Float64}(size(particles[i]))
   end
   newparticles
end


"""
    weightsinput!(input, input2, dbm, particles)
Computes the input that results only from the weights (without biases)
and the previous states in `particles` for all nodes in the DBM
`dbm` and stores it in `input`.
The state of the `particles` and the `dbm` is not altered.
`input2` must have the same size as `input` and `particle`.
For performance reasons, `input2` is used as preallocated space for storing
intermediate results.
"""
function weightsinput!(input::Particles, input2::Particles, dbm::BasicDBM,
      particles::Particles)

   # first layer gets input only from layer above
   A_mul_B!(input[1], particles[2], dbm[1].weights')

   # intermediate layers get input from layers above and below
   for i = 2:(length(particles) - 1)
      A_mul_B!(input2[i], particles[i+1], dbm[i].weights')
      A_mul_B!(input[i], particles[i-1], dbm[i-1].weights)
      input[i] .+= input2[i]
   end

   # last layer gets only input from layer below
   A_mul_B!(input[end], particles[end-1], dbm[end].weights)

   input
end


"""
    initparticles(dbm, nparticles)
Creates particles for Gibbs sampling in a DBM and initializes them with random
Bernoulli distributed (p=0.5) values.
(See also: `Particles`)

# Optional keyword arguments:
If the boolean flag `biased` is set to true, the values will be sampled
according to the biases of the `dbm`.
"""
function initparticles(dbm::BasicDBM, nparticles::Int; biased::Bool = false)
   nlayers = length(dbm) + 1
   particles = Particles(nlayers)
   if biased
      biases = combinedbiases(dbm)
      for i in 1:nlayers
         particles[i] = bernoulli!(repmat(sigm(biases[i])', nparticles))
      end
   else
      nunits = BMs.nunits(dbm)
      for i in 1:nlayers
         particles[i] = rand([0.0 1.0], nparticles, nunits[i])
      end
   end
   particles
end


"""
    meanfield(dbm, x)
    meanfield(dbm, x, eps)
Computes the mean-field approximation for the data set `x` and
returns a matrix of particles for the DBM.
The number of particles is equal to the number of samples in `x`.
`eps` is the convergence criterion for the fix-point iteration, default 0.001.
It is assumed that all nodes in in-between-layers are Bernoulli distributed.
"""
function meanfield(dbm::AbstractDBM, x::Array{Float64,2}, eps::Float64 = 0.001)

   nlayers = length(dbm) + 1
   mu = Particles(nlayers)

   # Initialization with single bottom-up pass using twice the weights for all
   # but the topmost layer (see [Salakhutdinov+Hinton, 2012], p. 1985)
   mu[1] = x
   for i=2:(nlayers-1)
      mu[i] = hiddenpotential(dbm[i-1], mu[i-1], 2.0)
   end
   mu[nlayers] = hiddenpotential(dbm[nlayers-1], mu[nlayers-1])

   newmu = newparticleslike(mu)
   newmu[1] = x
   input2 = newparticleslike(mu)

   # mean-field updates until convergence criterion is met
   delta = 1.0
   while delta > eps
      delta = 0.0

      for i = 2:(nlayers-1)
         # total input from layer below
         hiddeninput!(newmu[i], dbm[i-1], mu[i-1])
         # input from layer above
         visibleinput!(input2[i], dbm[i], mu[i+1])
         # combine input
         newmu[i] .+= input2[i]
         # By computing the potential,
         # the assumption is used that all nodes in in-between-layers
         # are Bernoulli-distributed:
         sigm!(newmu[i])

         delta = max(delta, maximum(abs.(mu[i] - newmu[i])))

         # swap new and old mu
         tmp = newmu[i]
         newmu[i] = mu[i]
         mu[i] = tmp
      end

      # last layer
      hiddenpotential!(newmu[end], dbm[end], mu[end-1])
      delta = max(delta, maximum(abs.(mu[end] - newmu[end])))

      # swap new and old mu
      tmp = newmu[end]
      newmu[end] = mu[end]
      mu[end] = tmp
   end

   mu
end


"""
    stackrbms(x; ...)
Performs greedy layerwise training for Deep Belief Networks or greedy layerwise
pretraining for Deep Boltzmann Machines and returns the trained model.

# Optional keyword arguments (ordered by importance):
* `predbm`: boolean indicating that the greedy layerwise training is
   pre-training for a DBM.
   If its value is false (default), a DBN is trained.
* `nhiddens`: vector containing the number of nodes of the i'th hidden layer in
   the i'th entry
* `epochs`: number of training epochs
* `learningrate`: learningrate, default 0.005
* `samplehidden`: boolean indicating that consequent layers are to be trained
   with sampled values instead of the deterministic potential,
   which is the default.
"""
function stackrbms(x::Array{Float64,2};
      nhiddens::Array{Int,1} = size(x,2)*ones(2),
      epochs::Int = 10,
      predbm::Bool = false,
      samplehidden::Bool = false,
      learningrate::Float64 = 0.005)

   nrbms = length(nhiddens)
   dbmn = Array{BernoulliRBM,1}(nrbms)

   upfactor = downfactor = 1.0
   if predbm
      upfactor = 2.0
   end

   dbmn[1] = BMs.fitrbm(x, nhidden = nhiddens[1], epochs = epochs,
         upfactor = upfactor, downfactor = downfactor, pcd = true,
         learningrate = learningrate)

   hiddenval = x
   for i=2:nrbms
      hiddenval = BMs.hiddenpotential(dbmn[i-1], hiddenval, upfactor)
      if samplehidden
         hiddenval = bernoulli(hiddenval)
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
            learningrate = learningrate)
   end

   dbmn
end


function sigm_bernoulli!(input::Particles)
   for i in eachindex(input)
      sigm_bernoulli!(input[i])
   end
   input
end

# const pgrid = collect(linspace(0.00001,0.99999,99999))
# const etagrid = log.(pgrid./(1.0-pgrid))

function sigm_bernoulli!(input::Matrix{Float64})
   for i in eachindex(input)
      @inbounds input[i] = 1.0*(rand() < 1.0/(1.0 + exp(-input[i])))
      # @inbounds input[i] = 1.0*(etagrid[Int(round(rand()*99998.0+1))] < input[i])
   end
   input
end


"""
    traindbm!(dbm, x; ...)
Trains the `dbm` (an `AbstractDBM`) using the learning procedure for a
general Boltzmann Machine with the training data set `x`.
A learning step consists of mean-field inference (positive phase),
stochastic approximation by Gibbs Sampling (negative phase) and the parameter
updates.

# Optional keyword arguments (ordered by importance):
* `epoch`: number of training epochs
* `learningrate`/`learningrates`: a vector of learning rates for each epoch to
   update the weights and biases. The learning rates should decrease with the
   epochs, e. g. like `a / (b + epoch)`. If only one value is given as
   `learningrate`, `a` and `b` are 11.0 and 10.0, respectively.
* `nparticles`: number of particles used for sampling, default 100
* `monitoring`: A function that is executed after each training epoch.
   It has to take the trained DBM and the current epoch as arguments.
"""
function traindbm!(dbm::AbstractDBM, x::Array{Float64,2};
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


"""
    traindbm!(dbm, x, particles, learningrate)
Trains the given `dbm` for one epoch.
"""
function traindbm!(dbm::BasicDBM, x::Array{Float64,2}, particles::Particles,
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
   dbmpart.visbias += learningrate*(lefta - righta)
   dbmpart.hidbias += learningrate*(leftb - rightb)
   nothing
end
