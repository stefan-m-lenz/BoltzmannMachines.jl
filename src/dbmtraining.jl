
typealias BasicDBM Array{BernoulliRBM,1}
typealias Particles Array{Array{Float64,2},1}
typealias Particle Array{Array{Float64,1},1}


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

   hiddbm::BasicDBM

   function MultivisionDBM(visrbms, hiddbm)
      # initialize ranges of hidden units for different RBMs
      nvisrbms = length(visrbms)

      # calculate visible ranges
      visrbmsvisranges = Array{UnitRange{Int}}(nvisrbms)
      offset = 0
      for i = 1:nvisrbms
         nvisibleofcurrentvisrbm = length(visrbms[i].visbias)
         visrbmsvisranges[i] = offset + (1:nvisibleofcurrentvisrbm)
         offset += nvisibleofcurrentvisrbm
      end

      # calculate hidden ranges
      visrbmshidranges = Array{UnitRange{Int}}(nvisrbms)
      offset = 0
      for i = 1:nvisrbms
         nhiddenofcurrentvisrbm = length(visrbms[i].hidbias)
         visrbmshidranges[i] = offset + (1:nhiddenofcurrentvisrbm)
         offset += nhiddenofcurrentvisrbm
      end

      new(visrbms, visrbmsvisranges, visrbmshidranges, hiddbm)
   end
end

typealias AbstractDBM Union{BasicDBM, MultivisionDBM}

function MultivisionDBM{T<:AbstractRBM}(visrbms::Vector{T})
   MultivisionDBM(visrbms, BasicDBM())
end

function MultivisionDBM(visrbm::AbstractRBM)
   MultivisionDBM([visrbm], BasicDBM())
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
      hh = hiddenpotential(mvdbm.hiddbm[i], hh, 2.0) # intermediate layer, factor is 2.0
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

function combinedbiases(dbm::BasicDBM)
   biases = Particle(length(dbm) + 1)
   biases[1] = dbm[1].visbias
   for i = 2:length(dbm)
      biases[i] = dbm[i].visbias + dbm[i-1].hidbias
   end
   biases[end] = dbm[end].hidbias
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


# TODO document
function gibbssample!(particles::Particles,
      mvdbm::MultivisionDBM,
      nsteps::Int = 5, beta::Float64 = 1.0)

   nhiddenlayers = length(mvdbm.hiddbm) + 1

   for step in 1:nsteps

      # save state of first hidden layer
      oldstate = copy(particles[2])

      # input of first hidden layer from second hidden layer
      hinputtop = particles[3] * mvdbm.hiddbm[1].weights'
      broadcast!(+, hinputtop, hinputtop, mvdbm.hiddbm[1].visbias')

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
         broadcast!(+, input, input, mvdbm.hiddbm[i-1].hidbias')
         if i < nhiddenlayers
            input += particles[i+2] * mvdbm.hiddbm[i].weights'
            broadcast!(+, input, input, mvdbm.hiddbm[i].visbias')
         end
         oldstate = copy(particles[i+1])
         particles[i+1] = bernoulli(sigm(beta * input))
      end
   end

   particles
end

function gibbssample!(particles::Particles, dbm::BasicDBM,
      steps::Int = 5, beta::Float64 = 1.0)

   for step in 1:steps
      oldstate = copy(particles[1])
      for i = 1:length(particles)
         input = zeros(particles[i])
         if i < length(particles)
            input += particles[i+1] * dbm[i].weights'
            broadcast!(+, input, input, dbm[i].visbias')
         end
         if i > 1
            input += oldstate * dbm[i-1].weights
            broadcast!(+, input, input, dbm[i-1].hidbias')
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
function initparticles(dbm::BasicDBM, nparticles::Int)
   nlayers = length(dbm) + 1
   particles = Particles(nlayers)
   particles[1] = rand([0.0 1.0], nparticles, length(dbm[1].visbias))
   for i in 2:nlayers
      particles[i] = rand([0.0 1.0], nparticles, length(dbm[i-1].hidbias))
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
      particles[1][:,visiblerange] = rand([0.0 1.0], nparticles, length(mvdbm.visrbms[i].visbias))
   end
   particles[2:end] = initparticles(mvdbm.hiddbm, nparticles)
   particles
end


"""
    meanfield(dbm, x)
    meanfield(dbm, x, eps)
Computes the mean-field approximation for the data set `x` and
returns a matrix of particles for the DBM.
The number of particles is equal to the number of samples in `x`.
`eps` is the convergence criterion for the fix-point iteration, default 0.001.
"""
function meanfield(dbm::BasicDBM, x::Array{Float64,2}, eps::Float64 = 0.001)

   nlayers = length(dbm) + 1
   mu = Particles(nlayers)

   # Initialization with single bottom-up pass using twice the weights for all
   # but the topmost layer (see [Salakhutdinov+Hinton, 2012], p. 1985)
   mu[1] = x
   for i=2:(nlayers-1)
      mu[i] = hiddenpotential(dbm[i-1], mu[i-1], 2.0)
   end
   mu[nlayers] = hiddenpotential(dbm[nlayers-1], mu[nlayers-1])

   # mean-field updates until convergence criterion is met
   delta = 1.0
   while delta > eps
      delta = 0.0

      for i=2:(nlayers-1)
         newmu = mu[i+1]*dbm[i].weights' + mu[i-1]*dbm[i-1].weights
         broadcast!(+, newmu, newmu, (dbm[i-1].hidbias + dbm[i].visbias)')
         newmu = sigm(newmu)
         newdelta = maximum(abs(mu[i] - newmu))
         if newdelta > delta
            delta = newdelta
         end
         mu[i] = newmu
      end

      # last layer
      newmu = hiddenpotential(dbm[nlayers-1], mu[nlayers-1])
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
   nfirsthidden = length(mvdbm.hiddbm[1].visbias)
   mu = Particles(nlayers)

   # Initialization with single bottom-up pass
   mu[1] = x
   mu[2] = visiblestofirsthidden(mvdbm, x)
   for i = 3:(nlayers-1) # intermediate hidden layers after second
      mu[i] = hiddenpotential(mvdbm.hiddbm[i-2], mu[i-1], 2.0)
   end
   mu[nlayers] = hiddenpotential(mvdbm.hiddbm[nlayers-2], mu[nlayers-1])

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
         broadcast!(+, input, input, mvdbm.hiddbm[i-2].hidbias')
         if i < nlayers
            input += mu[i+1] * mvdbm.hiddbm[i-1].weights'
            broadcast!(+, input, input, mvdbm.hiddbm[i-1].visbias')
         end
         newmu = sigm(input)
         delta = max(delta, maximum(abs(mu[i] - newmu)))
         mu[i] = newmu
      end
   end

   mu
end


# TODO document parameters
"""
Performs greedy layerwise training for Deep Belief Networks or greedy layerwise
pretraining for Deep Boltzmann Machines.
"""
function stackrbms(x::Array{Float64,2};
      nhiddens::Array{Int,1} = size(x,2)*ones(2),
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
      hiddenval = BMs.hiddenpotential(dbmn[i-1], hiddenval, upfactor)
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

