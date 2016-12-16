
typealias BasicDBM Array{BernoulliRBM,1}

"""
`Particles` are an array of matrices.
The i'th matrix contains in each row the vector of states of the nodes
of the i'th layer of an RBM or a DBM. The set of rows with the same index define
an activation state in a Boltzmann Machine.
Therefore, the size of the i'th matrix is
(number of samples/particles, number of nodes in layer i).
"""
typealias Particles Array{Array{Float64,2},1}

typealias Particle Array{Array{Float64,1},1}


"""
A MultivisionDBM consists of several visible layers (may have different
input types) and binary hidden layers.
Nodes of different visible layers are connected to non-overlapping parts
of the first hidden layer.
"""
type MultivisionDBM

   visrbms::Vector{AbstractXBernoulliRBM}
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

function MultivisionDBM{T<:AbstractXBernoulliRBM}(visrbms::Vector{T})
   MultivisionDBM(visrbms, BasicDBM())
end

function MultivisionDBM(visrbm::AbstractXBernoulliRBM)
   MultivisionDBM([visrbm], BasicDBM())
end

"""
    addlayer!(dbm, x)
Adds a pretrained layer to the BasicDBM `dbm`, given the dataset `x` as input
for the visible layer of the `dbm`.

# Optional keyword arguments (apply to BasicDBM and MultivisionDBM):
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
    addlayer!(mvdbm, x)
Adds a pretrained layer to the MultivisionDBM, given the dataset `x` as input
for the visible layers of the bottom RBMs.
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

   addlayer!(mvdbm.hiddbm, visiblestofirsthidden(mvdbm, x);
         isfirst = false,
         islast = islast, nhidden = nhidden, epochs = epochs,
         learningrate = learningrate, learningrates = learningrates, pcd = pcd,
         cdsteps = cdsteps, monitoring = monitoring)
   mvdbm
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


# TODO document
function gibbssample!(particles::Particles,
      mvdbm::MultivisionDBM,
      nsteps::Int = 5)

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
               bernoulli(sigm(input))

         # sample visible from old first hidden
         particles[1][:,visiblerange] =
               samplevisible(mvdbm.visrbms[i], oldstate[:,hiddenrange])
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
         particles[i+1] = bernoulli(sigm(input))
      end
   end

   particles
end


"""
    gibbssample!(particles, dbm, nsteps)
Performs Gibbs sampling on the `particles` in the DBM `dbm` for `nsteps` steps.
(See also: `Particles`.)
"""
function gibbssample!(particles::Particles, dbm::BasicDBM,
      nsteps::Int = 5,
      biases::Particle = BMs.combinedbiases(dbm))

   input = deepcopy(particles)
   input2 = deepcopy(particles)

   for step in 1:nsteps
      weightsinput!(input, input2, dbm, particles)
      for i in eachindex(input)
         broadcast!(+, input[i], input[i], biases[i]')
         particles[i] .= input[i]
      end
      sigm_bernoulli!(particles)
   end

   particles
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
         particles[i] = bernoulli(repmat(sigm(biases[i])', nparticles))
      end
   else
      nunits = BMs.nunits(dbm)
      for i in 1:nlayers
         particles[i] = rand([0.0 1.0], nparticles, nunits[i])
      end
   end
   particles
end

function initparticles(mvdbm::MultivisionDBM, nparticles::Int; biased::Bool = false)

   nhidrbms = length(mvdbm.hiddbm)
   if nhidrbms == 0
      error("Add layers to MultivisionDBM to be able to call `initparticles`")
   end

   particles = Particles(nhidrbms + 2)
   particles[1] = Matrix{Float64}(nparticles, mvdbm.visrbmsvisranges[end][end])
   particles[2:end] = initparticles(mvdbm.hiddbm, nparticles, biased = biased)
   for i = 1:length(mvdbm.visrbms)
      visiblerange = mvdbm.visrbmsvisranges[i]
      hiddenrange = mvdbm.visrbmshidranges[i]
      rbm = deepcopy(mvdbm.visrbms[i])
      rbm.weights .= 0.0
      if !biased
         rbm.visbias .= 0.0
      end
      particles[1][:,visiblerange] =
               samplevisible(rbm, particles[2][:,hiddenrange])
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


# TODO document parameters
"""
Performs greedy layerwise training for Deep Belief Networks or greedy layerwise
pretraining for Deep Boltzmann Machines, using a partition for lower layers.
"""
function stackpartrbms(x::Array{Float64,2};
      nhiddens::Array{Int,1} = size(x,2)*ones(2),
      visibleindex::Array{Array{Int,1}} = [collect(1:div(size(x,2),2)),collect((div(size(x,2),2)+1):size(x,2))],
      partlayers::Int = 1,
      epochs::Int = 10,
      predbm::Bool = false,
      samplehidden::Bool = false,
      learningrate::Float64 = 0.005,
      jointinitscale::Float64 = 0.0)

   if partlayers < 1
      return stackrbms(nhiddens,epochs,predbm,samplehidden,learningrate,layerwisemonitoring)
   end

   p = size(x,2)

   nhiddenspart = nhiddens[1:partlayers]
   nrbmsjoint = length(nhiddens) - partlayers

   nhiddensmat = parthiddens(nhiddenspart,visibleindex,p)

   partparams, hiddenvals = stackparts(x,nhiddensmat,nrbmsjoint,visibleindex,
                                      epochs,predbm,samplehidden,learningrate)

   bottomparams, hiddenval = joinparams(partparams,p,nhiddenspart,nhiddensmat,visibleindex,
                                        jointinitscale,hiddenvals)

   if nrbmsjoint  == 0
      return bottomparams
   end

   stackjoint(hiddenval,nhiddens,bottomparams,
              epochs,predbm,samplehidden,learningrate)
end

function stackjoint(hiddenval,nhiddens,bottomparams,epochs,predbm,samplehidden,learningrate)
   nrbms = length(nhiddens)
   nrbmspart = length(bottomparams)

   dbmn = Array{BMs.BernoulliRBM,1}(nrbms)

   dbmn[1:nrbmspart] = bottomparams

   for i=(nrbmspart+1):nrbms
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

      if i < nrbms
         hiddenval = BMs.hiddenpotential(dbmn[i], hiddenval, upfactor)
         if samplehidden
            hiddenval = BMs.bernoulli(hiddenval)
         end
      end
   end

   dbmn
end

function joinparams(partparams,p,nhiddens,nhiddensmat,visibleindex,jointinitscale,hiddenvals)
   nparts = length(visibleindex)

   params = Array{BMs.BernoulliRBM,1}(length(nhiddens))

   hiddenval = zeros(size(hiddenvals[1],1),sum(nhiddensmat[:,end]))

   for i=1:length(nhiddens)
      curin = (i == 1 ? p : nhiddens[i-1])
      curout = nhiddens[i]
      if jointinitscale == 0.0
         weights = zeros(curin,curout)
      else
         weights = randn(curin,curout) / curin * jointinitscale
      end
      visbias = zeros(curin)
      hidbias = zeros(curout)

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
         visbias[inindex] = partparams[j][i].visbias
         hidbias[outindex] = partparams[j][i].hidbias

         params[i] = BoltzmannMachines.BernoulliRBM(weights,visbias,hidbias)

         if i == length(nhiddens)
            hiddenval[:,outindex] = hiddenvals[j]
         end
      end
   end

   params, hiddenval
end

function parthiddens(nhiddens,visibleindex,p)
   nparts = length(visibleindex)

   nhiddensmat = zeros(Int,nparts,length(nhiddens))
   for i=1:(nparts-1)
      nhiddensmat[i,:] = round(nhiddens ./ (p/length(visibleindex[i])))
   end
   nhiddensmat[nparts,:] = nhiddens .- vec(sum(nhiddensmat,1))

   nhiddensmat
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


function stackparts(x,nhiddensmat,nrbmsjoint,visibleindex,epochs,predbm,samplehidden,learningrate)
   nparts, nrbmspart = size(nhiddensmat)
   n, p = size(x)

   partparams = Array{Array{BMs.BernoulliRBM,1},1}(nparts)

   hiddenvals = Array{Array{Float64,2},1}(nparts)

   for apart=1:nparts
      partparams[apart] = Array{BMs.BernoulliRBM,1}(nrbmspart)

      upfactor = downfactor = 1.0
      if predbm
         upfactor = 2.0
      end

      partparams[apart][1] = BMs.fitrbm(x[:,visibleindex[apart]], nhidden = nhiddensmat[apart,1], epochs = epochs,
            upfactor = upfactor, downfactor = downfactor, pcd = true,
            learningrate = learningrate)

      hiddenval = x[:,visibleindex[apart]]

      if nrbmspart > 1
         for i=2:nrbmspart
            hiddenval = BMs.hiddenpotential(partparams[apart][i-1], hiddenval, upfactor)
            if samplehidden
               hidden = bernoulli(hiddenval)
            end
            if predbm
               upfactor = downfactor = 2.0
               if i == nrbmspart && nrbmsjoint == 0
                  upfactor = 1.0
               end
            else
               upfactor = downfactor = 1.0
            end
            partparams[apart][i] = BMs.fitrbm(hiddenval, nhidden = nhiddensmat[apart,i], epochs = epochs,
                  upfactor = upfactor, downfactor = downfactor, pcd = true,
                  learningrate = learningrate)
         end
      end

      if predbm && nrbmsjoint > 1
         upfactor = 2.0
      else
         upfactor = 1.0
      end

      hiddenval = BMs.hiddenpotential(partparams[apart][end], hiddenval, upfactor)
      if samplehidden
         hiddenval = BMs.bernoulli(hiddenval)
      end

      hiddenvals[apart] = hiddenval
   end

   partparams, hiddenvals
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
