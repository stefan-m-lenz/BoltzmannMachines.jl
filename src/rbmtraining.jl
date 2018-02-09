function assert_enoughvaluesforepochs(vname::String, v::Vector, epochs::Int)
   if length(v) < epochs
      error("Not enough `$vname` (vector of length $(length(v))) " .*
         "for training epochs ($epochs)")
   end
end


"""
    fitrbm(x; ...)
Fits an RBM model to the data set `x`, using Stochastic Gradient Descent (SGD)
with Contrastive Divergence (CD), and returns it.

# Optional keyword arguments (ordered by importance):
* `rbmtype`: the type of the RBM that is to be trained
   This must be a subtype of `AbstractRBM` and defaults to `BernoulliRBM`.
* `nhidden`: number of hidden units for the returned RBM
* `epochs`: number of training epochs
* `learningrate`/`learningrates`: The learning rate for the weights and biases
   can be specified as single value, used throughout all epochs, or as a vector
   of `learningrates` that contains a value for each epoch. Defaults to 0.005.
* `batchsize`: number of samples that are used for making one step in the
   stochastic gradient descent optimization algorithm. Default is 1.
* `pcd`: indicating whether Persistent Contrastive Divergence (PCD) is to
   be used (true, default) or simple CD that initializes the Gibbs Chain with
   the training sample (false)
* `cdsteps`: number of Gibbs sampling steps for (persistent)
   contrastive divergence, defaults to 1
* `monitoring`: a function that is executed after each training epoch.
   It takes an RBM and the epoch as arguments.
* `upfactor`, `downfactor`: If this function is used for pretraining a part of
   a DBM, it is necessary to multiply the weights of the RBM with factors.
* `sdlearningrate`/`sdlearningrates`: learning rate(s) for the
   standard deviation if training a `GaussianBernoulliRBM` or
   `GaussianBernoulliRBM2`. Ignored for other types of RBMs.
   It usually must be much smaller than the learning rates for
   the weights. By default it is 0.0, which means that the standard deviation
   is not learned.
* `startrbm`: start training with the parameters of the given RBM.
   If this argument is specified, `nhidden` and `rbmtype` are ignored.
"""
function fitrbm(x::Matrix{Float64};
      nhidden::Int = size(x,2),
      epochs::Int = 10,
      upfactor::Float64 = 1.0,
      downfactor::Float64 = 1.0,
      learningrate::Float64 = 0.005,
      learningrates::Vector{Float64} = fill(learningrate, epochs),
      pcd::Bool = true,
      cdsteps::Int = 1,
      batchsize::Int = 1,
      rbmtype::DataType = BernoulliRBM,
      startrbm::AbstractRBM = NoRBM(),
      monitoring::Function = nomonitoring,

      # these arguments are only relevant for GaussianBernoulliRBMs:
      sdlearningrate::Float64 = 0.0,
      sdlearningrates::Vector{Float64} = fill(sdlearningrate, epochs),
      sdgradclipnorm::Float64 = 0.0,
      sdinitfactor::Float64 = 0.0)

   if startrbm === NoRBM()
      rbm = initrbm(x, nhidden, rbmtype)
   else
      rbm = deepcopy(startrbm)
      nhidden = nhiddennodes(startrbm)
   end

   assert_enoughvaluesforepochs("learningrates", learningrates, epochs)
   assert_enoughvaluesforepochs("sdlearningrates", sdlearningrates, epochs)

   if sdinitfactor > 0 &&
         (rbmtype == GaussianBernoulliRBM || rbmtype == GaussianBernoulliRBM2)
      rbm.sd .*= sdinitfactor
   end

   if pcd
      chainstate = rand(batchsize, nhidden)
   else
      chainstate = Matrix{Float64}(0, 0)
   end

   # allocate space for trainrbm!
   nvisible = size(x, 2)
   h = Matrix{Float64}(batchsize, nhidden)
   hmodel = Matrix{Float64}(batchsize, nhidden)
   vmodel = Matrix{Float64}(batchsize, nvisible)
   posupdate = Matrix{Float64}(nvisible, nhidden)
   negupdate = Matrix{Float64}(nvisible, nhidden)

   for epoch = 1:epochs

      # Train RBM on data set
      trainrbm!(rbm, x, cdsteps = cdsteps, chainstate = chainstate,
            upfactor = upfactor, downfactor = downfactor,
            learningrate = learningrates[epoch],
            sdlearningrate = sdlearningrates[epoch],
            sdgradclipnorm = sdgradclipnorm,
            batchsize = batchsize,
            h = h, hmodel = hmodel, vmodel = vmodel,
            posupdate = posupdate, negupdate = negupdate)

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
      #sd = fill(0.05, length(vec(std(x, 1))))
      GaussianBernoulliRBM(weights, visbias, hidbias, sd)

   elseif rbmtype == GaussianBernoulliRBM2
      visbias = vec(mean(x, 1))
      #sd = fill(0.05, length(vec(std(x, 1))))
      sd = vec(std(x, 1))
      weights .*= sd
      return GaussianBernoulliRBM2(weights, visbias, hidbias, sd)

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
    nomonitoring
Accepts a model and a number of epochs and returns nothing.
"""
function nomonitoring(bm, epoch)
end


"""
    randombatchmasks(nsamples, batchsize)
Returns BitArray-Sets for the sample indices when training on a dataset with
`nsamples` samples using minibatches of size `batchsize`.
"""
function randombatchmasks(nsamples::Int, batchsize::Int)
   if batchsize > nsamples
      error("Batchsize ($batchsize) exceeds number of samples ($nsamples).")
   end

   nfullbatches, nremainingbatch = divrem(nsamples, batchsize)
   batchsizes = fill(batchsize, nfullbatches)
   if nremainingbatch > 0
      push!(batchsizes, nremainingbatch)
   end
   randomsampleindices = randperm(nsamples)
   map(batchrange -> randomsampleindices[batchrange],
         ranges(batchsizes))
end


function sdupdateterm(gbrbm::GaussianBernoulliRBM,
         v::Matrix{Float64}, h::Matrix{Float64})
   vec(mean((v .- gbrbm.visbias').^2 ./ (gbrbm.sd' .^ 3) -
         h * gbrbm.weights' .* (v ./ (gbrbm.sd' .^ 2)), 1))
end


"""
    trainrbm!(rbm, x)
Trains the given `rbm` for one epoch using the data set `x`.
(See also function `fitrbm`.)

# Optional keyword arguments:
* `learningrate`, `cdsteps`, `sdlearningrate`, `upfactor`, `downfactor`:
   See documentation of function `fitrbm`.
* `chainstate`: a matrix for holding the states of the RBM's hidden nodes. If
   it is specified, PCD is used.
"""
function trainrbm!(rbm::AbstractRBM, x::Array{Float64,2};
      chainstate::Matrix{Float64} = Matrix{Float64}(0, 0),
      upfactor::Float64 = 1.0,
      downfactor::Float64 = 1.0,
      learningrate::Float64 = 0.005,
      cdsteps::Int = 1,
      batchsize::Int = 1,
      sdlearningrate::Float64 = 0.0,
      sdgradclipnorm::Float64 = 0.0,

      # write-only arguments for reusing allocated space:
      v::Matrix{Float64} = Matrix{Float64}(batchsize, length(rbm.visbias)),
      h::Matrix{Float64} = Matrix{Float64}(batchsize, length(rbm.hidbias)),
      hmodel::Matrix{Float64} = Matrix{Float64}(batchsize, length(rbm.hidbias)),
      vmodel::Matrix{Float64} = Matrix{Float64}(batchsize, length(rbm.visbias)),
      posupdate::Matrix{Float64} = Matrix{Float64}(size(rbm.weights)),
      negupdate::Matrix{Float64} = Matrix{Float64}(size(rbm.weights)))

   nsamples = size(x, 1)

   # perform PCD if a chain state is provided as parameter
   pcd = !isempty(chainstate)

   batchmasks = randombatchmasks(nsamples, batchsize)
   nbatchmasks = length(batchmasks)

   normalbatchsize = true

   for batchindex in eachindex(batchmasks)
      batchmask = batchmasks[batchindex]

      normalbatchsize = (batchindex < nbatchmasks || nsamples % batchsize == 0)

      if normalbatchsize
         v .= view(x, batchmask, :)
      else
         v = x[batchmask, :]
         thisbatchsize = nsamples % batchsize
         h = Matrix{Float64}(thisbatchsize, nhiddennodes(rbm))
         if !pcd
            vmodel = Matrix{Float64}(size(v))
            hmodel = Matrix{Float64}(size(h))
         end # in case of pcd, vmodel and hmodel are not downsized
      end

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

      if !normalbatchsize && pcd
         # remove additional samples
         # that are unnecessary for computing the gradient
         vmodel = vmodel[1:thisbatchsize, :]
         hmodel = hmodel[1:thisbatchsize, :]
      end

      updateparameters!(rbm, v, vmodel, h, hmodel,
            learningrate, sdlearningrate, sdgradclipnorm,
            posupdate, negupdate)
   end

   rbm
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
function updateparameters!(rbm::BernoulliRBM,
      v::Matrix{Float64}, vmodel::Matrix{Float64},
      h::Matrix{Float64}, hmodel::Matrix{Float64},
      learningrate::Float64,
      sdlearningrate::Float64,
      sdgradclipnorm::Float64,
      posupdate::Matrix{Float64}, negupdate::Matrix{Float64})

   At_mul_B!(posupdate, v, h)
   At_mul_B!(negupdate, vmodel, hmodel)
   posupdate .-= negupdate
   posupdate .*= learningrate
   rbm.weights .+= posupdate
   rbm.visbias .+= vec(mean(v, 1) - mean(vmodel, 1)) * learningrate
   rbm.hidbias .+= vec(mean(h, 1) - mean(hmodel, 1)) * learningrate
   nothing
end


function updateparameters!(rbm::Binomial2BernoulliRBM,
      v::Matrix{Float64}, vmodel::Matrix{Float64},
      h::Matrix{Float64}, hmodel::Matrix{Float64},
      learningrate::Float64,
      sdlearningrate::Float64,
      sdgradclipnorm::Float64,
      posupdate::Matrix{Float64}, negupdate::Matrix{Float64})

   # To train a Binomial2BernoulliRBM exactly like
   # training a BernoulliRBM where each two nodes share the weights,
   # use half the learning rate in the visible nodes.
   learningratehidden = learningrate
   learningrate /= 2.0

   At_mul_B!(posupdate, v, h)
   At_mul_B!(negupdate, vmodel, hmodel)
   posupdate .-= negupdate
   posupdate .*= learningrate
   rbm.weights .+= posupdate
   rbm.visbias .+= vec(mean(v, 1) - mean(vmodel, 1)) * learningrate
   rbm.hidbias .+= vec(mean(h, 1) - mean(hmodel, 1)) * learningratehidden
   nothing
end

function updateparameters!(gbrbm::GaussianBernoulliRBM,
      v::Matrix{Float64}, vmodel::Matrix{Float64},
      h::Matrix{Float64}, hmodel::Matrix{Float64},
      learningrate::Float64,
      sdlearningrate::Float64,
      sdgradclipnorm::Float64,
      posupdate::Matrix{Float64}, negupdate::Matrix{Float64})

   # See bottom of page 15 in [Krizhevsky, 2009].

   if sdlearningrate > 0.0
      sdgrad = sdupdateterm(gbrbm, v, h) - sdupdateterm(gbrbm, vmodel, hmodel)

      v ./= gbrbm.sd'
      vmodel ./= gbrbm.sd'

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
      v ./= gbrbm.sd'
      vmodel ./= gbrbm.sd'
   end

   At_mul_B!(posupdate, v, h)
   At_mul_B!(negupdate, vmodel, hmodel)
   posupdate .-= negupdate
   posupdate .*= learningrate
   gbrbm.weights .+= posupdate
   gbrbm.hidbias .+= vec(mean(h, 1) - mean(hmodel, 1)) * learningrate
   gbrbm.visbias .+= vec(mean(v, 1) - mean(vmodel, 1)) ./ gbrbm.sd *
         learningrate
   nothing
end

function updateparameters!(gbrbm::GaussianBernoulliRBM2,
      v::Matrix{Float64}, vmodel::Matrix{Float64},
      h::Matrix{Float64}, hmodel::Matrix{Float64},
      learningrate::Float64,
      sdlearningrate::Float64,
      sdgradclipnorm::Float64,
      posupdate::Matrix{Float64}, negupdate::Matrix{Float64})

   # See Cho,
   # "Improved learning of Gaussian-Bernoulli restricted Boltzmann machines"
   sdsq = gbrbm.sd .^ 2

   if sdlearningrate > 0.0
      sdgrads = vmodel .* (hmodel * gbrbm.weights')
      sdgrads .-= v .* (h * gbrbm.weights')
      sdgrads .*= 2.0
      sdgrads .+= (v .- gbrbm.visbias') .^ 2
      sdgrads .-= (vmodel .- gbrbm.visbias') .^ 2
      sdgrad = vec(mean(sdgrads, 1))
      sdgrad ./= sdsq
      sdgrad ./= gbrbm.sd

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
   end

   v ./= sdsq'
   vmodel ./= sdsq'

   At_mul_B!(posupdate, v, h)
   At_mul_B!(negupdate, vmodel, hmodel)
   posupdate .-= negupdate
   posupdate .*= learningrate
   gbrbm.weights .+= posupdate
   gbrbm.hidbias .+= vec(mean(h, 1) - mean(hmodel, 1)) * learningrate
   gbrbm.visbias .+= vec(mean(v, 1) - mean(vmodel, 1)) * learningrate
   nothing
end
