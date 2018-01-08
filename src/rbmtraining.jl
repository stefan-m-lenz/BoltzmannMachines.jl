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
   if sdinitfactor > 0 &&
         (rbmtype == GaussianBernoulliRBM || rbmtype == GaussianBernoulliRBM2)
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

   elseif rbmtype == GaussianBernoulliRBM2
      visbias = vec(mean(x, 1))
      sd = vec(std(x, 1))
      #hidbias = randn(nhidden)/sqrt(nhidden)
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

function updateparameters!(gbrbm::GaussianBernoulliRBM2,
      v::Vector{Float64}, vmodel::Vector{Float64},
      h::Vector{Float64}, hmodel::Vector{Float64},
      learningrate::Float64,
      sdlearningrate::Float64,
      sdgradclipnorm::Float64,
      posupdate::Matrix{Float64}, negupdate::Matrix{Float64})

   sdsq = gbrbm.sd .^ 2

   if sdlearningrate > 0.0
      sdgrad = vmodel .* (gbrbm.weights * hmodel)
      sdgrad .-= v.* (gbrbm.weights * h)
      sdgrad .*= 2.0
      sdgrad .+= (v - gbrbm.visbias) .^ 2
      sdgrad .-= (vmodel - gbrbm.visbias) .^ 2
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

   v ./= sdsq
   vmodel ./= sdsq

   A_mul_Bt!(posupdate, v, h)
   A_mul_Bt!(negupdate, vmodel, hmodel)
   posupdate .-= negupdate
   posupdate .*= learningrate
   gbrbm.weights .+= posupdate
   gbrbm.hidbias .+= (h - hmodel) * learningrate
   gbrbm.visbias .+= (v - vmodel) * learningrate
   nothing
end