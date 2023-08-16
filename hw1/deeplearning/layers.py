import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    #############################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You     #
    # will need to reshape the input into rows.                                 #
    #############################################################################
    # x gets reshaped to dim (N, D)
    # (N, D) @ (D, M) + (M,) = (N, M)
    out = x.reshape(-1, w.shape[0]) @ w + b
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: bias, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the affine backward pass.                                 #
    #############################################################################
    # for our dataset D = {(x_1, y_1), ..., (x_N, y_N), our loss function
    # is defined as L(theta) = - sum_i^N log prob (y_i | x_i)
    # where i indexes the dataset. for this reason we're just summing the gradients 
    # across the datapoints, i.e. dim N. 

    # we *should* be doing an averaging where we divide by 1/N, or in the future
    # 1/B where B is the size of the future minibatch. this will decouple the 
    # learning rate from the batch size, otherwise the gradient just accumulates.
    
    # db is the identity operation
    db = np.sum(1 * dout, axis=0)

    # dz/dw * dout = dout * a^T
    dw = x.reshape(-1, w.shape[0]).transpose() @ dout
    
    # dz/da * dout = W^T * dout
    # dim (N,D) then reshape 
    dx = (dout @ w.transpose()).reshape(x.shape)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    #############################################################################
    # TODO: Implement the ReLU forward pass.                                    #
    #############################################################################
    out = np.maximum(x, 0)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    #############################################################################
    # TODO: Implement the ReLU backward pass.                                   #
    #############################################################################
    # there is no dtheta (like dw or db for a linear layer) since relu isn't 
    # paramterized.
    # here we use dx = 1 if x>=0, and 0 otherwise, and then mult with dout
    dx = dout.copy()
    dx[x <= 0] = 0
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #############################################################################
        # TODO: Implement the training-time forward pass for batch normalization.   #
        # Use minibatch statistics to compute the mean and variance, use these      #
        # statistics to normalize the incoming data, and scale and shift the        #
        # normalized data using gamma and beta.                                     #
        #                                                                           #
        # You should store the output in the variable out. Any intermediates that   #
        # you need for the backward pass should be stored in the cache variable.    #
        #                                                                           #
        # You should also use your computed sample mean and variance together with  #
        # the momentum variable to update the running mean and running variance,    #
        # storing your result in the running_mean and running_var variables.        #
        #############################################################################

        sample_mean = 1/N * np.sum(x, axis=0) # mean per feature, across the training sample of size N
        sample_var = 1/N * np.sum((x-sample_mean)**2, axis=0)

        out = (x-sample_mean) / np.sqrt(sample_var + eps) * gamma + beta

        normed = (x-sample_mean) / np.sqrt(sample_var + eps)

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var    

        cache = (sample_mean, sample_var, normed, x, gamma, eps)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    elif mode == 'test':
        #############################################################################
        # TODO: Implement the test-time forward pass for batch normalization. Use   #
        # the running mean and variance to normalize the incoming data, then scale  #
        # and shift the normalized data using gamma and beta. Store the result in   #
        # the out variable.                                                         #
        #############################################################################
        out = (x-running_mean) / np.sqrt(running_var + eps) * gamma + beta 
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #############################################################################
    mu, var, normed, x, gamma, eps = cache
    N, D = dout.shape
    
    # work backward at each operation gate.

    # start with dout
    dy = dout
    
    # + gate: y = xhat * gamma + beta, outputs dbeta and dxhatgamma
    # dbeta = 1
    dbeta = np.sum(dout, axis=0)
    dxhatgamma = dout

    # * gate: xhatgamma = xhat * gamma, outputs dgamma and dxhat
    # note that xhat and the normed variable are the same thing
    dgamma = np.sum(dxhatgamma * normed, axis=0)
    dxhat = dxhatgamma * gamma

    # div gate: xhat = (x-mu) / sqrt(var)
    # dy/dsigma = dy/dxhat dxhat/dsigma
    # dy/d(x-mu) = dy/dxhat dxhat/d(x-mu) first branch only
    # dim D
    dsigma = np.sum(dxhat * -(x-mu) / var, axis=0)
    # dim (N,D)
    dxminusmu_xhat = dxhat * 1/np.sqrt(var)

    # sqrt gate: sigma = sqrt(var)
    # dim D
    dvar = dsigma * 1/2 * var**(-1/2)

    # mult 1/N gate: var = 1/N * SUM(x-mu)^2
    # dim D
    dvarunscaled = dvar * 1/N

    # SUM gate: varunscaled = SUM(x-mu)^2
    # dim (N,D)
    dxminusmusqd = dvarunscaled * np.ones(x.shape) 

    # ^2 gate: xminusmusqd = (x-mu)^2
    # dy/d(x-mu) = dy/(x-mu)^2 d(x-mu)^2/d(x-mu) second branch only
    # dim (N,D)
    dxminusmu_xminusmusqd = dxminusmusqd * 2 * (x-mu)

    # dy/d(x-mu) = dy/dxhat dxhat/d(x-mu) + dy/(x-mu)^2 d(x-mu)^2/d(x-mu)
    # output used in two branches, so sum the branches 
    # together per calculus sum rule
    # dim (N,D)
    dxminusmu = dxminusmu_xhat + dxminusmu_xminusmusqd

    # - gate: x-mu = x-mu. 
    # dy/mu = dy/(x-mu) d(x-mu)/dmu
    # dim D
    # dy/dx = dy/(x-mu) d(x-mu)/dx first branch only
    dmu = np.sum(-dxminusmu, axis=0)
    dx_xminusmu = dxminusmu
    
    # mult 1/N gate
    # mu = 1/N * SUM(x)
    # dy/dmu = dy/d(x-mu) d(x-mu)/dmu
    # dim D
    dsumx = dmu * 1/N

    # SUM gate
    # dy/dx = dy/dsumx dsumx/dx second branch only 
    # dim (N,D)
    dx_sum = dsumx * np.ones(x.shape)

    # dim (N,D)
    dx = dx_xminusmu + dx_sum
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #                                                                           #
    # After computing the gradient with respect to the centered inputs, you     #
    # should be able to compute gradients with respect to the inputs in a       #
    # single statement; our implementation fits on a single 80-character line.  #
    #############################################################################
    mu, var, normed, x, gamma, eps = cache
    N, D = dout.shape
    
    # dz/db_i = 1
    dbeta = np.sum(dout, axis=0)

    # dz/dgamma is the normalized unscaled input, retrieve it from cache of forward step
    dgamma = np.sum(normed * dout, axis=0)

    # dz/dx is complex... work backbward, following ideas here:
    # https://www.adityaagrawal.net/blog/deep_learning/bprop_batch_norm
    dx = np.zeros(dout.shape)
    dxhat = dout * gamma # (N,D)
    dvar = np.sum(dxhat * (x - mu) * (-1/2) * (var + eps) ** (-3/2), axis=0)
    dmu = np.sum(dxhat * (-1/np.sqrt(var + eps)), axis=0)
    
    # validating for first position only to ensure issue isn't here...
    # val = dxhat[0,0] * 1/np.sqrt(var[0]+eps) + dvar[0] * 2*(x[0,0]-mu[0]) / N + dmu[0] * 1/N
    # dx[0,0] = val
    
    dx = (dxhat * 1/np.sqrt(var + eps)
          + dvar * 2 * (x - mu) / N
          + dmu * 1/N
         )
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout
        and rescale the outputs to have the same mean as at test time.
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not in
        real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        ###########################################################################
        # TODO: Implement the training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                            #
        ###########################################################################
        mask = (np.random.rand(*x.shape) > p) / p
        out = x * mask
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    elif mode == 'test':
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        ###########################################################################
        # TODO: Implement the training phase backward pass for inverted dropout.  #
        ###########################################################################
        dx = dout * mask
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################
    stride = conv_param['stride']
    pad = conv_param['pad']
    N, C, H, W = x.shape
    F, C, FH, FW = w.shape # same C
    
    x_pad = np.pad(x, 
                   pad_width=((0,), (0,), (pad,), (pad,)), 
                   mode='constant', 
                   constant_values=0)

    H_out = 1 + (H + 2*pad - FH) // stride
    W_out = 1 + (W + 2*pad - FW) // stride

    out = np.zeros((N, F, H_out, W_out))

    # for each image in the batch
    for n in range(0, N):

        # iterate over each position in the output

        # for each filter
        for f in range(0, F): 
            # for each h,w in the output
            for out_h in range(0, H_out):
                for out_w in range(0, W_out):
                    
                    # first approach.
                    # original, hyperspecified operations. simplifying using list slicing below.
                    # # for each h,w in the filter
                    # val = 0
                    # for f_h in range(0, FH):
                    #     for f_w in range(0, FW):
                    #         # and for each channel
                    #         for c in range(0, C):
                    #             # accrue the filter value applied to the moving window over the input, 
                    #             # centered on the same position we're on in the output. god that's a bad explanation.
                    #             val += (w[f, c, f_h, f_w]
                    #                     * x_pad[n, c, (out_h*conv_param['stride'] + conv_param['pad'] - (w.shape[2]-1)//2 + f_h),
                    #                                   (out_w*conv_param['stride'] + conv_param['pad'] - (w.shape[3]-1)//2 + f_w)])
                    # out[n, f, out_h, out_w] = val + b[f]

                    # second approach.
                    # we don't need to specify every position in filter, can just use : list access with element-wise multiplication
                    # and better specify the window in the input that's being convolved
                    out[n, f, out_h, out_w] = np.sum(w[f, :, :, :] * x_pad[n, :, out_h*stride:out_h*stride+FH, out_w*stride:out_w*stride+FW]) + b[f]
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    x, w, b, conv_param = cache
    stride = conv_param['stride']
    pad = conv_param['pad']
    N, C, H, W = x.shape
    F, C, FH, FW = w.shape # same C
    # upstream deriv dout has shape (N, F, H_out, W_out)
    # H_out = 1 + (H + 2*pad - FH) // stride
    # W_out = 1 + (W + 2*pad - FW) // stride
    _, _, H_out, W_out = dout.shape
    
    x_pad = np.pad(x, 
                   pad_width=((0,), (0,), (pad,), (pad,)), 
                   mode='constant', 
                   constant_values=0)

    # convoluations cannot be represented as matrix multiplications! so,
    # consider one output unit z_{i,j,k} at a time.
    # 
    # dout = dL/dz
    #
    # for a single z_{i,j,k}, dz_{i,j,k}/db_k = 1
    # so by chain rule:
    #   dL/db_k = sum_i sum_j dL/dz_{i,j,k} * dz_{i,j,k}/db_k
    #           = sum_i sum_j dL/dz_{i,j,k}
    # AND we sum across the batch of training examples, so:
    #           = sum_n sum_i sum_j dL/dz_{i,j,k}
    db = np.sum(dout, axis=(0,2,3))

    # for a single z_{i,j,k}, dz_{i,j,k}/dw_{h,w,k,c} = a_{i+h,j+w,c}
    # so by chain rule:
    #   dL/db_k = sum_i sum_j dL/dz_{i,j,k} * dz_{i,j,k}/db_k
    #           = sum_i sum_j dL/dz_{i,j,k} * a_{i+h,j+w,c}
    # AND we sum across the batch of training examples, so:
    #           = sum_n sum_i sum_j dL/dz_{i,j,k} * a_{i+h,j+w,c}

    
    
    dw = np.zeros(w.shape)
    dx_pad = np.zeros(x_pad.shape)
    for n in range(0, N):
        for f in range(0, F): 
            for out_h in range(0, H_out):
                for out_w in range(0, W_out):
                    # out[n, f, out_h, out_w] = np.sum(w[f, :, :, :] * x_pad[n, :, out_h*stride:out_h*stride+FH, out_w*stride:out_w*stride+FW]) + b[f]
                    dw[f] += x_pad[n, :, out_h*stride:out_h*stride+FH, out_w*stride:out_w*stride+FW] * dout[n,f,out_h,out_w]
                    dx_pad[n, :, out_h*stride:out_h*stride+FH, out_w*stride:out_w*stride+FW] += w[f] * dout[n,f,out_h,out_w]
    
    dx = dx_pad[:, :, pad:H+pad, pad:W+pad]
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    H_out = H//pool_height
    W_out = W//pool_width
    out = np.zeros((N, C, H_out, W_out))

    for n in range(0, N):
        for c in range(0, C):
            for i in range(0, H_out):
                for j in range(0, W_out):
                    out[n,c,i,j] = np.max(x[n,c,i*stride:i*stride+pool_height, j*stride:j*stride+pool_width])
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_out = H//pool_height
    W_out = W//pool_width

    dx = np.zeros(x.shape)

    # each input pixel accumulates deriv 1 if it was the argmax in a given mask
    for n in range(0, N):
        for c in range(0, C):
            for i in range(0, H_out):
                for j in range(0, W_out):
                    x_mask = x[n,c,i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
                    max_pixel = np.unravel_index(np.argmax(x_mask, axis=None), x_mask.shape)
                    
                    dx[n,c,i*stride+max_pixel[0], j*stride+max_pixel[1]] += dout[n,c,i,j]
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    #############################################################################
    # TODO: Implement the forward pass for spatial batch normalization.         #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################
    N, C, H, W = x.shape
    # np.transpose gets me every time.
    # output i corresponds to axes[i]
    # TRANSPOSE+RESHAPE ORDER MATTERS! must be to (N, W, H, C)
    # can inspect visually by comparing 
    #     x.transpose(0,3,2,1).reshape(N*W*H, C)
    # and x.transpose(0,3,1,2).reshape(N*H*W, C)
    # and seeing that only the former plucks out the indices correctly. 
    # would love a compelling answer...
    x = x.transpose(0,3,2,1).reshape(N*W*H, C)
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    out = out.reshape(N, W, H, C).transpose(0,3,2,1)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    #############################################################################
    # TODO: Implement the backward pass for spatial batch normalization.        #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################
    N, C, H, W = dout.shape
    # TRANSPOSE+RESHAPE ORDER MATTERS! must be to (N, W, H, C)
    dx = dout.transpose(0,3,2,1).reshape(N*W*H, C)
    dx, dgamma, dbeta = batchnorm_backward_alt(dx, cache)
    dx = dx.reshape(N, W, H, C).transpose(0,3,2,1)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))    # Anton: why are we subtracting the max? seems unnecessary, math works out same. avoids degradation?
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
