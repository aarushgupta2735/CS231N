from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    N = x.shape[0]
    x_reshaped = x.reshape(N, -1)
    out = x_reshaped.dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    N = x.shape[0]
    x_reshaped = x.reshape(N, -1)
    
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x_reshaped.T.dot(dout)
    db = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout * (x > 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        #######################################################################
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        
        x_norm = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_norm + beta
        
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
        cache = (x, sample_mean, sample_var, x_norm, gamma, beta, eps)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        #######################################################################
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization.              #
    ###########################################################################
    x, mean, var, x_norm, gamma, beta, eps = cache
    N = x.shape[0]

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)

    dx_norm = dout * gamma
    dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + eps)**(-1.5), axis=0)
    dmean = np.sum(dx_norm * -1.0 / np.sqrt(var + eps), axis=0) + dvar * np.mean(-2.0 * (x - mean), axis=0)
    
    dx = (dx_norm / np.sqrt(var + eps)) + (dvar * 2.0 * (x - mean) / N) + (dmean / N)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization.              #
    ###########################################################################
    x, mean, var, x_norm, gamma, beta, eps = cache
    N = dout.shape[0]

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)

    # Simplified single-line expression for dx
    dx = (gamma / (N * np.sqrt(var + eps))) * (N * dout - np.sum(dout, axis=0) - x_norm * np.sum(dout * x_norm, axis=0))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    ###########################################################################
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    
    x_norm = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_norm + beta
    
    cache = (x, mean, var, x_norm, gamma, eps)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    ###########################################################################
    x, mean, var, x_norm, gamma, eps = cache
    N, D = x.shape

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)

    dx_norm = dout * gamma
    dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + eps)**(-1.5), axis=1, keepdims=True)
    dmean = np.sum(dx_norm * -1.0 / np.sqrt(var + eps), axis=1, keepdims=True) + dvar * np.mean(-2.0 * (x - mean), axis=1, keepdims=True)
    
    dx = (dx_norm / np.sqrt(var + eps)) + (dvar * 2.0 * (x - mean) / D) + (dmean / D)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
    else:
        # ADD THIS ELSE BLOCK TO FIX THE LINTER ERROR
        raise ValueError(f"Invalid mode {mode}")

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    ###########################################################################
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    H_prime = 1 + (H + 2 * pad - HH) // stride
    W_prime = 1 + (W + 2 * pad - WW) // stride

    out = np.zeros((N, F, H_prime, W_prime))
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    for n in range(N):
        for f in range(F):
            for i in range(H_prime):
                for j in range(W_prime):
                    # Define the window
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW

                    # Perform convolution
                    window = x_padded[n, :, h_start:h_end, w_start:w_end]
                    out[n, f, i, j] = np.sum(window * w[f]) + b[f]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    H_prime = 1 + (H + 2 * pad - HH) // stride
    W_prime = 1 + (W + 2 * pad - WW) // stride

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    dx_padded = np.zeros_like(x_padded)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    for n in range(N):
        for f in range(F):
            for i in range(H_prime):
                for j in range(W_prime):
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW

                    window = x_padded[n, :, h_start:h_end, w_start:w_end]
                    
                    db[f] += dout[n, f, i, j]
                    dw[f] += window * dout[n, f, i, j]
                    dx_padded[n, :, h_start:h_end, w_start:w_end] += w[f] * dout[n, f, i, j]

    dx = dx_padded[:, :, pad:-pad, pad:-pad]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    H_prime = 1 + (H - pool_height) // stride
    W_prime = 1 + (W - pool_width) // stride

    out = np.zeros((N, C, H_prime, W_prime))

    for n in range(N):
        for c in range(C):
            for i in range(H_prime):
                for j in range(W_prime):
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width

                    window = x[n, c, h_start:h_end, w_start:w_end]
                    out[n, c, i, j] = np.max(window)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    H_prime = 1 + (H - pool_height) // stride
    W_prime = 1 + (W - pool_width) // stride

    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for i in range(H_prime):
                for j in range(W_prime):
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width

                    window = x[n, c, h_start:h_end, w_start:w_end]
                    max_idx = np.unravel_index(np.argmax(window), window.shape)
                    
                    dx[n, c, h_start + max_idx[0], w_start + max_idx[1]] += dout[n, c, i, j]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.
    """
    out, cache = None, None
    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    ###########################################################################
    N, C, H, W = x.shape
    
    x_reshaped = x.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    
    out_reshaped, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    
    out = out_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    ###########################################################################
    N, C, H, W = dout.shape
    
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    
    dx_reshaped, dgamma, dbeta = batchnorm_backward_alt(dout_reshaped, cache)
    
    dx = dx_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    ###########################################################################
    N, C, H, W = x.shape
    
    x_reshaped = x.reshape(N * G, -1)
    
    mean = np.mean(x_reshaped, axis=1, keepdims=True)
    var = np.var(x_reshaped, axis=1, keepdims=True)
    
    x_norm = (x_reshaped - mean) / np.sqrt(var + eps)
    
    x_norm = x_norm.reshape(N, C, H, W)
    out = x_norm * gamma + beta
    
    cache = (x_reshaped, x_norm, mean, var, gamma, eps, G)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    ###########################################################################
    x_reshaped, x_norm, mean, var, gamma, eps, G = cache
    N, C, H, W = dout.shape
    
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    dgamma = np.sum(dout * x_norm, axis=(0, 2, 3), keepdims=True)

    dx_norm = dout * gamma
    dx_norm_reshaped = dx_norm.reshape(N * G, -1)
    
    D = dx_norm_reshaped.shape[1]
    
    dvar = np.sum(dx_norm_reshaped * (x_reshaped - mean) * -0.5 * (var + eps)**(-1.5), axis=1, keepdims=True)
    dmean = np.sum(dx_norm_reshaped * -1.0 / np.sqrt(var + eps), axis=1, keepdims=True) + dvar * np.mean(-2.0 * (x_reshaped - mean), axis=1, keepdims=True)
    
    dx_reshaped = (dx_norm_reshaped / np.sqrt(var + eps)) + (dvar * 2.0 * (x_reshaped - mean) / D) + (dmean / D)
    
    dx = dx_reshaped.reshape(N, C, H, W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.
    """
    loss, dx = None, None
    ###########################################################################
    # TODO: Copy over your solution from A1.                                  #
    ###########################################################################
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
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    """
    loss, dx = None, None
    ###########################################################################
    # TODO: Copy over your solution from A1.                                  #
    ###########################################################################
    N = x.shape[0]
    
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx