import random

import numpy as np

import torch
import torch.nn.functional as F


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Construct new tensor that requires gradient computation
    X = X.clone().detach().requires_grad_(True)
    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores, and then compute the gradients with torch.autograd.grad.           #
    ##############################################################################

    y_pred = model(X)
    # sum the unnormalized scores to a scalar
    unnorm_scores = y_pred.gather(1, y.view(-1, 1)).squeeze().sum()
    gradients = torch.autograd.grad(unnorm_scores, X)[0]

    # To compute the saliency map, we take the absolute value of this gradient, 
    # then take the maximum value over the 3 input channels
    saliency, _ = gradients.abs().max(dim=1)
    
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency


def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our fooling image to the input image.
    X_fooling = X.clone().detach().requires_grad_(True)

    learning_rate = 1
    ##############################################################################
    # TODO: Generate a fooling image X_fooling that the model will classify as   #
    # the class target_y. You should perform gradient ascent on the score of the #
    # target class, stopping when the model is fooled.                           #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop.                                          #
    #                                                                            #
    # HINT: For most examples, you should be able to generate a fooling image    #
    # in fewer than 100 iterations of gradient ascent.                           #
    # You can print your progress over iterations to check your algorithm.       #
    ##############################################################################

    for i in range(100):
        y_pred = model(X_fooling)
        y_pred_idx = y_pred.argmax()
        if y_pred_idx == target_y:
            print(f'stopping early at training step {i}')
            break
            
        # loss function is the score of our desired class
        score = y_pred[0, target_y]
        # could instead call score.backward() here, and then access the X_fooling.grad parameter
        grad = torch.autograd.grad(score, X_fooling)[0]

        dX = learning_rate * grad / grad.norm()

        # now ADD the gradient to get closer to target_y!
        with torch.no_grad():
            X_fooling += dX
            # X_fooling.grad.zero_()

    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_fooling.detach()


def update_class_visualization(model, target_y, l2_reg, learning_rate, img):
    """
    Perform one step of update on a image to maximize the score of target_y
    under a pretrained model.

    Inputs:
    - model: A pretrained CNN that will be used to generate the image
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    - img: the image tensor (1, C, H, W) to start from
    """

    # Create a copy of image tensor with gradient support
    img = img.clone().detach().requires_grad_(True)
    ########################################################################
    # TODO: Use the model to compute the gradient of the score for the     #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. Don't forget the #
    # L2 regularization term!                                              #
    # Be very careful about the signs of elements in your code.            #
    ########################################################################

    # this is just like the make_fooling_image, but only one step...
    y_pred = model(img)
        
    # loss function is the score of our desired class
    score = y_pred[0, target_y] + l2_reg * img.norm()
    score.backward()

    grad = img.grad
    dX = learning_rate * grad / grad.norm()

    # now ADD the gradient to get closer to target_y!
    with torch.no_grad():
        img += dX
        # X_fooling.grad.zero_()

    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################
    return img.detach()
