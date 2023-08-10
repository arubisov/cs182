"""
Do not modify this file.

This script reruns the basic tests that provided in the IPython notebooks to 
help you debug and check the correctness your implementations, collecting the
results of these checks for submission and grading.

This script should automatically be run in the CollectSubmission script.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from deeplearning.classifiers.fc_net import *
from deeplearning.data_utils import get_CIFAR10_data
from deeplearning.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from deeplearning.solver import Solver

# %matplotlib inline
# plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'
def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

### Feedforward Notebook Checks
def test_affine_forward():
    num_inputs = 2
    input_shape = (4, 5, 6)
    output_dim = 3

    input_size = num_inputs * np.prod(input_shape)
    weight_size = output_dim * np.prod(input_shape)

    x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
    w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
    b = np.linspace(-0.3, 0.1, num=output_dim)

    out, _ = affine_forward(x, w, b)
    correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
                                [ 3.25553199,  3.5141327,   3.77273342]])

    # Compare your output with ours. The error should be around 1e-9.
    print('Testing affine_forward function:')
    print('difference: ', rel_error(out, correct_out))

# 
"""
FC notebook (30 points) (currently 30 points)
    affine:
        test affine forward  +2
        test affine backward +2

    2 layer net:
        Test two layer network gradients + 2?
        solver: get to 50% acc, generate plots and result +3 (log)

    general FC net
        gradient check + 2
        check overfitting 3 layer + 3 (log)
        check overfitting 5 layer + 3 (log)

    Update Rules:
        check SGD + momentum +2 
        check momentum coverges faster + 1 (log)
        check RMSProp +2
        check Adam +2
        check all model plots +1 (log) 

    Final: train good model +5       

BN notebook (30 points), change to 20?
    bn basic:
        check bn forward +5
        check bn backward naive +5
        check bn backward fast +2 (check its at least twice as fast?)

        FC with batchnorm checks +2
        Compare batchnorm training (log), check training accuracy is higher
        Batchnorm initialization sensitivity: (log)


Dropout (10 points)
    check forward +3
    check backward +3
    check fc net with dropout +3
    compare dropout training val accuracies (log) +1

Conv (30 points) (change to 40)?
    check forward +3
    check backward +3
    check maxpool forward +2
    check maxpool backward +2

    3 layer:
        # check init loss?
        # grad check 
        overfit to small examples +3 (logs)
        train net to 40% acc in one epoch +3 (logs)
    spatial batchnorm:
        check forward +2
        check back +2


    final conv training +10?
"""
