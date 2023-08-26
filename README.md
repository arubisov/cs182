# CS182 (Spring 2021) at UC Berkeley

Self-study of CS182 (Spring 2021) at UC Berkeley - Designing, Visualizing and Understanding Deep Neural Networks. [Course homepage found here](https://cs182sp21.github.io/). This repo contains my solutions to the four homework assignments. Lecture videos can be found in [this YouTube playlist](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVmKkQHucjPAoRtIJYt8a5A). 

I began self-study on this course on 2023-08-02 thanks to a comment in [this Reddit discussion](https://www.reddit.com/r/deeplearning/comments/tsybv1/deep_learning_specialization_courses/) which described the course as "simply amazing." I agree.

## Homework 1
Lectures 1-7.  
Time spent: 20 hours.

- Implements a fully-connected neural network with arbitrarily many affine+ReLU hidden layers
- Implements convolutional neural networks
- Implements forward propagation and back propagations of the following layers:
  - affine
  - ReLU
  - batch normalization
  - spatial batch normalization
  - dropout
  - convolution
  - max pool
  - softmax loss
  - SVM loss
- Implements momentum, RMSProp, Adam
- For initial setup, compiling the Cython extension required manually setting `language_level = "2"` in the `setup.py` file because I was running Cython 3.0.0, whereas the course was released using Cython 0.29
- I added a cell in `FullyConnectedNets.ipynb` to visualize samples from the CIFAR10 dataset after dataload, always a useful step when working with image data. This revealed that each image is normalized by subtracting its mean. Visualizing required adding back the mean (by adding back the minimum)
- Some fantastic resources I used to help my understanding:
  - [How computational graphs help with the calculus of backprop](http://colah.github.io/posts/2015-08-Backprop/)
  - [Backprop for the convolution and max-pool layers](https://mukulrathi.com/demystifying-deep-learning/conv-net-backpropagation-maths-intuition-derivation/)
  - [Backprop calculus for the batch norm](https://www.adityaagrawal.net/blog/deep_learning/bprop_batch_norm)

## Homework 2
Lectures 8-10.  
Spring 2021 version not found, so this homework is forked from the [Spring 2022 version](https://github.com/cs182sp22/cs182_hw2_student).  
Time spent: 15 hours.

- Implements forward propagation and back propagations for:
  - word embedding
  - vanilla RNN single step
  - LSTM single step
  - RNN/LSTM train on entire sequence
  - RNN/LSTM inference for caption generation
- Implements neural network visualization techniques
  - saliency maps
  - DeepDream
  - class visualization from random noise
- Implements style transfer
  - Gram matrix
  - content loss
  - style loss
  - total variation loss
- Completed Intro to PyTorch notebook from the [Spring 2022 version of Homework 1](https://github.com/cs182sp22/cs182_hw1_student/blob/main/Pytorch.ipynb)
- Additional resources:
  - [Math for RNNs](https://www.oreilly.com/library/view/neural-networks-and/9781492037354/ch04.html) helped understand that in the notation y = W[hidden; input] that the layers gets summed after multiplication with their respective weight matrices, not concatenated and multiplied by a single weight matrix as aluded to in lecture.

## Homework 3
Lectures 11-13.  
Time spent: 0 hours.
