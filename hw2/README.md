# CS182/282 Assignment 2
In this assignment you will implement recurrent networks, and apply them to
image captioning on Microsoft COCO. You will also explore methods for visualizing
the features of a pretrained model on ImageNet, and also this model to implement
Style Transfer. The goals of this assignment are as follows:

- Understand the architecture of recurrent neural networks (RNNs) and how they operate on sequences by sharing weights over time
- Understand and implement both Vanilla RNNs and Long-Short Term Memory (LSTM) RNNs
- Understand how to sample from an RNN language model at test-time
- Understand how to combine convolutional neural nets and recurrent nets to implement an image captioning system
- Understand how a trained convolutional network can be used to compute gradients with respect to the input image
- Implement and different applications of image gradients, including saliency maps, fooling images, class visualizations.
- Understand and implement style transfer.


## Copy Solution from Homework 1
Please copy `layers.py` and `optim.py` from your homework 1 solution to the deeplearning directory. We will provide
reference files once the deadline of homework 1 is over.

## Setup
Make sure your machine is set up with the assignment dependencies.

### [Option 1] Install Anaconda and Required Packages
The preferred approach for installing all the assignment dependencies is to use
[Anaconda](https://www.anaconda.com/products/individual), which is a Python distribution
that includes many of the most popular Python packages for science, math,
engineering and data analysis. Once you install Anaconda you can run the following
command inside the homework directory to install the required packages for this homework:

```bash
conda env create -f environment.yml
```

Once you have all the packages installed, run the following command **every time**
to activate the environment when you work on the homework.
```bash
conda activate cs182_hw2
```


### [Option 2] Working on a Virtual Machine
This assignment is provided pre-setup with a VirtualBox image. Installation Instructions:
1. Follow [the instructions here](https://www.virtualbox.org/manual/ch02.html) to install VirtualBox if it is not already installed.
2. [Download the VirtualBox image here](https://drive.google.com/file/d/1uIAlrpIuXyHjJlVdNA0H3MsGubFvFn3x/view?usp=sharing)
3. Load the VirtualBox image using [the instructions here](https://docs.oracle.com/cd/E26217_01/E26796/html/qs-import-vm.html)
4. Start the VM. The username and password are both cs182. Required packages are pre-installed and the cs182_hw2 environment activated by default.
5. Download the assignment code onto the VM yourself.

#### FAQ
**I get an error "AMD-V is disabled in the BIOS" or "Intel-VT is disabled in the BIOS" or similar**

Solution: See [this link](https://docs.fedoraproject.org/en-US/Fedora/13/html/Virtualization_Guide/sect-Virtualization-Troubleshooting-Enabling_Intel_VT_and_AMD_V_virtualization_hardware_extensions_in_BIOS.html)


**The virtual machine won't boot**

Solutions:

- Try increasing the number of allocated CPUs: Under Settings→System→Processor
- Try [increasing the amount of allocated memory:](https://superuser.com/questions/926339/how-to-change-the-ram-allocated-to-an-os-in-virtualbox)

### Download Data
Once you have the starter code, you will need to download the CIFAR-10 dataset.
Run the following from the homework 2 directory:

```bash
cd deeplearning/datasets
./get_assignment2_data.sh
```

If you don't have wget installed, you can also try 

```bash
./get_assignment2_data_curl.sh
```


### Start Jupyter Notebook
After you download data, you should start the IPython notebook server
from the homework 2 directory with the following command:

```bash
jupyter notebook
```

If you are unfamiliar with IPython, you should
read our [IPython tutorial](http://cs231n.github.io/ipython-tutorial/).



## Submitting your work:
Once you are done working run the `collect_submission.sh` script;
this will produce a file called `assignment2.zip`.
Upload this file to Gradescope.
Note that Gradescope will run an autograder on the files you submit. For some
test cases, there is a nonzero (but should be very low) probability that correct
implementations may fail due to randomness. If you think your implementation is
correct, then you can simply resubmit to rerun the autograder to check whether
it really is just a particularly unlucky seed..


## Q1: Image Captioning with Recurrent Neural Network (34 points)
The IPython notebook `RNN_Captioning.ipynb` will introduce you to the implementation
of vanilla recurrent neural networks for image captioning. Follow the instructions
in the notebook to complete this part.


## Q2: Image Captioning with LSTM (25 points)
The IPython notebook `LSTM_Captioning.ipynb` will introduce you to the implementation
of LSTM for image captioning. Follow the instructions in the notebook to complete this part.


## Q3: Network Visualization (18 points)
The IPython notebook `NetworkVisualization.ipynb` will introduce you to various techniques
for visualizing neural network internals. Follow the instructions in the notebook to complete this part.
We will use PyTorch for this part.

## Q4: Style Transfer (23 points)
The IPython notebook `StyleTransfer.ipynb` will introduce you to image style transfer.
Follow the instructions in the notebook to complete this part. We will use PyTorch for this part.

