# Computer Vision


NOTE
we are not telling our model what to learn
it will learn those pattern by itself by seeing examples

video, text, audio

nutrify.app

each category works at different level of performance

OBJECT DETECTION
where is the thing we are looking for?

WHAT ARE THE DIFFERENT SECTIONS OF AN IMAGE
computational photography apple uses it
person 1, 2 , ... n,
skin
hair
sky


TESLA SELF DRIVING CAR


---

torch vision, domain library
cnn architecture
multi class image classification

---

input output

X width
Y height
C color chnnel

tensor(X;Y;C)

1 ouput per each class

how to improve model? mode DIVERSE images, balanced dataset

CNN - convolutional neural network or transformers

---

tensor (batch_size, width, height, colour_channels)

32 is a very common batch size

224 224 very common dimensions for norla computers

---

easier with grayscale images

tensor (32, 28, 28, 1)

NHWC oppure NCHW 

---

WHAT IS A CNN

model quite good at recognizing images

---

torchvision - base domain library for computer vision


torchvision - base domain library for PyTorch computer vision
torchvision.datasets - get datasets and data loading functions for computer vision here
torchvision.models - get pretrained computer vision models that you can leverage for your own problems
torchvision.transforms - functions for manipulating your vision data (images) to be suitable for use with an ML model
torch.utils.data.Dataset - Base dataset class for PyTorch.
torch.utils.data.DataLoader - Creates a Python iterable over a dataset

---

transforming and augmenting images

PIL - Python Imaging Library

MNIST - Modified National Institute of Standards and Technology dataset

it is the hello world in computer vision

1988

The original MNIST dataset contains a lot of handwritten digits. Members of the AI/ML/Data Science community love this dataset and use it as a benchmark to validate their algorithms. In fact, MNIST is often the first dataset researchers try. "If it doesn't work on MNIST, it won't work at all", they said. "Well, if it does work on MNIST, it may still fail on others."

replacing MNSIT


IMGAGENET is the gold standard of CV evaluation

---

transform image to tensor

torchvision.transform.ToTensor()

--- 
warning
numpy array  N HWC [0,255]
torch tensor N CHW [0,1]

for gray scale we need to red rid of the extra 1 dimension
tensor.squeeze()

chances are, if we get confused on our dataset
our model might get confused later on

---

Dataloader - turn dataset into a python iterable (turn data into minibatches)

fashion MNIST is 60k images, modern DL have billions
if computer hardware need memory to hanfle all images
can't store all images into ram
so we do it in ram

more computationally efficient

Right now, our data is in the form of PyTorch Datasets.

DataLoader turns our dataset into a Python iterable.

More specifically, we want to turn our data into batches (or mini-batches).

Why would we do this?

It is more computationally efficient, as in, your computing hardware may not be able to look (store in memory) at 60000 images in one hit. So we break it down to 32 images at a time (batch size of 32).
It gives our neural network more chances to update its gradients per epoch.
For more on mini-batches, see here: https://youtu.be/l4lSUAcvHFs

if we see 60k images in one time, only 1 change per epoch, with smaller abtch we update more frequently

shuffle=true, we don't want NN to remember order in our data, there may be patterns

this principles is used in a lot of DL problems

DON't SUFFLE TRAINING DATA
we need the same order to easily evaluate different models


PIN DATA - load data faster
DROP LAST - overlapp get rid of last batch

---


When starting to build a series of machine learning modelling experiments, it's best practice to start with a baseline model.

A baseline model is a simple model you will try and improve upon with subsequent models/experiments.

In other words: start simply and add complexity when necessary.

NN are too powerful, they tend to do too well on our data


---


FLATTEN LAYER

no learnale parameterss

# we condense info down to a single vector space, baseline model,
# linear layer can't handle multidimensional data

we want to comrpess our image into a single vector

one logit per class is the ouput

most common error: vector shape missmatchs

---

Machine learning is very experimental.

Two of the main things you'll often want to track are:

Model's performance (loss and accuracy values etc)
How fast it runs

---

Creating a training loop and training a model on batches of data
Loop through epochs.
Loop through training batches, perform training steps, calculate the train loss per batch.
Loop through testing batches, perform testing steps, calculate the test loss per batch.
Print out what's happening.
Time it all (for fun).

---
TQDM fantastic progress bar envionment versatile

tqdm.auto find best one


---

accumulate train loss

get train loss per batch

at the end of the loop divide cumulative loss by the number of batches => average training loss per batch

oprimizer optimizes once per batch and not epoch

---

why relu has no argumenrs, it does not change the shape of our data

normally you have linea followed by non linar layer

---

Note: Sometimes, depending on your data/hardware you might find that your model trains faster on CPU than GPU.

Why is this?

It could be that the overhead for copying data/model to and from the GPU outweighs the compute benefits offered by the GPU.
The hardware you're using has a better CPU in terms compute capability than the GPU.
For more on how to make your models compute faster, see here: https://horace.io/brrr_intro.html

the second one is super rare

bigger speds up when you arre running larger mdoels and datasets

---

what is a CNN
how does it woek
visualize visualize visualize
become one with the data
read stuff, helps you know how to find things


PIN DATA - load data faster
DROP LAST - overlapp get rid of last batch
dataloader

---
CNNs

CONVOLUTIONAL NEURAL NETWORKS

CNN's are also known ConvNets.

CNN's are known for their capabilities to find patterns in visual data.

To find out what's happening inside a CNN, see this website: https://poloclub.github.io/cnn-explainer/


standard order
conv - activation - pooling

combination of those, they can be combined in different ways

linear output

(conv - activation - pooling ) * 2


---

kernel
stride
padding
HYPERPARAETERS

conv 2D, 1d,3d

CNN, imput starts to get smallser
model lersn compressed representation of image

mdodel learns most generalizable pattern
---

model_2 = FashionModel2(input_shape = 1, # color channels

while imput_shape=3 if RGB


--- 

CNN

A layer is simply a collection of neurons with the same operation, including the same hyperparameters.

CNNs can be used for many different computer vision tasks, such as image processing, classification, segmentation, and object detection.

he network architecture, Tiny VGG, used in CNN Explainer contains many of the same layers and operations used in state-of-the-art CNNs today, but on a smaller scale. This way, it will be easier to understand getting started.

PADDING
 the most commonly used approach is zero-padding because of its performance, simplicity, and computational efficiency. The technique involves adding zeros symmetrically around the edges of an input. This approach is adopted by many high-performing CNN


KERNEL
often also referred to as filter size, refers to the dimensions of the sliding window over the inputù
smaller kernel sizes lead to better performance for the image classification task
less kernels, faster trainign

STRIDE
ndicates how many pixels the kernel should be shifted over at a time. 
As stride is decreased, more features are learned because more data is extracted, which also leads to larger output layers. 
ensure that the kernel slides across the input symmetrically when implementing a CNN


RELU
Part of the reason these groundbreaking CNNs are able to achieve such tremendous accuracies is because of their non-linearity. ReLU applies much-needed non-linearity into the model.
CNNs using ReLU are faster to train than their counterparts.
This activation function is applied elementwise on every value from the input tensor.

SOFTMAX
A softmax operation serves a key purpose: making sure the CNN outputs sum to 1.


---

compress inut in some representationt hat best suits the data

intelligence is compression

output feature vector

convolution layer find most important features of out image
we pass it trough relu to introduce nonlinearities
we compress even fouther with a maxpooling layer


---

performance / speed tradeoff
- accuracy of the model vs inference/training time


confusion matrix

A confusion matrix is a fantastic way of evaluating your classification models visually: https://www.learnpytorch.io/02_pytorch_classification/#9-more-classification-evaluation-metrics

Make predictions with our trained model on the test dataset
Make a confusion matrix torchmetrics.ConfusionMatrix - https://torchmetrics.readthedocs.io/en/stable/classification/confusion_matrix.html
Plot the confusion matrix using mlxtend.plotting.plot_confusion_matrix() - http://rasbt.github.io/mlxtend/user_guide/plotting/plot_confusion_matrix/



---

Self-driving cars, such as Tesla using computer vision to percieve what's happening on the road. See Tesla AI day for more - https://youtu.be/j0z4FweCy4M
Healthcare imaging, such as using computer vision to help interpret X-rays. Google also uses computer vision for detecting polyps in the intenstines - https://ai.googleblog.com/2021/08/improved-detection-of-elusive-polyps.html
Security, computer vision can be used to detect whether someone is invading your home or not - https://store.google.com/au/product/nest_cam_battery?hl=en-GB


---

how to avid overfitting


Note: there are lots of these, so don't worry too much about all of them, just pick 3 and start with those.

See this article for some ideas: https://elitedatascience.com/overfitting-in-machine-learning

3 ways to prevent overfitting:

Regularization techniques - You could use dropout on your neural networks, dropout involves randomly removing neurons in different layers so that the remaining neurons hopefully learn more robust weights/patterns.
Use a different model - maybe the model you're using for a specific problem is too complicated, as in, it's learning the data too well because it has so many layers. You could remove some layers to simplify your model. Or you could pick a totally different model altogether, one that may be more suited to your particular problem. Or... you could also use transfer learning (taking the patterns from one model and applying them to your own problem).
Reduce noise in data/cleanup dataset/introduce data augmentation techniques - If the model is learning the data too well, it might be just memorizing the data, including the noise. One option would be to remove the noise/clean up the dataset or if this doesn't, you can introduce artificial noise through the use of data augmentation to artificially increase the diversity of your training dataset.