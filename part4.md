CUTSOM DATASET

can I use my own dataset?
yes but it needs to be compatible

CUSTOM LIBRARY

TorchVision
TorchText
TorchRec
TorchAudio

torchX.datasets

foodvision mini

torch.utils.tensorboard 

- cutom data
- transforming data
. data augmentation

Depending on what you're working on, vision, text, audio, recommendation, you'll want to look into each of the PyTorch domain libraries for existing data loading functions and customizable data loading functions.



Our dataset is a subset of the Food101 dataset.

Food101 starts 101 different classes of food and 1000 images per class (750 training, 250 testing).

Our dataset starts with 3 classes of food and only 10% of the images (~75 training, 25 testing).

Why do this?

When starting out ML projects, it's important to try things on a small scale and then increase the scale when necessary.

The whole point is to speed up how fast you can experiment.

 increase the rate of experiement


--- 


PIL PILLOW

way to  - image processing laibrary


---

 Transforming data
Before we can use our image data with PyTorch:

Turn your target data into tensors (in our case, numerical representation of our images).
Turn it into a torch.utils.data.Dataset and subsequently a torch.utils.data.DataLoader, we'll call these Dataset and DataLoader.



data augmentation, you never trasform test data

midifided or new synthetic
artificialy addting diversity in data
applying image transformations


artificially increase diversity of a dataset


/ harder to learn
/ new perspecrive


trivial augment



OPERATION FUSION; Faster
not reassingn x, 
most important optimization


torchINFO
print summary og model


softmax is not necessary
argmax wors to get the msot important prediction
with softmax you get normalized probabilities


BEST LOSS CURVES
train and test loss is similar

UNDERFITTIN - your loss could be lowet
OVERFITTING; train loss lower than loss, (learning too much)


LRS learning rate scheduling,
initial big staep, final close steps
dinamic learning rate, linear decreasing


- wrong dt
wrong shape
wrong device


Note, to make a prediction on a custom image we had to:

Load the image and turn it into a tensor
Make sure the image was the same datatype as the model (torch.float32)
Make sure the image was the same shape as the data the model was trained on (3, 64, 64) with a batch size... (1, 3, 64, 64)
Make sure the image was on the same device as our model


We have to make sure our custom image is in the same format as the data our model was trained on.



When starting out ML projects, it's important to try things on a small scale and then increase the scale when necessary.

The whole point is to speed up how fast you can experiment.



Before we can use our image data with PyTorch:
1. Turn your target data into tensors (in our case, numerical representation of our images).
2. Turn it into a `torch.utils.data.Dataset` and subsequently a `torch.utils.data.DataLoader`, we'll call these `Dataset` and `DataLoader`.




5 Option 2: Loading Image Data with a Custom Dataset
Want to be able to load images from file
Want to be able to get class names from the Dataset
Want to be able to get classes as dictionary from the Dataset
Pros:

Can create a Dataset out of almost anything
Not limited to PyTorch pre-built Dataset functions
Cons:

Even though you could create Dataset out of almost anything, it doesn't mean it will work...
Using a custom Dataset often results in us writing more code, which could be prone to errors or performance issues
All custom datasets in PyTorch, often subclass - https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset