
### Training the model

```python

# instance of nn.Model
model = ModelClass()

# check inside your model
list(model.parameters())
model.state_dict()

# Make predictions with model
with torch.inference_mode():
  y_preds = model_0(X_test)


# Setup a loss function
loss_fn = nn.L1Loss()

# Setup an optimizer (stochastic gradient descent)
optimizer = torch.optim.SGD(params=model_0.parameters(),  lr=0.01) 

# reproducibility
torch.manual_seed(42)

```

TRAINING
moving from unknownd parameters to known parameters

to a poor reperesentation of the data to a better one

LOSS FUNCTION (or cost function or criterion)
measure how poor or wrong your model is
most common one is least square errors


move parameters in order to minimize the loss function

L1 loss - absolute error MAE mean absolute error
MSE loss - L2^2 mean squared error 
CE loss - cross entropy, used in classification
BCE loss - binary cross entropy, binary classification
cost function is problem specific

MAE_Loss = torch.mean(torch.abs(y_pred - y_test))

OR

torch.nn.L1Loss

LOSS FUNCTION + OPTIMIZER


TORCH:OPTIM
SGD - stochastic gradient descent
ADAM 


LOSS = measure how wrong our model is
OPTIM = adjust model parameters to reduce the lo

lr=0.01) # lr = learning rate = possibly the most important hyperparameter you can set
you get this value with experience

weight, bias (a parameter is a value that the model sets itself)

the more the learning rate the more it asust the parameters in one hit

lr (learning rate) - the learning rate is a hyperparameter that defines how big/small the optimizer changes the parameters with each step (a small lr results in small changes, a large lr results in large changes)

parameter - the model sets itself
hyperparameter - we ML engineers set it

BUILDING A TRAINING LOOP
 
 foward pass, foward propagation


HYPERPARAMETERS
epochs

```python
### Training

# An epoch is one loop through the data
epochs = 200 
epoch_count = [] 
loss_values = []
test_loss_values = [] 

for epoch in range(epochs): 

  # Training Mode (ets all parameters to require grad)
  model_0.train() 

  # 1. Forward pass (feeds training data to forward())
  y_pred = model_0(X_train)

  # 2. Calculate the loss (how wrong is the model)
  loss = loss_fn(y_pred, y_train)

  # 3. Clears the gradients (they accumulate by default)
  optimizer.zero_grad() 

  # 4. Back-propagation (calculates grad of each parameter)
  loss.backward()

  # 5. Gradient-descent (updates model parameters)
  optimizer.step() 

  ### Testing
  model_0.eval() 
  with torch.inference_mode():
    # 1. Do the forward pass 
    test_pred = model_0(X_test)

    # 2. Calculate the loss
    test_loss = loss_fn(test_pred, y_test)

  # Print out what's happening
  if epoch % 10 == 0:
    epoch_count.append(epoch)
    loss_values.append(loss)
    test_loss_values.append(test_loss)
    print(f"\nEpoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
    print(model_0.state_dict())

```

optimizer always after backpropagation
optimizers makes adjustments in parameters in of in regards of the backpropagation of the loss

by default how the optimizer changes will acculumate through the loop so... we have to zero them above in step 3 for the next iteration of the loop

backprob
we take the gradients of the loss function with respect to each parameter

  
model_0.eval() : turns off different settings in the model not needed for evaluation/testing (dropout/batch norm layers) 
with torch.inference_mode(): turns off gradient tracking, not needed when predicting

TRAINING AND TESTING LOOPS ARE SIMILAR
the only different is that  in training you keep track of the gradient so you adjust with backprop and grad descent

il loss dipende dai dati
loss and test_loss   

KEEPING TRACK OP MODEL PROGRESS
we keep the value so we can compare past experiments with future ones


the model is goof if train loss and test loss match up
model is converging

 with torch.inference_mode():
 good practice, better performance
 stop grad tracking during inference

 if you got a big model, you probably wanna save it and reuse it


There are three main methods you should about for saving and loading models in PyTorch.

torch.save() - allows you save a PyTorch object in Python's pickle format
torch.load() - allows you load a saved PyTorch object
torch.nn.Module.load_state_dict() - this allows to load a model's saved state dictionary

serializing (saving) deserializing (loading)


recommanded way tos save and load is to save the sate dich

Because state_dict objects are Python dictionaries, they can be easily saved, updated, altered, and restored, adding a great deal of modularity to PyTorch models and optimizers.

When saving a model for inference, it is only necessary to save the trained model’s learned parameters. Saving the model’s state_dict with the torch.save() function will give you the most flexibility for restoring the model later, which is why it is the recommended method for saving models.

you don't init parametes but layers with parametes in tehre

nn.Linear for creating linear parameters


linear layer
proving layer
linear transform
dense layer
fully connected layer

using preexisting layers in DEEP LERNING MATHEMATICAL TRANSFORMATION


TRAINING
- loss function
- optimizer
- training loop
- tesing loop


-----

2 CLASSIFICATION

second biggest problem in machine learning

email spam or not spam?
food classification app?

imagenet dataset
popular for computervision

imagenet1000 - 1000 class classification


MULTICLASS CLASSIFICATIOn (1 lable to each)
MULTILABLE CLASSIFICATION (multiple lable and classes)


classification input and output

input = numerical representation of images
output = prediction probability

W, H = 224 C =3 (RGB)

32 3 224 224 most used

batchsize always less than 32
looks at 32 images at a time
 
 🍕🧁 AI for Food Recognition 🍩🥗


pytorch DL
scikitlearn ML

out_feature needs to match to in_feature of the next layer
if not layer mixmatch layer

the more hidden units the more opportunity our model has the possibility to learn patterns in the data


log enropy

logit in deep learning

optimizers SDG and adam


BCE with LOGIT LOSS
combies BCE with sigmoid layer in one


```python


class CircleModelV0(nn.Module):
  def __init__(self):
    super().__init__()
    # 2. Create 2 nn.Linear layers capable of handling the shapes of our data
    self.layer_1 = nn.Linear(in_features=2, out_features=5) # takes in 2 features and upscales to 5 features 
    self.layer_2 = nn.Linear(in_features=5, out_features=1) # takes in 5 features from previous layer and outputs a single feature (same shape as y)

  # 3. Define a forward() method that outlines the forward pass
  def forward(self, x):
    return self.layer_2(self.layer_1(x)) # x -> layer_1 ->  layer_2 -> output

# 4. Instantiate an instance of our model class and send it to the target device
model = CircleModelV0().to(device)


# is equal to

model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)


```



```python

nn.Sequential(
    nn.Sigmoid(),
    nn.BCELoss()
)


# is euqal to n
# more numerically stable

loss_fn = nn.BCEWithLogitsLoss()



```


```python

# Calculate accuracy - out of 100 examples, what percentage does our model get right? 
def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item() 
  acc = (correct/len(y_pred)) * 100
  return acc


  appuracy: true_positive/ (total)
  where total is true positives + true negatives
```

accuracy - evaluation function
comparing prediction to ground truth label


model output raw logit

logit goes from 0 1 to -inf +inf
logit is the inverse of the sigmoid logistic curve


the vector of raw (non-normalized) predictions that a classification model generates, which is ordinarily then passed to a normalization function. If the model is solving a multi-class classification problem, logits typically become an input to the softmax function. The softmax function then generates a vector of (normalized) probabilities with one value for each possible class.



I couldn't find a formal definition anywhere, but logit basically means:

The raw predictions which come out of the last layer of the neural network.
1. This is the very tensor on which you apply the argmax function to get the predicted class.
2. This is the very tensor which you feed into the softmax function to get the probabilities for the predicted classes.


n context of deep learning the logits layer means the layer that feeds in to softmax (or other such normalization). The output of the softmax are the probabilities for the classification task and its input is logits layer. The logits layer typically produces values from -infinity to +infinity and the softmax layer transforms it to values from 0 to 1.


Unfortunately the term logits is abused in deep learning. From pure mathematical perspective logit is a function that performs above mapping. In deep learning people started calling the layer "logits layer" that feeds in to logit function. Then people started calling the output values of this layer "logit" creating the confusion with logit the function.



We can convert these logits into prediction probabilities by passing them to some kind of activation function (e.g. sigmoid for binary classification and softmax for multiclass classification).

Then we can convert our model's prediction probabilities to prediction labels by either rounding them or taking the argmax().

Going from raw logits -> prediction probabilities -> prediction labels 


layers and actication functions
some form of mathematical oeprations




dense layer (logit) - sigmoid (pred probs) - rounf (pred lables)
###  In full (logits -> pred probs -> pred labels)


```python

y_logits = model_0(X_train).squeeze()
y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labels

# same but the second is more numerically stable

loss = nn.BCELoss(torch.sigmoid(y_logits), y_train)

loss = nn.BCEWithLogitsLoss(y_logits, y_train)




```

ideal acc 100
ideal loss 0



FROM A MODEL PRESPECTIVE:
since you can improve the model even from a daata perspective

HYPERPARAMETERS


Improving a model (from a model perspective)
Add more layers - give the model more chances to learn about patterns in the data
Add more hidden units - go from 5 hidden units to 10 hidden units
Fit for longer
Changing the activation functions
Change the learning rate
Change the loss function


when you are doing EXPERIMENT TRACKING you change one parameter per time
like a real scientist


logit is represented with z


  def forward(self, x):
    # z = self.layer_1(x)
    # z = self.layer_2(z)
    # z = self.layer_3(z) 
    return self.layer_3(self.layer_2(self.layer_1(x))) # this way of writing operations leverages speed ups where possible behind the scenes



accuraci is optional, loss is not




trick to troubleshoot in a different projcet
can our model, LEARN ANYTHING AT ALL??????
use easier data???

do we have something fundamentally wrong?


ACCURACY IS ONLY DEFINED IN CLASSIFICATION




NON LINEAR ACTIVATIONS
you can't model nonlinearities even with lots of linear layers


"What patterns could you draw if you were given an infinite amount of a straight and non-straight lines?"

Or in machine learning terms, an infinite (but really it is finite) of linear and non-linear functions?



def relu(x: torch.Tensor) -> torch.Tensor:
  return torch.maximum(torch.tensor(0), x) # inputs must be tensors

relu(A)

relu impplementation


HYPERPARAMETERS ARE ALL IN CAPS, start of the notebook


ASSIGNING WEIGHT TO EVERY CLASS is there are unbalanced training set

most data is not linearly saparable


Accuracy - out of 100 samples, how many does our model get right?
Precision
Recall
F1-score
Confusion matrix
Classification report

ACCURACY is not grat for imbalances
PRECISION AND RECOL are amazing for imbalances

precision recall tradeoff

f1 combines precions and recol