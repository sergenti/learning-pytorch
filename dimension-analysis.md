```python
import torch
from torch import nn

# layers
input_layer = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=0)
conv_layer = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=0)
relu_layer = nn.ReLU()
pooling_layer = nn.MaxPool2d(kernel_size = 2, stride=2)
flatten_layer =  nn.Flatten()
linear_layer = nn.Linear(in_features = 10 * 13 * 13, out_features=10)

# computations
image = torch.rand(1, 3, 64, 64)
conv1 = input_layer(image)
relu1 = relu_layer(conv1)
conv2 = conv_layer(relu1)
relu2 = relu_layer(conv2)
pool1 = pooling_layer(relu2)
conv3 = conv_layer(pool1)
relu3 = relu_layer(conv3)
conv4 = conv_layer(relu3)
relu4 = relu_layer(conv4)
pool2 = pooling_layer(relu4)
flat = flatten_layer(pool2)
out = linear_layer(flat)

print(
f'''
Tiny VGG Architecture

Input Layer
image: {list(image.size())}

Convolutional Block 1
conv1: {list(conv1.size())}
relu1: {list(relu1.size())}
conv2: {list(conv2.size())}
relu2: {list(relu2.size())}
pool1: {list(pool1.size())}

Convolutional Block 2
conv3: {list(conv3.size())}
relu3: {list(relu3.size())}
conv4: {list(conv4.size())}
relu4: {list(relu4.size())}
pool2: {list(pool2.size())}

Classifier Block
flat:  {list(flat.size())}
out:  {list(out.size())}

'''
)
```