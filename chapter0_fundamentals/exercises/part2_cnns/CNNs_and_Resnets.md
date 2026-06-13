# `NN.Module`
```Python
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self, arg1, arg2, ...):
        super().__init__()
        # Initialization code - setting up of attributes
		self.weights = nn.Parameter(weights) # wrapping a tensor in nn.Parameter
        self.biases = nn.Parameter(biases)

    def forward(self, x: Tensor) -> Tensor:
        # Forward pass code - uses init attributes and does whatever to them in the forward pass
        
	def extra_repr(self) -> str:
        return f"arg1={self.arg1}, arg2={self.arg2}, ..." # extra string representations
```
- 2 basic functions of `NN.Module` used by all classes that inherit it: `NN.relu`, `nn.softmax` etc
- you don't explicitly call the forward method ever and you don't define your layers there, cause otherwise they will get reinstantiated each time forward() is called, which is once per batch.
## `nn.Parameter`
- Torch class for storing weights and biases of a Module.
- If a `Parameter` is set as an attribute of a `Module`, it will be auto-detected by torch and returned when you call `module.parameters()` (along with all the other `Parameters` associated with the `Module`, or any of the `Module`'s sub-modules!).
- This makes it easy to pass all the parameters of a model into an optimizer and update them all at once.
## RELU
- simplest nonlinear activation function
- element-wise maximum between the input and a tensor of zeros.
- **Universal Approximation Theorem**: we can approximate any continuous function using a sufficiently large neural network, if we use nonlinear activation functions
- **Uniform Kaiming Initialization**: this is what we use to initialize bias/weight values for a Linear layer instead of randomizing them, defined as follows: $$[\frac{-1}{\sqrt{N_{in}}}, \frac{1}{\sqrt{N_{in}}}]$$
	- where $N_{in}$ is the input range (in_features).
## Flatten
- `flatten(start_dim=1, end_dim=-1)` allows you to flatten into single dimension over all input layers but the first batch dimension layer.
## MLP
- Simple architecture of linear and non linear alternating layers

# Jan's Talk
- feature maps - hidden layer dimensions (basically moving of the convolution)