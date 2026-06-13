[Golden resource for quick revision of Probability Theory](https://medium.com/jun94-devpblog/prob-stats-3-expected-value-variance-and-standard-deviation-bce9303d8da8)
# Information Theory
## Expected Surprise
> $D_{KL}(P||Q)$= how much more surprised you expect to be when observing data with distribution P, if you falsely believe the distribution is Q vs if you know the true distribution. 
- Simple: Prob of surprise observing P if you falsely assumed it to be Q.
## Hypothesis Testing
> $D_{KL}(P||Q)=$ the amount of evidence we expect to get for P over Q in hypothesis testing, if P is true.
- 
## MLEs
> If P is an empirical distribution of data, $D_{KL}(P||Q)$ is minimised (over Q) when Q is the maximum likelihood estimator for P.

## Suboptimal Coding
> $D_{KL}(P||Q)$= the number of bits we're wasting, if we try and compress a data source with distribution P using a code which is actually optimized for Q (i.e. a code which would have minimum expected message length if Q were the true data source distribution).
# PyTorch
- PyTorch gives 2 primitives to work with datasets:  `torch.utils.data.DataLoader` and `torch.utils.data.Dataset`. `Dataset` stores samples and their labels, and `DataLoader` gives you the iterable for it.
- `nn.Module` class is inherited to write your own Neural Network in Pytorch.
### Broadcasting
- When two tensors are involved in an elementwise operation, NumPy/PyTorch tries to broadcast them (i.e. copying them along dimensions) so that they both have the same shape.
- Rules:
	- You can prepend dummy dimensions of size 1 to the start of a tensor until they both have same dimensions
	- After this point, if some dimension has size 1 in one of the tensors, it can be repeated until it matches the size of the corresponding dimension in the other tensor
- Example: 
	- if we have a 2-D data of `data.shape = (N, k)` (i.e. we have `N` separate datapoints, each being a vector of length `k`).
	- If we want to add vector `vec` of length `k` to each datapoint, we first append a dummy dimension to `vec` so it gets the shape `(1, k)`
	- Then `vec` is repeated across 1st dimension so it gets final shape `(N, k)`
	- So now output has shape `(N, k)` and elements `output[i, j] = data[i, j] + vec[j]`
- ==Basic tip ==of how to see if 2 tensors align for broadcasting - they have same number of dimensions (or you can append a dimension of 1), and from left to right, ***either each dimension is the same number or one of them is 1.*** (the only rule actually)
- `tensor.unsqueeze(dim)` will give you a new tensor with a dummy dimension of size 1 inserted at position `dim` in the new tensor, and `tensor.squeeze(dim)` will give you a tensor with the dimension at position `dim` removed (if it had size 1, otherwise nothing happens).
- `normed = t.norm(matrix, dim=1, keepdim=True)` where matrix of size `(m, n)` ka L2 norm nikaal rhe hain. `dim=1` means L2 norm is calculated against x axis (row elements) and `keepdim=True` means the dimension of normed stays `(m, 1)` and so the dimensions are broadcasted.
## Classic important operations
- `tensor.argmax(dim)` will return a tensor of the index containing the maximum value along the dimension `dim` (i.e. the shape of this output will be the same as the shape of `tensor` except for the dimension `dim`).
```Python
def sample_distribution(probs: Tensor, n: int) -> Tensor:
    """Return n random samples from probs, where probs is a normalized probability distribution.
    probs: shape (k,) where probs[i] is the probability of event i occurring.
    n: number of random samples
    Return: shape (n,) where out[i] is an integer indicating which event was sampled.
    Use t.rand and t.cumsum to do this without any explicit loops.
    """
    cumsum = t.cumsum(probs, dim=0)
    random_values = t.rand(n, 1)
    return (random_values > cumsum).sum(dim=-1)
```
### LogSumExp and Underflow/Overflow
[Great Blog](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/): can easily follow along the code/formulae
## Einops
[Tutorial link](https://einops.rocks/1-einops-basics/)
- With einops you can do things like `rearrange(ims, "b h w c -> (b h w c)").shape` (flattened 4d dimensions into 1d).
 ```Python
# resulting dimensions are computed very simply
# length of newly composed axis is a product of components
# [6, 96, 96, 3] -> [96, (6 * 96), 3]
rearrange(ims, "b h w c -> h (b w) c").shape # output is (96, 576, 3)

# decomposition is the inverse process - represent an axis as a combination of new axes
# several decompositions possible, so b1=2 is to decompose 6 to b1=2 and b2=3
rearrange(ims, "(b1 b2) h w c -> b1 b2 h w c ", b1=2).shape # output is (2, 3, 96, 96, 3)

# average over batch
reduce(ims, "b h w c -> h w c", "mean") # same as ims.mean(axis=0)

# repeat along a new axis. New axis can be placed anywhere
repeat(ims[0], "h w c -> h new_axis w c", new_axis=5).shape # output is (96, 5, 96, 3)

# shortcut
repeat(ims[0], "h w c -> h 5 w c").shape
  ```
