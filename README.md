# mkltorch
MKLDNN library is designed to accelerate Deep Neural Network(DNN) computation. The repo of mkltorch provides MKLTensor to be called in MKL DNN library in convenience.
# relation with torch
__Torch__ is the main package in [Torch7](http://torch.ch) where data
structures for multi-dimensional tensors and mathematical operations
over these are defined. Additionally, it provides many utilities for
accessing files, serializing objects of arbitrary types and other
useful utilities.

__mklTorch__ warps the data structure of tensor which is privided by
__Torch__ and offers a Tensor which can be named as mklTensor. It is 
necessary to convert the regular tensor to mklTensor if you want to 
use MKL DNN to boost your neural networks on CPU, in particular Intel® 
Xeon processors (HSW, BDW, Xeon Phi).

If there is a regular tensor variable named as A, you can call its mkl() 
method and you will a corresponding mklTensor. It is not hard.
```lua
require 'mkltorch'
a = torch.FloatTensor(2,3)
b = a:mkl()
```   

