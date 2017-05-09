# mkltorch
MKLDNN library is designed to accelerate Deep Neural Network(DNN) 
computation on CPU, in particular Intel® Xeon processors (HSW, 
BDW, Xeon Phi). The repo of mkltorch provides mklTensor to be called
 in MKLDNN library in [Torch7](http://torch.ch) for convenience .
# relation with torch
__Torch__ is the main package in Torch7 where data structures for 
multi-dimensional tensors and mathematical operationsover these are 
defined. Additionally, it provides many utilities for accessing files,
 serializing objects of arbitrary types and other useful utilities.

__mklTorch__ warps the data structure of tensor which is provided by
__Torch__ and offers a Tensor which can be named as mklTensor. It is 
necessary to convert the regular tensor to mklTensor if you want to 
use MKLDNN to boost your neural networksi computation.

If there is a regular tensor variable named as a, you can call its mkl() 
method and you will get the corresponding mklTensor. It is not very hard.
```lua
require 'mkltorch'
a = torch.FloatTensor(2,3)
b = a:mkl()
```   
Accordingly, it is easy to convert a mklTensor to a regular tensor just 
by using th() method. Just like:
```lua
require 'mkltorch'
a = torch.FloatTensor(2,3)
b = a:mkl()
c = b:th()
```   
__NOTE:__ 

  * mklTensor doesn't provide any basic mathematical operations. You can 
transfer it to a regular tensor if necessary.
  * Any convertions only occur between the tweo types tentors which have
 same data type.In other word, a regular tensor with float type only can 
be converted to MKLFloatTensor. It is impossible to get a corresponding 
MKLDoubleTensor.
  * There are only two type data structures of mklTensor(float and double)
 currently.   

## mklTensor Library ##
It offers mklTensor some basic operation to create, copy, convert or query some infos.
   * new()               create a new mklTensor and return it.
   * th()                convert a mklTensor to the regular tensor and return the regular tensor. 
   * float()             convert a MKLFloatTensor to the regular float tensor and return the regular float tensor.
   * double()            convert a MKLDoubleTensor to the regular double tensor and return the regular double tensor.
   * MKL2TH(A)           convert a mklTensor to the regular tensor and return the regular tensor. And also assigned it to A
   * TH2MKL(A)           convert a regular tensor A to the mklTensor and return the mklTensor. 
   * nElement()          return the ammount of the mklTensor.


The package also provide a regular tensor conversion methods to get the corresponding mklTensor.
   * mkl()               convert a regular tensor to the mklTensor and return the mklTensor.
   * mklFloat()          convert a regular float tensor to the MKLFloatTensor and return the MKLFloatTensor.
   * mklDouble()         convert a regular double tensor to the MKLDoubleTensor and return the MKLDoubleTensor.


