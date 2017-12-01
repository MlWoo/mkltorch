# mkltorch
[MKLDNN](https://github.com/01org/mkl-dnn) library is designed to accelerate Deep Neural Network(DNN) computation on CPU, in particular Intel® Xeon processors (HSW, BDW, Xeon Phi). The repo of mkltorch provides mklTensor and lower level API to be called in MKLDNN library in [Torch](http://torch.ch) for convenience .
# Relation with torch
__Torch__ is the main package in Torch7 where data structures for multi-dimensional tensors and mathematical operationsover these are defined. Additionally, it provides many utilities for accessing files, serializing objects of arbitrary types and other useful utilities.

__mklTorch__ warps the data structure of tensor which is provided by __Torch__ and offers a new Tensor which can be named as mklTensor. It is necessary to convert the regular tensor to mklTensor if you want to use MKLDNN to boost your neural network computation.

If there is a regular tensor variable named as a, you can call its mkl() method and you will get the corresponding mklTensor. It is not very hard.
```lua
require 'mkltorch'
a = torch.FloatTensor(2,3)
b = a:mkl()
```   
Accordingly, it is easy to convert a mklTensor to a regular tensor just by using th() method. Just like:
```lua
require 'mkltorch'
a = torch.FloatTensor(2,3)
b = a:mkl()
c = b:th()
```   
__NOTE:__  

  * Any basic mathematical operations of mklTensor are __potentially dangerous__. It should be converted a mkltensor to a regular tensor if it is necessary to handle with these basic mathematical operations.
  * Any convertions only occur between two types tentors which have __the same data type__. In other word, a regular tensor with float type only can be converted to MKLFloatTensor. It is impossible to get a corresponding MKLDoubleTensor.
  * There are only three type data structures of mklTensor(single float, double float and long) currently. Single and double float types of mklTensor are provided in float computation like neural network. However, the long type of mklTensor are __never__ used in you programs explicitly. It is a supporting tensor type which is only used in neural network Operations.   

## mklTensor Library ##
It offers mklTensor some basic operation to create, copy, convert or query some infos.
   * new()               
     create a new mklTensor and return it.
   * th()                
     convert a mklTensor to the regular tensor and return the regular tensor. 
   * float()             
     convert a MKLFloatTensor to the regular float tensor and return the regular float tensor.
   * double()            
     convert a MKLDoubleTensor to the regular double tensor and return the regular double tensor.
   * MKL2TH(A)           
     convert a mklTensor to the regular tensor and return the regular tensor. And also assigned it to A
   * TH2MKL(A)           
     convert a regular tensor A to the mklTensor and return the mklTensor. 
   * nElement()          
     Return the ammount of the mklTensor.
   * directTH()          
     fetch the tensor from a MKLFloatTensor. It is __NOT__ recommended to use this method by user in order to avoid potential problems.
   * size([dim])         
     Returns the size of the specified dimension dim or the sizes of all dimensions if param dim is ignored.        

The package also provide a regular tensor conversion methods to get the corresponding mklTensor.
   * mkl()               
     convert a regular tensor to the mklTensor and return the mklTensor.
   * mklFloat()          
     convert a regular float tensor to the MKLFloatTensor and return the MKLFloatTensor.
   * mklDouble()         
     convert a regular double tensor to the MKLDoubleTensor and return the MKLDoubleTensor.


