# pybind11_cupy

An example of passing cupy arrays directly to C++/CUDA. The current example allocates three 1D cupy device arrays, fills two of them with values, and then stores the sum of the first two arrays in the third. The summation operation is carried out in a CUDA kernel defined in `demo.cu`.


## Compilation/Invocation 
```
$ cmake .
$ make
$ python test.py
```
