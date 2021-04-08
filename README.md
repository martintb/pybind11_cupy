# pybind11_cupy

An example of passing cupy arrays directly to C++/CUDA. The current example allocates three 1D cupy device arrays, fills two of them with values, and then stores the sum of the first two arrays in the third. The summation operation is carried out in a CUDA kernel defined in `demo.cu`.

## Compilation/Invocation 
```
$ cmake .
$ make
$ python test.py
`100000000``
```


## Example Output
```
(base) [tbm@sulphur:/home/tbm/cuda] $ python test.py
Allocated array x on <CUDA Device 0>: (100000000,) elements,  381 Mb
Allocated array y on <CUDA Device 0>: (100000000,) elements,  381 Mb
Allocated array z on <CUDA Device 0>: (100000000,) elements,  381 Mb
Used memory in CuPy memory pool: 1144 Mb
Total memory in CuPy memory pool: 1144 Mb

Adding x to y and storing result in z...
x: [1. 1. 1. ... 1. 1. 1.]
y: [3.14 3.14 3.14 ... 3.14 3.14 3.14]
z: [4.1400003 4.1400003 4.1400003 ... 4.1400003 4.1400003 4.1400003]

Attempting to free all memory on GPU...
Used memory in CuPy memory pool:    0 Mb
Total memory in CuPy memory pool:    0 Mb
```
