#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Kernel function to add the elements of two arrays
__global__
void cuadd(int n, float *x, float *y, float *z)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    z[i] = x[i] + y[i];
}

int pyadd(int N, size_t px, size_t py, size_t pz)
{

  float *x = (float*) px;
  float *y = (float*) py;
  float *z = (float*) pz;

  // Run kernel on GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  cuadd<<<numBlocks, blockSize>>>(N,x,y,z);

  // Wait for GPU to finish before accessing on host
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}

PYBIND11_MODULE(demo, m) {
        m.doc() = "pybind11 example plugin"; // optional module docstring
        m.def("pyadd", &pyadd, "A function which adds two cupy arrays");
}
