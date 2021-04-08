#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

// Error checking macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// CUDA kernel function to add the elements of two arrays
__global__
void cuadd(int n, float *x, float *y, float *z)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    z[i] = x[i] + y[i];
}

// Function to be called by Python
void pyadd(int N, size_t px, size_t py, size_t pz)
{

  float *x = reinterpret_cast<float*> (px);
  float *y = reinterpret_cast<float*> (py);
  float *z = reinterpret_cast<float*> (pz);

  // Run kernel on GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  cuadd<<<numBlocks, blockSize>>>(N,x,y,z);

  // Wait for GPU to finish before accessing on host
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
}

// This is the code to expose the pyadd function to Python
PYBIND11_MODULE(demo, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring
  m.def("pyadd", &pyadd, "A function which adds two cupy arrays");
}

