import demo
import numpy as np
import cupy as cp

#get memory pool object for memory profiling
mempool = cp.get_default_memory_pool()

x = cp.ones(100000000,dtype=np.float32)
y = cp.ones(100000000,dtype=np.float32)*3.14
z = cp.empty(100000000,dtype=np.float32)
print(f'Allocated array x on {x.device}: {x.shape} elements, {x.nbytes/1024/1024:4.0f} Mb')
print(f'Allocated array y on {y.device}: {x.shape} elements, {x.nbytes/1024/1024:4.0f} Mb')
print(f'Allocated array z on {z.device}: {x.shape} elements, {x.nbytes/1024/1024:4.0f} Mb')
print(f'Used memory in CuPy memory pool: {mempool.used_bytes()/1024/1024:4.0f} Mb')
print(f'Total memory in CuPy memory pool: {mempool.total_bytes()/1024/1024:4.0f} Mb')
print()

print('Adding x to y and storing result in z...')
demo.pyadd(len(x),x.data.ptr,y.data.ptr,z.data.ptr)
print('x:',x)
print('y:',y)
print('z:',z)
print()

print('Attempting to free all memory on GPU...')
del x, y, z
mempool.free_all_blocks()
print(f'Used memory in CuPy memory pool: {mempool.used_bytes()/1024/1024:4.0f} Mb')
print(f'Total memory in CuPy memory pool: {mempool.total_bytes()/1024/1024:4.0f} Mb')
