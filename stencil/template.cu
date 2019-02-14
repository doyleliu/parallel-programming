#include <cstdio>
#include <cstdlib>

#include "helper.hpp"

#define TILE_SIZE 32

__global__ void kernel(int *A0, int *Anext, int nx, int ny, int nz) {

  // INSERT KERNEL CODE HERE
  #define A0(i, j, k) A0[((k)*ny + (j))*nx + (i)]
  #define Anext(i, j, k) Anext[((k)*ny + (j))*nx + (i)]
  __shared__ int ds_A[TILE_SIZE][TILE_SIZE];
  
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;

  int bottom = A0(idx, idy, 0);
  int current = A0(idx, idy, 1);
  int top = A0(idx, idy, 2);
  ds_A[threadIdx.y][threadIdx.x] = current;

  __syncthreads();

  for(int k = 1; k< nz -1;k ++){
    // if((idx == 0) && (idy ==0) || (idx ==0 && idy == ny-1)||(idx == nx-1 && idy == 0)||(idx == nx-1 && idy == ny -1)){
    //   ds_A[threadIdx.y][threadIdx.x] = A0(idx,idy,k);
    // }
    if(idx > 0 && idx < nx && idy >0 && idy < ny){
      // __syncthreads();
      Anext(idx, idy, k) = bottom + top +
        ((threadIdx.x > 0)? ds_A[threadIdx.y][threadIdx.x-1]:A0(idx-1, idy, k)) + 
        ((threadIdx.x < blockDim.x -1)? ds_A[threadIdx.y][threadIdx.x+1]:A0(idx+1, idy, k)) +
        ((threadIdx.y > 0)? ds_A[threadIdx.y-1][threadIdx.x]:A0(idx, idy-1, k))+ 
        ((threadIdx.y < blockDim.y - 1)? ds_A[threadIdx.y+1][threadIdx.x]:A0(idx, idy+1, k)) - 6* current;
        // A0(idx-1, idy, k) +A0(idx+1, idy, k) +A0(idx, idy-1, k)+ A0(idx, idy+1, k) - 6*current;
    }
    
    __syncthreads();
    bottom = current; 
    ds_A[threadIdx.y][threadIdx.x] = top;
    current = top;
    
    if(k + 2 < nz)
      top = A0(idx, idy, k+2);
    else
      top = 0;
    __syncthreads();
  }
  
}

void launchStencil(int* A0, int* Anext, int nx, int ny, int nz) {

  // INSERT CODE HERE
  dim3 DimGrid(ceil(nx / (TILE_SIZE * 1.0)), ceil(ny / (TILE_SIZE * 1.0)), 1);
  dim3 DimBlock(TILE_SIZE,TILE_SIZE,1);
  kernel<<<DimGrid,DimBlock>>>(A0, Anext, nx, ny, nz);


  // CPU compute test

}


static int eval(const int nx, const int ny, const int nz) {

  // Generate model
  const auto conf_info = std::string("stencil[") + std::to_string(nx) + "," + 
                                                   std::to_string(ny) + "," + 
                                                   std::to_string(nz) + "]";
  INFO("Running "  << conf_info);

  // generate input data
  timer_start("Generating test data");
  std::vector<int> hostA0(nx * ny * nz);
  generate_data(hostA0.data(), nx, ny, nz);
  std::vector<int> hostAnext(nx * ny * nz);

  timer_start("Allocating GPU memory.");
  int *deviceA0 = nullptr, *deviceAnext = nullptr;
  CUDA_RUNTIME(cudaMalloc((void **)&deviceA0, nx * ny * nz * sizeof(int)));
  CUDA_RUNTIME(cudaMalloc((void **)&deviceAnext, nx * ny * nz * sizeof(int)));
  timer_stop();

  timer_start("Copying inputs to the GPU.");
  CUDA_RUNTIME(cudaMemcpy(deviceA0, hostA0.data(), nx * ny * nz * sizeof(int), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  //////////////////////////////////////////
  // GPU Gather Computation
  //////////////////////////////////////////
  timer_start("Performing GPU convlayer");
  launchStencil(deviceA0, deviceAnext, nx, ny, nz);
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  timer_start("Copying output to the CPU");
  CUDA_RUNTIME(cudaMemcpy(hostAnext.data(), deviceAnext, nx * ny * nz * sizeof(int), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  // verify with provided implementation
  timer_start("Verifying results");
  verify(hostAnext.data(), hostA0.data(), nx, ny, nz);
  timer_stop();

  CUDA_RUNTIME(cudaFree(deviceA0));
  CUDA_RUNTIME(cudaFree(deviceAnext));

  return 0;
}



TEST_CASE("Stencil", "[stencil]") {

  SECTION("[dims:32,32,32]") {
    eval(32,32,32);
  }
  SECTION("[dims:30,30,30]") {
    eval(30,30,30);
  }
  SECTION("[dims:29,29,29]") {
    eval(29,29,29);
  }
  SECTION("[dims:31,31,31]") {
    eval(31,31,31);
  }
  SECTION("[dims:29,29,2]") {
    eval(29,29,29);
  }
  SECTION("[dims:1,1,2]") {
    eval(1,1,2);
  }
  SECTION("[dims:512,512,64]") {
    eval(512,512,64);
  }

}
