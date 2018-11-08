// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[BLOCK_SIZE];
  int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if(i < len){
    T[threadIdx.x] =  input[i];
  }else T[threadIdx.x] = 0;
  __syncthreads();
  int stride = 1;
  while(stride < BLOCK_SIZE){
    __syncthreads();
    int id = (threadIdx.x + 1) * stride * 2 - 1;
    if(id < BLOCK_SIZE){
      T[id] += T[id - stride];
    }
    stride = stride * 2;
  }
  stride = BLOCK_SIZE / 2;
  while(stride > 0){
    __syncthreads();
    int id = (threadIdx.x + 1) * stride * 2 - 1;
    if((id + stride) < BLOCK_SIZE){
      T[id + stride] += T[id]; 
    }
    stride = stride / 2;
  }
  __syncthreads();
  if(i < len) output[i] = T[threadIdx.x];
  
}

__global__ void block_scan(float *input, float *output, int len, int blockLen) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if( i < blockLen){
    if((i + 1) * BLOCK_SIZE - 1 < len){
      output[i] = input[(i + 1) * BLOCK_SIZE - 1];
    }else{
      output[i] = input[len - 1];
    }
  }
}

__global__ void getResult(float *output, float *blockVal, int len, int blockLen ){
  int i = threadIdx.x + blockIdx.x * BLOCK_SIZE;  
  if(blockIdx.x > 0 && i < len && blockIdx.x < blockLen){
    output[i] += blockVal[blockIdx.x - 1]; 
  } 
  
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  // Block list
  float *deviceBlockInput;
  float *deviceBlockOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  int numBlock = ceil(numElements / (1.0 * BLOCK_SIZE));
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceBlockInput, numBlock * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceBlockOutput, numBlock * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  
  dim3 DimGrid(ceil(numElements / (1.0 * BLOCK_SIZE)), 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  dim3 DimGridBlock(ceil(numBlock/(BLOCK_SIZE* 1.0)),1,1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numElements);
  cudaDeviceSynchronize();
  block_scan<<<DimGridBlock, DimBlock>>>(deviceOutput, deviceBlockInput, numElements, numBlock);
  cudaDeviceSynchronize();
  dim3 DimGrid2(ceil(numBlock/(BLOCK_SIZE* 1.0)),1,1);
  scan<<<DimGrid2, DimBlock>>>(deviceBlockInput, deviceBlockOutput, numBlock);
  cudaDeviceSynchronize();
  
  getResult<<<DimGrid, DimBlock>>>(deviceOutput, deviceBlockOutput, numElements, numBlock);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

