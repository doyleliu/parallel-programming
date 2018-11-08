#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define tileWidth 8
#define maskSize 3
#define maskRadius maskSize/2
#define W (tileWidth + (maskSize - 1))

//@@ Define constant memory for device kernel here
__constant__ float mask[maskSize][maskSize][maskSize];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int x_out = blockIdx.x * tileWidth + tx;
  int y_out = blockIdx.y * tileWidth + ty;
  int z_out = blockIdx.z * tileWidth + tz;
  
  int x_in = x_out - maskRadius;
  int y_in = y_out - maskRadius;
  int z_in = z_out - maskRadius;
  
  float result = 0.0f;
  
  __shared__ float N_s[W][W][W];
  if((x_in >= 0) && (x_in < x_size)&& (y_in >= 0) && (y_in < y_size) && (z_in >= 0) && (z_in < z_size)){
    N_s[tz][ty][tx] = input[z_in * (x_size*y_size) + y_in * x_size + x_in];
  }else{
    N_s[tz][ty][tx] = 0;
  }
  
  __syncthreads();
  
  int x_id, y_id, z_id;
  
  if(tx < tileWidth && ty < tileWidth && tz < tileWidth){
    for(int i = 0; i < maskSize; i++ ){
      for(int j = 0; j < maskSize; j ++){
        for(int k = 0 ; k < maskSize; k ++){
          result = result + mask[i][j][k] * N_s[i+tz][j+ty][k+tx];
        }
      }
    }
    if(x_out < x_size && y_out < y_size && z_out < z_size){
      output[z_out * (x_size*y_size) + y_out * x_size + x_out] = result;
    }
  }
    
  
  
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  int inputSize = z_size * y_size * x_size;
  cudaMalloc((void **) &deviceInput, inputSize * sizeof(float));
  cudaMalloc((void **) &deviceOutput, inputSize * sizeof(float));
  
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, &hostInput[3] , inputSize * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(mask, hostKernel,  maskSize* maskSize * maskSize * sizeof(float));
  
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 DimGrid(ceil(x_size/(1.0 * tileWidth)), ceil(y_size/(1.0 * tileWidth)), ceil(z_size/(1.0 * tileWidth)));
  dim3 DimBlock(W, W, W);
  
  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[3], deviceOutput, inputSize * sizeof(float), cudaMemcpyDeviceToHost);
  
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

