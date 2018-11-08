// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here
__global__ void floattoUnsignedChar(float *input , unsigned char *output, int width, int height, int channels){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < width * height * channels) output[id] = (unsigned char)(input[id] * 255);
}
  
__global__ void rgbtoGray(unsigned char * input, unsigned char * output, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < width && y < height){
        int idx = y * width + x;
        unsigned char r = input[3 * idx];
        unsigned char g = input[3 * idx + 1];
        unsigned char b = input[3 * idx + 2];
        output[idx] = (unsigned char) (0.21 * r + 0.71 * g + 0.07 * b);
    }

}

  
__global__ void computeHistogram(unsigned char * input, unsigned int * output ,int width, int height){
    __shared__ unsigned int histogram[HISTOGRAM_LENGTH];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(threadIdx.x < HISTOGRAM_LENGTH){
      histogram[threadIdx.x] = 0;
    }
    __syncthreads();
    // int x = blockIdx.x * blockDim.x + threadIdx.x;
    // int y = blockIdx.y * blockDim.y + threadIdx.y;

    // if(x < width && y < height){
    //   unsigned char tmpval = input[y * width + x];
    //   atomicAdd(&(histogram[tmpval]), 1);
    // }
    if(id < width * height){
      unsigned char tmpval = input[id];
        atomicAdd(&(histogram[tmpval]), 1);
    }
      
    __syncthreads();
    if(threadIdx.x < HISTOGRAM_LENGTH){
      //output[id] = histogram[id];
      atomicAdd(&(output[threadIdx.x]), histogram[threadIdx.x]);
    }
    
}

  
__global__ void getCDFfromHistogram(unsigned int * input, float * output, int width , int height ){
    __shared__ unsigned int CDF[HISTOGRAM_LENGTH];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadIdx.x < HISTOGRAM_LENGTH ){
        //CDF[ i] = (1.0 *input[i]) / (width * height);
      CDF[ threadIdx.x] = input[threadIdx.x];
    }
    __syncthreads();

    int stride = 1;
    while(stride < HISTOGRAM_LENGTH){
      __syncthreads();
      int id = (threadIdx.x + 1) * stride * 2 - 1;
      if(id < HISTOGRAM_LENGTH){
        CDF[id] += CDF[id - stride];
      }
      stride = stride * 2;
    }
    stride = HISTOGRAM_LENGTH / 2;
    while(stride > 0){
      __syncthreads();
      int id = (threadIdx.x + 1) * stride * 2 - 1;
      if((id + stride) < HISTOGRAM_LENGTH){
        CDF[id + stride] += CDF[id]; 
      }
      stride = stride / 2;
    }
    __syncthreads();
    //if(i < HISTOGRAM_LENGTH)
        output[threadIdx.x] = CDF[threadIdx.x] / (1.0 * width * height);
} 
  
__global__ void equalize(unsigned char *image, float * CDFInput, int width , int height, int channels){

    __shared__ float CDF[HISTOGRAM_LENGTH];

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadIdx.x < HISTOGRAM_LENGTH ){
        CDF[ threadIdx.x] = CDFInput[threadIdx.x];
    }
    __syncthreads();

    if(id < width * height * channels){
        unsigned char element = image[id];
        image[id]  = (unsigned char) (min( max(255.0 * (CDF[element] - CDF[0]) / (1.0 -CDF[0] ), 0.0), 255.0));
    }

}
  
  
__global__ void unsignedChartoFloat(unsigned char * input, float * output, int width, int height, int channels){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < width * height * channels) output[id] = (float)input[id] / 255.0;
}


int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInput;
  unsigned char *deviceInputChar;
  unsigned char *deviceInputGray;
  unsigned int *deviceInputHistogram;
  float * deviceCDF;
  float *deviceOutput;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage); 
  hostOutputImageData = wbImage_getData(outputImage); 
  wbTime_stop(Generic, "Importing data and creating memory on host");
  printf("The hostInputImageData is %d, %d \n",imageWidth, imageHeight);

  //@@ insert code here
  cudaMalloc((void **) &deviceInput, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceInputChar, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void **) &deviceInputGray, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **) &deviceInputHistogram,  HISTOGRAM_LENGTH* sizeof(unsigned int));
  cudaMalloc((void **) &deviceCDF,  HISTOGRAM_LENGTH* sizeof(float));
  cudaMalloc((void **) &deviceOutput, imageWidth * imageHeight * imageChannels * sizeof(float)); 

  cudaMemcpy(deviceInput , hostInputImageData , imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  
  cudaMemset(deviceInputHistogram , 0 , HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset(deviceCDF , 0 , HISTOGRAM_LENGTH * sizeof(float));
  

  dim3 DimGrid(ceil((imageWidth * imageHeight * imageChannels)  / (1.0 * 512)), 1, 1);
  dim3 DimBlock(512, 1 , 1);
  floattoUnsignedChar<<<DimGrid, DimBlock>>>(deviceInput, deviceInputChar, imageWidth, imageHeight , imageChannels);
  cudaDeviceSynchronize();

  dim3 DimGrid2(ceil(imageWidth / (1.0 * 32)), ceil(imageHeight / (1.0 * 32)), 1);
  dim3 DimBlock2(32, 32 , 1);
  rgbtoGray<<<DimGrid2, DimBlock2>>>(deviceInputChar, deviceInputGray, imageWidth, imageHeight);
  cudaDeviceSynchronize();

//   unsigned char *testdevice = new unsigned char[imageWidth * imageHeight * sizeof(unsigned char)];
//   cudaMemcpy(testdevice, deviceInputGray, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);
//   for(int i = 0 ; i < imageWidth * imageHeight; i ++)
//     printf("%d ",testdevice[i]);
  
  dim3 DimGrid3(ceil((imageWidth * imageHeight)  / (1.0 * 512)), 1, 1);
  dim3 DimBlock3(512, 1 , 1);
  computeHistogram<<<DimGrid3, DimBlock3>>>(deviceInputGray, deviceInputHistogram, imageWidth, imageHeight);
  cudaDeviceSynchronize();

//   unsigned int *testdevice = new unsigned int[HISTOGRAM_LENGTH* sizeof(unsigned int)];
//   cudaMemcpy(testdevice, deviceInputHistogram, HISTOGRAM_LENGTH* sizeof(unsigned int), cudaMemcpyDeviceToHost);
//   for(int i = 0 ; i < HISTOGRAM_LENGTH; i ++)
//     printf("%d ",testdevice[i]);
  


  dim3 DimGrid4(1, 1, 1);
  dim3 DimBlock4(HISTOGRAM_LENGTH, 1 , 1);
  getCDFfromHistogram<<<DimGrid4, DimBlock4>>>(deviceInputHistogram, deviceCDF, imageWidth, imageHeight);
  cudaDeviceSynchronize();
    
//   float *testdevice = new float[HISTOGRAM_LENGTH* sizeof(float)];
//   cudaMemcpy(testdevice, deviceCDF, HISTOGRAM_LENGTH* sizeof(float), cudaMemcpyDeviceToHost);
//   for(int i = 0 ; i < HISTOGRAM_LENGTH; i ++)
//     printf("%f ",testdevice[i]);
    //printf("The first element is %d, %d , %d \n",testdevice[130],testdevice[129] , testdevice[131]);


  dim3 DimGrid5(ceil((imageWidth * imageHeight *imageChannels )  / (1.0 * 512)), 1, 1);
  dim3 DimBlock5(512, 1 , 1);
  equalize<<<DimGrid5, DimBlock5>>>(deviceInputChar, deviceCDF, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  
//   float *testdevice = new float[HISTOGRAM_LENGTH* sizeof(float)];
//   cudaMemcpy(testdevice, deviceCDF, HISTOGRAM_LENGTH* sizeof(float), cudaMemcpyDeviceToHost);
//   for(int i = 0 ; i < HISTOGRAM_LENGTH; i ++)
//     printf("%f ",testdevice[i]);
  
// deviceInputChar
//   unsigned char *testdevice = new unsigned char[imageWidth * imageHeight * imageChannels * sizeof(unsigned char)];
//   cudaMemcpy(testdevice, deviceInputChar, imageWidth * imageHeight * imageChannels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
//   for(int i = 0 ; i < imageWidth * imageHeight * imageChannels; i ++)
//     printf("%d ",testdevice[i]);

  dim3 DimGrid6(ceil((imageWidth * imageHeight * imageChannels)  / (1.0 * 512)), 1, 1);
  dim3 DimBlock6(512, 1 , 1);
  unsignedChartoFloat<<<DimGrid6, DimBlock6>>>(deviceInputChar, deviceOutput, imageWidth, imageHeight,imageChannels);
  cudaDeviceSynchronize();
  
//   float *testdevice = new float[imageWidth * imageHeight * imageChannels * sizeof(float)];
//   cudaMemcpy(testdevice, deviceOutput, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
//   for(int i = 0 ; i < imageWidth * imageHeight * imageChannels; i ++)
//     printf("%f ",testdevice[i]);
 
  
  
  
  cudaMemcpy(hostOutputImageData, deviceOutput, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

 

  wbSolution(args, outputImage);
  
  cudaFree(deviceInput); cudaFree(deviceInputChar); 
  cudaFree(deviceInputGray) ; cudaFree(deviceInputHistogram); 
  cudaFree(deviceCDF); cudaFree(deviceOutput);

  //@@ insert code here

  return 0;
}







