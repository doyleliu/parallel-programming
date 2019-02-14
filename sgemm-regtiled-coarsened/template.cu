#include <cstdio>
#include <cstdlib>

#include "template.hu"

#define TILE_SZ_A 128
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A/TILE_SZ_B)

#define tileWidth 1

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

  /********************************************************************
  *
  * Compute C = A x B
  *   where A is a (m x k) matrix
  *   where B is a (k x n) matrix
  *   where C is a (m x n) matrix
  *
  * Use register and shared memory tiling and thread coarsening
  *
  * NOTE: A and C are column major, B is row major
  *
  ********************************************************************/

  // Macros for accessing flattened matrices
  #define A(row,col) A[(row) + (col)*m]
  #define B(row,col) B[(row)*n + (col)]
  #define C(row,col) C[(row) + (col)*m]

  // INSERT KERNEL CODE HERE
  __shared__ float Bds[TILE_SZ_RATIO][TILE_SZ_B];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  //Pvalue
  float P[TILE_SZ_B] = {0.0};

  // Tiling iterations
  for(int i= 0 ; i < ceil(k / (TILE_SZ_RATIO * 1.0)); i ++){
        if(i*TILE_SZ_RATIO+ty < k && bx * TILE_SZ_B + tx < n )
            Bds[ty][tx] = B(i*TILE_SZ_RATIO+ty, bx * TILE_SZ_B + tx);
        else
            Bds[ty][tx] = 0;
        __syncthreads();

        int j = 0;
        for(j = 0 ; j < TILE_SZ_RATIO; j ++){
            int p = 0;
            for(p = 0; p < TILE_SZ_B; p ++){
                if(by * TILE_SZ_A + ty * TILE_SZ_B + tx < m && i*TILE_SZ_RATIO+j < k)
                    P[p] += A(by * TILE_SZ_A + ty * TILE_SZ_B + tx, i*TILE_SZ_RATIO+j) * Bds[j][p];
            }
        }
        __syncthreads();
    
  }

  int i = 0;
  for(i = 0; i < TILE_SZ_B; i ++){
      if( by * TILE_SZ_A + ty * TILE_SZ_B + tx< m && bx * TILE_SZ_B + i < n)
        C(by * TILE_SZ_A + ty * TILE_SZ_B + tx, bx * TILE_SZ_B + i) = P[i];
  }



   //origin tiled matrix multiplication
    // __shared__ float TileMatrixA[tileWidth][tileWidth];
    // __shared__ float TileMatrixB[tileWidth][tileWidth];

    // int bx = blockIdx.x;
    // int by = blockIdx.y;
    // int tx = threadIdx.x;
    // int ty = threadIdx.y;

    // int row = by * tileWidth + ty;
    // int col = bx * tileWidth + tx;

    // float Cvalue = 0;

    // for(int i = 0; i < ceil(k / (float)tileWidth) ; i++){
    //     if((row < m) && (i * tileWidth + tx) < k)
    //         TileMatrixA[ty][tx] = A(row, i * tileWidth + tx);
    //     if((i * tileWidth + ty) < k && col < n)
    //         TileMatrixB[ty][tx] = B(i * tileWidth + ty,col);
    //     __syncthreads();
        
    //     for(int kk = 0; kk < tileWidth; kk ++){
    //         Cvalue = Cvalue + TileMatrixA[ty][kk]* TileMatrixB[kk][tx];
    //     }
    //     __syncthreads();
    // }

    // C(row, col) = Cvalue;

}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'T') && (transb != 't')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    //INSERT CODE HERE
    //test2
    // dim3 DimGrid(ceil(m / (TILE_SZ_B * 1.0)), 1, 1);
    // dim3 DimBlock(TILE_SZ_B,1,1);
    // dim3 DimGrid(ceil(n/ double(tileWidth)), ceil(m/double(tileWidth)), 1);
    // dim3 DimBlock(tileWidth, tileWidth, 1);
    dim3 DimBlock(TILE_SZ_B, TILE_SZ_RATIO, 1);
    dim3 DimGrid(ceil(n/(1.0*TILE_SZ_B)), ceil(m/(1.0*TILE_SZ_RATIO)), 1);

    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE
    mysgemm<<<DimGrid,DimBlock>>>(m, n, k, A, B, C);

}

