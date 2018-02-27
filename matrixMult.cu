/*
EXECUTION INSTRUCTIONS:-
1. Compile the solution with command – “ nvcc programName.cu -lcublas outputfile”
2. Run the out put file.

To change the matrix size and kernel configuration, change the defined values at the beginning of the code
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda.h"
#include "cublas.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 1024  //Matrix Size
#define TILE_WIDTH 16 //Tile Size
#define TN2D 16 //Threads per block for 2D
#define TN1D 256 //Threads per block for 1D

//Naive 1D approach of matrix multiplication
__global__ void matrixMultNaive1D(float *a, float *b, float *c) {
	int width = N;
	float sum = 0.0;
	//Getting the id of the corresponding matrix element
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int x = floorf(i / width); //Calculating the row number
	int y = i % width; //Calculating the column number

	//Getting the value of ith element of c by multiplying corresponding values of a and b
	for (int k = 0; k < width; k++) {
		sum += a[x * width + k] * b[k * width + y];
	}
	c[i] = sum;
}


//Naive 2D approach of matrix multiplication
__global__ void matrixMultNaive2D(float *a, float *b, float *c) {

	int width = N;
	float sum = 0.0;
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	//if (col < width && row < width) {
	for (int k = 0; k < width; k++)
		sum += a[row * width + k] * b[k * width + col];
	c[row * width + col] = sum;
	//}
}


//Tiled + Shared Memory approach of matrix multiplication
__global__ void matrixMultTiled(float *a, float *b, float *c) {

	//Initilaize matrices in shared memory 
	__shared__ float As[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

	int width = N;
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	float sum = 0.0;

	// Loop over the As and Bs tiles required to compute the Cd element
	for (int m = 0; m < width / TILE_WIDTH; ++m) {
		// Collaborative loading of a and b tiles into shared memory
		As[ty][tx] = a[Row*width + (m*TILE_WIDTH + tx)];
		Bs[ty][tx] = b[Col + (m*TILE_WIDTH + ty)*width];
		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k)
			sum += As[ty][k] * Bs[k][tx];
		__syncthreads();
	}
	c[Row*width + Col] = sum;
}


//Tiling + Shared memory + Loop unrolling approach of matrix multiplication
__global__ void matrixMultUnrolled(float *a, float *b, float *c) {

	//Initilaize matrices in shared memory 
	__shared__ float As[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

	int width = N;
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	float sum = 0.0;

	// Loop over the As and Bs tiles required to compute the C element
	for (int m = 0; m < width / TILE_WIDTH; ++m) {
		// Collaborative loading of a and b tiles into shared memory
		As[ty][tx] = a[Row*width + (m*TILE_WIDTH + tx)];
		Bs[ty][tx] = b[Col + (m*TILE_WIDTH + ty)*width];
		__syncthreads();
		//Unrolling th loop 4 times
		for (int k = 0; k < TILE_WIDTH; k += 4)
			sum += As[ty][k + 0] * Bs[k + 0][tx] + As[ty][k + 1] * Bs[k + 1][tx] + As[ty][k + 2] * Bs[k + 2][tx] + As[ty][k + 3] * Bs[k + 3][tx];
		__syncthreads();
	}
	c[Row*width + Col] = sum;
}

//Getting random float values for input matrix
void random_float(float* a, int n)
{
	int i;
	for (i = 0; i < n; i++)
		a[i] = float(rand() % 10);
}

int main(void) {
	float *a, *b, *c1d, *c2d, *ct, *clu, *ccuB;// host copies of a, b, c
	float *d_a, *d_b, *d_c1d, *d_c2d, *d_ct, *d_clu;// device copies of a, b, c
	int size = (N * N) * sizeof(int);

	cudaEvent_t naive1dCompStart, naive1dCompStop, naive2dCompStart, naive2dCompStop, tiledCompStart, tiledCompStop, cubCompStart, cubCompStop, luCompStart, luCompStop, memStart, memStop;
	cudaEventCreate(&naive1dCompStart);
	cudaEventCreate(&naive1dCompStop);
	cudaEventCreate(&naive2dCompStart);
	cudaEventCreate(&naive2dCompStop);
	cudaEventCreate(&tiledCompStart);
	cudaEventCreate(&tiledCompStop);
	cudaEventCreate(&luCompStart);
	cudaEventCreate(&luCompStop);
	cudaEventCreate(&cubCompStart);
	cudaEventCreate(&cubCompStop);
	cudaEventCreate(&memStart);
	cudaEventCreate(&memStop);

	//Allocspace for device copies of a, b, c
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c1d, size);
	cudaMalloc((void**)&d_c2d, size);
	cudaMalloc((void**)&d_ct, size);
	cudaMalloc((void**)&d_clu, size);

	//Allocspace for host copies of a, b, c and setup input values
	a = (float*)malloc(size); random_float(a, N*N);
	b = (float*)malloc(size); random_float(b, N*N);
	c1d = (float*)malloc(size);
	c2d = (float*)malloc(size);
	ct = (float*)malloc(size);
	clu = (float*)malloc(size);
	ccuB = (float*)malloc(size);

	// Copy inputs to device
	cudaEventRecord(memStart, 0);
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	cudaEventRecord(memStop, 0);
	cudaEventSynchronize(memStop);
	float elapsedTimeMem1 = 0.0f;
	cudaEventElapsedTime(&elapsedTimeMem1, memStart, memStop);

	//Define dimension of grid and block
	dim3 gridDim1d((N*N) / TN1D, 1);
	dim3 blockDim1d(TN1D, 1);
	dim3 gridDim2d(N / TN2D, N / TN2D);
	dim3 blockDim2d(TN2D, TN2D);
	dim3 gridDimT(N / TILE_WIDTH, N / TILE_WIDTH);
	dim3 blockDimT(TILE_WIDTH, TILE_WIDTH);
	dim3 gridDimlu(N / TILE_WIDTH, N / TILE_WIDTH);
	dim3 blockDimlu(TILE_WIDTH, TILE_WIDTH);

	//---------------------KERNEL LAUNCHES--------------------------

	cudaEventRecord(naive1dCompStart, 0);

	//Naive 1D
	matrixMultNaive1D << <gridDim1d, blockDim1d >> >(d_a, d_b, d_c1d);

	cudaEventRecord(naive1dCompStop, 0);
	cudaEventSynchronize(naive1dCompStop);
	float elapsedTime1d = 0.0f;
	cudaEventElapsedTime(&elapsedTime1d, naive1dCompStart, naive1dCompStop);

	cudaEventRecord(memStart, 0);

	cudaMemcpy(c1d, d_c1d, size, cudaMemcpyDeviceToHost);

	cudaEventRecord(memStop, 0);
	cudaEventSynchronize(memStop);
	float elapsedTimeMem2 = 0.0f;
	cudaEventElapsedTime(&elapsedTimeMem2, memStart, memStop);

	//----------------------------------------------------------
	cudaEventRecord(naive2dCompStart, 0);

	//Naive 2D
	matrixMultNaive2D << <gridDim2d, blockDim2d >> >(d_a, d_b, d_c2d);

	cudaEventRecord(naive2dCompStop, 0);
	cudaEventSynchronize(naive2dCompStop);
	float elapsedTime2d = 0.0f;
	cudaEventElapsedTime(&elapsedTime2d, naive2dCompStart, naive2dCompStop);

	cudaMemcpy(c2d, d_c2d, size, cudaMemcpyDeviceToHost);

	//----------------------------------------------------------
	cudaEventRecord(tiledCompStart, 0);

	//Tiled + Shared
	matrixMultTiled << <gridDimT, blockDimT >> >(d_a, d_b, d_ct);

	cudaEventRecord(tiledCompStop, 0);
	cudaEventSynchronize(tiledCompStop);
	float elapsedTimeT = 0.0f;
	cudaEventElapsedTime(&elapsedTimeT, tiledCompStart, tiledCompStop);

	cudaMemcpy(ct, d_ct, size, cudaMemcpyDeviceToHost);

	//----------------------------------------------------------
	cudaEventRecord(luCompStart, 0);

	//Tiled + Shared + Loop unrolling
	matrixMultUnrolled << <gridDimlu, blockDimlu >> >(d_a, d_b, d_clu);

	cudaEventRecord(luCompStop, 0);
	cudaEventSynchronize(luCompStop);
	float elapsedTimelu = 0.0f;
	cudaEventElapsedTime(&elapsedTimelu, luCompStart, luCompStop);

	cudaMemcpy(clu, d_clu, size, cudaMemcpyDeviceToHost);


	//---------------------------CUBLAS START---------------------------------

	float *cu_a, *cu_b, *cu_c;
	cudaMalloc((void**)&cu_a, size);
	cudaMalloc((void**)&cu_b, size);
	cudaMalloc((void**)&cu_c, size);


	cublasSetMatrix(N, N, sizeof(*a), a, N, cu_a, N);
	cublasSetMatrix(N, N, sizeof(*b), b, N, cu_b, N);
	cublasSetMatrix(N, N, sizeof(*ccuB), ccuB, N, cu_c, N);

	cudaEventRecord(cubCompStart, 0);

	cublasSgemm('n', 'n', N, N, N, 1.0f, cu_b, N, cu_a, N, 0.0f, cu_c, N);

	cudaEventRecord(cubCompStop, 0);
	cudaEventSynchronize(cubCompStop);
	float elapsedTimecub = 0.0f;
	cudaEventElapsedTime(&elapsedTimecub, cubCompStart, cubCompStop);

	cublasGetMatrix(N, N, sizeof(*ccuB), cu_c, N, ccuB, N);

	//Verifying correctness of each approach with respect to cuBLAS
	float difference1 = 0, difference2 = 0, difference3 = 0, difference4 = 0;
	for (int i = 0; i<N*N; i++) {
		//printf ( "c1d[%d]=%f cublas[%d]=%f\n", i, c1d[i], i, ccuB[i]);
		if (ccuB[i] != c1d[i]) {
			difference1 += ccuB[i] - c1d[i];
		}
		if (ccuB[i] != c2d[i]) {
			difference2 += ccuB[i] - c2d[i];
		}
		if (ccuB[i] != ct[i]) {
			difference3 += ccuB[i] - ct[i];
		}
		if (ccuB[i] != clu[i]) {
			difference4 += ccuB[i] - clu[i];
		}
	}

	//----------------------CUBLAS END----------------------------------------------

	//Checking for device errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	printf("Matrix Size: %d\nThreads per Block (1D): %d\nTile Size: %d\n\n", N, TN1D, TILE_WIDTH);
	
	printf("Residual for Naive1D: %f\n", difference1);
	printf("Residual for Naive2D: %f\n", difference2);
	printf("Residual for Tiled: %f\n", difference3);
	printf("Residual for Loop Unrolling: %f\n\n", difference4);

	printf("Execution Time (Naive 1D) Computation : %f ms\n", elapsedTime1d * 1000);
	printf("Execution Time (Naive 1D) All: %f ms\n\n", (elapsedTime1d+elapsedTimeMem1+elapsedTimeMem2) * 1000);
	printf("Execution Time (Naive 2D) Computation: %f ms\n", elapsedTime2d * 1000);
	printf("Execution Time (Naive 2D) All: %f ms\n\n", (elapsedTime2d + elapsedTimeMem1 + elapsedTimeMem2) * 1000);
	printf("Execution Time (Tiled) Computation: %f ms\n", elapsedTimeT * 1000);
	printf("Execution Time (Tiled) All: %f ms\n\n", (elapsedTimeT + elapsedTimeMem1 + elapsedTimeMem2) * 1000);
	printf("Execution Time (Loop Unrolled) Computation: %f ms\n", elapsedTimelu * 1000);
	printf("Execution Time (Loop Unrolled) All: %f ms\n\n", (elapsedTimelu + elapsedTimeMem1 + elapsedTimeMem2) * 1000);
	printf("Execution Time (cuBLAS) Computation: %f ms\n", elapsedTimecub * 1000);
	printf("Execution Time (cuBLAS) All: %f ms\n\n", (elapsedTimecub + elapsedTimeMem1 + elapsedTimeMem2) * 1000);

	//Cleanup
	free(a); free(b); free(c1d); free(c2d); free(ct); free(clu);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c1d); cudaFree(d_c2d); cudaFree(d_ct); cudaFree(d_clu);

	return 0;
}

