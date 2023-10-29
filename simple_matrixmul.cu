#include <stdio.h>
// #include <cuda_runtime.h>

const int N = 1024;

__global__ void sgemm(float *a, float *b, float *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;
    if( col < n && row < n) {
        for(int i =0; i<n ;i++ ) {
            sum += a[row*n + i] * b[i*n + col];
        }
        c[row*n + col] = sum;
    }
}

__global__ void tilegemm(float *a, float *b, float *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x, ty = threadIdx.y;

    const int TILE_DIM = 32;
    float sum=0;
    __shared__ float a_tile[TILE_DIM][TILE_DIM];
    __shared__ float b_tile[TILE_DIM][TILE_DIM];
    for(int tile = 0; tile < n/TILE_DIM; tile++) {
        a_tile[ty][tx] = a[row * n + (tile*TILE_DIM) + tx];
        b_tile[ty][tx] = b[(tile*TILE_DIM+ty) * n + col];
        __syncthreads();
        
    #pragma unroll
        for(int i=0; i<TILE_DIM; i++) {
            sum += a_tile[ty][i] * b_tile[i][tx];
        }
        __syncthreads();
    }
    c[row*n + col] = sum;
}

int main() {
    cudaError_t err = cudaSuccess;

    int num_row = N, num_col = N;
    float *h_A, *h_B, *h_C;
    size_t size = num_row*num_col*sizeof(float);

    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    for(int i=0; i<num_row*num_col; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // h_A[0] = 1;    h_A[1] = 2;
    // h_A[2] = 3;    h_A[3] = 4;

    // h_B[0] = 1;    h_B[1] = 2;
    // h_B[2] = 3;    h_B[3] = 4;

    printf("h_A and h_B malloc value.\n");

    
    float *d_A = NULL;
    cudaMalloc((void**)&d_A, size);
    float *d_B = NULL;
    cudaMalloc((void**)&d_B, size);    
    float *d_C = NULL;
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    err = cudaGetLastError();

    if (err != cudaSuccess) {
      printf("Failed to cudaMemcpy (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    printf("d_A and d_B malloc value.\n");

    const int block_size = 32;
    dim3 threads(block_size, block_size);
    dim3 grid((num_row + threads.x - 1) / threads.x, (num_col + threads.y - 1) / threads.y);

    printf("Compiting result using CUDA Kernel...\n");

    sgemm<<<grid, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    tilegemm<<<grid, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    // const int T = 1000;
    // for(int i=0;i < T; i++) {
        // tilegemm<<<grid, threads>>>(d_A, d_B, d_C, N);
    // }

    err = cudaGetLastError();

    if (err != cudaSuccess) {
      printf("Failed to launch kernel (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    printf("Checking result...\n");
    for(int i=0; i<num_row; i++) {
        for(int j=0; j<num_col; j++) {
            float sum=0;
            for(int k=0; k<N; k++) sum += h_A[i*num_col+k]*h_B[k*num_col+j];
            if(fabs(h_C[i*num_col+j]-sum)>1e-2) {
                printf("result error!\n");
                printf("i=%d, j=%d, h_C=%f, sum=%f\n",i,j,h_C[i*num_col+j],sum);
                return 0;
            }
        }
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    printf("result correct!\n");
    
    return 0;
}