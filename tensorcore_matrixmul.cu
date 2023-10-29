#include <stdio.h>
#include <cuda.h>
#include <mma.h>

using namespace nvcuda;

const int N = 1024;

__global__ void simple_wmma(const half *__restrict__ A,
    const half *__restrict__ B, float *__restrict__ C, size_t M, size_t N, size_t K) {
    int warp_in_a_row = K / 16;
    int warp_id_global = (blockIdx.x*blockDim.x + threadIdx.x) / 32;

    int warp_id_x = warp_id_global / warp_in_a_row;
    int warp_id_y = warp_id_global % warp_in_a_row;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int rowId = 16 * warp_id_x;
    int colId = 16 * warp_id_y;

    for(int kk=0; kk<K; kk+=16) {
        wmma::load_matrix_sync(a_frag, A + rowId * K + kk, K);
        wmma::load_matrix_sync(b_frag, B + kk * N + colId, K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(C + rowId * N + colId, c_frag, N, wmma::mem_row_major);
}

int main() {
    cudaError_t err = cudaSuccess;

    int num_row = N, num_col = N;
    half *h_A, *h_B;
    float *h_C;
    size_t size_AB = num_row*num_col*sizeof(half);
    size_t size_C = num_row*num_col*sizeof(float);

    h_A = (half *)malloc(size_AB);
    h_B = (half *)malloc(size_AB);
    h_C = (float *)malloc(size_C);

    for(int i=0; i<num_row*num_col; i++) {
        h_A[i] = (half)(rand() / (float)RAND_MAX);
        h_B[i] = (half)(rand() / (float)RAND_MAX);
    }

    // h_A[0] = 1;    h_A[1] = 2;
    // h_A[2] = 3;    h_A[3] = 4;

    // h_B[0] = 1;    h_B[1] = 2;
    // h_B[2] = 3;    h_B[3] = 4;

    printf("h_A and h_B malloc value.\n");

    
    half *d_A = NULL;
    cudaMalloc((void**)&d_A, size_AB);
    half *d_B = NULL;
    cudaMalloc((void**)&d_B, size_AB);    
    float *d_C = NULL;
    cudaMalloc((void**)&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_AB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_AB, cudaMemcpyHostToDevice);

    err = cudaGetLastError();

    if (err != cudaSuccess) {
      printf("Failed to cudaMemcpy (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    printf("d_A and d_B malloc value.\n");

    printf("Compiting result using CUDA Kernel...\n");

    // compute C[16*16] per warp
    // 4 warp per block
    // (N*N) / (16*16) / 4 block per grid 
    dim3 block_size(32 * 4, 1, 1);
    dim3 grid_size((N * N) /(16*16) / 4, 1, 1);

    simple_wmma<<<grid_size, block_size>>>(d_A, d_B, d_C, N, N, N);

    err = cudaGetLastError();

    if (err != cudaSuccess) {
      printf("Failed to launch kernel (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    printf("Checking result...\n");
    for(int i=0; i<num_row; i++) {
        for(int j=0; j<num_col; j++) {
            float sum=0;
            for(int k=0; k<N; k++) sum += __half2float(h_A[i*num_col+k])*__half2float(h_B[k*num_col+j]);
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