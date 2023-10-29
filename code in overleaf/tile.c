const int TILE_DIM = 16;
Type_C sum=0;
__shared__ Type_A a_tile[TILE_DIM][TILE_DIM];
__shared__ Type_B b_tile[TILE_DIM][TILE_DIM];
for(int tile = 0; tile < K/TILE_DIM; tile++) {
    a_tile[ty][tx] = A[row * K + (tile*TILE_DIM) + tx];
    b_tile[ty][tx] = B[(tile*TILE_DIM+ty) * N + col];
    __syncthreads();
    
#pragma unroll
    for(int i=0; i<TILE_DIM; i++) {
        sum += __half2float(a_tile[ty][i]) * __half2float(b_tile[i][tx]);
    }
    __syncthreads();
}
C[row*N + col] = sum;