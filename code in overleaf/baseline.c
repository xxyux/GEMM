int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

float sum = 0;
if( col < M && row < N) {
    for(int i =0; i<K; i++ ) {
        sum += __half2float(A[row*N + i]) * __half2float(B[i*N + col]);
    }
    C[row*N + col] = sum;
}