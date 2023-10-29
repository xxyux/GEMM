all: bin/gemm_baseline

bin/gemm_baseline: utils.cuh gemm.cuh gemm_kernel.cuh gemm_imp_baseline.cu 
	mkdir -p bin
	nvcc -arch=sm_89 -O3 gemm_imp_baseline.cu -o bin/gemm_shared_tensor_v1_3

gemm: utils.cuh gemm.cuh gemm_kernel.cuh gemm_imp_baseline.cu 
	mkdir -p bin
	nvcc -arch=compute_89 -code=sm_89 -lcublas --ptxas-options=-v -O3 gemm_imp_baseline.cu -o bin/gemm



# nvcc -arch=sm_89 -O3 gemm_imp_baseline.cu -o bin/gemm_simple_tensor
# nvcc -arch=sm_89 -O3 gemm_imp_baseline.cu -o bin/gemm_shared_tensor_v1
# nvcc -arch=sm_89 -O3 gemm_imp_baseline.cu -o bin/gemm_shared_tensor_v1_1
# nvcc -arch=sm_89 -O3 gemm_imp_baseline.cu -o bin/gemm_shared_tensor_v1_2
# add your versions as targets in here. 
# you can also try different compiler options here. 

clean:
	rm -rf bin

.PHONY : clean

