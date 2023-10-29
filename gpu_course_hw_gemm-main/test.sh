#! /bin/bash
echo "Start running..."

version=(baseline tilegemm v0 v1 v1_1 v3 v3_1 v3_2)
size=(512 1024 1536 2048 3072 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384)

for ((i=0; i<8; i++))
do
    for ((j=0; j<18; j++))
    do
        echo "version is ${version[$i]}, size is ${size[$j]}"
        CUDA_VISIBLE_DEVICES=1 ./bin/gemm ${size[$j]} ${version[$i]}
    done
done