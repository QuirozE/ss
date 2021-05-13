#include <iostream>

__global__ cu_add(int a, int b, int *c) {
    *c = a + b;
}

int main(void) {
    int res;
    int* cuda_res;

    cudaMalloc((void**)&cuda_res, sizeof(int));

    cu_add(10, 7, cuda_res);

    cudaMemcpy(&res, cuda_res, cudaMemxpyDeviceToHost);

    cudaFree(cuda_res);

    printf("10 + 7 = %d, using CUDA!!!", res);
    return 0;
}
