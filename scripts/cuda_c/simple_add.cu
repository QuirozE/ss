#include <iostream>

__global__ void cu_add(int a, int b, int *c) {
    *c = a + b;
}

int main(void) {
    int res;
    int* cuda_res;

    cudaMalloc((void**)&cuda_res, sizeof(int));

    cu_add<<<1, 1>>>(10, 7, cuda_res);

    cudaMemcpy(&res, cuda_res, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(cuda_res);

    printf("10 + 7 = %d, using a CUDA kernel!!!\n", res);
    return 0;
}
