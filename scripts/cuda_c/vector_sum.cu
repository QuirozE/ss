#include <stdio.h>

#define N 1000000 /* Vector size */

/* Some macros to make CUDA memory manipulation less verbose */
#define cu_malloc_v(v) cudaMalloc((void**)&v, sizeof(int) * N)
#define cu_cpy_v(in, out, mode) cudaMemcpy(in, out, sizeof(int) * N, mode)
#define cu_cpy_v_to_dev(dev, host) cu_cpy_v(\
        dev, host, cudaMemcpyHostToDevice)
#define cu_cpy_v_to_host(host, dev) cu_cpy_v(\
        host, dev, cudaMemcpyDeviceToHost)

#define print_error(err) printf("%s\n", cudaGetErrorString(err))

/* Vector parallel sum */
__global__ void v_sum(int *a, int *b, int *c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i+= offset) {
        c[i] = a[i] + b[i];
    }
}

int main(void) {
    int *a = (int*) malloc(sizeof(int)*N),
        *b = (int*) malloc(sizeof(int)*N),
        *c = (int*) malloc(sizeof(int)*N);

    int *dev_a, *dev_b, *dev_c;

    for(int i = 0; i< N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    cu_malloc_v(dev_a);
    cu_malloc_v(dev_b);
    cu_malloc_v(dev_c);

    cu_cpy_v_to_dev(dev_a, a);
    cu_cpy_v_to_dev(dev_b, b);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    v_sum<<<blocks, threads>>>(dev_a, dev_b, dev_c);

    cu_cpy_v_to_host(c, dev_c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    for(int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    free(a);
    free(b);
    free(c);

    return 0;
}
