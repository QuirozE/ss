#include <stdio.h>

#define N 50 /* Vector size */

/* Some macros to make CUDA memory manipulation less verbose */
#define cu_malloc_v(v) cudaMalloc((void**)&v, sizeof(int) * N)
#define cu_cpy_v(in, out, mode) cudaMemcpy(in, out, sizeof(int) * N, mode)
#define cu_cpy_v_to_dev(dev, host) {\
    cu_cpy_v(dev, host, cudaMemcpyHostToDevice);\
}
#define cu_cpy_v_to_host(host, dev) {\
    cu_cpy_v(host, dev, cudaMemcpyDeviceToHost);\
}

/* Vector parallel sum */
__global__ void v_sum(int *a, int *b, int *c) {
    int tid = blockIdx.x;

    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main(void) {
    int a[N], b[N], c[N];
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

    v_sum<<<N, 1>>>(dev_a, dev_b, dev_c);

    cu_cpy_v_to_host(c, dev_c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    for(int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    return 0;
}
