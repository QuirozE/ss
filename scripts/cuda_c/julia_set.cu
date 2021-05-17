#include <fstream>
#include <iostream>

using std::cout;
using std::endl;
using std::ofstream;
using std::string;

struct CuCmplx {
  double r, i;

public:
  __device__ CuCmplx(double r, double i) : r(r), i(i) {}
  __device__ CuCmplx operator+(const CuCmplx &other) {
    return CuCmplx(r + other.r, i + other.i);
  }
  __device__ CuCmplx operator*(const CuCmplx &other) {
    return CuCmplx(r * other.r - i * other.i, r * other.i + i * other.r);
  }
  __device__ double abs(void) { return r * r + i * i; }
};

const int X = 1000;
const double scale_const = 1.5;
const int iters = 200;
const int threshold = 2000;

__device__ double scale(int x) { return scale_const * (x - X / 2) / X; }

__device__ int is_in_julia(const CuCmplx &p) {
  CuCmplx C(-0.8, 0.156);
  CuCmplx nth = p;
  for (int i = 0; i < iters; i++) {
    nth = nth * nth + C;
    if (nth.abs() > threshold) {
      return 0;
    }
  }
  return 1;
}

__global__ void kernel(int *space) {
  int tid_x = blockIdx.x, tid_y = blockIdx.y,
      offset = tid_x + tid_y * gridDim.x;
  CuCmplx point(scale(tid_x), scale(tid_y));
  space[offset] = is_in_julia(point);
}

double scale_host(int x) { return scale_const * (x - X / 2) / X; }

void save_space(const int (&space)[X][X], const string &filename) {
  ofstream file(filename);
  file << "x,y" << endl;
  for (int i = 0; i < X; i++) {
    for (int j = 0; j < X; j++) {
      if (space[i][j]) {
        double x = scale_host(i), y = scale_host(j);
        file << x << "," << y << endl;
      }
    }
  }
  file.close();
}

int main(void) {
  int *dev_space, space[X][X];
  cudaMalloc((void **)&dev_space, sizeof(int) * X * X);

  dim3 grid(X, X);

  kernel<<<grid, 1>>>(dev_space);

  cudaMemcpy(space, dev_space, sizeof(int) * X * X, cudaMemcpyDeviceToHost);

  cudaFree(dev_space);

  save_space(space, "julia_set_X=10000_s=1.5_cu.csv");
  return 0;
}
