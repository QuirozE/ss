/**
 * Sequential dot product of two arrays. First, the product of paired values is
 * computed, and then all the results are added togheter. For this, a convoluted
 * reduction that runs in $O(N)$ is used. A straight forward parallelization
 * yields a $O(\log{N})$ reduction.
 */

#include <array>
#include <iostream>

using std::cout;
using std::endl;

const int N = 3;

void redux(int *arr, int fold(int, int)) {
  int i = (N - 1) / 2;
  int lim = N;
  while (i >= 0) {
    for (int j = 0; j <= i; j++) {
      if (i + 1 + j < lim) {
        arr[j] = fold(arr[j], arr[i + 1 + j]);
      }
    }
    if (!i)
      break;

    lim = i + 1;
    i /= 2;
  }
}

void dot(int *a, int *b, int *c) {
  for (int i = 0; i < N; i++) {
    c[i] = a[i] * b[i];
  }
}

int sum(int a, int b) { return a + b; }

long first_n_cubes(int n) { return n * n * (n + 1) * (n + 1) / 4; }

int main(void) {
  int *a = new int[N + 1], *b = new int[N + 1], *c = new int[N + 1];

  for (int i = 1; i <= N; i++) {
    a[i - 1] = i * i;
    b[i - 1] = i;
  }

  dot(a, b, c);

  delete[] a;
  delete[] b;

  redux(c, sum);

  cout << "Real value: " << c[0] << endl;
  cout << "Expected value: " << first_n_cubes(N) << endl;

  delete[] c;
  return 0;
}
