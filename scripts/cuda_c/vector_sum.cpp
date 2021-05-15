#include <iostream>

#define N 900000

void suma(int *a, int *b, int *c) {
  for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}

int main(void) {
  int *a = new int[N], *b = new int[N], *c = new int[N];

  for (int i = 0; i < N; i++) {
    a[i] = -i;
    b[i] = i * i;
  }

  suma(a, b, c);

  for (int i = 0; i < N; i++) {
    printf("%d + %d = %d", a[i], b[i], c[i]);
  }

  delete[] a;
  delete[] b;
  delete[] c;
  return 0;
}
