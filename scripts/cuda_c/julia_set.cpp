#include <iostream>

struct Cmplx {
  double r;
  double i;

public:
  Cmplx(double r, double i) : r(r), i(i) {}
  Cmplx operator+(const Cmplx &other) {
    return Cmplx(r + other.r, i + other.i);
  }
  Cmplx operator*(const Cmplx &other) {
    return Cmplx(r * other.r - i * other.i, r * other.i + i * other.r);
  }
  double abs(void) { return r * r + i * i; }
};

const int X = 100;
const int iters = 200;
const int threshold = 2000;
const Cmplx C(-0.8, 0.156);
const double scale_const = 1.5;

double scale(int x) { return scale_const * (x - X) / X; }

int is_in_julia(Cmplx c) {
  Cmplx nth = c;
  for (int i = 0; i < iters; i++) {
    nth = nth * nth + C;
    if (nth.abs() > threshold) {
      return 0;
    }
  }
  return 1;
}

int main(void) { return 0; }
