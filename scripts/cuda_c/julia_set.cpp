#include <fstream>
#include <iostream>

using std::cout;
using std::endl;
using std::ofstream;
using std::string;

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

const int X = 1000;
const int iters = 200;
const int threshold = 2000;
const Cmplx C(-0.8, 0.156);
const double scale_const = 1.5;

double scale(int x) { return scale_const * (x - X / 2) / X; }

int is_in_julia(const Cmplx &c) {
  Cmplx nth = c;
  for (int i = 0; i < iters; i++) {
    nth = nth * nth + C;
    if (nth.abs() > threshold) {
      return 0;
    }
  }
  return 1;
}

void kernel(int (&space)[X][X]) {
  for (int i = 0; i < X; i++) {
    for (int j = 0; j < X; j++) {
      Cmplx point(scale(i), scale(j));
      space[i][j] = is_in_julia(point);
    }
  }
}

void print_space(const int (&space)[X][X]) {
  for (auto &row : space) {
    for (auto point : row) {
      if (point) {
        cout << "*";
      } else {
        cout << " ";
      }
    }
    cout << endl;
  }
}

void save_space(const int (&space)[X][X], const string &filename) {
  ofstream file(filename);
  file << "x,y" << endl;
  for (int i = 0; i < X; i++) {
    for (int j = 0; j < X; j++) {
      if (space[i][j]) {
        double x = scale(i), y = scale(j);
        file << x << "," << y << endl;
      }
    }
  }
  file.close();
}

int main(void) {
  int space[X][X];
  kernel(space);
  save_space(space, "julia_set_X=1000_s=1.5_cpp.csv");
  return 0;
}
