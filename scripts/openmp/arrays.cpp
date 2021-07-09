#include <algorithm>
#include <iostream>
#include <omp.h>
#include <vector>

using std::cout;
using std::endl;
using std::generate;
using std::vector;

const size_t N = 1 << 25;
const unsigned int seed = 0;

vector<double> add(vector<double> a, vector<double> b) {
  double t0 = omp_get_wtime();
  size_t n = a.size();
  vector<double> c(n);
  for (int i = 0; i < n; i++)
    c[i] = a[i] + b[i];
  cout << "adding " << n << " elements: " << omp_get_wtime() - t0 << endl;
  return c;
}

vector<double> mp_add(vector<double> a, vector<double> b) {
  double t0 = omp_get_wtime();
  size_t n = a.size();
  vector<double> c(n);
#pragma omp parallel for
  {
    for (int i = 0; i < n; i++)
      c[i] = a[i] + b[i];
  }
  cout << "mp adding " << n << " elements: " << omp_get_wtime() - t0 << endl;
  return c;
}

double sum(vector<double> a, vector<double> b) {
  double t0 = omp_get_wtime();
  size_t n = a.size();
  double c = 0;
  for (int i = 0; i < n; i++)
    c += a[i] + b[i];
  cout << "sum of " << n << " elements: " << omp_get_wtime() - t0 << endl;
  return c;
}

double mp_sum(vector<double> a, vector<double> b) {
  double t0 = omp_get_wtime();
  size_t n = a.size();
  double c = 0;
#pragma omp parallel for reduction(+ : c)
  {
    for (int i = 0; i < n; i++)
      c += a[i] + b[i];
  }
  cout << "mp sum of " << n << " elements: " << omp_get_wtime() - t0 << endl;
  return c;
}

bool find(vector<int> a, int b) {
  int n = a.size();
  bool found = false;
  double t0 = omp_get_wtime();
  for (int i = 0; i < n && !found; i++)
    if (a[i] == b)
      found = true;

  cout << "searching in" << n << " elements: " << omp_get_wtime() - t0 << endl;
  return false;
}

int main() {
  srand(seed);

  vector<double> a(N);
  generate(a.begin(), a.end(), rand);

  vector<double> b(N);
  generate(b.begin(), b.end(), rand);

  vector<int> c;
  generate(c.begin(), c.end(), rand);

  add(a, b);
  mp_add(a, b);
  sum(a, b);
  mp_sum(a, b);
  find(c, rand());

  return 0;
}
