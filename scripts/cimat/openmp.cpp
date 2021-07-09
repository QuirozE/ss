#include <algorithm>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

using std::cout;
using std::endl;
using std::generate;
using std::string;
using std::to_string;
using std::vector;

const size_t N = 1 << 25;
const unsigned int seed = 0;

#define NE(s) s + to_string(N) + " elements"
#define TIME(mess, f, ...)                                                     \
  do {                                                                         \
    double t0 = omp_get_wtime();                                               \
    f(__VA_ARGS__);                                                            \
    cout << mess << " took : " << omp_get_wtime() - t0 << endl;                \
  } while (0)

void hello() {
  int n_t, id;
#pragma omp parallel private(n_t, id)
  {
    n_t = omp_get_num_threads();
    id = omp_get_thread_num();
    if (!id) {
      cout << n_t << " cores available" << endl;
    }
    cout << "hi from core " << id << endl;
  }
}

vector<double> add(vector<double> a, vector<double> b) {
  size_t n = a.size();
  vector<double> c(n);
  for (int i = 0; i < n; i++)
    c[i] = a[i] + b[i];
  return c;
}

vector<double> mp_add(vector<double> a, vector<double> b) {
  size_t n = a.size();
  vector<double> c(n);
#pragma omp parallel for
  {
    for (int i = 0; i < n; i++)
      c[i] = a[i] + b[i];
  }
  return c;
}

double sum(vector<double> a, vector<double> b) {
  size_t n = a.size();
  double c = 0;
  for (int i = 0; i < n; i++)
    c += a[i] + b[i];
  return c;
}

double mp_sum(vector<double> a, vector<double> b) {
  size_t n = a.size();
  double c = 0;
#pragma omp parallel for reduction(+ : c)
  {
    for (int i = 0; i < n; i++)
      c += a[i] + b[i];
  }
  return c;
}

bool find(vector<int> a, int b) {
  int n = a.size();
  bool found = false;
  for (int i = 0; i < n && !found; i++)
    if (a[i] == b)
      found = true;
  return false;
}

bool mp_find(vector<int> a, int b) {
  size_t n = a.size();
  bool found = false;
  int n_t, id, i, i0, stride;
#pragma omp parallel private(i, id, n_t, i0, stride) shared(found)
  {
    n_t = omp_get_num_threads();
    id = omp_get_thread_num();
    stride = n / n_t;
    i0 = stride * id;
    for (int i = 0; i < stride && !found; i++)
      if (a[i0 + i] == b)
        found = true;
  }
  return found;
}

int main() {
  cout << "OpenMP basic examples" << endl;

  srand(seed);

  TIME("saying 'hi'", hello);

  vector<double> a(N);
  generate(a.begin(), a.end(), rand);

  vector<double> b(N);
  generate(b.begin(), b.end(), rand);

  vector<int> c(N);
  generate(c.begin(), c.end(), rand);

  TIME(NE("adding "), add, a, b);
  TIME(NE("mp adding "), mp_add, a, b);
  TIME(NE("sum of "), mp_sum, a, b);
  TIME(NE("mp sum of "), mp_sum, a, b);
  TIME(NE("searching "), find, c, rand());
  TIME(NE("mp searching "), mp_find, c, rand());

  return 0;
}
