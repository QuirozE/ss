#include <algorithm>
#include <iostream>
#include <vector>
#include <mpi.h>

using std::cout;
using std::vector;
using std::endl;

const int N = 10;

void hello() {
  int n, id;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);

  cout << "(" << id << "/" << n - 1 << ") Hi!" << endl;
}

vector<double> mpi_sum(vector<double> a, vector<double> b) {
  int n = a.size();
  vector<double> c(n);

  int n_t, id;
  MPI_Comm_size(MPI_COMM_WORLD, &n_t);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);

  int stride = n/n_t, init = id * stride;

  for(int i = init; i < init + stride; i++)
    c[i] = a[i] + b[i];

  if(id == n_t - 1)
    for(int i = init + stride; i < n; i++)
      c[i] = a[i] + b[i];

  return c;
}

int main() {

  vector<double> a(N);
  generate(a.begin(), a.end(), rand);

  vector<double> b(N);
  generate(b.begin(), b.end(), rand);

  MPI_Init(NULL, NULL);

  hello();
  mpi_sum(a, b);

  MPI_Finalize();
  return 0;
}
