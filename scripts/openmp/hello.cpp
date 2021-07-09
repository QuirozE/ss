#include <iostream>
#include <omp.h>

using std::cout;
using std::endl;

int main() {
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
  return 0;
}
