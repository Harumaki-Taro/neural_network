#ifndef INCLUDE_openmp_h_
#define INCLUDE_openmp_h_

#include <omp.h>

void use_openmp(void) {
    // openMP
    int n = Eigen::nbThreads( );
    omp_set_num_threads(n-2);
    Eigen::setNbThreads(n-2);
}


#endif // INCLUDE_openmp_h_
