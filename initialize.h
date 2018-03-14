#include <iostream>
#include <omp.h>



void use_openmp(void) {
    // openMP
    int n = Eigen::nbThreads( );
    if ( n > 3 ) {
        n -= 2;
    } else if ( n == 3 ) {
        n = 2;
    } else if ( n == 2 ) {
        n = 1;
    } else if ( n == 1 ) {
        std::cout << "このデバイスはシングルスレッドです。OpenMPは使えません。" << std::endl;
        exit(1);
    } else {
        std::cout << "スレッド数が異常です。" << std::endl;
        exit(1);
    }
    omp_set_num_threads(n);
    Eigen::setNbThreads(n);
}


void initialize(void) {
    std::srand((unsigned int) time(0));
    use_openmp();
}
