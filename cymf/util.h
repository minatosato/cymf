
#ifdef _OPENMP
#include <omp.h>
#endif

namespace cymf {
#ifdef _OPENMP
inline int threadid() { return omp_get_thread_num(); }
#else
inline int threadid() { return 0; }
#endif
} 
