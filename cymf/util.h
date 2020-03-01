//
// Copyright (c) 2020 Minato Sato
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//

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
