#include <string.h>
#include <omp.h>

template<class T> inline void
interpadjoint_helper(
    T       *m,
    T       *d,
    T      **h,
    size_t   nh,
    size_t   nm,
    size_t   i,
    size_t   lo,
    size_t   hi)
{
    size_t j = i*nh;
    size_t im3 = i-3;
    for (size_t k = 0; k < nh; k++) {
        size_t jk = j+k;
        if (jk >= nm) {
            continue;
        }
        for (size_t l = lo; l <= hi; l++) {
            m[jk] += h[k][l] * d[im3+l];
        }
    }
}

template<class T> void
interpadjoint_1d(
    T     **h,
    T      *m,
    T      *d,
    size_t  nh,
    size_t  nm,
    size_t  nd)
{
    memset(m, '\0', nm*sizeof(T));
    interpadjoint_helper(m,d,h,nh,nm,0,3,7); // i=0
    interpadjoint_helper(m,d,h,nh,nm,1,2,7); // i=1
    interpadjoint_helper(m,d,h,nh,nm,2,1,7); // i=2
    for (size_t i = 3; i < nd-4; i++) {
        interpadjoint_helper(m,d,h,nh,nm,i,0,7);
    }
    interpadjoint_helper(m,d,h,nh,nm,nd-4,0,6); // i=nd-4
    interpadjoint_helper(m,d,h,nh,nm,nd-3,0,5); // i=nd-3
    interpadjoint_helper(m,d,h,nh,nm,nd-2,0,4); // i=nd-2
}

template<class T> void
interpadjoint_nd(
    T      **h,
    T       *m,
    T       *d,
    size_t   nh,
    size_t   nm,
    size_t   nd,
    size_t   nslow,
    size_t   nthreads)
{
#pragma omp parallel for num_threads(nthreads) default(shared)
    for (size_t i = 0; i < nslow; i++) {
        interpadjoint_1d(h, m+i*nm, d+i*nd, nh, nm, nd);
    }
}

template<class T> inline void
interpforward_helper(
    T       *d,
    T       *m,
    T      **h,
    size_t   nh,
    size_t   nm,
    size_t   i,
    size_t   lo,
    size_t   hi)
{
    size_t j = i*nh;
    size_t im3 = i-3;
    for (size_t k = 0; k < nh; k++) {
        size_t jk = j+k;
        if (jk >= nm) {
            continue;
        }
        for (size_t l = lo; l <= hi; l++) {
            d[im3+l] += h[k][l]*m[jk];
        }
    }
}

template<class T> void
interpforward_1d(
    T      **h,
    T       *d,
    T       *m,
    size_t   nh,
    size_t   nd,
    size_t   nm)
{
    memset(d, '\0', nd*sizeof(T));
    interpforward_helper(d,m,h,nh,nm,0,3,7); // i=0
    interpforward_helper(d,m,h,nh,nm,1,2,7); // i=1
    interpforward_helper(d,m,h,nh,nm,2,1,7); // i=2
    for (size_t i = 3; i < nd-4; i++) {
        interpforward_helper(d,m,h,nh,nm,i,0,7);
    }
    interpforward_helper(d,m,h,nh,nm,nd-4,0,6); // i=nd-4
    interpforward_helper(d,m,h,nh,nm,nd-3,0,5); // i=nd-3
    interpforward_helper(d,m,h,nh,nm,nd-2,0,4); // i=nd-2
}

template<class T> void
interpforward_nd(
    T      **h,
    T       *d,
    T       *m,
    size_t   nh,
    size_t   nd,
    size_t   nm,
    size_t   nslow,
    size_t   nthreads)
{
    memset(d, '\0', nd*nslow*sizeof(T));
#pragma omp parallel for num_threads(nthreads) default(shared)
    for (size_t i = 0; i < nslow; i++) {
        interpforward_1d(h, d+i*nd, m+i*nm, nh, nd, nm);
    }
}

extern "C"
{

struct SourcePoint32 {
    long  iu;
    long  ir;
    float c;
};

struct SourcePoint64 {
    long iu;
    long ir;
    double c;
};

void
injectdata_float(
    float          *field,
    float          *data,
    size_t          it,
    size_t          nt,
    SourcePoint32 **partitions,
    size_t          npartitions,
    long           *npoints_per_partition,
    size_t          nthreads)
{
#pragma omp parallel for num_threads(nthreads)
    for (size_t ipartition = 0; ipartition < npartitions; ipartition++) {
        for (size_t ipoint = 0; ipoint < npoints_per_partition[ipartition]; ipoint++) {
            struct SourcePoint32 p = partitions[ipartition][ipoint];
            field[p.iu-1] += p.c*data[(p.ir-1)*nt+it-1];
        }
    }
}

void
injectdata_double(
    double         *field,
    double         *data,
    size_t          it,
    size_t          nt,
    SourcePoint64 **partitions,
    size_t          npartitions,
    long           *npoints_per_partition,
    size_t          nthreads)
{
#pragma omp parallel for num_threads(nthreads)
    for (size_t ipartition = 0; ipartition < npartitions; ipartition++) {
        for (size_t ipoint = 0; ipoint < npoints_per_partition[ipartition]; ipoint++) {
            struct SourcePoint64 p = partitions[ipartition][ipoint];
            field[p.iu-1] += p.c*data[(p.ir-1)*nt+it-1];
        }
    }
}

void
extractdata_float(
    float          *data,
    float          *field,
    SourcePoint32 **partitions,
    size_t          npartitions,
    long           *npoints_per_partition,
    size_t          it,
    size_t          nt,
    size_t          nthreads)
{
#pragma omp parallel for num_threads(nthreads)
    for (size_t ipartition = 0; ipartition < npartitions; ipartition++) {
        struct SourcePoint32 *partition = partitions[ipartition];
#pragma omp simd
        for (size_t ipoint = 0; ipoint < npoints_per_partition[ipartition]; ipoint++) {
            struct SourcePoint32 p = partition[ipoint];
            data[(p.ir-1)*nt+it-1] += p.c*field[p.iu-1];
        }
    }
}

void
extractdata_double(
    double         *data,
    double         *field,
    SourcePoint64 **partitions,
    size_t          npartitions,
    long           *npoints_per_partition,
    size_t          it,
    size_t          nt,
    size_t          nthreads)
{
#pragma omp parallel for num_threads(nthreads)
    for (size_t ipartition = 0; ipartition < npartitions; ipartition++) {
        struct SourcePoint64 *partition = partitions[ipartition];
#pragma omp simd
        for (size_t ipoint = 0; ipoint < npoints_per_partition[ipartition]; ipoint++) {
            struct SourcePoint64 p = partition[ipoint];
            data[(p.ir-1)*nt+it-1] += p.c*field[p.iu-1];
        }
    }
}

void
interpadjoint_1d_float(
    float **h,
    float  *m,
    float  *d,
    size_t  nh,
    size_t  nm,
    size_t  nd)
{
    interpadjoint_1d(h, m, d, nh, nm, nd);
}

void
interpadjoint_1d_double(
    double **h,
    double  *m,
    double  *d,
    size_t   nh,
    size_t   nm,
    size_t   nd)
{
    interpadjoint_1d(h, m, d, nh, nm, nd);
}

void
interpforward_1d_float(
    float **h,
    float  *d,
    float  *m,
    size_t  nh,
    size_t  nd,
    size_t  nm)
{
    interpforward_1d(h, d, m, nh, nd, nm);
}

void
interpforward_1d_double(
    double **h,
    double  *d,
    double  *m,
    size_t   nh,
    size_t   nd,
    size_t   nm)
{
    interpforward_1d(h, d, m, nh, nd, nm);
}

void
interpadjoint_nd_float(
    float  **h,
    float   *m,
    float   *d,
    size_t   nh,
    size_t   nm,
    size_t   nd,
    size_t   nslow,
    size_t   nthreads)
{
    interpadjoint_nd(h, m, d, nh, nm, nd, nslow, nthreads);
}

void
interpadjoint_nd_double(
    double **h,
    double  *m,
    double  *d,
    size_t   nh,
    size_t   nm,
    size_t   nd,
    size_t   nslow,
    size_t   nthreads)
{
    interpadjoint_nd(h, m, d, nh, nm, nd, nslow, nthreads);
}

void
interpforward_nd_float(
    float  **h,
    float   *d,
    float   *m,
    size_t   nh,
    size_t   nd,
    size_t   nm,
    size_t   nslow,
    size_t   nthreads)
{
    interpforward_nd(h, d, m, nh, nd, nm, nslow, nthreads);
}

void
interpforward_nd_double(
    double **h,
    double  *d,
    double  *m,
    size_t   nh,
    size_t   nd,
    size_t   nm,
    size_t   nslow,
    size_t   nthreads)
{
    interpforward_nd(h, d, m, nh, nd, nm, nslow, nthreads);
}

} // extern C
