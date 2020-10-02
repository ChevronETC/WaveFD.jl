#include <string.h>
#include <omp.h>

template<class T> void
injectdata_2d_ongrid(
    T      *field,
    T      *data,
    T      *c,
    long   *iz,
    long   *ix,
    size_t  it,
    size_t  nt,
    size_t  nc,
    size_t  nz,
    size_t  nx,
    size_t  nthreads)
{
#pragma omp parallel for num_threads(nthreads) default(shared)
    for (size_t i = 0; i < nc; i++) {
        field[(ix[i]-1)*nz+iz[i]-1] += c[i]*data[i*nt+it-1];
    }
}

template<class T> void
injectdata_3d_ongrid(
    T      *field,
    T      *data,
    T      *c,
    long   *iz,
    long   *iy,
    long   *ix,
    size_t  it,
    size_t  nt,
    size_t  nc,
    size_t  nz,
    size_t  ny,
    size_t  nx,
    size_t  nthreads)
{
#pragma omp parallel for num_threads(nthreads) default(shared)
    for (size_t i = 0; i < nc; i++) {
        field[(ix[i]-1)*ny*nz+(iy[i]-1)*nz+iz[i]-1] += c[i]*data[i*nt+it-1];
    }
}

template<class T> void
extractdata_2d_ongrid(
    T      *data,
    T      *field,
    T      *c,
    long   *iz,
    long   *ix,
    size_t  it,
    size_t  nt,
    size_t  nc,
    size_t  nz,
    size_t  nx,
    size_t  nthreads)
{
#pragma omp parallel for num_threads(nthreads) default(shared)
    for (size_t i = 0; i < nc; i++) {
        data[i*nt+it-1] += c[i]*field[(ix[i]-1)*nz+iz[i]-1];
    }
}

template<class T> void
extractdata_3d_ongrid(
    T      *data,
    T      *field,
    T      *c,
    long   *iz,
    long   *iy,
    long   *ix,
    size_t  it,
    size_t  nt,
    size_t  nc,
    size_t  nz,
    size_t  ny,
    size_t  nx,
    size_t  nthreads)
{
#pragma omp parallel for num_threads(nthreads) default(shared)
    for (size_t i = 0; i < nc; i++) {
        data[i*nt+it-1] += c[i]*field[(ix[i]-1)*ny*nz+(iy[i]-1)*nz+iz[i]-1];
    }
}

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

void
injectdata_2d_ongrid_float(
    float  *field,
    float  *data,
    float  *c,
    long   *iz,
    long   *ix,
    size_t  it,
    size_t  nt,
    size_t  nc,
    size_t  nz,
    size_t  nx,
    size_t  nthreads)
{
    injectdata_2d_ongrid(field, data, c, iz, ix, it, nt, nc, nz, nx, nthreads);
}

void
injectdata_2d_ongrid_double(
    double  *field,
    double  *data,
    double  *c,
    long    *iz,
    long    *ix,
    size_t   it,
    size_t   nt,
    size_t   nc,
    size_t   nz,
    size_t   nx,
    size_t   nthreads)
{
    injectdata_2d_ongrid(field, data, c, iz, ix, it, nt, nc, nz, nx, nthreads);
}

void
injectdata_3d_ongrid_float(
    float  *field,
    float  *data,
    float  *c,
    long   *iz,
    long   *iy,
    long   *ix,
    size_t  it,
    size_t  nt,
    size_t  nc,
    size_t  nz,
    size_t  ny,
    size_t  nx,
    size_t  nthreads)
{
    injectdata_3d_ongrid(field, data, c, iz, iy, ix, it, nt, nc, nz, ny, nx, nthreads);
}

void
injectdata_3d_ongrid_double(
    double *field,
    double *data,
    double *c,
    long   *iz,
    long   *iy,
    long   *ix,
    size_t  it,
    size_t  nt,
    size_t  nc,
    size_t  nz,
    size_t  ny,
    size_t  nx,
    size_t  nthreads)
{
    injectdata_3d_ongrid(field, data, c, iz, iy, ix, it, nt, nc, nz, ny, nx, nthreads);
}

void
extractdata_2d_ongrid_float(
    float   *data,
    float   *field,
    float   *c,
    long    *iz,
    long    *ix,
    size_t   it,
    size_t   nt,
    size_t   nc,
    size_t   nz,
    size_t   nx,
    size_t   nthreads)
{
    extractdata_2d_ongrid(data, field, c, iz, ix, it, nt, nc, nz, nx, nthreads);
}

void
extractdata_2d_ongrid_double(
    double  *data,
    double  *field,
    double  *c,
    long    *iz,
    long    *ix,
    size_t   it,
    size_t   nt,
    size_t   nc,
    size_t   nz,
    size_t   nx,
    size_t   nthreads)
{
    extractdata_2d_ongrid(data, field, c, iz, ix, it, nt, nc, nz, nx, nthreads);
}

void
extractdata_3d_ongrid_float(
    float  *data,
    float  *field,
    float  *c,
    long   *iz,
    long   *iy,
    long   *ix,
    size_t  it,
    size_t  nt,
    size_t  nc,
    size_t  nz,
    size_t  ny,
    size_t  nx,
    size_t  nthreads)
{
    extractdata_3d_ongrid(data, field, c, iz, iy, ix, it, nt, nc, nz, ny, nx, nthreads);
}

void
extractdata_3d_ongrid_double(
    double *data,
    double *field,
    double *c,
    long   *iz,
    long   *iy,
    long   *ix,
    size_t  it,
    size_t  nt,
    size_t  nc,
    size_t  nz,
    size_t  ny,
    size_t  nx,
    size_t  nthreads)
{
    extractdata_3d_ongrid(data, field, c, iz, iy, ix, it, nt, nc, nz, ny, nx, nthreads);
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
