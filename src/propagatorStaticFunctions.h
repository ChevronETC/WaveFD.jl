#ifndef PROPAGATOR_STATIC_FUNCTIONS_H
#define PROPAGATOR_STATIC_FUNCTIONS_H

#define MIN(x,y) ((x)<(y)?(x):(y))

template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
inline void applyFirstDerivatives2D_PlusHalf(
        const long freeSurface,
        const long nx,
        const long nz,
        const long nthread,
        const Type c8_1,
        const Type c8_2,
        const Type c8_3,
        const Type c8_4,
        const Type invDx,
        const Type invDz,
        const Type * __restrict__ const inX,
        const Type * __restrict__ const inZ,
        Type * __restrict__ outX,
        Type * __restrict__ outZ,
        const long BX_2D,
        const long BZ_2D) {

    const long nx4 = nx - 4;
    const long nz4 = nz - 4;

    // zero output array
#pragma omp parallel for collapse(2) num_threads(nthread) schedule(static)
    for (long bx = 0; bx < nx; bx += BX_2D) {
        for (long bz = 0; bz < nz; bz += BZ_2D) {
            const long kxmax = MIN(bx + BX_2D, nx);
            const long kzmax = MIN(bz + BZ_2D, nz);

            for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                for (long kz = bz; kz < kzmax; kz++) {
                    outX[kx * nz + kz] = 0;
                    outZ[kx * nz + kz] = 0;
                }
            }
        }
    }

    // interior
#pragma omp parallel for collapse(2) num_threads(nthread) schedule(static)
    for (long bx = 4; bx < nx4; bx += BX_2D) {
        for (long bz = 4; bz < nz4; bz += BZ_2D) { /* cache blocking */

            const long kxmax = MIN(bx + BX_2D, nx4);
            const long kzmax = MIN(bz + BZ_2D, nz4);

            for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                for (long kz = bz; kz < kzmax; kz++) {

                    const Type stencilDx =
                            c8_1 * (- inX[(kx+0) * nz + kz] + inX[(kx+1) * nz + kz]) +
                            c8_2 * (- inX[(kx-1) * nz + kz] + inX[(kx+2) * nz + kz]) +
                            c8_3 * (- inX[(kx-2) * nz + kz] + inX[(kx+3) * nz + kz]) +
                            c8_4 * (- inX[(kx-3) * nz + kz] + inX[(kx+4) * nz + kz]);

                    const Type stencilDz =
                            c8_1 * (- inZ[kx * nz + (kz+0)] + inZ[kx * nz + (kz+1)]) +
                            c8_2 * (- inZ[kx * nz + (kz-1)] + inZ[kx * nz + (kz+2)]) +
                            c8_3 * (- inZ[kx * nz + (kz-2)] + inZ[kx * nz + (kz+3)]) +
                            c8_4 * (- inZ[kx * nz + (kz-3)] + inZ[kx * nz + (kz+4)]);

                    outX[kx * nz + kz] = invDx * stencilDx;
                    outZ[kx * nz + kz] = invDz * stencilDz;
                }
            }
        }
    }

    // roll on free surface
    if (freeSurface) {
#pragma omp parallel for num_threads(nthread) schedule(static)
        for (long kx = 4; kx < nx4; kx++) {

            // kz = 0 -- 1/2 cells below free surface for Z derivative, at free surface for X/Y derivative
            // X and Y derivatives are identically zero
            // [kx * nz + 0]
            {
                const Type stencilDPz0 =
                        c8_1 * (- inZ[kx * nz + 0] + inZ[kx * nz + 1]) +
                        c8_2 * (+ inZ[kx * nz + 1] + inZ[kx * nz + 2]) +
                        c8_3 * (+ inZ[kx * nz + 2] + inZ[kx * nz + 3]) +
                        c8_4 * (+ inZ[kx * nz + 3] + inZ[kx * nz + 4]);

                const long k = kx * nz + 0;
                outX[k] = 0;
                outZ[k] = invDz * stencilDPz0;
            }

            // kz = 1 -- 1 1/2 cells below free surface for Z derivative, 1 cells below for X/Y derivative
            // [kx * nz + 1]
            {
                const Type stencilDPx1 =
                        c8_1 * (- inX[(kx+0) * nz + 1] + inX[(kx+1) * nz + 1]) +
                        c8_2 * (- inX[(kx-1) * nz + 1] + inX[(kx+2) * nz + 1]) +
                        c8_3 * (- inX[(kx-2) * nz + 1] + inX[(kx+3) * nz + 1]) +
                        c8_4 * (- inX[(kx-3) * nz + 1] + inX[(kx+4) * nz + 1]);

                const Type stencilDPz1 =
                        c8_1 * (- inZ[kx * nz + 1] + inZ[kx * nz + 2]) +
                        c8_2 * (- inZ[kx * nz + 0] + inZ[kx * nz + 3]) +
                        c8_3 * (+ inZ[kx * nz + 1] + inZ[kx * nz + 4]) +
                        c8_4 * (+ inZ[kx * nz + 2] + inZ[kx * nz + 5]);

                const long k = kx * nz + 1;
                outX[k] = invDx * stencilDPx1;
                outZ[k] = invDz * stencilDPz1;
            }

            // kz = 2 -- 2 1/2 cells below free surface for Z derivative, 2 cells below for X/Y derivative
            // [kx * nz + 2]
            {
                const Type stencilDPx2 =
                        c8_1 * (- inX[(kx+0) * nz + 2] + inX[(kx+1) * nz + 2]) +
                        c8_2 * (- inX[(kx-1) * nz + 2] + inX[(kx+2) * nz + 2]) +
                        c8_3 * (- inX[(kx-2) * nz + 2] + inX[(kx+3) * nz + 2]) +
                        c8_4 * (- inX[(kx-3) * nz + 2] + inX[(kx+4) * nz + 2]);

                const Type stencilDPz2 =
                        c8_1 * (- inZ[kx * nz + 2] + inZ[kx * nz + 3]) +
                        c8_2 * (- inZ[kx * nz + 1] + inZ[kx * nz + 4]) +
                        c8_3 * (- inZ[kx * nz + 0] + inZ[kx * nz + 5]) +
                        c8_4 * (+ inZ[kx * nz + 1] + inZ[kx * nz + 6]);

                const long k = kx * nz + 2;
                outX[k] = invDx * stencilDPx2;
                outZ[k] = invDz * stencilDPz2;
            }

            // kz = 3 -- 3 1/2 cells below free surface for Z derivative, 3 cells below for X/Y derivative
            // [kx * nz + 3]
            {
                const Type stencilDPx3 =
                        c8_1 * (- inX[(kx+0) * nz + 3] + inX[(kx+1) * nz + 3]) +
                        c8_2 * (- inX[(kx-1) * nz + 3] + inX[(kx+2) * nz + 3]) +
                        c8_3 * (- inX[(kx-2) * nz + 3] + inX[(kx+3) * nz + 3]) +
                        c8_4 * (- inX[(kx-3) * nz + 3] + inX[(kx+4) * nz + 3]);

                const Type stencilDPz3 =
                        c8_1 * (- inZ[kx * nz + 3] + inZ[kx * nz + 4]) +
                        c8_2 * (- inZ[kx * nz + 2] + inZ[kx * nz + 5]) +
                        c8_3 * (- inZ[kx * nz + 1] + inZ[kx * nz + 6]) +
                        c8_4 * (- inZ[kx * nz + 0] + inZ[kx * nz + 7]);

                const long k = kx * nz + 3;
                outX[k] = invDx * stencilDPx3;
                outZ[k] = invDz * stencilDPz3;
            }

        }
    }
}

template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
inline void applyFirstDerivatives2D_MinusHalf(
        const long freeSurface,
        const long nx,
        const long nz,
        const long nthread,
        const Type c8_1,
        const Type c8_2,
        const Type c8_3,
        const Type c8_4,
        const Type invDx,
        const Type invDz,
        const Type * __restrict__ const inX,
        const Type * __restrict__ const inZ,
        Type * __restrict__ outX,
        Type * __restrict__ outZ,
        const long BX_2D,
        const long BZ_2D) {

    const long nx4 = nx - 4;
    const long nz4 = nz - 4;

    // zero output array
#pragma omp parallel for collapse(2) num_threads(nthread) schedule(static)
    for (long bx = 0; bx < nx; bx += BX_2D) {
        for (long bz = 0; bz < nz; bz += BZ_2D) {
            const long kxmax = MIN(bx + BX_2D, nx);
            const long kzmax = MIN(bz + BZ_2D, nz);

            for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                for (long kz = bz; kz < kzmax; kz++) {
                    outX[kx * nz + kz] = 0;
                    outZ[kx * nz + kz] = 0;
                }
            }
        }
    }

    // interior
#pragma omp parallel for collapse(2) num_threads(nthread) schedule(static)
    for (long bx = 4; bx < nx4; bx += BX_2D) {
        for (long bz = 4; bz < nz4; bz += BZ_2D) { /* cache blocking */

            const long kxmax = MIN(bx + BX_2D, nx4);
            const long kzmax = MIN(bz + BZ_2D, nz4);

            for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                for (long kz = bz; kz < kzmax; kz++) {

                    const Type stencilDx =
                            c8_1 * (- inX[(kx-1) * nz + kz] + inX[(kx+0) * nz + kz]) +
                            c8_2 * (- inX[(kx-2) * nz + kz] + inX[(kx+1) * nz + kz]) +
                            c8_3 * (- inX[(kx-3) * nz + kz] + inX[(kx+2) * nz + kz]) +
                            c8_4 * (- inX[(kx-4) * nz + kz] + inX[(kx+3) * nz + kz]);

                    const Type stencilDz =
                            c8_1 * (- inZ[kx * nz + (kz-1)] + inZ[kx * nz + (kz+0)]) +
                            c8_2 * (- inZ[kx * nz + (kz-2)] + inZ[kx * nz + (kz+1)]) +
                            c8_3 * (- inZ[kx * nz + (kz-3)] + inZ[kx * nz + (kz+2)]) +
                            c8_4 * (- inZ[kx * nz + (kz-4)] + inZ[kx * nz + (kz+3)]);

                    outX[kx * nz + kz] = invDx * stencilDx;
                    outZ[kx * nz + kz] = invDz * stencilDz;
                }
            }
        }
    }

    // roll on free surface
    if (freeSurface) {
#pragma omp parallel for num_threads(nthread) schedule(static)
        for (long kx = 4; kx < nx4; kx++) {


            // kz = 0 -- at the free surface -- p = 0
            // [kx * nz + 0]
            {
                const long k = kx * nz + 0;
                outX[k] = 0;
                outZ[k] = 0;
            }

            // kz = 1 -- one cell below the free surface
            // [kx * nz + 1]
            {
                const Type stencilDPx1 =
                        c8_1 * (- inX[(kx-1) * nz + 1] + inX[(kx+0) * nz + 1]) +
                        c8_2 * (- inX[(kx-2) * nz + 1] + inX[(kx+1) * nz + 1]) +
                        c8_3 * (- inX[(kx-3) * nz + 1] + inX[(kx+2) * nz + 1]) +
                        c8_4 * (- inX[(kx-4) * nz + 1] + inX[(kx+3) * nz + 1]);

                const Type stencilDPz1 =
                        c8_1 * (- inZ[kx * nz + 0] + inZ[kx * nz + 1]) +
                        c8_2 * (- inZ[kx * nz + 0] + inZ[kx * nz + 2]) +
                        c8_3 * (- inZ[kx * nz + 1] + inZ[kx * nz + 3]) +
                        c8_4 * (- inZ[kx * nz + 2] + inZ[kx * nz + 4]);

                const long k = kx * nz + 1;
                outX[k] = invDx * stencilDPx1;
                outZ[k] = invDz * stencilDPz1;
            }

            // kz = 2 -- two cells below the free surface
            // [kx * nz + 2]
            {
                const Type stencilDPx2 =
                        c8_1 * (- inX[(kx-1) * nz + 2] + inX[(kx+0) * nz + 2]) +
                        c8_2 * (- inX[(kx-2) * nz + 2] + inX[(kx+1) * nz + 2]) +
                        c8_3 * (- inX[(kx-3) * nz + 2] + inX[(kx+2) * nz + 2]) +
                        c8_4 * (- inX[(kx-4) * nz + 2] + inX[(kx+3) * nz + 2]);

                const Type stencilDPz2 =
                        c8_1 * (- inZ[kx * nz + 1] + inZ[kx * nz + 2]) +
                        c8_2 * (- inZ[kx * nz + 0] + inZ[kx * nz + 3]) +
                        c8_3 * (- inZ[kx * nz + 0] + inZ[kx * nz + 4]) +
                        c8_4 * (- inZ[kx * nz + 1] + inZ[kx * nz + 5]);

                const long k = kx * nz + 2;
                outX[k] = invDx * stencilDPx2;
                outZ[k] = invDz * stencilDPz2;
            }

            // kz = 3 -- three cells below the free surface
            // [kx * nz + 3]
            {
                const Type stencilDPx3 =
                        c8_1 * (- inX[(kx-1) * nz + 3] + inX[(kx+0) * nz + 3]) +
                        c8_2 * (- inX[(kx-2) * nz + 3] + inX[(kx+1) * nz + 3]) +
                        c8_3 * (- inX[(kx-3) * nz + 3] + inX[(kx+2) * nz + 3]) +
                        c8_4 * (- inX[(kx-4) * nz + 3] + inX[(kx+3) * nz + 3]);

                const Type stencilDPz3 =
                        c8_1 * (- inZ[kx * nz + 2] + inZ[kx * nz + 3]) +
                        c8_2 * (- inZ[kx * nz + 1] + inZ[kx * nz + 4]) +
                        c8_3 * (- inZ[kx * nz + 0] + inZ[kx * nz + 5]) +
                        c8_4 * (- inZ[kx * nz + 0] + inZ[kx * nz + 6]);

                const long k = kx * nz + 3;
                outX[k] = invDx * stencilDPx3;
                outZ[k] = invDz * stencilDPz3;
            }
        }
    }
}

template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
inline void applyFirstDerivatives3D_PlusHalf(
        const long freeSurface,
        const long nx,
        const long ny,
        const long nz,
        const long nthread,
        const Type c8_1,
        const Type c8_2,
        const Type c8_3,
        const Type c8_4,
        const Type invDx,
        const Type invDy,
        const Type invDz,
        Type * __restrict__ inX,
        Type * __restrict__ inY,
        Type * __restrict__ inZ,
        Type * __restrict__ outX,
        Type * __restrict__ outY,
        Type * __restrict__ outZ,
        const long BX_3D,
        const long BY_3D,
        const long BZ_3D) {

    const long nx4 = nx - 4;
    const long ny4 = ny - 4;
    const long nz4 = nz - 4;
    const long nynz = ny * nz;

	// zero output array: note only the annulus that is in the absorbing boundary needs to be zeroed
	for (long k = 0; k < 4; k++) {

#pragma omp parallel for num_threads(nthread) schedule(static)
        for (long kx = 0; kx < nx; kx++) {
#pragma omp simd
            for (long ky = 0; ky < ny; ky++) {
                long kindex1 = kx * ny * nz + ky * nz + k;
                long kindex2 = kx * ny * nz + ky * nz + (nz - 1 - k);
                outX[kindex1] = outX[kindex2] = 0;
                outY[kindex1] = outY[kindex2] = 0;
                outZ[kindex1] = outZ[kindex2] = 0;
            }
        }

#pragma omp parallel for num_threads(nthread) schedule(static)
        for (long kx = 0; kx < nx; kx++) {
#pragma omp simd
            for (long kz = 0; kz < nz; kz++) {
                long kindex1 = kx * ny * nz + k * nz + kz;
                long kindex2 = kx * ny * nz + (ny - 1 - k) * nz + kz;
                outX[kindex1] = outX[kindex2] = 0;
                outY[kindex1] = outY[kindex2] = 0;
                outZ[kindex1] = outZ[kindex2] = 0;
            }
        }

#pragma omp parallel for num_threads(nthread) schedule(static)
        for (long ky = 0; ky < ny; ky++) {
#pragma omp simd
            for (long kz = 0; kz < nz; kz++) {
                long kindex1 = k * ny * nz + ky * nz + kz;
                long kindex2 = (nx - 1 - k) * ny * nz + ky * nz + kz;
                outX[kindex1] = outX[kindex2] = 0;
                outY[kindex1] = outY[kindex2] = 0;
                outZ[kindex1] = outZ[kindex2] = 0;
            }
        }

    }

    // interior
#pragma omp parallel for collapse(3) num_threads(nthread) schedule(static)
    for (long bx = 4; bx < nx4; bx += BX_3D) {
        for (long by = 4; by < ny4; by += BY_3D) {
            for (long bz = 4; bz < nz4; bz += BZ_3D) {
                const long kxmax = MIN(bx + BX_3D, nx4);
                const long kymax = MIN(by + BY_3D, ny4);
                const long kzmax = MIN(bz + BZ_3D, nz4);

                for (long kx = bx; kx < kxmax; kx++) {
                    const long kxnynz = kx * nynz;

                    for (long ky = by; ky < kymax; ky++) {
                        const long kynz = ky * nz;
                        const long kxnynz_kynz = kxnynz + kynz;

#pragma omp simd
                        for (long kz = bz; kz < kzmax; kz++) {
                            const long kynz_kz = + kynz + kz;

                            const Type stencilDx =
                                    c8_1 * (- inX[(kx+0) * nynz + kynz_kz] + inX[(kx+1) * nynz + kynz_kz]) +
                                    c8_2 * (- inX[(kx-1) * nynz + kynz_kz] + inX[(kx+2) * nynz + kynz_kz]) +
                                    c8_3 * (- inX[(kx-2) * nynz + kynz_kz] + inX[(kx+3) * nynz + kynz_kz]) +
                                    c8_4 * (- inX[(kx-3) * nynz + kynz_kz] + inX[(kx+4) * nynz + kynz_kz]);

                            const Type stencilDy =
                                    c8_1 * (- inY[kxnynz + (ky+0) * nz + kz] + inY[kxnynz + (ky+1) * nz + kz]) +
                                    c8_2 * (- inY[kxnynz + (ky-1) * nz + kz] + inY[kxnynz + (ky+2) * nz + kz]) +
                                    c8_3 * (- inY[kxnynz + (ky-2) * nz + kz] + inY[kxnynz + (ky+3) * nz + kz]) +
                                    c8_4 * (- inY[kxnynz + (ky-3) * nz + kz] + inY[kxnynz + (ky+4) * nz + kz]);

                            const Type stencilDz =
                                    c8_1 * (- inZ[kxnynz_kynz + (kz+0)] + inZ[kxnynz_kynz + (kz+1)]) +
                                    c8_2 * (- inZ[kxnynz_kynz + (kz-1)] + inZ[kxnynz_kynz + (kz+2)]) +
                                    c8_3 * (- inZ[kxnynz_kynz + (kz-2)] + inZ[kxnynz_kynz + (kz+3)]) +
                                    c8_4 * (- inZ[kxnynz_kynz + (kz-3)] + inZ[kxnynz_kynz + (kz+4)]);

                            outX[kxnynz_kynz + kz] = invDx * stencilDx;
                            outY[kxnynz_kynz + kz] = invDy * stencilDy;
                            outZ[kxnynz_kynz + kz] = invDz * stencilDz;
                        }
                    }
                }
            }
        }
    }

    // roll on free surface
    if (freeSurface) {
#pragma omp parallel for num_threads(nthread) schedule(static)
        for (long kx = 4; kx < nx4; kx++) {
            const long kxnynz = kx * nynz;

#pragma omp simd
            for (long ky = 4; ky < ny4; ky++) {
                const long kynz = ky * nz;
                const long kxnynz_kynz = kxnynz + kynz;

                // kz = 0 -- 1/2 cells below free surface for Z derivative, at free surface for X/Y derivative
                // X and Y derivatives are identically zero
                const Type stencilDz0 =
                        c8_1 * (- inZ[kxnynz_kynz + 0] + inZ[kxnynz_kynz + 1]) +
                        c8_2 * (+ inZ[kxnynz_kynz + 1] + inZ[kxnynz_kynz + 2]) +
                        c8_3 * (+ inZ[kxnynz_kynz + 2] + inZ[kxnynz_kynz + 3]) +
                        c8_4 * (+ inZ[kxnynz_kynz + 3] + inZ[kxnynz_kynz + 4]);

                outX[kxnynz_kynz + 0] = 0;
                outY[kxnynz_kynz + 0] = 0;
                outZ[kxnynz_kynz + 0] = invDz * stencilDz0;

                // kz = 1 -- 1 1/2 cells below free surface for Z derivative, 1 cells below for X/Y derivative
                const Type stencilDx1 =
                        c8_1 * (- inX[(kx+0) * nynz + kynz + 1] + inX[(kx+1) * nynz + kynz + 1]) +
                        c8_2 * (- inX[(kx-1) * nynz + kynz + 1] + inX[(kx+2) * nynz + kynz + 1]) +
                        c8_3 * (- inX[(kx-2) * nynz + kynz + 1] + inX[(kx+3) * nynz + kynz + 1]) +
                        c8_4 * (- inX[(kx-3) * nynz + kynz + 1] + inX[(kx+4) * nynz + kynz + 1]);

                const Type stencilDy1 =
                        c8_1 * (- inY[kxnynz + (ky+0) * nz + 1] + inY[kxnynz + (ky+1) * nz + 1]) +
                        c8_2 * (- inY[kxnynz + (ky-1) * nz + 1] + inY[kxnynz + (ky+2) * nz + 1]) +
                        c8_3 * (- inY[kxnynz + (ky-2) * nz + 1] + inY[kxnynz + (ky+3) * nz + 1]) +
                        c8_4 * (- inY[kxnynz + (ky-3) * nz + 1] + inY[kxnynz + (ky+4) * nz + 1]);

                const Type stencilDz1 =
                        c8_1 * (- inZ[kxnynz_kynz + 1] + inZ[kxnynz_kynz + 2]) +
                        c8_2 * (- inZ[kxnynz_kynz + 0] + inZ[kxnynz_kynz + 3]) +
                        c8_3 * (+ inZ[kxnynz_kynz + 1] + inZ[kxnynz_kynz + 4]) +
                        c8_4 * (+ inZ[kxnynz_kynz + 2] + inZ[kxnynz_kynz + 5]);

                outX[kxnynz_kynz + 1] = invDx * stencilDx1;
                outY[kxnynz_kynz + 1] = invDy * stencilDy1;
                outZ[kxnynz_kynz + 1] = invDz * stencilDz1;

                // kz = 2 -- 2 1/2 cells below free surface for Z derivative, 2 cells below for X/Y derivative
                const Type stencilDx2 =
                        c8_1 * (- inX[(kx+0) * nynz + kynz + 2] + inX[(kx+1) * nynz + kynz + 2]) +
                        c8_2 * (- inX[(kx-1) * nynz + kynz + 2] + inX[(kx+2) * nynz + kynz + 2]) +
                        c8_3 * (- inX[(kx-2) * nynz + kynz + 2] + inX[(kx+3) * nynz + kynz + 2]) +
                        c8_4 * (- inX[(kx-3) * nynz + kynz + 2] + inX[(kx+4) * nynz + kynz + 2]);

                const Type stencilDy2 =
                        c8_1 * (- inY[kxnynz + (ky+0) * nz + 2] + inY[kxnynz + (ky+1) * nz + 2]) +
                        c8_2 * (- inY[kxnynz + (ky-1) * nz + 2] + inY[kxnynz + (ky+2) * nz + 2]) +
                        c8_3 * (- inY[kxnynz + (ky-2) * nz + 2] + inY[kxnynz + (ky+3) * nz + 2]) +
                        c8_4 * (- inY[kxnynz + (ky-3) * nz + 2] + inY[kxnynz + (ky+4) * nz + 2]);

                const Type stencilDz2 =
                        c8_1 * (- inZ[kxnynz_kynz + 2] + inZ[kxnynz_kynz + 3]) +
                        c8_2 * (- inZ[kxnynz_kynz + 1] + inZ[kxnynz_kynz + 4]) +
                        c8_3 * (- inZ[kxnynz_kynz + 0] + inZ[kxnynz_kynz + 5]) +
                        c8_4 * (+ inZ[kxnynz_kynz + 1] + inZ[kxnynz_kynz + 6]);

                outX[kxnynz_kynz + 2] = invDx * stencilDx2;
                outY[kxnynz_kynz + 2] = invDy * stencilDy2;
                outZ[kxnynz_kynz + 2] = invDz * stencilDz2;

                // kz = 3 -- 3 1/2 cells below free surface for Z derivative, 3 cells below for X/Y derivative
                const Type stencilDx3 =
                        c8_1 * (- inX[(kx+0) * nynz + kynz + 3] + inX[(kx+1) * nynz + kynz + 3]) +
                        c8_2 * (- inX[(kx-1) * nynz + kynz + 3] + inX[(kx+2) * nynz + kynz + 3]) +
                        c8_3 * (- inX[(kx-2) * nynz + kynz + 3] + inX[(kx+3) * nynz + kynz + 3]) +
                        c8_4 * (- inX[(kx-3) * nynz + kynz + 3] + inX[(kx+4) * nynz + kynz + 3]);

                const Type stencilDy3 =
                        c8_1 * (- inY[kxnynz + (ky+0) * nz + 3] + inY[kxnynz + (ky+1) * nz + 3]) +
                        c8_2 * (- inY[kxnynz + (ky-1) * nz + 3] + inY[kxnynz + (ky+2) * nz + 3]) +
                        c8_3 * (- inY[kxnynz + (ky-2) * nz + 3] + inY[kxnynz + (ky+3) * nz + 3]) +
                        c8_4 * (- inY[kxnynz + (ky-3) * nz + 3] + inY[kxnynz + (ky+4) * nz + 3]);

                const Type stencilDz3 =
                        c8_1 * (- inZ[kxnynz_kynz + 3] + inZ[kxnynz_kynz + 4]) +
                        c8_2 * (- inZ[kxnynz_kynz + 2] + inZ[kxnynz_kynz + 5]) +
                        c8_3 * (- inZ[kxnynz_kynz + 1] + inZ[kxnynz_kynz + 6]) +
                        c8_4 * (- inZ[kxnynz_kynz + 0] + inZ[kxnynz_kynz + 7]);

                outX[kxnynz_kynz + 3] = invDx * stencilDx3;
                outY[kxnynz_kynz + 3] = invDy * stencilDy3;
                outZ[kxnynz_kynz + 3] = invDz * stencilDz3;
            }
        }
    }
}

template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
inline void applyFirstDerivatives3D_MinusHalf(
        const long freeSurface,
        const long nx,
        const long ny,
        const long nz,
        const long nthread,
        const Type c8_1,
        const Type c8_2,
        const Type c8_3,
        const Type c8_4,
        const Type invDx,
        const Type invDy,
        const Type invDz,
        Type * __restrict__ inX,
        Type * __restrict__ inY,
        Type * __restrict__ inZ,
        Type * __restrict__ outX,
        Type * __restrict__ outY,
        Type * __restrict__ outZ,
        const long BX_3D,
        const long BY_3D,
        const long BZ_3D) {

    const long nx4 = nx - 4;
    const long ny4 = ny - 4;
    const long nz4 = nz - 4;
    const long nynz = ny * nz;

    // zero output array: note only the annulus that is in the absorbing boundary needs to be zeroed
    for (long k = 0; k < 4; k++) {

#pragma omp parallel for num_threads(nthread) schedule(static)
        for (long kx = 0; kx < nx; kx++) {
#pragma omp simd
            for (long ky = 0; ky < ny; ky++) {
                long kindex1 = kx * ny * nz + ky * nz + k;
                long kindex2 = kx * ny * nz + ky * nz + (nz - 1 - k);
                outX[kindex1] = outX[kindex2] = 0;
                outY[kindex1] = outY[kindex2] = 0;
                outZ[kindex1] = outZ[kindex2] = 0;
            }
        }

#pragma omp parallel for num_threads(nthread) schedule(static)
        for (long kx = 0; kx < nx; kx++) {
#pragma omp simd
            for (long kz = 0; kz < nz; kz++) {
                long kindex1 = kx * ny * nz + k * nz + kz;
                long kindex2 = kx * ny * nz + (ny - 1 - k) * nz + kz;
                outX[kindex1] = outX[kindex2] = 0;
                outY[kindex1] = outY[kindex2] = 0;
                outZ[kindex1] = outZ[kindex2] = 0;
            }
        }

#pragma omp parallel for num_threads(nthread) schedule(static)
        for (long ky = 0; ky < ny; ky++) {
#pragma omp simd
            for (long kz = 0; kz < nz; kz++) {
                long kindex1 = k * ny * nz + ky * nz + kz;
                long kindex2 = (nx - 1 - k) * ny * nz + ky * nz + kz;
                outX[kindex1] = outX[kindex2] = 0;
                outY[kindex1] = outY[kindex2] = 0;
                outZ[kindex1] = outZ[kindex2] = 0;
            }
        }

    }

    // interior
#pragma omp parallel for collapse(3) num_threads(nthread) schedule(static)
    for (long bx = 4; bx < nx4; bx += BX_3D) {
        for (long by = 4; by < ny4; by += BY_3D) {
            for (long bz = 4; bz < nz4; bz += BZ_3D) {
                const long kxmax = MIN(bx + BX_3D, nx4);
                const long kymax = MIN(by + BY_3D, ny4);
                const long kzmax = MIN(bz + BZ_3D, nz4);

                for (long kx = bx; kx < kxmax; kx++) {
                    const long kxnynz = kx * nynz;

                    for (long ky = by; ky < kymax; ky++) {
                        const long kynz = ky * nz;
                        const long kxnynz_kynz = kxnynz + kynz;

#pragma omp simd
                        for (long kz = bz; kz < kzmax; kz++) {
                            const long kynz_kz = + kynz + kz;

                            const Type stencilDx =
                                    c8_1 * (- inX[(kx-1) * nynz + kynz_kz] + inX[(kx+0) * nynz + kynz_kz]) +
                                    c8_2 * (- inX[(kx-2) * nynz + kynz_kz] + inX[(kx+1) * nynz + kynz_kz]) +
                                    c8_3 * (- inX[(kx-3) * nynz + kynz_kz] + inX[(kx+2) * nynz + kynz_kz]) +
                                    c8_4 * (- inX[(kx-4) * nynz + kynz_kz] + inX[(kx+3) * nynz + kynz_kz]);

                            const Type stencilDy =
                                    c8_1 * (- inY[kxnynz + (ky-1) * nz + kz] + inY[kxnynz + (ky+0) * nz + kz]) +
                                    c8_2 * (- inY[kxnynz + (ky-2) * nz + kz] + inY[kxnynz + (ky+1) * nz + kz]) +
                                    c8_3 * (- inY[kxnynz + (ky-3) * nz + kz] + inY[kxnynz + (ky+2) * nz + kz]) +
                                    c8_4 * (- inY[kxnynz + (ky-4) * nz + kz] + inY[kxnynz + (ky+3) * nz + kz]);

                            const Type stencilDz =
                                    c8_1 * (- inZ[kxnynz_kynz + (kz-1)] + inZ[kxnynz_kynz + (kz+0)]) +
                                    c8_2 * (- inZ[kxnynz_kynz + (kz-2)] + inZ[kxnynz_kynz + (kz+1)]) +
                                    c8_3 * (- inZ[kxnynz_kynz + (kz-3)] + inZ[kxnynz_kynz + (kz+2)]) +
                                    c8_4 * (- inZ[kxnynz_kynz + (kz-4)] + inZ[kxnynz_kynz + (kz+3)]);

                            outX[kxnynz_kynz + kz] = invDx * stencilDx;
                            outY[kxnynz_kynz + kz] = invDy * stencilDy;
                            outZ[kxnynz_kynz + kz] = invDz * stencilDz;
                        }
                    }
                }
            }
        }
    }

    // roll on free surface
    if (freeSurface) {
#pragma omp parallel for num_threads(nthread) schedule(static)
        for (long kx = 4; kx < nx4; kx++) {
            const long kxnynz = kx * nynz;

#pragma omp simd
            for (long ky = 4; ky < ny4; ky++) {
                const long kynz = ky * nz;
                const long kxnynz_kynz = kxnynz + kynz;

                // kz = 0 -- at the free surface -- p = 0
                outX[kxnynz_kynz + 0] = 0;
                outY[kxnynz_kynz + 0] = 0;
                outZ[kxnynz_kynz + 0] = 0;

                // kz = 1 -- one cell below the free surface
                const Type stencilDx1 =
                        c8_1 * (- inX[(kx-1) * nynz + kynz + 1] + inX[(kx+0) * nynz + kynz + 1]) +
                        c8_2 * (- inX[(kx-2) * nynz + kynz + 1] + inX[(kx+1) * nynz + kynz + 1]) +
                        c8_3 * (- inX[(kx-3) * nynz + kynz + 1] + inX[(kx+2) * nynz + kynz + 1]) +
                        c8_4 * (- inX[(kx-4) * nynz + kynz + 1] + inX[(kx+3) * nynz + kynz + 1]);

                const Type stencilDy1 =
                        c8_1 * (- inY[kxnynz + (ky-1) * nz + 1] + inY[kxnynz + (ky+0) * nz + 1]) +
                        c8_2 * (- inY[kxnynz + (ky-2) * nz + 1] + inY[kxnynz + (ky+1) * nz + 1]) +
                        c8_3 * (- inY[kxnynz + (ky-3) * nz + 1] + inY[kxnynz + (ky+2) * nz + 1]) +
                        c8_4 * (- inY[kxnynz + (ky-4) * nz + 1] + inY[kxnynz + (ky+3) * nz + 1]);

                const Type stencilDz1 =
                        c8_1 * (- inZ[kxnynz_kynz + 0] + inZ[kxnynz_kynz + 1]) +
                        c8_2 * (- inZ[kxnynz_kynz + 0] + inZ[kxnynz_kynz + 2]) +
                        c8_3 * (- inZ[kxnynz_kynz + 1] + inZ[kxnynz_kynz + 3]) +
                        c8_4 * (- inZ[kxnynz_kynz + 2] + inZ[kxnynz_kynz + 4]);

                outX[kxnynz_kynz + 1] = invDx * stencilDx1;
                outY[kxnynz_kynz + 1] = invDy * stencilDy1;
                outZ[kxnynz_kynz + 1] = invDz * stencilDz1;

                // kz = 2 -- two cells below the free surface
                const Type stencilDx2 =
                        c8_1 * (- inX[(kx-1) * nynz + kynz + 2] + inX[(kx+0) * nynz + kynz + 2]) +
                        c8_2 * (- inX[(kx-2) * nynz + kynz + 2] + inX[(kx+1) * nynz + kynz + 2]) +
                        c8_3 * (- inX[(kx-3) * nynz + kynz + 2] + inX[(kx+2) * nynz + kynz + 2]) +
                        c8_4 * (- inX[(kx-4) * nynz + kynz + 2] + inX[(kx+3) * nynz + kynz + 2]);

                const Type stencilDy2 =
                        c8_1 * (- inY[kxnynz + (ky-1) * nz + 2] + inY[kxnynz + (ky+0) * nz + 2]) +
                        c8_2 * (- inY[kxnynz + (ky-2) * nz + 2] + inY[kxnynz + (ky+1) * nz + 2]) +
                        c8_3 * (- inY[kxnynz + (ky-3) * nz + 2] + inY[kxnynz + (ky+2) * nz + 2]) +
                        c8_4 * (- inY[kxnynz + (ky-4) * nz + 2] + inY[kxnynz + (ky+3) * nz + 2]);

                const Type stencilDz2 =
                        c8_1 * (- inZ[kxnynz_kynz + 1] + inZ[kxnynz_kynz + 2]) +
                        c8_2 * (- inZ[kxnynz_kynz + 0] + inZ[kxnynz_kynz + 3]) +
                        c8_3 * (- inZ[kxnynz_kynz + 0] + inZ[kxnynz_kynz + 4]) +
                        c8_4 * (- inZ[kxnynz_kynz + 1] + inZ[kxnynz_kynz + 5]);

                outX[kxnynz_kynz + 2] = invDx * stencilDx2;
                outY[kxnynz_kynz + 2] = invDy * stencilDy2;
                outZ[kxnynz_kynz + 2] = invDz * stencilDz2;

                // kz = 3 -- three cells below the free surface
                const Type stencilDx3 =
                        c8_1 * (- inX[(kx-1) * nynz + kynz + 3] + inX[(kx+0) * nynz + kynz + 3]) +
                        c8_2 * (- inX[(kx-2) * nynz + kynz + 3] + inX[(kx+1) * nynz + kynz + 3]) +
                        c8_3 * (- inX[(kx-3) * nynz + kynz + 3] + inX[(kx+2) * nynz + kynz + 3]) +
                        c8_4 * (- inX[(kx-4) * nynz + kynz + 3] + inX[(kx+3) * nynz + kynz + 3]);

                const Type stencilDy3 =
                        c8_1 * (- inY[kxnynz + (ky-1) * nz + 3] + inY[kxnynz + (ky+0) * nz + 3]) +
                        c8_2 * (- inY[kxnynz + (ky-2) * nz + 3] + inY[kxnynz + (ky+1) * nz + 3]) +
                        c8_3 * (- inY[kxnynz + (ky-3) * nz + 3] + inY[kxnynz + (ky+2) * nz + 3]) +
                        c8_4 * (- inY[kxnynz + (ky-4) * nz + 3] + inY[kxnynz + (ky+3) * nz + 3]);

                const Type stencilDz3 =
                        c8_1 * (- inZ[kxnynz_kynz + 2] + inZ[kxnynz_kynz + 3]) +
                        c8_2 * (- inZ[kxnynz_kynz + 1] + inZ[kxnynz_kynz + 4]) +
                        c8_3 * (- inZ[kxnynz_kynz + 0] + inZ[kxnynz_kynz + 5]) +
                        c8_4 * (- inZ[kxnynz_kynz + 0] + inZ[kxnynz_kynz + 6]);

                outX[kxnynz_kynz + 3] = invDx * stencilDx3;
                outY[kxnynz_kynz + 3] = invDy * stencilDy3;
                outZ[kxnynz_kynz + 3] = invDz * stencilDz3;
            }
        }
    }
}

#endif