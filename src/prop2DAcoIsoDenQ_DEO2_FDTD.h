#ifndef PROP2DACOISODENQ_DEO2_FDTD_H
#define PROP2DACOISODENQ_DEO2_FDTD_H

#include <omp.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <fftw3.h>
#include <complex>

#include "propagatorStaticFunctions.h"

#define MIN(x,y) ((x)<(y)?(x):(y))

class Prop2DAcoIsoDenQ_DEO2_FDTD {

public:
    const bool _freeSurface;
    const long _nbx, _nbz, _nthread, _nx, _nz, _nsponge;
    const float _dx, _dz, _dt;
    const float _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz;

    float * __restrict__ _v = NULL;
    float * __restrict__ _b = NULL;
    float * __restrict__ _dtOmegaInvQ = NULL;
    float * __restrict__ _pSpace = NULL;
    float * __restrict__ _tmpPx1 = NULL;
    float * __restrict__ _tmpPz1 = NULL;
    float * __restrict__ _tmpPx2 = NULL;
    float * __restrict__ _tmpPz2 = NULL;
    float * _pOld = NULL;
    float * _pCur = NULL;

    Prop2DAcoIsoDenQ_DEO2_FDTD(
        bool freeSurface,
        long nthread,
        long nx,
        long nz,
        long nsponge,
        float dx,
        float dz,
        float dt,
        const long nbx,
        const long nbz) :
            _freeSurface(freeSurface),
            _nthread(nthread),
            _nx(nx),
            _nz(nz),
            _nsponge(nsponge),
            _nbx(nbx),
            _nbz(nbz),
            _dx(dx),
            _dz(dz),
            _dt(dt),
            _c8_1(+1225.0 / 1024.0),
            _c8_2(-245.0 / 3072.0),
            _c8_3(+49.0 / 5120.0),
            _c8_4(-5.0 / 7168.0),
            _invDx(1.0 / _dx),
            _invDz(1.0 / _dz) {

        // Allocate arrays
        _v           = new float[_nx * _nz];
        _b           = new float[_nx * _nz];
        _dtOmegaInvQ = new float[_nx * _nz];
        _pSpace      = new float[_nx * _nz];
        _tmpPx1      = new float[_nx * _nz];
        _tmpPz1      = new float[_nx * _nz];
        _tmpPx2      = new float[_nx * _nz];
        _tmpPz2      = new float[_nx * _nz];
        _pOld        = new float[_nx * _nz];
        _pCur        = new float[_nx * _nz];

        numaFirstTouch(_nx, _nz, _nthread, _v, _b,
            _dtOmegaInvQ, _pSpace, _tmpPx1, _tmpPz1, _tmpPx2, _tmpPz2, _pOld, _pCur,
            _nbx, _nbz);
    }

#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void numaFirstTouch(
            const long nx,
            const long nz,
            const long nthread,
            float * __restrict__ v,
            float * __restrict__ b,
            float * __restrict__ dtOmegaInvQ,
            float * __restrict__ pSpace,
            float * __restrict__ tmpPx1,
            float * __restrict__ tmpPz1,
            float * __restrict__ tmpPx2,
            float * __restrict__ tmpPz2,
            float * __restrict__ pOld,
            float * __restrict__ pCur,
            const long BX_2D,
            const long BZ_2D) {

        const long nx4 = nx - 4;
        const long nz4 = nz - 4;

#pragma omp parallel for collapse(2) num_threads(nthread) schedule(static)
        for (long bx = 4; bx < nx4; bx += BX_2D) {
            for (long bz = 4; bz < nz4; bz += BZ_2D) {
                const long kxmax = MIN(bx + BX_2D, nx4);
                const long kzmax = MIN(bz + BZ_2D, nz4);

                for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                    for (long kz = bz; kz < kzmax; kz++) {
                        const long k = kx * _nz + kz;

                        v[k] = 0;
                        b[k] = 0;
                        dtOmegaInvQ[k] = 0;
                        pSpace[k] = 0;
                        tmpPx1[k] = 0;
                        tmpPz1[k] = 0;
                        tmpPx2[k] = 0;
                        tmpPz2[k] = 0;
                        pOld[k] = 0;
                        pCur[k] = 0;
                    }
                }
            }
        }

        // zero annulus
#pragma omp parallel for num_threads(nthread) schedule(static)
        for (long kz = 0; kz < 4; kz++) {
#pragma omp simd
            for (long kx = 0; kx < nx; kx++) {
                const long k = kx * _nz + kz;
                v[k] = b[k] = dtOmegaInvQ[k] = tmpPx1[k] = tmpPz1[k] = tmpPx2[k] = tmpPz2[k] = pOld[k] = pCur[k] = 0;
            }
        }
#pragma omp parallel for num_threads(nthread) schedule(static)
        for (long kz = nz4; kz < nz; kz++) {
#pragma omp simd
            for (long kx = 0; kx < nx; kx++) {
                const long k = kx * _nz + kz;
                v[k] = b[k] = dtOmegaInvQ[k] = tmpPx1[k] = tmpPz1[k] = tmpPx2[k] = tmpPz2[k] = pOld[k] = pCur[k] = 0;
            }
        }

#pragma omp parallel for num_threads(nthread) schedule(static)
        for (long kx = 0; kx < 4; kx++) {
#pragma omp simd
            for (long kz = 0; kz < nz; kz++) {
                const long k = kx * _nz + kz;
                v[k] = b[k] = dtOmegaInvQ[k] = tmpPx1[k] = tmpPz1[k] = tmpPx2[k] = tmpPz2[k] = pOld[k] = pCur[k] = 0;
            }
        }
#pragma omp parallel for num_threads(nthread) schedule(static)
        for (long kx = nx4; kx < nx; kx++) {
#pragma omp simd
            for (long kz = 0; kz < nz; kz++) {
                const long k = kx * _nz + kz;
                v[k] = b[k] = dtOmegaInvQ[k] = tmpPx1[k] = tmpPz1[k] = tmpPx2[k] = tmpPz2[k] = pOld[k] = pCur[k] = 0;
            }
        }
    }

    ~Prop2DAcoIsoDenQ_DEO2_FDTD() {
        if (_v != NULL) delete [] _v;
        if (_b != NULL) delete [] _b;
        if (_dtOmegaInvQ != NULL) delete [] _dtOmegaInvQ;
        if (_pSpace != NULL) delete [] _pSpace;
        if (_tmpPx1 != NULL) delete [] _tmpPx1;
        if (_tmpPz1 != NULL) delete [] _tmpPz1;
        if (_tmpPx2 != NULL) delete [] _tmpPx2;
        if (_tmpPz2 != NULL) delete [] _tmpPz2;
        if (_pOld != NULL) delete [] _pOld;
        if (_pCur != NULL) delete [] _pCur;
    }

#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    void info() {
        printf("\n");
        printf("Prop2DAcoIsoDenQ_DEO2_FDTD\n");
        printf("  nx,nz;              %5ld %5ld\n", _nx, _nz);
        printf("  nthread,nsponge,fs; %5ld %5ld %5d\n", _nthread, _nsponge, _freeSurface);
        printf("  X min,max,inc;    %+16.6f %+16.6f %+16.6f\n", 0.0, _dx * (_nx - 1), _dx);
        printf("  Z min,max,inc;    %+16.6f %+16.6f %+16.6f\n", 0.0, _dz * (_nz - 1), _dz);
    }

    /**
     * Notes
     * - User must have called setupDtOmegaInvQ_2D to initialize the array _dtOmegaInvQ
     * - wavefield arrays are switched in this call
     *     pCur -> pOld
     *     pOld -> pCur
     */
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void timeStep() {

        applyFirstDerivatives2D_PlusHalf_Sandwich(
                _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
                _pCur, _pCur, _b, _tmpPx1, _tmpPz1, _nbx, _nbz);

        applyFirstDerivatives2D_MinusHalf_TimeUpdate_Nonlinear(
                _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz, _dt,
                _tmpPx1, _tmpPz1, _v, _b, _dtOmegaInvQ, _pCur, _pSpace, _pOld, _nbx, _nbz);

        // swap pointers
        float *pswap = _pOld;
        _pOld = _pCur;
        _pCur = pswap;
    }

    /**
     * Scale spatial derivatives by v^2/b to make them temporal derivs
     */
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void scaleSpatialDerivatives() {
#pragma omp parallel for collapse(2) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long bz = 0; bz < _nz; bz += _nbz) {
                const long kxmax = MIN(bx + _nbx, _nx);
                const long kzmax = MIN(bz + _nbz, _nz);

                for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                    for (long kz = bz; kz < kzmax; kz++) {
                        const long k = kx * _nz + kz;
                        const float v2OverB = _v[k] * _v[k] / _b[k];
                        _pSpace[k] *= v2OverB;
                    }
                }
            }
        }
    }

    /**
     * Add the Born source for velocity only model-space at the current time
     *
     * User must have:
     *   - called the nonlinear forward
     *   - saved 2nd time derivative of pressure at corresponding time index in array dp2
     *   - Born source term will be injected into the _pCur array
     */
template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void forwardBornInjection_V(Type *dVel, Type *wavefieldDP) {      
  
#pragma omp parallel for collapse(2) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long bz = 0; bz < _nz; bz += _nbz) {
                const long kxmax = MIN(bx + _nbx, _nx);
                const long kzmax = MIN(bz + _nbz, _nz);

                for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                    for (long kz = bz; kz < kzmax; kz++) {
                        const long k = kx * _nz + kz;

                        const Type V  = _v[k];
                        const Type B  = _b[k];
                        const Type dV = dVel[k];

                        // const Type dt2v2OverB = _dt * _dt * V * V / B;
                        // const Type factorV = 2 * B * dV / (V * V * V);

                        const Type factor = 2 * _dt * _dt * dV  / V;

                        _pCur[k] += factor * wavefieldDP[k];
                    }
                }
            }
        }
    }

    /**
     * Add the Born source for buoyancy and velocity model-space at the current time
     *
     * User must have:
     *   - called the nonlinear forward
     *   - saved 2nd time derivative of pressure at corresponding time index in array dp2
     *   - Born source term will be injected into the _pCur array
     *
     * TODO: if these second derivative call and following loop is expensive,
     *       could consider fusing the two derivative loops with the final loop
     */
template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void forwardBornInjection_VB(Type *dVel, Type *dBuoy, Type *wavefieldP, Type *wavefieldDP) {

        applyFirstDerivatives2D_PlusHalf(
            _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz, 
            wavefieldP, wavefieldP, _tmpPx1, _tmpPz1, _nbx, _nbz);

#pragma omp parallel for collapse(2) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long bz = 0; bz < _nz; bz += _nbz) {
                const long kxmax = MIN(bx + _nbx, _nx);
                const long kzmax = MIN(bz + _nbz, _nz);

                for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                    for (long kz = bz; kz < kzmax; kz++) {
                        const long k = kx * _nz + kz;

                        const Type dB = dBuoy[k];

                        _tmpPx2[k] = dB * _tmpPx1[k];
                        _tmpPz2[k] = dB * _tmpPz1[k];
                    }
                }
            }
        }

        applyFirstDerivatives2D_MinusHalf(
            _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz, 
            _tmpPx2, _tmpPz2, _tmpPx1, _tmpPz1, _nbx, _nbz);

#pragma omp parallel for collapse(2) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long bz = 0; bz < _nz; bz += _nbz) {
                const long kxmax = MIN(bx + _nbx, _nx);
                const long kzmax = MIN(bz + _nbz, _nz);

                for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                    for (long kz = bz; kz < kzmax; kz++) {
                        const long k = kx * _nz + kz;

                        const Type V  = _v[k];
                        const Type B  = _b[k];
                        const Type dV = dVel[k];
                        const Type dB = dBuoy[k];

                        const Type V2 = V * V;
                        const Type dt2v2OverB = _dt * _dt * V2 / B;
                        // const Type factorV = 2 * B * dV / (V * V * V);
                        // const Type factorB = - dB / (V * V);

                        _pCur[k] += dt2v2OverB * (wavefieldDP[k] / V2 * (2 * B * dV / V - dB) + _tmpPx1[k] + _tmpPz1[k]);
                    }
                }
            }
        }
    }

    /**
     * Add the Born source for buoyancy model-space at the current time
     *
     * User must have:
     *   - called the nonlinear forward
     *   - saved 2nd time derivative of pressure at corresponding time index in array dp2
     *   - Born source term will be injected into the _pCur array
     *
     * TODO: if these second derivatice call and following loop is expensive, 
     *       could consider fusing the two derivative loops with the final loop
     */
template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void forwardBornInjection_B(Type *dBuoy, Type *wavefieldP, Type *wavefieldDP) {

        applyFirstDerivatives2D_PlusHalf(
            _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz, 
            wavefieldP, wavefieldP, _tmpPx1, _tmpPz1, _nbx, _nbz);

#pragma omp parallel for collapse(2) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long bz = 0; bz < _nz; bz += _nbz) {
                const long kxmax = MIN(bx + _nbx, _nx);
                const long kzmax = MIN(bz + _nbz, _nz);

                for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                    for (long kz = bz; kz < kzmax; kz++) {
                        const long k = kx * _nz + kz;

                        const Type dB = dBuoy[k];

                        _tmpPx2[k] = dB * _tmpPx1[k];
                        _tmpPz2[k] = dB * _tmpPz1[k];
                    }
                }
            }
        }

        applyFirstDerivatives2D_MinusHalf(
            _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz, 
            _tmpPx2, _tmpPz2, _tmpPx1, _tmpPz1, _nbx, _nbz);

#pragma omp parallel for collapse(2) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long bz = 0; bz < _nz; bz += _nbz) {
                const long kxmax = MIN(bx + _nbx, _nx);
                const long kzmax = MIN(bz + _nbz, _nz);

                for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                    for (long kz = bz; kz < kzmax; kz++) {
                        const long k = kx * _nz + kz;

                        const Type V  = _v[k];
                        const Type B  = _b[k];
                        const Type dB = dBuoy[k];

                        const Type V2 = V * V;
                        const Type dt2v2OverB = _dt * _dt * V2 / B;
                        const Type factorB = - dB / V2;

                        _pCur[k] += dt2v2OverB * (wavefieldDP[k] * factorB + _tmpPx1[k] + _tmpPz1[k]);
                    }
                }
            }
        }
    }

    /**
     * Accumulate the Born image term at the current time for velocity only model-space
     *
     * User must have:
     *   - called the nonlinear forward
     *   - saved 2nd time derivative of pressure at corresponding time index in array dp2
     *   - Born image term will be accumulated iu the _dm array
     *
     *   - velocity term: [+ 2B/V^3 LtP r                                 ]
     *   - buoyancy  term: [- 1/V^2  LtP r - dx' P dx - dy' P dy - dz' P dz]
     *
     * TODO: if these adjoint accumulations are expensive, could consider fusing the 
     *       two derivative loops with the final loop
     */
template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void adjointBornAccumulation_V(Type *dVel, Type *wavefieldDP) {

#pragma omp parallel for num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long bz = 0; bz < _nz; bz += _nbz) {
                const long kxmax = MIN(bx + _nbx, _nx);
                const long kzmax = MIN(bz + _nbz, _nz);

                for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                    for (long kz = bz; kz < kzmax; kz++) {
                        const long k = kx * _nz + kz;

                        const Type V = _v[k];
                        const Type B = _b[k];

                        const Type factorV = + 2 * B / (V * V * V);

                        dVel[k]  += (factorV * wavefieldDP[k] * _pOld[k]);
                    }
                }
            }
        }
    }

    /**
     * Accumulate the Born image term at the current time for velocity and buoyancy model-space
     *
     * User must have:
     *   - called the nonlinear forward
     *   - saved 2nd time derivative of pressure at corresponding time index in array dp2
     *   - Born image term will be accumulated iu the _dm array
     *
     *   - velocity term: [+ 2B/V^3 LtP r                                 ]
     *   - buoyancy  term: [- 1/V^2  LtP r - dx' P dx - dy' P dy - dz' P dz]
     */
template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void adjointBornAccumulation_VB(Type *dVel, Type *dBuoy, Type *wavefieldP, Type *wavefieldDP) {

        applyFirstDerivatives2D_PlusHalf(
            _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
            wavefieldP, wavefieldP, _tmpPx1, _tmpPz1, _nbx, _nbz);

        applyFirstDerivatives2D_PlusHalf(
            _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
            _pOld, _pOld, _tmpPx2, _tmpPz2, _nbx, _nbz);

#pragma omp parallel for num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long bz = 0; bz < _nz; bz += _nbz) {
                const long kxmax = MIN(bx + _nbx, _nx);
                const long kzmax = MIN(bz + _nbz, _nz);

                for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                    for (long kz = bz; kz < kzmax; kz++) {
                        const long k = kx * _nz + kz;

                        const Type V = _v[k];
                        const Type B = _b[k];

                        const Type factorV = + 2 * B / (V * V * V);
                        const Type factorB = - 1 / (V * V);

                        dVel[k]  += factorV * wavefieldDP[k] * _pOld[k];
                        dBuoy[k] += factorB * wavefieldDP[k] * _pOld[k]
                            - _tmpPx1[k] * _tmpPx2[k] - _tmpPz1[k] * _tmpPz2[k];
                    }
                }
            }
        }
    }

    /**
     * Accumulate the Born image term at the current time for buoyancy only model-space
     *
     * User must have:
     *   - called the nonlinear forward
     *   - saved 2nd time derivative of pressure at corresponding time index in array dp2
     *   - Born image term will be accumulated iu the _dm array
     *
     *   - velocity term: [+ 2B/V^3 LtP r                                 ]
     *   - buoyancy  term: [- 1/V^2  LtP r - dx' P dx - dy' P dy - dz' P dz]
     */
template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void adjointBornAccumulation_B(Type *dBuoy, Type *wavefieldP, Type *wavefieldDP) {

        applyFirstDerivatives2D_PlusHalf(
            _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
            wavefieldP, wavefieldP, _tmpPx1, _tmpPz1, _nbx, _nbz);

        applyFirstDerivatives2D_PlusHalf(
            _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
            _pOld, _pOld, _tmpPx2, _tmpPz2, _nbx, _nbz);

#pragma omp parallel for num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long bz = 0; bz < _nz; bz += _nbz) {
                const long kxmax = MIN(bx + _nbx, _nx);
                const long kzmax = MIN(bz + _nbz, _nz);

                for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                    for (long kz = bz; kz < kzmax; kz++) {
                        const long k = kx * _nz + kz;

                        const Type V = _v[k];
                        const Type B = _b[k];

                        const Type factorB = - 1 / (V * V);

                        dBuoy[k] += factorB * wavefieldDP[k] * _pOld[k] 
                            - _tmpPx1[k] * _tmpPx2[k] - _tmpPz1[k] * _tmpPz2[k];
                    }
                }
            }
        }
    }

    /**
     * Apply Kz wavenumber filter for up/down wavefield seperation
     * Faqi, 2011, Geophysics https://library.seg.org/doi/full/10.1190/1.3533914
     * 
     * We handle the FWI and RTM imaging conditions with a condition inside the OMP loop
     * 
     * Example Kz filtering with 8 samples 
     * frequency | +0 | +1 | +2 | +3 |  N | -3 | -2 | -1 |
     * original  |  0 |  1 |  2 |  3 |  4 |  5 |  6 |  7 |
     * upgoing   |  0 |  X |  X |  X |  4 |  5 |  6 |  7 |
     * dngoing   |  0 |  1 |  2 |  3 |  4 |  X |  X |  X |
     */
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void adjointBornAccumulation_wavefieldsep(float *dVel, float *wavefieldDP, const long isFWI) {
        const long nfft = 2 * _nz;
        const float scale = 1.0f / (float)(nfft);

        // FWI: adj wavefield is dngoing
        // RTM: adj wavefield is upgoing
        const long kfft_adj = (isFWI) ? 0 : nfft / 2;

        std::complex<float> * __restrict__ tmp = new std::complex<float>[nfft];

        fftwf_plan planForward = fftwf_plan_dft_1d(nfft,
		    reinterpret_cast<fftwf_complex*>(tmp),
		    reinterpret_cast<fftwf_complex*>(tmp), +1, FFTW_ESTIMATE);

		fftwf_plan planInverse = fftwf_plan_dft_1d(nfft,
		    reinterpret_cast<fftwf_complex*>(tmp),
		    reinterpret_cast<fftwf_complex*>(tmp), -1, FFTW_ESTIMATE);

        delete [] tmp;

#pragma omp parallel num_threads(_nthread)
        {
            std::complex<float> * __restrict__ tmp_nlf = new std::complex<float>[nfft];
            std::complex<float> * __restrict__ tmp_adj = new std::complex<float>[nfft];

#pragma omp for schedule(static)
            for (long bx = 0; bx < _nx; bx += _nbx) {
                const long kxmax = MIN(bx + _nbx, _nx);
                for (long kx = bx; kx < kxmax; kx++) {

#pragma omp simd
                    for (long kfft = 0; kfft < nfft; kfft++) {
                        tmp_nlf[kfft] = 0;
                        tmp_adj[kfft] = 0;
                    }  

#pragma omp simd
                    for (long kz = 0; kz < _nz; kz++) {
                        const long k = kx * _nz + kz;
                        tmp_nlf[kz] = scale * wavefieldDP[k];
                        tmp_adj[kz] = scale * _pOld[k];
                    }  

                    fftwf_execute_dft(planForward,
                        reinterpret_cast<fftwf_complex*>(tmp_nlf),
                        reinterpret_cast<fftwf_complex*>(tmp_nlf));

                    fftwf_execute_dft(planForward,
                        reinterpret_cast<fftwf_complex*>(tmp_adj),
                        reinterpret_cast<fftwf_complex*>(tmp_adj));

                    // upgoing: zero the positive frequencies, excluding Nyquist
                    // dngoing: zero the negative frequencies, excluding Nyquist
#pragma omp simd
                    for (long k = 1; k < nfft / 2; k++) {
                        tmp_nlf[nfft / 2 + k] = 0;
                        tmp_adj[kfft_adj + k] = 0;
                    }

                    fftwf_execute_dft(planInverse,
                        reinterpret_cast<fftwf_complex*>(tmp_nlf),
                        reinterpret_cast<fftwf_complex*>(tmp_nlf));

                    fftwf_execute_dft(planInverse,
                        reinterpret_cast<fftwf_complex*>(tmp_adj),
                        reinterpret_cast<fftwf_complex*>(tmp_adj));

                    // Faqi eq 10
                    // Applied to FWI: [Sup * Rdn]
                    // Applied to RTM: [Sup * Rup]
#pragma omp simd
                    for (long kz = 0; kz < _nz; kz++) {
                        const long k = kx * _nz + kz;
                        const float V = _v[k];
                        const float B = _b[k];
                        const float factor = 2 * B / (V * V * V);
                        dVel[k] +=  (factor * real(tmp_nlf[kz] * tmp_adj[kz]));
                    }

                } // end loop over kx
            } // end loop over bx

            delete [] tmp_nlf;
            delete [] tmp_adj;
        } // end parallel region

        fftwf_destroy_plan(planForward);
        fftwf_destroy_plan(planInverse);
    }

// Mixed IC 
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void adjointBornAccumulation_wavefieldsep_mix(float *dVel, float *wavefieldDP, const float weight) {
 
        // Apply standard IC 
        #pragma omp parallel for num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long bz = 0; bz < _nz; bz += _nbz) {
                const long kxmax = MIN(bx + _nbx, _nx);
                const long kzmax = MIN(bz + _nbz, _nz);

                for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                    for (long kz = bz; kz < kzmax; kz++) {
                        const long k = kx * _nz + kz;

                        const Type V = _v[k];
                        const Type B = _b[k];

                        const Type factorV = + 2 * B / (V * V * V);

                        dVel[k]  += (1-abs(weight))* (factorV * wavefieldDP[k] * _pOld[k]);
                    }
                }
            }
        }


        if weight < 0.0 ? isFWI = true : isFWI = false 


        const long nfft = 2 * _nz;
        const float scale = 1.0f / (float)(nfft);

        // FWI: adj wavefield is dngoing
        // RTM: adj wavefield is upgoing
        const long kfft_adj = (isFWI) ? 0 : nfft / 2;

        std::complex<float> * __restrict__ tmp = new std::complex<float>[nfft];

        fftwf_plan planForward = fftwf_plan_dft_1d(nfft,
		    reinterpret_cast<fftwf_complex*>(tmp),
		    reinterpret_cast<fftwf_complex*>(tmp), +1, FFTW_ESTIMATE);

		fftwf_plan planInverse = fftwf_plan_dft_1d(nfft,
		    reinterpret_cast<fftwf_complex*>(tmp),
		    reinterpret_cast<fftwf_complex*>(tmp), -1, FFTW_ESTIMATE);

        delete [] tmp;

#pragma omp parallel num_threads(_nthread)
        {
            std::complex<float> * __restrict__ tmp_nlf = new std::complex<float>[nfft];
            std::complex<float> * __restrict__ tmp_adj = new std::complex<float>[nfft];

#pragma omp for schedule(static)
            for (long bx = 0; bx < _nx; bx += _nbx) {
                const long kxmax = MIN(bx + _nbx, _nx);
                for (long kx = bx; kx < kxmax; kx++) {

#pragma omp simd
                    for (long kfft = 0; kfft < nfft; kfft++) {
                        tmp_nlf[kfft] = 0;
                        tmp_adj[kfft] = 0;
                    }  

#pragma omp simd
                    for (long kz = 0; kz < _nz; kz++) {
                        const long k = kx * _nz + kz;
                        tmp_nlf[kz] = scale * wavefieldDP[k];
                        tmp_adj[kz] = scale * _pOld[k];
                    }  

                    fftwf_execute_dft(planForward,
                        reinterpret_cast<fftwf_complex*>(tmp_nlf),
                        reinterpret_cast<fftwf_complex*>(tmp_nlf));

                    fftwf_execute_dft(planForward,
                        reinterpret_cast<fftwf_complex*>(tmp_adj),
                        reinterpret_cast<fftwf_complex*>(tmp_adj));

                    // upgoing: zero the positive frequencies, excluding Nyquist
                    // dngoing: zero the negative frequencies, excluding Nyquist
#pragma omp simd
                    for (long k = 1; k < nfft / 2; k++) {
                        tmp_nlf[nfft / 2 + k] = 0;
                        tmp_adj[kfft_adj + k] = 0;
                    }

                    fftwf_execute_dft(planInverse,
                        reinterpret_cast<fftwf_complex*>(tmp_nlf),
                        reinterpret_cast<fftwf_complex*>(tmp_nlf));

                    fftwf_execute_dft(planInverse,
                        reinterpret_cast<fftwf_complex*>(tmp_adj),
                        reinterpret_cast<fftwf_complex*>(tmp_adj));

                    // Faqi eq 10
                    // Applied to FWI: [Sup * Rdn]
                    // Applied to RTM: [Sup * Rup]
#pragma omp simd
                    for (long kz = 0; kz < _nz; kz++) {
                        const long k = kx * _nz + kz;
                        const float V = _v[k];
                        const float B = _b[k];
                        const float factor = 2 * B / (V * V * V);
                        dVel[k] += abs(weight) * (factor * real(tmp_nlf[kz] * tmp_adj[kz]));
                    }

                } // end loop over kx
            } // end loop over bx

            delete [] tmp_nlf;
            delete [] tmp_adj;
        } // end parallel region

        fftwf_destroy_plan(planForward);
        fftwf_destroy_plan(planInverse);
    }

template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline static void applyFirstDerivatives2D_PlusHalf_Sandwich(
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
            const Type * __restrict__ const inPX,
            const Type * __restrict__ const inPZ,
            const Type * __restrict__ const fieldBuoy,
            Type * __restrict__ tmpPX,
            Type * __restrict__ tmpPZ,
            const long BX_2D,
            const long BZ_2D) {

        // zero output arrays
#pragma omp parallel for collapse(2) num_threads(nthread) schedule(static)
        for (long bx = 0; bx < nx; bx += BX_2D) {
            for (long bz = 0; bz < nz; bz += BZ_2D) {
                const long kxmax = MIN(bx + BX_2D - 1, nx);
                const long kzmax = MIN(bz + BZ_2D - 1, nz);

                for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                    for (long kz = bz; kz < kzmax; kz++) {
                        long k = kx * nz + kz;
                        tmpPX[k] = 0;
                        tmpPZ[k] = 0;
                    }
                }
            }
        }

        const long nx4 = nx - 4;
        const long nz4 = nz - 4;

        // interior
#pragma omp parallel for collapse(2) num_threads(nthread) schedule(static)
        for (long bx = 4; bx < nx4; bx += BX_2D) {
            for (long bz = 4; bz < nz4; bz += BZ_2D) { /* cache blocking */

                const long kxmax = MIN(bx + BX_2D, nx4);
                const long kzmax = MIN(bz + BZ_2D, nz4);

                for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                    for (long kz = bz; kz < kzmax; kz++) {
                        const long kxnz = kx * nz;
                        const long k = kxnz + kz;

                        const Type stencilDPx =
                                c8_1 * (- inPX[(kx+0) * nz + kz] + inPX[(kx+1) * nz + kz]) +
                                c8_2 * (- inPX[(kx-1) * nz + kz] + inPX[(kx+2) * nz + kz]) +
                                c8_3 * (- inPX[(kx-2) * nz + kz] + inPX[(kx+3) * nz + kz]) +
                                c8_4 * (- inPX[(kx-3) * nz + kz] + inPX[(kx+4) * nz + kz]);

                        const Type stencilDPz =
                                c8_1 * (- inPZ[kxnz + (kz+0)] + inPZ[kxnz + (kz+1)]) +
                                c8_2 * (- inPZ[kxnz + (kz-1)] + inPZ[kxnz + (kz+2)]) +
                                c8_3 * (- inPZ[kxnz + (kz-2)] + inPZ[kxnz + (kz+3)]) +
                                c8_4 * (- inPZ[kxnz + (kz-3)] + inPZ[kxnz + (kz+4)]);

                        const Type dPx = invDx * stencilDPx;
                        const Type dPz = invDz * stencilDPz;

                        const Type B = fieldBuoy[k];

                        tmpPX[k] = B * dPx;
                        tmpPZ[k] = B * dPz;
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
                            c8_1 * (- inPZ[kx * nz + 0] + inPZ[kx * nz + 1]) +
                            c8_2 * (+ inPZ[kx * nz + 1] + inPZ[kx * nz + 2]) +
                            c8_3 * (+ inPZ[kx * nz + 2] + inPZ[kx * nz + 3]) +
                            c8_4 * (+ inPZ[kx * nz + 3] + inPZ[kx * nz + 4]);

                    const Type dPx = 0;
                    const Type dPz = invDz * stencilDPz0;

                    const long k = kx * nz + 0;

                    const Type B = fieldBuoy[k];

                    tmpPX[k] = B * dPx;
                    tmpPZ[k] = B * dPz;
                }

                // kz = 1 -- 1 1/2 cells below free surface for Z derivative, 1 cells below for X/Y derivative
                // [kx * nz + 1]
                {
                    const Type stencilDPx1 =
                            c8_1 * (- inPX[(kx+0) * nz + 1] + inPX[(kx+1) * nz + 1]) +
                            c8_2 * (- inPX[(kx-1) * nz + 1] + inPX[(kx+2) * nz + 1]) +
                            c8_3 * (- inPX[(kx-2) * nz + 1] + inPX[(kx+3) * nz + 1]) +
                            c8_4 * (- inPX[(kx-3) * nz + 1] + inPX[(kx+4) * nz + 1]);

                    const Type stencilDPz1 =
                            c8_1 * (- inPZ[kx * nz + 1] + inPZ[kx * nz + 2]) +
                            c8_2 * (- inPZ[kx * nz + 0] + inPZ[kx * nz + 3]) +
                            c8_3 * (+ inPZ[kx * nz + 1] + inPZ[kx * nz + 4]) +
                            c8_4 * (+ inPZ[kx * nz + 2] + inPZ[kx * nz + 5]);

                    const Type dPx = invDx * stencilDPx1;
                    const Type dPz = invDz * stencilDPz1;

                    const long k = kx * nz + 1;

                    const Type B = fieldBuoy[k];

                    tmpPX[k] = B * dPx;
                    tmpPZ[k] = B * dPz;
                }

                // kz = 2 -- 2 1/2 cells below free surface for Z derivative, 2 cells below for X/Y derivative
                // [kx * nz + 2]
                {
                    const Type stencilDPx2 =
                            c8_1 * (- inPX[(kx+0) * nz + 2] + inPX[(kx+1) * nz + 2]) +
                            c8_2 * (- inPX[(kx-1) * nz + 2] + inPX[(kx+2) * nz + 2]) +
                            c8_3 * (- inPX[(kx-2) * nz + 2] + inPX[(kx+3) * nz + 2]) +
                            c8_4 * (- inPX[(kx-3) * nz + 2] + inPX[(kx+4) * nz + 2]);

                    const Type stencilDPz2 =
                            c8_1 * (- inPZ[kx * nz + 2] + inPZ[kx * nz + 3]) +
                            c8_2 * (- inPZ[kx * nz + 1] + inPZ[kx * nz + 4]) +
                            c8_3 * (- inPZ[kx * nz + 0] + inPZ[kx * nz + 5]) +
                            c8_4 * (+ inPZ[kx * nz + 1] + inPZ[kx * nz + 6]);

                    const Type dPx = invDx * stencilDPx2;
                    const Type dPz = invDz * stencilDPz2;

                    const long k = kx * nz + 2;

                    const Type B = fieldBuoy[k];

                    tmpPX[k] = B * dPx;
                    tmpPZ[k] = B * dPz;
                }

                // kz = 3 -- 3 1/2 cells below free surface for Z derivative, 3 cells below for X/Y derivative
                // [kx * nz + 3]
                {
                    const Type stencilDPx3 =
                            c8_1 * (- inPX[(kx+0) * nz + 3] + inPX[(kx+1) * nz + 3]) +
                            c8_2 * (- inPX[(kx-1) * nz + 3] + inPX[(kx+2) * nz + 3]) +
                            c8_3 * (- inPX[(kx-2) * nz + 3] + inPX[(kx+3) * nz + 3]) +
                            c8_4 * (- inPX[(kx-3) * nz + 3] + inPX[(kx+4) * nz + 3]);

                    const Type stencilDPz3 =
                            c8_1 * (- inPZ[kx * nz + 3] + inPZ[kx * nz + 4]) +
                            c8_2 * (- inPZ[kx * nz + 2] + inPZ[kx * nz + 5]) +
                            c8_3 * (- inPZ[kx * nz + 1] + inPZ[kx * nz + 6]) +
                            c8_4 * (- inPZ[kx * nz + 0] + inPZ[kx * nz + 7]);

                    const Type dPx = invDx * stencilDPx3;
                    const Type dPz = invDz * stencilDPz3;

                    const long k = kx * nz + 3;

                    const Type B = fieldBuoy[k];

                    tmpPX[k] = B * dPx;
                    tmpPZ[k] = B * dPz;
                }
            }
        }
    }

    template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline static void applyFirstDerivatives2D_MinusHalf_TimeUpdate_Nonlinear(
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
            const Type dtMod,
            const Type * __restrict__ const tmpPX,
            const Type * __restrict__ const tmpPZ,
            const Type * __restrict__ const fieldVel,
            const Type * __restrict__ const fieldBuoy,
            const Type * __restrict__ const dtOmegaInvQ,
            Type * __restrict__ pCur,
            Type * __restrict__ pSpace,
            Type * __restrict__ pOld,
            const long BX_2D,
            const long BZ_2D) {

        const long nx4 = nx - 4;
        const long nz4 = nz - 4;

        const Type dt2 = dtMod * dtMod;

        // zero output arrays
#pragma omp parallel for collapse(2) num_threads(nthread) schedule(static)
        for (long bx = 0; bx < nx; bx += BX_2D) {
            for (long bz = 0; bz < nz; bz += BZ_2D) {
                const long kxmax = MIN(bx + BX_2D, nx);
                const long kzmax = MIN(bz + BZ_2D, nz);

                for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                    for (long kz = bz; kz < kzmax; kz++) {
                        long k = kx * nz + kz;
                        pSpace[k] = 0;
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

                        const Type stencilDPx =
                                c8_1 * (- tmpPX[(kx-1) * nz + kz] + tmpPX[(kx+0) * nz + kz]) +
                                c8_2 * (- tmpPX[(kx-2) * nz + kz] + tmpPX[(kx+1) * nz + kz]) +
                                c8_3 * (- tmpPX[(kx-3) * nz + kz] + tmpPX[(kx+2) * nz + kz]) +
                                c8_4 * (- tmpPX[(kx-4) * nz + kz] + tmpPX[(kx+3) * nz + kz]);

                        const Type stencilDPz =
                                c8_1 * (- tmpPZ[kx * nz + (kz-1)] + tmpPZ[kx * nz + (kz+0)]) +
                                c8_2 * (- tmpPZ[kx * nz + (kz-2)] + tmpPZ[kx * nz + (kz+1)]) +
                                c8_3 * (- tmpPZ[kx * nz + (kz-3)] + tmpPZ[kx * nz + (kz+2)]) +
                                c8_4 * (- tmpPZ[kx * nz + (kz-4)] + tmpPZ[kx * nz + (kz+3)]);

                        const Type dPX = invDx * stencilDPx;
                        const Type dPZ = invDz * stencilDPz;

                        const long k = kx * nz + kz;

                        const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                        pOld[k] = dt2V2_B * (dPX + dPZ) - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                        pSpace[k] = dPX + dPZ;
                    }
                }
            }
        }

        // roll on free surface
        if (freeSurface) {
#pragma omp parallel for num_threads(nthread) schedule(guided)
            for (long kx = 4; kx < nx4; kx++) {

                // kz = 0 -- at the free surface -- p = 0
                // [kx * nz + 0]
                {
                    const Type dPX = 0;
                    const Type dPZ = 0;

                    const long k = kx * nz + 0;
                    const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                    pOld[k] = dt2V2_B * (dPX + dPZ) - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                    pSpace[k] = dPX + dPZ;
                }

                // kz = 1 -- one cell below the free surface
                // [kx * nz + 1]
                {
                    const Type stencilDPx1 =
                            c8_1 * (- tmpPX[(kx-1) * nz + 1] + tmpPX[(kx+0) * nz + 1]) +
                            c8_2 * (- tmpPX[(kx-2) * nz + 1] + tmpPX[(kx+1) * nz + 1]) +
                            c8_3 * (- tmpPX[(kx-3) * nz + 1] + tmpPX[(kx+2) * nz + 1]) +
                            c8_4 * (- tmpPX[(kx-4) * nz + 1] + tmpPX[(kx+3) * nz + 1]);

                    const Type stencilDPz1 =
                            c8_1 * (- tmpPZ[kx * nz + 0] + tmpPZ[kx * nz + 1]) +
                            c8_2 * (- tmpPZ[kx * nz + 0] + tmpPZ[kx * nz + 2]) +
                            c8_3 * (- tmpPZ[kx * nz + 1] + tmpPZ[kx * nz + 3]) +
                            c8_4 * (- tmpPZ[kx * nz + 2] + tmpPZ[kx * nz + 4]);

                    const Type dPx = invDx * stencilDPx1;
                    const Type dPz = invDz * stencilDPz1;

                    const long k = kx * nz + 1;
                    const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                    pOld[k] = dt2V2_B * (dPx + dPz) - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                    pSpace[k] = dPx + dPz;
                }

                // kz = 2 -- two cells below the free surface
                // [kx * nz + 2]
                {
                    const Type stencilDPx2 =
                            c8_1 * (- tmpPX[(kx-1) * nz + 2] + tmpPX[(kx+0) * nz + 2]) +
                            c8_2 * (- tmpPX[(kx-2) * nz + 2] + tmpPX[(kx+1) * nz + 2]) +
                            c8_3 * (- tmpPX[(kx-3) * nz + 2] + tmpPX[(kx+2) * nz + 2]) +
                            c8_4 * (- tmpPX[(kx-4) * nz + 2] + tmpPX[(kx+3) * nz + 2]);

                    const Type stencilDPz2 =
                            c8_1 * (- tmpPZ[kx * nz + 1] + tmpPZ[kx * nz + 2]) +
                            c8_2 * (- tmpPZ[kx * nz + 0] + tmpPZ[kx * nz + 3]) +
                            c8_3 * (- tmpPZ[kx * nz + 0] + tmpPZ[kx * nz + 4]) +
                            c8_4 * (- tmpPZ[kx * nz + 1] + tmpPZ[kx * nz + 5]);

                    const Type dPx = invDx * stencilDPx2;
                    const Type dPz = invDz * stencilDPz2;

                    const long k = kx * nz + 2;
                    const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                    pOld[k] = dt2V2_B * (dPx + dPz) - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                    pSpace[k] = dPx + dPz;
                }

                // kz = 3 -- three cells below the free surface
                // [kx * nz + 3]
                {
                    const Type stencilDPx3 =
                            c8_1 * (- tmpPX[(kx-1) * nz + 3] + tmpPX[(kx+0) * nz + 3]) +
                            c8_2 * (- tmpPX[(kx-2) * nz + 3] + tmpPX[(kx+1) * nz + 3]) +
                            c8_3 * (- tmpPX[(kx-3) * nz + 3] + tmpPX[(kx+2) * nz + 3]) +
                            c8_4 * (- tmpPX[(kx-4) * nz + 3] + tmpPX[(kx+3) * nz + 3]);

                    const Type stencilDPz3 =
                            c8_1 * (- tmpPZ[kx * nz + 2] + tmpPZ[kx * nz + 3]) +
                            c8_2 * (- tmpPZ[kx * nz + 1] + tmpPZ[kx * nz + 4]) +
                            c8_3 * (- tmpPZ[kx * nz + 0] + tmpPZ[kx * nz + 5]) +
                            c8_4 * (- tmpPZ[kx * nz + 0] + tmpPZ[kx * nz + 6]);

                    const Type dPx = invDx * stencilDPx3;
                    const Type dPz = invDz * stencilDPz3;

                    const long k = kx * nz + 3;
                    const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                    pOld[k] = dt2V2_B * (dPx + dPz) - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                    pSpace[k] = dPx + dPz;
                }
            }
        }
    }

};

#endif
