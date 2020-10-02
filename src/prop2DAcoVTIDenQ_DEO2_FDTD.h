#ifndef PROP2DACOVTIDENQ_DEO2_FDTD_H
#define PROP2DACOVTIDENQ_DEO2_FDTD_H

#include <omp.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define MIN(x,y) ((x)<(y)?(x):(y))

class Prop2DAcoVTIDenQ_DEO2_FDTD {

public:
    const bool _freeSurface;
    const long _nbx, _nbz, _nthread, _nx, _nz, _nsponge;
    const float _dx, _dz, _dt;
    const float _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz;
    const float _fDefault = 0.85f;

    float * __restrict__ _v = NULL;
    float * __restrict__ _eps = NULL;
    float * __restrict__ _eta = NULL;
    float * __restrict__ _b = NULL;
    float * __restrict__ _f = NULL;
    float * __restrict__ _dtOmegaInvQ = NULL;
    float * __restrict__ _pSpace = NULL;
    float * __restrict__ _mSpace = NULL;
    float * __restrict__ _tmpPx1 = NULL;
    float * __restrict__ _tmpPz1 = NULL;
    float * __restrict__ _tmpMx1 = NULL;
    float * __restrict__ _tmpMz1 = NULL;
    float * __restrict__ _tmpPx2 = NULL;
    float * __restrict__ _tmpPz2 = NULL;
    float * __restrict__ _tmpMx2 = NULL;
    float * __restrict__ _tmpMz2 = NULL;
    float * _pOld = NULL;
    float * _pCur = NULL;
    float * _mOld = NULL;
    float * _mCur = NULL;

    Prop2DAcoVTIDenQ_DEO2_FDTD(
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
        _eps         = new float[_nx * _nz];
        _eta         = new float[_nx * _nz];
        _b           = new float[_nx * _nz];
        _f           = new float[_nx * _nz];
        _dtOmegaInvQ = new float[_nx * _nz];
        _pSpace      = new float[_nx * _nz];
        _mSpace      = new float[_nx * _nz];
        _tmpPx1      = new float[_nx * _nz];
        _tmpPz1      = new float[_nx * _nz];
        _tmpMx1      = new float[_nx * _nz];
        _tmpMz1      = new float[_nx * _nz];
        _tmpPx2      = new float[_nx * _nz];
        _tmpPz2      = new float[_nx * _nz];
        _tmpMx2      = new float[_nx * _nz];
        _tmpMz2      = new float[_nx * _nz];
        _pOld        = new float[_nx * _nz];
        _pCur        = new float[_nx * _nz];
        _mOld        = new float[_nx * _nz];
        _mCur        = new float[_nx * _nz];

        numaFirstTouch(_nx, _nz, _nthread, _v, _eps, _eta, _b,
            _f, _dtOmegaInvQ, _pSpace, _mSpace, _tmpPx1,
            _tmpPz1, _tmpMx1, _tmpMz1, _tmpPx2, _tmpPz2,
            _tmpMx2, _tmpMz2, _pOld, _pCur, _mOld, _mCur, _nbx, _nbz);
    }

#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void numaFirstTouch(
            const long nx,
            const long nz,
            const long nthread,
            float * __restrict__ v,
            float * __restrict__ eps,
            float * __restrict__ eta,
            float * __restrict__ b,
            float * __restrict__ f,
            float * __restrict__ dtOmegaInvQ,
            float * __restrict__ pSpace,
            float * __restrict__ mSpace,
            float * __restrict__ tmpPx1,
            float * __restrict__ tmpPz1,
            float * __restrict__ tmpMx1,
            float * __restrict__ tmpMz1,
            float * __restrict__ tmpPx2,
            float * __restrict__ tmpPz2,
            float * __restrict__ tmpMx2,
            float * __restrict__ tmpMz2,
            float * __restrict__ pOld,
            float * __restrict__ pCur,
            float * __restrict__ mOld,
            float * __restrict__ mCur,
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
                        const long k = kx * nz + kz;

                        v[k] = 0;
                        eps[k] = 0;
                        eta[k] = 0;
                        b[k] = 0;
                        f[k] = 0;
                        dtOmegaInvQ[k] = 0;
                        pSpace[k] = 0;
                        mSpace[k] = 0;
                        tmpPx1[k] = 0;
                        tmpPz1[k] = 0;
                        tmpMx1[k] = 0;
                        tmpMz1[k] = 0;
                        tmpPx2[k] = 0;
                        tmpPz2[k] = 0;
                        tmpMx2[k] = 0;
                        tmpMz2[k] = 0;
                        pOld[k] = 0;
                        pCur[k] = 0;
                        mOld[k] = 0;
                        mCur[k] = 0;
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
                v[k] = eps[k] = eta[k] = b[k] = f[k] = dtOmegaInvQ[k] = pSpace[k] = mSpace[k] =
                    tmpPx1[k] = tmpPz1[k] = tmpMx1[k] = tmpMz1[k] = tmpPx2[k] = tmpPz2[k] =
                    tmpMx2[k] = tmpMz2[k] = pOld[k] = pCur[k] = mOld[k] = mCur[k] = 0;
            }
        }
#pragma omp parallel for num_threads(nthread) schedule(static)
        for (long kz = nz4; kz < nz; kz++) {
#pragma omp simd
            for (long kx = 0; kx < nx; kx++) {
                const long k = kx * _nz + kz;
                v[k] = eps[k] = eta[k] = b[k] = f[k] = dtOmegaInvQ[k] = pSpace[k] = mSpace[k] =
                    tmpPx1[k] = tmpPz1[k] = tmpMx1[k] = tmpMz1[k] = tmpPx2[k] = tmpPz2[k] =
                    tmpMx2[k] = tmpMz2[k] = pOld[k] = pCur[k] = mOld[k] = mCur[k] = 0;
            }
        }

#pragma omp parallel for num_threads(nthread) schedule(static)
        for (long kx = 0; kx < 4; kx++) {
#pragma omp simd
            for (long kz = 0; kz < nz; kz++) {
                const long k = kx * _nz + kz;
                v[k] = eps[k] = eta[k] = b[k] = f[k] = dtOmegaInvQ[k] = pSpace[k] = mSpace[k] =
                    tmpPx1[k] = tmpPz1[k] = tmpMx1[k] = tmpMz1[k] = tmpPx2[k] = tmpPz2[k] =
                    tmpMx2[k] = tmpMz2[k] = pOld[k] = pCur[k] = mOld[k] = mCur[k] = 0;
            }
        }
#pragma omp parallel for num_threads(nthread) schedule(static)
        for (long kx = nx4; kx < nx; kx++) {
#pragma omp simd
            for (long kz = 0; kz < nz; kz++) {
                const long k = kx * _nz + kz;
                v[k] = eps[k] = eta[k] = b[k] = f[k] = dtOmegaInvQ[k] = pSpace[k] = mSpace[k] =
                    tmpPx1[k] = tmpPz1[k] = tmpMx1[k] = tmpMz1[k] = tmpPx2[k] = tmpPz2[k] =
                    tmpMx2[k] = tmpMz2[k] = pOld[k] = pCur[k] = mOld[k] = mCur[k] = 0;
            }
        }
    }

    ~Prop2DAcoVTIDenQ_DEO2_FDTD() {
        if (_v != NULL) delete [] _v;
        if (_eps != NULL) delete [] _eps;
        if (_eta != NULL) delete [] _eta;
        if (_b != NULL) delete [] _b;
        if (_f != NULL) delete [] _f;
        if (_dtOmegaInvQ != NULL) delete [] _dtOmegaInvQ;
        if (_pSpace != NULL) delete [] _pSpace;
        if (_mSpace != NULL) delete [] _mSpace;
        if (_tmpPx1 != NULL) delete [] _tmpPx1;
        if (_tmpPz1 != NULL) delete [] _tmpPz1;
        if (_tmpMx1 != NULL) delete [] _tmpMx1;
        if (_tmpMz1 != NULL) delete [] _tmpMz1;
        if (_tmpPx2 != NULL) delete [] _tmpPx2;
        if (_tmpPz2 != NULL) delete [] _tmpPz2;
        if (_tmpMx2 != NULL) delete [] _tmpMx2;
        if (_tmpMz2 != NULL) delete [] _tmpMz2;
        if (_pOld != NULL) delete [] _pOld;
        if (_pCur != NULL) delete [] _pCur;
        if (_mOld != NULL) delete [] _mOld;
        if (_mCur != NULL) delete [] _mCur;
    }

#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    void info() {
        printf("\n");
        printf("Prop2DAcoVTIDenQ_DEO2_FDTD\n");
        printf("  nx,nz;              %5ld %5ld\n", _nx, _nz);
        printf("  nthread,nsponge,fs; %5ld %5ld %5d\n", _nthread, _nsponge, _freeSurface);
        printf("  X min,max,inc;    %+16.8f %+16.8f %+16.8f\n", 0.0, _dx * (_nx - 1), _dx);
        printf("  Z min,max,inc;    %+16.8f %+16.8f %+16.8f\n", 0.0, _dz * (_nz - 1), _dz);
    }

    /**
     * Notes
     * - User must have called setupDtOmegaInvQ_2D to initialize the array _dtOmegaInvQ
     * - wavefield arrays are switched in this call
     *     pCur -> pOld
     *     pOld -> pCur
     *     mCur -> mOld
     *     mOld -> mCur
     */
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void timeStep() {

        applyFirstDerivatives2D_PlusHalf_Sandwich(
                _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
                _pCur, _pCur, _mCur, _mCur, _eps, _eta, _f, _b,
                _tmpPx1, _tmpPz1, _tmpMx1, _tmpMz1, _nbx, _nbz);

        applyFirstDerivatives2D_MinusHalf_TimeUpdate_Nonlinear(
                _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz, _dt,
                _tmpPx1, _tmpPz1, _tmpMx1, _tmpMz1, _v, _b, _dtOmegaInvQ, _pCur, _mCur,
                _pSpace, _mSpace, _pOld, _mOld, _nbx, _nbz);

        // swap pointers to be consistent with mjolnir kernel
        float *pswap = _pOld;
        _pOld = _pCur;
        _pCur = pswap;

        float *mswap = _mOld;
        _mOld = _mCur;
        _mCur = mswap;
    }

    /**
     * Same as above, but does not collect the spatial derivatives
     * Note this is only used in the PSD operators, where the first (transient) time steps do
     *   not need to save the P'' term
     */
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void timeStepLinear() {

        applyFirstDerivatives2D_PlusHalf_Sandwich(
                _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
                _pCur, _pCur, _mCur, _mCur, _eps, _eta, _f, _b,
                _tmpPx1, _tmpPz1, _tmpMx1, _tmpMz1, _nbx, _nbz);

        applyFirstDerivatives2D_MinusHalf_TimeUpdate_Linear(
                _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz, _dt,
                _tmpPx1, _tmpPz1, _tmpMx1, _tmpMz1, _v, _b, _dtOmegaInvQ, _pCur, _mCur, _pOld, _mOld, _nbx, _nbz);

        // swap pointers to be consistent with mjolnir kernel
        float *pswap = _pOld;
        _pOld = _pCur;
        _pCur = pswap;

        float *mswap = _mOld;
        _mOld = _mCur;
        _mCur = mswap;
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
                        _mSpace[k] *= v2OverB;
                    }
                }
            }
        }
    }

    /**
     * Add the Born source at the current time
     *
     * Note: "dmodel" is the three components of the model consecutively [vel,eps,eta]
     *
     * User must have:
     *   - called the nonlinear forward
     *   - saved 2nd time derivative of pressure at corresponding time index in array dp2
     *   - Born source term will be injected into the _pCur array
     */
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
     inline void forwardBornInjection_V(
            float *dV,
            float *wavefieldDP, float *wavefieldDM) {
#pragma omp parallel for collapse(2) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long bz = 0; bz < _nz; bz += _nbz) {
                const long kxmax = MIN(bx + _nbx, _nx);
                const long kzmax = MIN(bz + _nbz, _nz);

                for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                    for (long kz = bz; kz < kzmax; kz++) {
                        const long k = kx * _nz + kz;

                        const float V  = _v[k];
                        const float B  = _b[k];
                        const float _dV = dV[k];

                        // V^2/b factor to "clear" the b/V^2 factor on L_tP and L_tM
                        // _dt^2 factor is from the finite difference approximation
                        // 2B_dV/V^3 factor is from the linearization
                        const float factor = 2 * _dt * _dt * _dV / V;

                        _pCur[k] += factor * wavefieldDP[k];
                        _mCur[k] += factor * wavefieldDM[k];
                    }
                }
            }
        }
     }

#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void forwardBornInjection_VEA(
            float *dV, float *dE, float *dA, /* vel, epsilon, eta */
            float *wavefieldP, float *wavefieldM, float *wavefieldDP, float *wavefieldDM) {

        // Right side spatial derivatives for the Born source
        applyFirstDerivatives2D_PlusHalf(
                _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
                wavefieldP, wavefieldP, _tmpPx1, _tmpPz1, _nbx, _nbz);

        applyFirstDerivatives2D_PlusHalf(
                _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
                wavefieldM, wavefieldM, _tmpMx1, _tmpMz1, _nbx, _nbz);

        // Sandwich terms for the Born source
        // note flipped sign for Z derivative term between P and M
#pragma omp parallel for collapse(2) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long bz = 0; bz < _nz; bz += _nbz) {
                const long kxmax = MIN(bx + _nbx, _nx);
                const long kzmax = MIN(bz + _nbz, _nz);

                for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                    for (long kz = bz; kz < kzmax; kz++) {

                        const long k = kx * _nz + kz;

                        const float V  = _v[k];
                        const float E  = _eps[k];
                        const float A  = _eta[k];
                        const float B  = _b[k];

                        const float _dA = dA[k];
                        const float _dE = dE[k];

                        const float F  = _f[k];

                        _tmpPx2[k] = (+2 * B * _dE) *_tmpPx1[k];
                        _tmpPz2[k] = (-2 * B * F * A * _dA) *_tmpPz1[k] +
                                (_dA * B * F * (1 - 2 * A * A) / sqrt(1 - A * A)) * _tmpMz1[k];

                        _tmpMx2[k] = 0;
                        _tmpMz2[k] = (+2 * B * F * A * _dA) *_tmpMz1[k] +
                                (_dA * B * F * (1 - 2 * A * A) / sqrt(1 - A * A)) * _tmpPz1[k];
                    }
                }
            }
        }

        // Left side spatial derivatives for the Born source
        applyFirstDerivatives2D_MinusHalf(
                _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
                _tmpPx2, _tmpPz2, _tmpPx1, _tmpPz1, _nbx, _nbz);

        applyFirstDerivatives2D_MinusHalf(
                _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
                _tmpMx2, _tmpMz2, _tmpMx1, _tmpMz1, _nbx, _nbz);

        // add the born source at the current time
#pragma omp parallel for collapse(2) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long bz = 0; bz < _nz; bz += _nbz) {
                const long kxmax = MIN(bx + _nbx, _nx);
                const long kzmax = MIN(bz + _nbz, _nz);

                for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                    for (long kz = bz; kz < kzmax; kz++) {
                        const long k = kx * _nz + kz;

                        const float V  = _v[k];
                        const float B  = _b[k];
                        const float _dV = dV[k];

                        const float dt2v2OverB = _dt * _dt * V * V / B;

                        const float factor = 2 * B * _dV / (V * V * V);

                        _pCur[k] += dt2v2OverB * (factor * wavefieldDP[k] + _tmpPx1[k] + _tmpPz1[k]);
                        _mCur[k] += dt2v2OverB * (factor * wavefieldDM[k] + _tmpMx1[k] + _tmpMz1[k]);
                    }
                }
            }
        }
    }

    /**
     * Accumulate the Born image term at the current time
     *
     * User must have:
     *   - called the nonlinear forward
     *   - saved 2nd time derivative of pressure at corresponding time index in array dp2
     *   - Born image term will be accumulated iu the _dm array
     */
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
     inline void adjointBornAccumulation_V(float *dV,
             float *wavefieldDP, float *wavefieldDM) {
 #pragma omp parallel for collapse(2) num_threads(_nthread) schedule(static)
         for (long bx = 0; bx < _nx; bx += _nbx) {
             for (long bz = 0; bz < _nz; bz += _nbz) {
                 const long kxmax = MIN(bx + _nbx, _nx);
                 const long kzmax = MIN(bz + _nbz, _nz);

                 for (long kx = bx; kx < kxmax; kx++) {
 #pragma omp simd
                     for (long kz = bz; kz < kzmax; kz++) {
                         const long k = kx * _nz + kz;

                         const float V = _v[k];
                         const float B = _b[k];

                         const float factor = 2 * B / (V * V * V);

                         dV[k] += factor * (wavefieldDP[k] * _pOld[k] + wavefieldDM[k] * _mOld[k]);
                     }
                 }
             }
         }
     }

#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void adjointBornAccumulation_VEA(float *dV, float *dE, float *dA, /* vel, epsilon, eta */
            float *wavefieldP, float *wavefieldM, float *wavefieldDP, float *wavefieldDM) {

        // Right side spatial derivatives for the adjoint accumulation
        applyFirstDerivatives2D_PlusHalf(
                _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
                wavefieldP, wavefieldP, _tmpPx1, _tmpPz1, _nbx, _nbz);

        applyFirstDerivatives2D_PlusHalf(
                _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
                wavefieldM, wavefieldM, _tmpMx1, _tmpMz1, _nbx, _nbz);

        applyFirstDerivatives2D_PlusHalf(
                _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
                _pOld, _pOld, _tmpPx2, _tmpPz2, _nbx, _nbz);

        applyFirstDerivatives2D_PlusHalf(
                _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
                _mOld, _mOld, _tmpMx2, _tmpMz2, _nbx, _nbz);

        // Sandwich terms for the adjoint accumulation
#pragma omp parallel for collapse(2) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long bz = 0; bz < _nz; bz += _nbz) {
                const long kxmax = MIN(bx + _nbx, _nx);
                const long kzmax = MIN(bz + _nbz, _nz);

                for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                    for (long kz = bz; kz < kzmax; kz++) {
                        const long k = kx * _nz + kz;

                        const float V = _v[k];
                        const float E = _eps[k];
                        const float A = _eta[k];
                        const float B = _b[k];
                        const float F = _f[k];

                        const float factor = 2 * B / (V * V * V);

                        dV[k] +=
                                factor * wavefieldDP[k] * _pOld[k] +
                                factor * wavefieldDM[k] * _mOld[k];

                        dE[k] += -2 * B * _tmpPx1[k] * _tmpPx2[k];

                        const float partP =
                                2 * B * F * A * _tmpPz1[k] - (B * F * (1 - 2 * A * A) / sqrt(1 - A * A)) * _tmpMz1[k];

                        const float partM =
                                2 * B * F * A * _tmpMz1[k] + (B * F * (1 - 2 * A * A) / sqrt(1 - A * A)) * _tmpPz1[k];

                        dA[k] += partP * _tmpPz2[k] - partM * _tmpMz2[k];
                    }
                }
            }
        }
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
            const Type * __restrict__ const inMX,
            const Type * __restrict__ const inMZ,
            const Type * __restrict__ const fieldEps,
            const Type * __restrict__ const fieldEta,
            const Type * __restrict__ const fieldVsVp,
            const Type * __restrict__ const fieldBuoy,
            Type * __restrict__ tmpPX,
            Type * __restrict__ tmpPZ,
            Type * __restrict__ tmpMX,
            Type * __restrict__ tmpMZ,
            const long BX_2D,
            const long BZ_2D) {

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
                        tmpPX[k] = 0;
                        tmpPZ[k] = 0;
                        tmpMX[k] = 0;
                        tmpMZ[k] = 0;
                    }
                }
            }
        }

        const long nx4 = nx - 4;
        const long nz4 = nz - 4;

        // interior
#pragma omp parallel for collapse(2) num_threads(nthread) schedule(guided)
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

                        const Type stencilDMx =
                                c8_1 * (- inMX[(kx+0) * nz + kz] + inMX[(kx+1) * nz + kz]) +
                                c8_2 * (- inMX[(kx-1) * nz + kz] + inMX[(kx+2) * nz + kz]) +
                                c8_3 * (- inMX[(kx-2) * nz + kz] + inMX[(kx+3) * nz + kz]) +
                                c8_4 * (- inMX[(kx-3) * nz + kz] + inMX[(kx+4) * nz + kz]);

                        const Type stencilDMz =
                                c8_1 * (- inMZ[kxnz + (kz+0)] + inMZ[kxnz + (kz+1)]) +
                                c8_2 * (- inMZ[kxnz + (kz-1)] + inMZ[kxnz + (kz+2)]) +
                                c8_3 * (- inMZ[kxnz + (kz-2)] + inMZ[kxnz + (kz+3)]) +
                                c8_4 * (- inMZ[kxnz + (kz-3)] + inMZ[kxnz + (kz+4)]);

                        const Type dPx = invDx * stencilDPx;
                        const Type dPz = invDz * stencilDPz;
                        const Type dMx = invDx * stencilDMx;
                        const Type dMz = invDz * stencilDMz;

                        const Type E = 1 + 2 * fieldEps[k];
                        const Type A = fieldEta[k];
                        const Type F = fieldVsVp[k];
                        const Type B = fieldBuoy[k];

                        tmpPX[k] = B * E * dPx;
                        tmpPZ[k] = B * (1 - F * A * A) * dPz + B * F * A * sqrt(1 - A * A) * dMz;
                        tmpMX[k] = B * (1 - F) * dMx;
                        tmpMZ[k] = B * F * A * sqrt(1 - A * A) * dPz + B * (1 - F + F * A * A) * dMz;
                    }
                }
            }
        }

        // roll on free surface
        if (freeSurface) {
#pragma omp parallel for num_threads(nthread) schedule(guided)
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

                    const Type stencilDMz0 =
                            c8_1 * (- inMZ[kx * nz + 0] + inMZ[kx * nz + 1]) +
                            c8_2 * (+ inMZ[kx * nz + 1] + inMZ[kx * nz + 2]) +
                            c8_3 * (+ inMZ[kx * nz + 2] + inMZ[kx * nz + 3]) +
                            c8_4 * (+ inMZ[kx * nz + 3] + inMZ[kx * nz + 4]);

                    const Type dPx = 0;
                    const Type dPz = invDz * stencilDPz0;
                    const Type dMx = 0;
                    const Type dMz = invDz * stencilDMz0;

                    const long k = kx * nz + 0;

                    const Type E = 1 + 2 * fieldEps[k];
                    const Type A = fieldEta[k];
                    const Type F = fieldVsVp[k];
                    const Type B = fieldBuoy[k];

                    tmpPX[k] = B * E * dPx;
                    tmpPZ[k] = B * (1 - F * A * A) * dPz + B * F * A * sqrt(1 - A * A) * dMz;
                    tmpMX[k] = B * (1 - F) * dMx;
                    tmpMZ[k] = B * F * A * sqrt(1 - A * A) * dPz + B * (1 - F + F * A * A) * dMz;
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

                    const Type stencilDMx1 =
                            c8_1 * (- inMX[(kx+0) * nz + 1] + inMX[(kx+1) * nz + 1]) +
                            c8_2 * (- inMX[(kx-1) * nz + 1] + inMX[(kx+2) * nz + 1]) +
                            c8_3 * (- inMX[(kx-2) * nz + 1] + inMX[(kx+3) * nz + 1]) +
                            c8_4 * (- inMX[(kx-3) * nz + 1] + inMX[(kx+4) * nz + 1]);

                    const Type stencilDMz1 =
                            c8_1 * (- inMZ[kx * nz + 1] + inMZ[kx * nz + 2]) +
                            c8_2 * (- inMZ[kx * nz + 0] + inMZ[kx * nz + 3]) +
                            c8_3 * (+ inMZ[kx * nz + 1] + inMZ[kx * nz + 4]) +
                            c8_4 * (+ inMZ[kx * nz + 2] + inMZ[kx * nz + 5]);

                    const Type dPx = invDx * stencilDPx1;
                    const Type dPz = invDz * stencilDPz1;
                    const Type dMx = invDx * stencilDMx1;
                    const Type dMz = invDz * stencilDMz1;

                    const long k = kx * nz + 1;

                    const Type E = 1 + 2 * fieldEps[k];
                    const Type A = fieldEta[k];
                    const Type F = fieldVsVp[k];
                    const Type B = fieldBuoy[k];

                    tmpPX[k] = B * E * dPx;
                    tmpPZ[k] = B * (1 - F * A * A) * dPz + B * F * A * sqrt(1 - A * A) * dMz;
                    tmpMX[k] = B * (1 - F) * dMx;
                    tmpMZ[k] = B * F * A * sqrt(1 - A * A) * dPz + B * (1 - F + F * A * A) * dMz;
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

                    const Type stencilDMx2 =
                            c8_1 * (- inMX[(kx+0) * nz + 2] + inMX[(kx+1) * nz + 2]) +
                            c8_2 * (- inMX[(kx-1) * nz + 2] + inMX[(kx+2) * nz + 2]) +
                            c8_3 * (- inMX[(kx-2) * nz + 2] + inMX[(kx+3) * nz + 2]) +
                            c8_4 * (- inMX[(kx-3) * nz + 2] + inMX[(kx+4) * nz + 2]);

                    const Type stencilDMz2 =
                            c8_1 * (- inMZ[kx * nz + 2] + inMZ[kx * nz + 3]) +
                            c8_2 * (- inMZ[kx * nz + 1] + inMZ[kx * nz + 4]) +
                            c8_3 * (- inMZ[kx * nz + 0] + inMZ[kx * nz + 5]) +
                            c8_4 * (+ inMZ[kx * nz + 1] + inMZ[kx * nz + 6]);

                    const Type dPx = invDx * stencilDPx2;
                    const Type dPz = invDz * stencilDPz2;
                    const Type dMx = invDx * stencilDMx2;
                    const Type dMz = invDz * stencilDMz2;

                    const long k = kx * nz + 2;

                    const Type E = 1 + 2 * fieldEps[k];
                    const Type A = fieldEta[k];
                    const Type F = fieldVsVp[k];
                    const Type B = fieldBuoy[k];

                    tmpPX[k] = B * E * dPx;
                    tmpPZ[k] = B * (1 - F * A * A) * dPz + B * F * A * sqrt(1 - A * A) * dMz;
                    tmpMX[k] = B * (1 - F) * dMx;
                    tmpMZ[k] = B * F * A * sqrt(1 - A * A) * dPz + B * (1 - F + F * A * A) * dMz;
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

                    const Type stencilDMx3 =
                            c8_1 * (- inMX[(kx+0) * nz + 3] + inMX[(kx+1) * nz + 3]) +
                            c8_2 * (- inMX[(kx-1) * nz + 3] + inMX[(kx+2) * nz + 3]) +
                            c8_3 * (- inMX[(kx-2) * nz + 3] + inMX[(kx+3) * nz + 3]) +
                            c8_4 * (- inMX[(kx-3) * nz + 3] + inMX[(kx+4) * nz + 3]);

                    const Type stencilDMz3 =
                            c8_1 * (- inMZ[kx * nz + 3] + inMZ[kx * nz + 4]) +
                            c8_2 * (- inMZ[kx * nz + 2] + inMZ[kx * nz + 5]) +
                            c8_3 * (- inMZ[kx * nz + 1] + inMZ[kx * nz + 6]) +
                            c8_4 * (- inMZ[kx * nz + 0] + inMZ[kx * nz + 7]);

                    const Type dPx = invDx * stencilDPx3;
                    const Type dPz = invDz * stencilDPz3;
                    const Type dMx = invDx * stencilDMx3;
                    const Type dMz = invDz * stencilDMz3;

                    const long k = kx * nz + 3;

                    const Type E = 1 + 2 * fieldEps[k];
                    const Type A = fieldEta[k];
                    const Type F = fieldVsVp[k];
                    const Type B = fieldBuoy[k];

                    tmpPX[k] = B * E * dPx;
                    tmpPZ[k] = B * (1 - F * A * A) * dPz + B * F * A * sqrt(1 - A * A) * dMz;
                    tmpMX[k] = B * (1 - F) * dMx;
                    tmpMZ[k] = B * F * A * sqrt(1 - A * A) * dPz + B * (1 - F + F * A * A) * dMz;
                }
            }
        }
    }

    /**
     * Combines
     *   applyFirstDerivatives_MinusHalf(P)
     *   secondOrderTimeUpdate_BubeConservation(P)
     *   applyFirstDerivatives_MinusHalf(M)
     *   secondOrderTimeUpdate_BubeConservation(M)
     *
     * Updates pOld and mOld with second order time update
     *
     * Nonlinear method: outputs the spatial derivatives for source wavefield serialization
     * Linear    method: does not output the spatial derivatives
     */
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
            const Type * __restrict__ const tmpMX,
            const Type * __restrict__ const tmpMZ,
            const Type * __restrict__ const fieldVel,
            const Type * __restrict__ const fieldBuoy,
            const Type * __restrict__ const dtOmegaInvQ,
            Type * __restrict__ pCur,
            Type * __restrict__ mCur,
            Type * __restrict__ pSpace,
            Type * __restrict__ mSpace,
            Type * __restrict__ pOld,
            Type * __restrict__ mOld,
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
                        mSpace[k] = 0;
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

                        const Type stencilDMx =
                                c8_1 * (- tmpMX[(kx-1) * nz + kz] + tmpMX[(kx+0) * nz + kz]) +
                                c8_2 * (- tmpMX[(kx-2) * nz + kz] + tmpMX[(kx+1) * nz + kz]) +
                                c8_3 * (- tmpMX[(kx-3) * nz + kz] + tmpMX[(kx+2) * nz + kz]) +
                                c8_4 * (- tmpMX[(kx-4) * nz + kz] + tmpMX[(kx+3) * nz + kz]);

                        const Type stencilDMz =
                                c8_1 * (- tmpMZ[kx * nz + (kz-1)] + tmpMZ[kx * nz + (kz+0)]) +
                                c8_2 * (- tmpMZ[kx * nz + (kz-2)] + tmpMZ[kx * nz + (kz+1)]) +
                                c8_3 * (- tmpMZ[kx * nz + (kz-3)] + tmpMZ[kx * nz + (kz+2)]) +
                                c8_4 * (- tmpMZ[kx * nz + (kz-4)] + tmpMZ[kx * nz + (kz+3)]);

                        const Type dPX = invDx * stencilDPx;
                        const Type dPZ = invDz * stencilDPz;
                        const Type dMX = invDx * stencilDMx;
                        const Type dMZ = invDz * stencilDMz;

                        const long k = kx * nz + kz;
                        const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                        pSpace[k] = dPX + dPZ;
                        mSpace[k] = dMX + dMZ;

                        pOld[k] = dt2V2_B * pSpace[k] - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                        mOld[k] = dt2V2_B * mSpace[k] - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
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
                    const Type dPX = 0;
                    const Type dPZ = 0;
                    const Type dMX = 0;
                    const Type dMZ = 0;

                    const long k = kx * nz + 0;
                    const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                    pSpace[k] = dPX + dPZ;
                    mSpace[k] = dMX + dMZ;

                    pOld[k] = dt2V2_B * pSpace[k] - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                    mOld[k] = dt2V2_B * mSpace[k] - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
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

                    const Type stencilDMx1 =
                            c8_1 * (- tmpMX[(kx-1) * nz + 1] + tmpMX[(kx+0) * nz + 1]) +
                            c8_2 * (- tmpMX[(kx-2) * nz + 1] + tmpMX[(kx+1) * nz + 1]) +
                            c8_3 * (- tmpMX[(kx-3) * nz + 1] + tmpMX[(kx+2) * nz + 1]) +
                            c8_4 * (- tmpMX[(kx-4) * nz + 1] + tmpMX[(kx+3) * nz + 1]);

                    const Type stencilDMz1 =
                            c8_1 * (- tmpMZ[kx * nz + 0] + tmpMZ[kx * nz + 1]) +
                            c8_2 * (- tmpMZ[kx * nz + 0] + tmpMZ[kx * nz + 2]) +
                            c8_3 * (- tmpMZ[kx * nz + 1] + tmpMZ[kx * nz + 3]) +
                            c8_4 * (- tmpMZ[kx * nz + 2] + tmpMZ[kx * nz + 4]);

                    const Type dPx = invDx * stencilDPx1;
                    const Type dPz = invDz * stencilDPz1;
                    const Type dMx = invDx * stencilDMx1;
                    const Type dMz = invDz * stencilDMz1;

                    const long k = kx * nz + 1;
                    const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                    pSpace[k] = dPx + dPz;
                    mSpace[k] = dMx + dMz;

                    pOld[k] = dt2V2_B * pSpace[k] - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                    mOld[k] = dt2V2_B * mSpace[k] - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
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

                    const Type stencilDMx2 =
                            c8_1 * (- tmpMX[(kx-1) * nz + 2] + tmpMX[(kx+0) * nz + 2]) +
                            c8_2 * (- tmpMX[(kx-2) * nz + 2] + tmpMX[(kx+1) * nz + 2]) +
                            c8_3 * (- tmpMX[(kx-3) * nz + 2] + tmpMX[(kx+2) * nz + 2]) +
                            c8_4 * (- tmpMX[(kx-4) * nz + 2] + tmpMX[(kx+3) * nz + 2]);

                    const Type stencilDMz2 =
                            c8_1 * (- tmpMZ[kx * nz + 1] + tmpMZ[kx * nz + 2]) +
                            c8_2 * (- tmpMZ[kx * nz + 0] + tmpMZ[kx * nz + 3]) +
                            c8_3 * (- tmpMZ[kx * nz + 0] + tmpMZ[kx * nz + 4]) +
                            c8_4 * (- tmpMZ[kx * nz + 1] + tmpMZ[kx * nz + 5]);

                    const Type dPx = invDx * stencilDPx2;
                    const Type dPz = invDz * stencilDPz2;
                    const Type dMx = invDx * stencilDMx2;
                    const Type dMz = invDz * stencilDMz2;

                    const long k = kx * nz + 2;
                    const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                    pSpace[k] = dPx + dPz;
                    mSpace[k] = dMx + dMz;

                    pOld[k] = dt2V2_B * pSpace[k] - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                    mOld[k] = dt2V2_B * mSpace[k] - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
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

                    const Type stencilDMx3 =
                            c8_1 * (- tmpMX[(kx-1) * nz + 3] + tmpMX[(kx+0) * nz + 3]) +
                            c8_2 * (- tmpMX[(kx-2) * nz + 3] + tmpMX[(kx+1) * nz + 3]) +
                            c8_3 * (- tmpMX[(kx-3) * nz + 3] + tmpMX[(kx+2) * nz + 3]) +
                            c8_4 * (- tmpMX[(kx-4) * nz + 3] + tmpMX[(kx+3) * nz + 3]);

                    const Type stencilDMz3 =
                            c8_1 * (- tmpMZ[kx * nz + 2] + tmpMZ[kx * nz + 3]) +
                            c8_2 * (- tmpMZ[kx * nz + 1] + tmpMZ[kx * nz + 4]) +
                            c8_3 * (- tmpMZ[kx * nz + 0] + tmpMZ[kx * nz + 5]) +
                            c8_4 * (- tmpMZ[kx * nz + 0] + tmpMZ[kx * nz + 6]);

                    const Type dPx = invDx * stencilDPx3;
                    const Type dPz = invDz * stencilDPz3;
                    const Type dMx = invDx * stencilDMx3;
                    const Type dMz = invDz * stencilDMz3;

                    const long k = kx * nz + 3;
                    const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                    pSpace[k] = dPx + dPz;
                    mSpace[k] = dMx + dMz;

                    pOld[k] = dt2V2_B * pSpace[k] - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                    mOld[k] = dt2V2_B * mSpace[k] - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
                }
            }
        }
    }

    template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline static void applyFirstDerivatives2D_MinusHalf_TimeUpdate_Linear(
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
            const Type * __restrict__ const tmpMX,
            const Type * __restrict__ const tmpMZ,
            const Type * __restrict__ const fieldVel,
            const Type * __restrict__ const fieldBuoy,
            const Type * __restrict__ const dtOmegaInvQ,
            Type * __restrict__ pCur,
            Type * __restrict__ mCur,
            Type * __restrict__ pOld,
            Type * __restrict__ mOld,
            const long BX_2D,
            const long BZ_2D) {

        const long nx4 = nx - 4;
        const long nz4 = nz - 4;

        const Type dt2 = dtMod * dtMod;

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
                                c8_1 * (- tmpPX[(kx-1) * nz + kz] + tmpPX[(kx+0) * nz + kz]) +
                                c8_2 * (- tmpPX[(kx-2) * nz + kz] + tmpPX[(kx+1) * nz + kz]) +
                                c8_3 * (- tmpPX[(kx-3) * nz + kz] + tmpPX[(kx+2) * nz + kz]) +
                                c8_4 * (- tmpPX[(kx-4) * nz + kz] + tmpPX[(kx+3) * nz + kz]);

                        const Type stencilDPz =
                                c8_1 * (- tmpPZ[kxnz + (kz-1)] + tmpPZ[kxnz + (kz+0)]) +
                                c8_2 * (- tmpPZ[kxnz + (kz-2)] + tmpPZ[kxnz + (kz+1)]) +
                                c8_3 * (- tmpPZ[kxnz + (kz-3)] + tmpPZ[kxnz + (kz+2)]) +
                                c8_4 * (- tmpPZ[kxnz + (kz-4)] + tmpPZ[kxnz + (kz+3)]);

                        const Type stencilDMx =
                                c8_1 * (- tmpMX[(kx-1) * nz + kz] + tmpMX[(kx+0) * nz + kz]) +
                                c8_2 * (- tmpMX[(kx-2) * nz + kz] + tmpMX[(kx+1) * nz + kz]) +
                                c8_3 * (- tmpMX[(kx-3) * nz + kz] + tmpMX[(kx+2) * nz + kz]) +
                                c8_4 * (- tmpMX[(kx-4) * nz + kz] + tmpMX[(kx+3) * nz + kz]);

                        const Type stencilDMz =
                                c8_1 * (- tmpMZ[kxnz + (kz-1)] + tmpMZ[kxnz + (kz+0)]) +
                                c8_2 * (- tmpMZ[kxnz + (kz-2)] + tmpMZ[kxnz + (kz+1)]) +
                                c8_3 * (- tmpMZ[kxnz + (kz-3)] + tmpMZ[kxnz + (kz+2)]) +
                                c8_4 * (- tmpMZ[kxnz + (kz-4)] + tmpMZ[kxnz + (kz+3)]);

                        const Type dPx = invDx * stencilDPx;
                        const Type dPz = invDz * stencilDPz;
                        const Type dMx = invDx * stencilDMx;
                        const Type dMz = invDz * stencilDMz;

                        const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                        pOld[k] = dt2V2_B * (dPx + dPz) - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                        mOld[k] = dt2V2_B * (dMx + dMz) - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
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
                    const Type dPX = 0;
                    const Type dPZ = 0;

                    const Type dMX = 0;
                    const Type dMZ = 0;

                    const long k = kx * nz + 0;
                    const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                    pOld[k] = dt2V2_B * (dPX + dPZ) - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                    mOld[k] = dt2V2_B * (dMX + dMZ) - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
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

                    const Type stencilDMx1 =
                            c8_1 * (- tmpMX[(kx-1) * nz + 1] + tmpMX[(kx+0) * nz + 1]) +
                            c8_2 * (- tmpMX[(kx-2) * nz + 1] + tmpMX[(kx+1) * nz + 1]) +
                            c8_3 * (- tmpMX[(kx-3) * nz + 1] + tmpMX[(kx+2) * nz + 1]) +
                            c8_4 * (- tmpMX[(kx-4) * nz + 1] + tmpMX[(kx+3) * nz + 1]);

                    const Type stencilDMz1 =
                            c8_1 * (- tmpMZ[kx * nz + 0] + tmpMZ[kx * nz + 1]) +
                            c8_2 * (- tmpMZ[kx * nz + 0] + tmpMZ[kx * nz + 2]) +
                            c8_3 * (- tmpMZ[kx * nz + 1] + tmpMZ[kx * nz + 3]) +
                            c8_4 * (- tmpMZ[kx * nz + 2] + tmpMZ[kx * nz + 4]);

                    const Type dPx = invDx * stencilDPx1;
                    const Type dPz = invDz * stencilDPz1;
                    const Type dMx = invDx * stencilDMx1;
                    const Type dMz = invDz * stencilDMz1;

                    const long k = kx * nz + 1;
                    const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                    pOld[k] = dt2V2_B * (dPx + dPz) - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                    mOld[k] = dt2V2_B * (dMx + dMz) - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
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

                    const Type stencilDMx2 =
                            c8_1 * (- tmpMX[(kx-1) * nz + 2] + tmpMX[(kx+0) * nz + 2]) +
                            c8_2 * (- tmpMX[(kx-2) * nz + 2] + tmpMX[(kx+1) * nz + 2]) +
                            c8_3 * (- tmpMX[(kx-3) * nz + 2] + tmpMX[(kx+2) * nz + 2]) +
                            c8_4 * (- tmpMX[(kx-4) * nz + 2] + tmpMX[(kx+3) * nz + 2]);

                    const Type stencilDMz2 =
                            c8_1 * (- tmpMZ[kx * nz + 1] + tmpMZ[kx * nz + 2]) +
                            c8_2 * (- tmpMZ[kx * nz + 0] + tmpMZ[kx * nz + 3]) +
                            c8_3 * (- tmpMZ[kx * nz + 0] + tmpMZ[kx * nz + 4]) +
                            c8_4 * (- tmpMZ[kx * nz + 1] + tmpMZ[kx * nz + 5]);

                    const Type dPx = invDx * stencilDPx2;
                    const Type dPz = invDz * stencilDPz2;
                    const Type dMx = invDx * stencilDMx2;
                    const Type dMz = invDz * stencilDMz2;

                    const long k = kx * nz + 2;
                    const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                    pOld[k] = dt2V2_B * (dPx + dPz) - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                    mOld[k] = dt2V2_B * (dMx + dMz) - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
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

                    const Type stencilDMx3 =
                            c8_1 * (- tmpMX[(kx-1) * nz + 3] + tmpMX[(kx+0) * nz + 3]) +
                            c8_2 * (- tmpMX[(kx-2) * nz + 3] + tmpMX[(kx+1) * nz + 3]) +
                            c8_3 * (- tmpMX[(kx-3) * nz + 3] + tmpMX[(kx+2) * nz + 3]) +
                            c8_4 * (- tmpMX[(kx-4) * nz + 3] + tmpMX[(kx+3) * nz + 3]);

                    const Type stencilDMz3 =
                            c8_1 * (- tmpMZ[kx * nz + 2] + tmpMZ[kx * nz + 3]) +
                            c8_2 * (- tmpMZ[kx * nz + 1] + tmpMZ[kx * nz + 4]) +
                            c8_3 * (- tmpMZ[kx * nz + 0] + tmpMZ[kx * nz + 5]) +
                            c8_4 * (- tmpMZ[kx * nz + 0] + tmpMZ[kx * nz + 6]);

                    const Type dPx = invDx * stencilDPx3;
                    const Type dPz = invDz * stencilDPz3;
                    const Type dMx = invDx * stencilDMx3;
                    const Type dMz = invDz * stencilDMz3;

                    const long k = kx * nz + 3;
                    const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                    pOld[k] = dt2V2_B * (dPx + dPz) - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                    mOld[k] = dt2V2_B * (dMx + dMz) - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
                }
            }
        }
    }

    template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline static void applyFirstDerivatives2D_PlusHalf(
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
    inline static void applyFirstDerivatives2D_MinusHalf(
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

};

#endif
