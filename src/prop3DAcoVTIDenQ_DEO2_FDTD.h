#ifndef PROP3DACOVTIDENQ_DEO2_FDTD_H
#define PROP3DACOVTIDENQ_DEO2_FDTD_H

#include <omp.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <fftw3.h>
#include <complex>

#define MIN(x,y) ((x)<(y)?(x):(y))

class Prop3DAcoVTIDenQ_DEO2_FDTD {

public:
    const bool _freeSurface;
    const long _nbx, _nby, _nbz, _nthread, _nx, _ny, _nz, _nsponge;
    const float _dx, _dy, _dz, _dt;
    const float _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz;
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
    float * __restrict__ _tmpPy1 = NULL;
    float * __restrict__ _tmpPz1 = NULL;
    float * __restrict__ _tmpMx1 = NULL;
    float * __restrict__ _tmpMy1 = NULL;
    float * __restrict__ _tmpMz1 = NULL;
    float * __restrict__ _tmpPx2 = NULL;
    float * __restrict__ _tmpPy2 = NULL;
    float * __restrict__ _tmpPz2 = NULL;
    float * __restrict__ _tmpMx2 = NULL;
    float * __restrict__ _tmpMy2 = NULL;
    float * __restrict__ _tmpMz2 = NULL;
    float * _pOld = NULL;
    float * _pCur = NULL;
    float * _mOld = NULL;
    float * _mCur = NULL;

    Prop3DAcoVTIDenQ_DEO2_FDTD(
        bool freeSurface,
        long nthread,
        long nx,
        long ny,
        long nz,
        long nsponge,
        float dx,
        float dy,
        float dz,
        float dt,
        const long nbx,
        const long nby,
        const long nbz) :
            _freeSurface(freeSurface),
            _nthread(nthread),
            _nx(nx),
            _ny(ny),
            _nz(nz),
            _nsponge(nsponge),
            _nbx(nbx),
            _nby(nby),
            _nbz(nbz),
            _dx(dx),
            _dy(dy),
            _dz(dz),
            _dt(dt),
            _c8_1(+1225.0 / 1024.0),
            _c8_2(-245.0 / 3072.0),
            _c8_3(+49.0 / 5120.0),
            _c8_4(-5.0 / 7168.0),
            _invDx(1.0 / _dx),
            _invDy(1.0 / _dy),
            _invDz(1.0 / _dz) {

        // Allocate arrays
        _v           = new float[_nx * _ny * _nz];
        _eps         = new float[_nx * _ny * _nz];
        _eta         = new float[_nx * _ny * _nz];
        _b           = new float[_nx * _ny * _nz];
        _f           = new float[_nx * _ny * _nz];
        _dtOmegaInvQ = new float[_nx * _ny * _nz];
        _pSpace      = new float[_nx * _ny * _nz];
        _mSpace      = new float[_nx * _ny * _nz];
        _tmpPx1      = new float[_nx * _ny * _nz];
        _tmpPy1      = new float[_nx * _ny * _nz];
        _tmpPz1      = new float[_nx * _ny * _nz];
        _tmpMx1      = new float[_nx * _ny * _nz];
        _tmpMy1      = new float[_nx * _ny * _nz];
        _tmpMz1      = new float[_nx * _ny * _nz];
        _tmpPx2      = new float[_nx * _ny * _nz];
        _tmpPy2      = new float[_nx * _ny * _nz];
        _tmpPz2      = new float[_nx * _ny * _nz];
        _tmpMx2      = new float[_nx * _ny * _nz];
        _tmpMy2      = new float[_nx * _ny * _nz];
        _tmpMz2      = new float[_nx * _ny * _nz];
        _pOld        = new float[_nx * _ny * _nz];
        _pCur        = new float[_nx * _ny * _nz];
        _mOld        = new float[_nx * _ny * _nz];
        _mCur        = new float[_nx * _ny * _nz];

        numaFirstTouch(_nx, _ny, _nz, _nthread, _v, _eps, _eta, _b,
            _f, _dtOmegaInvQ, _pSpace, _mSpace, 
            _tmpPx1, _tmpPy1, _tmpPz1, _tmpMx1, _tmpMy1, _tmpMz1, 
            _tmpPx2, _tmpPy2, _tmpPz2, _tmpMx2, _tmpMy2, _tmpMz2, 
            _pOld, _pCur, _mOld, _mCur, _nbx, _nby, _nbz);
    }

#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void numaFirstTouch(
            const long nx,
            const long ny,
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
            float * __restrict__ tmpPy1,
            float * __restrict__ tmpPz1,
            float * __restrict__ tmpMx1,
            float * __restrict__ tmpMy1,
            float * __restrict__ tmpMz1,
            float * __restrict__ tmpPx2,
            float * __restrict__ tmpPy2,
            float * __restrict__ tmpPz2,
            float * __restrict__ tmpMx2,
            float * __restrict__ tmpMy2,
            float * __restrict__ tmpMz2,
            float * __restrict__ pOld,
            float * __restrict__ pCur,
            float * __restrict__ mOld,
            float * __restrict__ mCur,
            const long BX_3D,
            const long BY_3D,
            const long BZ_3D) {

        const long nx4 = nx - 4;
        const long ny4 = ny - 4;
        const long nz4 = nz - 4;

#pragma omp parallel for collapse(3) num_threads(nthread) schedule(static)
        for (long bx = 4; bx < nx4; bx += BX_3D) {
            for (long by = 4; by < ny4; by += BY_3D) {
                for (long bz = 4; bz < nz4; bz += BZ_3D) {
                    const long kxmax = MIN(bx + BX_3D, nx4);
                    const long kymax = MIN(by + BY_3D, ny4);
                    const long kzmax = MIN(bz + BZ_3D, nz4);

                    for (long kx = bx; kx < kxmax; kx++) {
                        for (long ky = by; ky < kymax; ky++) {
#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long k = kx * _ny * _nz + ky * _nz + kz;

                                v[k] = 0;
                                eps[k] = 0;
                                eta[k] = 0;
                                b[k] = 0;
                                f[k] = 0;
                                dtOmegaInvQ[k] = 0;
                                pSpace[k] = 0;
                                mSpace[k] = 0;
                                tmpPx1[k] = 0;
                                tmpPy1[k] = 0;
                                tmpPz1[k] = 0;
                                tmpMx1[k] = 0;
                                tmpMy1[k] = 0;
                                tmpMz1[k] = 0;
                                tmpPx2[k] = 0;
                                tmpPy2[k] = 0;
                                tmpPz2[k] = 0;
                                tmpMx2[k] = 0;
                                tmpMy2[k] = 0;
                                tmpMz2[k] = 0;
                                pOld[k] = 0;
                                pCur[k] = 0;
                                mOld[k] = 0;
                                mCur[k] = 0;
                            }
                        }
                    }
                }
            }
        }

        // annulus
        for (long k = 0; k < 4; k++) {

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long kx = 0; kx < nx; kx++) {
#pragma omp simd
                for (long ky = 0; ky < ny; ky++) {
                    const long kindex1 = kx * ny * nz + ky * nz + k;
                    const long kindex2 = kx * ny * nz + ky * nz + (nz - 1 - k);
                    v[kindex1] = eps[kindex1] = eta[kindex1] = b[kindex1] = f[kindex1] =
                        dtOmegaInvQ[kindex1] = pSpace[kindex1] = mSpace[kindex1] =
                        tmpPx1[kindex1] = tmpPy1[kindex1] = tmpPz1[kindex1] = tmpMx1[kindex1] =
                        tmpMy1[kindex1] = tmpMz1[kindex1] = tmpPx2[kindex1] = tmpPy2[kindex1] =
                        tmpPz2[kindex1] = tmpMx2[kindex1] = tmpMy2[kindex1] = tmpMz2[kindex1] =
                        pOld[kindex1] = pCur[kindex1] = mOld[kindex1] = mCur[kindex1] = 0;

                    v[kindex2] = eps[kindex2] = eta[kindex2] = b[kindex2] = f[kindex2] =
                        dtOmegaInvQ[kindex2] = pSpace[kindex2] = mSpace[kindex2] =
                        tmpPx1[kindex2] = tmpPy1[kindex2] = tmpPz1[kindex2] = tmpMx1[kindex2] =
                        tmpMy1[kindex2] = tmpMz1[kindex2] = tmpPx2[kindex2] = tmpPy2[kindex2] =
                        tmpPz2[kindex2] = tmpMx2[kindex2] = tmpMy2[kindex2] = tmpMz2[kindex2] =
                        pOld[kindex2] = pCur[kindex2] = mOld[kindex2] = mCur[kindex2] = 0;
                }
            }

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long kx = 0; kx < nx; kx++) {
#pragma omp simd
                for (long kz = 0; kz < nz; kz++) {
                    const long kindex1 = kx * ny * nz + k * nz + kz;
                    const long kindex2 = kx * ny * nz + (ny - 1 - k) * nz + kz;
                    v[kindex1] = eps[kindex1] = eta[kindex1] = b[kindex1] = f[kindex1] =
                        dtOmegaInvQ[kindex1] = pSpace[kindex1] = mSpace[kindex1] =
                        tmpPx1[kindex1] = tmpPy1[kindex1] = tmpPz1[kindex1] = tmpMx1[kindex1] =
                        tmpMy1[kindex1] = tmpMz1[kindex1] = tmpPx2[kindex1] = tmpPy2[kindex1] =
                        tmpPz2[kindex1] = tmpMx2[kindex1] = tmpMy2[kindex1] = tmpMz2[kindex1] =
                        pOld[kindex1] = pCur[kindex1] = mOld[kindex1] = mCur[kindex1] = 0;

                    v[kindex2] = eps[kindex2] = eta[kindex2] = b[kindex2] = f[kindex2] =
                        dtOmegaInvQ[kindex2] = pSpace[kindex2] = mSpace[kindex2] =
                        tmpPx1[kindex2] = tmpPy1[kindex2] = tmpPz1[kindex2] = tmpMx1[kindex2] =
                        tmpMy1[kindex2] = tmpMz1[kindex2] = tmpPx2[kindex2] = tmpPy2[kindex2] =
                        tmpPz2[kindex2] = tmpMx2[kindex2] = tmpMy2[kindex2] = tmpMz2[kindex2] =
                        pOld[kindex2] = pCur[kindex2] = mOld[kindex2] = mCur[kindex2] = 0;
                }
            }

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long ky = 0; ky < ny; ky++) {
#pragma omp simd
                for (long kz = 0; kz < nz; kz++) {
                    const long kindex1 = k * ny * nz + ky * nz + kz;
                    const long kindex2 = (nx - 1 - k) * ny * nz + ky * nz + kz;
                    v[kindex1] = eps[kindex1] = eta[kindex1] = b[kindex1] = f[kindex1] =
                        dtOmegaInvQ[kindex1] = pSpace[kindex1] = mSpace[kindex1] =
                        tmpPx1[kindex1] = tmpPy1[kindex1] = tmpPz1[kindex1] = tmpMx1[kindex1] =
                        tmpMy1[kindex1] = tmpMz1[kindex1] = tmpPx2[kindex1] = tmpPy2[kindex1] =
                        tmpPz2[kindex1] = tmpMx2[kindex1] = tmpMy2[kindex1] = tmpMz2[kindex1] =
                        pOld[kindex1] = pCur[kindex1] = mOld[kindex1] = mCur[kindex1] = 0;

                    v[kindex2] = eps[kindex2] = eta[kindex2] = b[kindex2] = f[kindex2] =
                        dtOmegaInvQ[kindex2] = pSpace[kindex2] = mSpace[kindex2] =
                        tmpPx1[kindex2] = tmpPy1[kindex2] = tmpPz1[kindex2] = tmpMx1[kindex2] =
                        tmpMy1[kindex2] = tmpMz1[kindex2] = tmpPx2[kindex2] = tmpPy2[kindex2] =
                        tmpPz2[kindex2] = tmpMx2[kindex2] = tmpMy2[kindex2] = tmpMz2[kindex2] =
                        pOld[kindex2] = pCur[kindex2] = mOld[kindex2] = mCur[kindex2] = 0;
                }
            }
        }
    }

    ~Prop3DAcoVTIDenQ_DEO2_FDTD() {
        if (_v != NULL) delete [] _v;
        if (_eps != NULL) delete [] _eps;
        if (_eta != NULL) delete [] _eta;
        if (_b != NULL) delete [] _b;
        if (_f != NULL) delete [] _f;
        if (_dtOmegaInvQ != NULL) delete [] _dtOmegaInvQ;
        if (_pSpace != NULL) delete [] _pSpace;
        if (_mSpace != NULL) delete [] _mSpace;
        if (_tmpPx1 != NULL) delete [] _tmpPx1;
        if (_tmpPy1 != NULL) delete [] _tmpPy1;
        if (_tmpPz1 != NULL) delete [] _tmpPz1;
        if (_tmpMx1 != NULL) delete [] _tmpMx1;
        if (_tmpMy1 != NULL) delete [] _tmpMy1;
        if (_tmpMz1 != NULL) delete [] _tmpMz1;
        if (_tmpPx2 != NULL) delete [] _tmpPx2;
        if (_tmpPy2 != NULL) delete [] _tmpPy2;
        if (_tmpPz2 != NULL) delete [] _tmpPz2;
        if (_tmpMx2 != NULL) delete [] _tmpMx2;
        if (_tmpMy2 != NULL) delete [] _tmpMy2;
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
        printf("Prop3DAcoVTIDenQ_DEO2_FDTD\n");
        printf("  nx,ny,nz;           %5ld %5ld %5ld\n", _nx, _ny, _nz);
        printf("  nthread,nsponge,fs; %5ld %5ld %5d\n", _nthread, _nsponge, _freeSurface);
        printf("  X min,max,inc;    %+16.8f %+16.8f %+16.8f\n", 0.0, _dx * (_nx - 1), _dx);
        printf("  Y min,max,inc;    %+16.8f %+16.8f %+16.8f\n", 0.0, _dy * (_ny - 1), _dy);
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

        applyFirstDerivatives3D_PlusHalf_Sandwich(
                _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz,
                _pCur, _pCur, _pCur, _mCur, _mCur, _mCur, _eps, _eta, _f, _b,
                _tmpPx1, _tmpPy1, _tmpPz1, _tmpMx1, _tmpMy1, _tmpMz1, _nbx, _nby, _nbz);

        applyFirstDerivatives3D_MinusHalf_TimeUpdate_Nonlinear(
                _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz, _dt,
                _tmpPx1, _tmpPy1, _tmpPz1, _tmpMx1, _tmpMy1, _tmpMz1, _v, _b, _dtOmegaInvQ,
                _pCur, _mCur, _pSpace, _mSpace, _pOld, _mOld, _nbx, _nby, _nbz);

        // swap pointers
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

        applyFirstDerivatives3D_PlusHalf_Sandwich(
                _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz,
                _pCur, _pCur, _pCur, _mCur, _mCur, _mCur, _eps, _eta, _f, _b,
                _tmpPx1, _tmpPy1, _tmpPz1, _tmpMx1, _tmpMy1, _tmpMz1, _nbx, _nby, _nbz);

        applyFirstDerivatives3D_MinusHalf_TimeUpdate_Linear(
                _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz, _dt,
                _tmpPx1, _tmpPy1, _tmpPz1, _tmpMx1, _tmpMy1, _tmpMz1, _v, _b, _dtOmegaInvQ,
                _pCur, _mCur, _pOld, _mOld, _nbx, _nby, _nbz);

        // swap pointers
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
#pragma omp parallel for collapse(3) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long by = 0; by < _ny; by += _nby) {
                for (long bz = 0; bz < _nz; bz += _nbz) {
                    const long kxmax = MIN(bx + _nbx, _nx);
                    const long kymax = MIN(by + _nby, _ny);
                    const long kzmax = MIN(bz + _nbz, _nz);

                    for (long kx = bx; kx < kxmax; kx++) {
                        for (long ky = by; ky < kymax; ky++) {
#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long k = kx * _ny * _nz + ky * _nz + kz;
                                const float v2OverB = _v[k] * _v[k] / _b[k];
                                _pSpace[k] *= v2OverB;
                                _mSpace[k] *= v2OverB;
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Add the Born source at the current time
     *
     * User must have:
     *   - called the nonlinear forward
     *   - saved 2nd time derivative of pressure at corresponding time index in array dp2
     *   - Born source term will be injected into the _pCur array
     */
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void forwardBornInjection_V(float *dVel, float *wavefieldDP, float *wavefieldDM) {
#pragma omp parallel for collapse(3) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long by = 0; by < _ny; by += _nby) {
                for (long bz = 0; bz < _nz; bz += _nbz) {
                    const long kxmax = MIN(bx + _nbx, _nx);
                    const long kymax = MIN(by + _nby, _ny);
                    const long kzmax = MIN(bz + _nbz, _nz);

                    for (long kx = bx; kx < kxmax; kx++) {
                        for (long ky = by; ky < kymax; ky++) {
#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long k = kx * _ny * _nz + ky * _nz + kz;

                                const float V  = _v[k];
                                const float B  = _b[k];
                                const float dV = dVel[k];

                                // V^2/b factor to "clear" the b/V^2 factor on L_tP and L_tM
                                // _dt^2 factor is from the finite difference approximation
                                // 2B_dV/V^3 factor is from the linearization
                                const float factor = 2 * _dt * _dt * dV / V;

                                _pCur[k] += factor * wavefieldDP[k];
                                _mCur[k] += factor * wavefieldDM[k];
                            }
                        }
                    }
                }
            }
        }
    }

#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void forwardBornInjection_VEA(float *dVel, float *dEps, float *dEta,
            float *wavefieldP, float *wavefieldM, float *wavefieldDP, float *wavefieldDM) {

        // Right side spatial derivatives for the Born source
        applyFirstDerivatives3D_PlusHalf(
                _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz,
                wavefieldP, wavefieldP, wavefieldP, _tmpPx1, _tmpPy1, _tmpPz1, _nbx, _nby, _nbz);

        applyFirstDerivatives3D_PlusHalf(
                _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz,
                wavefieldM, wavefieldM, wavefieldM, _tmpMx1, _tmpMy1, _tmpMz1, _nbx, _nby, _nbz);

        // Sandwich terms for the Born source
        // note flipped sign for Z derivative term between P and M
#pragma omp parallel for collapse(3) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long by = 0; by < _ny; by += _nby) {
                for (long bz = 0; bz < _nz; bz += _nbz) {
                    const long kxmax = MIN(bx + _nbx, _nx);
                    const long kymax = MIN(by + _nby, _ny);
                    const long kzmax = MIN(bz + _nbz, _nz);

                    for (long kx = bx; kx < kxmax; kx++) {
                        for (long ky = by; ky < kymax; ky++) {
#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long k = kx * _ny * _nz + ky * _nz + kz;

                                const float V  = _v[k];
                                const float E  = _eps[k];
                                const float A  = _eta[k];
                                const float B  = _b[k];
                                const float F  = _f[k];

                                const float dV = dVel[k];
                                const float dE = dEps[k];
                                const float dA = dEta[k];

                                _tmpPx2[k] = (+2 * B * dE) *_tmpPx1[k];
                                _tmpPy2[k] = (+2 * B * dE) *_tmpPy1[k];
                                _tmpPz2[k] = (-2 * B * F * A * dA) *_tmpPz1[k] +
                                    (dA * B * F * (1 - 2 * A * A) / sqrt(1 - A * A)) * _tmpMz1[k];

                                _tmpMx2[k] = 0;
                                _tmpMy2[k] = 0;
                                _tmpMz2[k] = (+2 * B * F * A * dA) *_tmpMz1[k] +
                                    (dA * B * F * (1 - 2 * A * A) / sqrt(1 - A * A)) * _tmpPz1[k];
                            }
                        }
                    }
                }
            }
        }

        // Left side spatial derivatives for the Born source
        applyFirstDerivatives3D_MinusHalf(
                _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz,
                _tmpPx2, _tmpPy2, _tmpPz2, _tmpPx1, _tmpPy1, _tmpPz1, _nbx, _nby, _nbz);

        applyFirstDerivatives3D_MinusHalf(
                _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz,
                _tmpMx2, _tmpMy2, _tmpMz2, _tmpMx1, _tmpMy1, _tmpMz1, _nbx, _nby, _nbz);

        // add the born source at the current time
#pragma omp parallel for collapse(3) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long by = 0; by < _ny; by += _nby) {
                for (long bz = 0; bz < _nz; bz += _nbz) {
                    const long kxmax = MIN(bx + _nbx, _nx);
                    const long kymax = MIN(by + _nby, _ny);
                    const long kzmax = MIN(bz + _nbz, _nz);

                    for (long kx = bx; kx < kxmax; kx++) {
                        for (long ky = by; ky < kymax; ky++) {
#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long k = kx * _ny * _nz + ky * _nz + kz;

                                const float V  = _v[k];
                                const float B  = _b[k];
                                const float dV = dVel[k];

                                const float dt2v2OverB = _dt * _dt * V * V / B;

                                const float factor = 2 * B * dV / (V * V * V);

                                _pCur[k] += dt2v2OverB * (factor * wavefieldDP[k] + _tmpPx1[k] + _tmpPy1[k] + _tmpPz1[k]);
                                _mCur[k] += dt2v2OverB * (factor * wavefieldDM[k] + _tmpMx1[k] + _tmpMy1[k] + _tmpMz1[k]);
                            }
                        }
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
    inline void adjointBornAccumulation_V(float *dVel,
            float *wavefieldDP, float *wavefieldDM) {
#pragma omp parallel for collapse(3) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long by = 0; by < _ny; by += _nby) {
                for (long bz = 0; bz < _nz; bz += _nbz) {
                    const long kxmax = MIN(bx + _nbx, _nx);
                    const long kymax = MIN(by + _nby, _ny);
                    const long kzmax = MIN(bz + _nbz, _nz);

                    for (long kx = bx; kx < kxmax; kx++) {
                        for (long ky = by; ky < kymax; ky++) {
#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long k = kx * _ny * _nz + ky * _nz + kz;
                                const float V = _v[k];
                                const float B = _b[k];
                                const float factor = 2 * B / (V * V * V);
                                dVel[k] += factor * (wavefieldDP[k] * _pOld[k] + wavefieldDM[k] * _mOld[k]);
                            }
                        }
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
     inline void adjointBornAccumulation_wavefieldsep_V(float *dVel, 
            float *wavefieldDP, float *wavefieldDM, const long isFWI) {
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
            std::complex<float> * __restrict__ tmp_nlf_p = new std::complex<float>[nfft];
            std::complex<float> * __restrict__ tmp_adj_p = new std::complex<float>[nfft];
            std::complex<float> * __restrict__ tmp_nlf_m = new std::complex<float>[nfft];
            std::complex<float> * __restrict__ tmp_adj_m = new std::complex<float>[nfft];

#pragma omp for collapse(2) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long by = 0; by < _ny; by += _nby) {
                const long kxmax = MIN(bx + _nbx, _nx);
                const long kymax = MIN(by + _nby, _ny);

                for (long kx = bx; kx < kxmax; kx++) {
                    for (long ky = by; ky < kymax; ky++) {
#pragma omp simd
                        for (long kfft = 0; kfft < nfft; kfft++) {
                            tmp_nlf_p[kfft] = 0;
                            tmp_adj_p[kfft] = 0;
                            tmp_nlf_m[kfft] = 0;
                            tmp_adj_m[kfft] = 0;
                        }  

#pragma omp simd
                        for (long kz = 0; kz < _nz; kz++) {
                            const long k = kx * _ny * _nz + ky * _nz + kz;
                            tmp_nlf_p[kz] = scale * wavefieldDP[k];
                            tmp_adj_p[kz] = scale * _pOld[k];
                            tmp_nlf_m[kz] = scale * wavefieldDM[k];
                            tmp_adj_m[kz] = scale * _mOld[k];
                        }  

                        fftwf_execute_dft(planForward,
                            reinterpret_cast<fftwf_complex*>(tmp_nlf_p),
                            reinterpret_cast<fftwf_complex*>(tmp_nlf_p));

                        fftwf_execute_dft(planForward,
                            reinterpret_cast<fftwf_complex*>(tmp_adj_p),
                            reinterpret_cast<fftwf_complex*>(tmp_adj_p));

                        fftwf_execute_dft(planForward,
                            reinterpret_cast<fftwf_complex*>(tmp_nlf_m),
                            reinterpret_cast<fftwf_complex*>(tmp_nlf_m));

                        fftwf_execute_dft(planForward,
                            reinterpret_cast<fftwf_complex*>(tmp_adj_m),
                            reinterpret_cast<fftwf_complex*>(tmp_adj_m));

                        // upgoing: zero the positive frequencies, excluding Nyquist
                        // dngoing: zero the negative frequencies, excluding Nyquist
#pragma omp simd
                        for (long k = 1; k < nfft / 2; k++) {
                            tmp_nlf_p[nfft / 2 + k] = 0;
                            tmp_adj_p[kfft_adj + k] = 0;
                            tmp_nlf_m[nfft / 2 + k] = 0;
                            tmp_adj_m[kfft_adj + k] = 0;
                        }

                        fftwf_execute_dft(planInverse,
                            reinterpret_cast<fftwf_complex*>(tmp_nlf_p),
                            reinterpret_cast<fftwf_complex*>(tmp_nlf_p));

                        fftwf_execute_dft(planInverse,
                            reinterpret_cast<fftwf_complex*>(tmp_adj_p),
                            reinterpret_cast<fftwf_complex*>(tmp_adj_p));

                        fftwf_execute_dft(planInverse,
                            reinterpret_cast<fftwf_complex*>(tmp_nlf_m),
                            reinterpret_cast<fftwf_complex*>(tmp_nlf_m));

                        fftwf_execute_dft(planInverse,
                            reinterpret_cast<fftwf_complex*>(tmp_adj_m),
                            reinterpret_cast<fftwf_complex*>(tmp_adj_m));

                        // Faqi eq 10
                        // Applied to FWI: [Sup * Rdn]
                        // Applied to RTM: [Sup * Rup]
#pragma omp simd
                        for (long kz = 0; kz < _nz; kz++) {
                            const long k = kx * _ny * _nz + ky * _nz + kz;
                            const float V = _v[k];
                            const float B = _b[k];
                            const float factor = 2 * B / (V * V * V);
                            dVel[k] += factor * (real(tmp_nlf_p[kz] * tmp_adj_p[kz]) + real(tmp_nlf_m[kz] * tmp_adj_m[kz]));
                        }
                    } // end loop over ky
                } // end loop over kx
            } // end loop over by
        } // end loop over bx

        delete [] tmp_nlf_p;
        delete [] tmp_adj_p;
        delete [] tmp_nlf_m;
        delete [] tmp_adj_m;
    } // end parallel region

    fftwf_destroy_plan(planForward);
    fftwf_destroy_plan(planInverse);
 }

#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void adjointBornAccumulation_VEA(float *dVel, float *dEps, float *dEta,
            float *wavefieldP, float *wavefieldM, float *wavefieldDP, float *wavefieldDM) {

        // Right side spatial derivatives for the adjoint accumulation
        applyFirstDerivatives3D_PlusHalf(
                _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz,
                wavefieldP, wavefieldP, wavefieldP, _tmpPx1, _tmpPy1, _tmpPz1, _nbx, _nby, _nbz);

        applyFirstDerivatives3D_PlusHalf(
                _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz,
                wavefieldM, wavefieldM, wavefieldM, _tmpMx1, _tmpMy1, _tmpMz1, _nbx, _nby, _nbz);

        applyFirstDerivatives3D_PlusHalf(
                _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz,
                _pOld, _pOld, _pOld, _tmpPx2, _tmpPy2, _tmpPz2, _nbx, _nby, _nbz);

        applyFirstDerivatives3D_PlusHalf(
                _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz,
                _mOld, _mOld, _mOld, _tmpMx2, _tmpMy2, _tmpMz2, _nbx, _nby, _nbz);

        // Sandwich terms for the adjoint accumulation
#pragma omp parallel for collapse(3) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long by = 0; by < _ny; by += _nby) {
                for (long bz = 0; bz < _nz; bz += _nbz) {
                    const long kxmax = MIN(bx + _nbx, _nx);
                    const long kymax = MIN(by + _nby, _ny);
                    const long kzmax = MIN(bz + _nbz, _nz);

                    for (long kx = bx; kx < kxmax; kx++) {
                        for (long ky = by; ky < kymax; ky++) {
#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long k = kx * _ny * _nz + ky * _nz + kz;

                                const float V = _v[k];
                                const float E = _eps[k];
                                const float A = _eta[k];
                                const float B = _b[k];
                                const float F = _f[k];

                                const float factor = 2 * B / (V * V * V);

                                dVel[k] += factor * (wavefieldDP[k] * _pOld[k] + wavefieldDM[k] * _mOld[k]);

                                dEps[k] += (-2 * B * _tmpPx1[k] * _tmpPx2[k] -2 * B * _tmpPy1[k] * _tmpPy2[k]);

                                const float partP = 2 * B * F * A * _tmpPz1[k] - (B * F * (1 - 2 * A * A) / sqrt(1 - A * A)) * _tmpMz1[k];
                                const float partM = 2 * B * F * A * _tmpMz1[k] + (B * F * (1 - 2 * A * A) / sqrt(1 - A * A)) * _tmpPz1[k];

                                dEta[k] += (partP * _tmpPz2[k] - partM * _tmpMz2[k]);
                            }
                        }
                    }
                }
            }
        }
    }

    template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline static void applyFirstDerivatives3D_PlusHalf_Sandwich(
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
            const Type * __restrict__ const inPX,
            const Type * __restrict__ const inPY,
            const Type * __restrict__ const inPZ,
            const Type * __restrict__ const inMX,
            const Type * __restrict__ const inMY,
            const Type * __restrict__ const inMZ,
            const Type * __restrict__ const fieldEps,
            const Type * __restrict__ const fieldEta,
            const Type * __restrict__ const fieldVsVp,
            const Type * __restrict__ const fieldBuoy,
            Type * __restrict__ tmpPX,
            Type * __restrict__ tmpPY,
            Type * __restrict__ tmpPZ,
            Type * __restrict__ tmpMX,
            Type * __restrict__ tmpMY,
            Type * __restrict__ tmpMZ,
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
                    const long kindex1 = kx * ny * nz + ky * nz + k;
                    const long kindex2 = kx * ny * nz + ky * nz + (nz - 1 - k);
                    tmpPX[kindex1] = tmpPX[kindex2] = 0;
                    tmpPY[kindex1] = tmpPY[kindex2] = 0;
                    tmpPZ[kindex1] = tmpPZ[kindex2] = 0;
                    tmpMX[kindex1] = tmpMX[kindex2] = 0;
                    tmpMY[kindex1] = tmpMY[kindex2] = 0;
                    tmpMZ[kindex1] = tmpMZ[kindex2] = 0;
                }
            }

#pragma omp parallel for num_threads(nthread) schedule(static)
        for (long kx = 0; kx < nx; kx++) {
#pragma omp simd
            for (long kz = 0; kz < nz; kz++) {
                const long kindex1 = kx * ny * nz + k * nz + kz;
                const long kindex2 = kx * ny * nz + (ny - 1 - k) * nz + kz;
                tmpPX[kindex1] = tmpPX[kindex2] = 0;
                tmpPY[kindex1] = tmpPY[kindex2] = 0;
                tmpPZ[kindex1] = tmpPZ[kindex2] = 0;
                tmpMX[kindex1] = tmpMX[kindex2] = 0;
                tmpMY[kindex1] = tmpMY[kindex2] = 0;
                tmpMZ[kindex1] = tmpMZ[kindex2] = 0;
            }
        }

#pragma omp parallel for num_threads(nthread) schedule(static)
        for (long ky = 0; ky < ny; ky++) {
#pragma omp simd
            for (long kz = 0; kz < nz; kz++) {
                const long kindex1 = k * ny * nz + ky * nz + kz;
                const long kindex2 = (nx - 1 - k) * ny * nz + ky * nz + kz;
                tmpPX[kindex1] = tmpPX[kindex2] = 0;
                tmpPY[kindex1] = tmpPY[kindex2] = 0;
                tmpPZ[kindex1] = tmpPZ[kindex2] = 0;
                tmpMX[kindex1] = tmpMX[kindex2] = 0;
                tmpMY[kindex1] = tmpMY[kindex2] = 0;
                tmpMZ[kindex1] = tmpMZ[kindex2] = 0;
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
                                const long k = kxnynz_kynz + kz;
                                const long kynz_kz = + kynz + kz;


                                const float stencilDPx =
                                        c8_1 * (- inPX[(kx+0) * nynz + kynz_kz] + inPX[(kx+1) * nynz + kynz_kz]) +
                                        c8_2 * (- inPX[(kx-1) * nynz + kynz_kz] + inPX[(kx+2) * nynz + kynz_kz]) +
                                        c8_3 * (- inPX[(kx-2) * nynz + kynz_kz] + inPX[(kx+3) * nynz + kynz_kz]) +
                                        c8_4 * (- inPX[(kx-3) * nynz + kynz_kz] + inPX[(kx+4) * nynz + kynz_kz]);

                                const float stencilDPy =
                                        c8_1 * (- inPY[kxnynz + (ky+0) * nz + kz] + inPY[kxnynz + (ky+1) * nz + kz]) +
                                        c8_2 * (- inPY[kxnynz + (ky-1) * nz + kz] + inPY[kxnynz + (ky+2) * nz + kz]) +
                                        c8_3 * (- inPY[kxnynz + (ky-2) * nz + kz] + inPY[kxnynz + (ky+3) * nz + kz]) +
                                        c8_4 * (- inPY[kxnynz + (ky-3) * nz + kz] + inPY[kxnynz + (ky+4) * nz + kz]);

                                const float stencilDPz =
                                        c8_1 * (- inPZ[kxnynz_kynz + (kz+0)] + inPZ[kxnynz_kynz + (kz+1)]) +
                                        c8_2 * (- inPZ[kxnynz_kynz + (kz-1)] + inPZ[kxnynz_kynz + (kz+2)]) +
                                        c8_3 * (- inPZ[kxnynz_kynz + (kz-2)] + inPZ[kxnynz_kynz + (kz+3)]) +
                                        c8_4 * (- inPZ[kxnynz_kynz + (kz-3)] + inPZ[kxnynz_kynz + (kz+4)]);

                                const float stencilDMx =
                                        c8_1 * (- inMX[(kx+0) * nynz + kynz_kz] + inMX[(kx+1) * nynz + kynz_kz]) +
                                        c8_2 * (- inMX[(kx-1) * nynz + kynz_kz] + inMX[(kx+2) * nynz + kynz_kz]) +
                                        c8_3 * (- inMX[(kx-2) * nynz + kynz_kz] + inMX[(kx+3) * nynz + kynz_kz]) +
                                        c8_4 * (- inMX[(kx-3) * nynz + kynz_kz] + inMX[(kx+4) * nynz + kynz_kz]);

                                const float stencilDMy =
                                        c8_1 * (- inMY[kxnynz + (ky+0) * nz + kz] + inMY[kxnynz + (ky+1) * nz + kz]) +
                                        c8_2 * (- inMY[kxnynz + (ky-1) * nz + kz] + inMY[kxnynz + (ky+2) * nz + kz]) +
                                        c8_3 * (- inMY[kxnynz + (ky-2) * nz + kz] + inMY[kxnynz + (ky+3) * nz + kz]) +
                                        c8_4 * (- inMY[kxnynz + (ky-3) * nz + kz] + inMY[kxnynz + (ky+4) * nz + kz]);

                                const float stencilDMz =
                                        c8_1 * (- inMZ[kxnynz_kynz + (kz+0)] + inMZ[kxnynz_kynz + (kz+1)]) +
                                        c8_2 * (- inMZ[kxnynz_kynz + (kz-1)] + inMZ[kxnynz_kynz + (kz+2)]) +
                                        c8_3 * (- inMZ[kxnynz_kynz + (kz-2)] + inMZ[kxnynz_kynz + (kz+3)]) +
                                        c8_4 * (- inMZ[kxnynz_kynz + (kz-3)] + inMZ[kxnynz_kynz + (kz+4)]);

                                const float dPx = invDx * stencilDPx;
                                const float dPy = invDy * stencilDPy;
                                const float dPz = invDz * stencilDPz;
                                const float dMx = invDx * stencilDMx;
                                const float dMy = invDy * stencilDMy;
                                const float dMz = invDz * stencilDMz;

                                const float E = 1 + 2 * fieldEps[k];
                                const float A = fieldEta[k];
                                const float F = fieldVsVp[k];
                                const float B = fieldBuoy[k];
                                const float SA2 = sqrt(1 - A * A);

                                tmpPX[k] = B * E * dPx;
                                tmpPY[k] = B * E * dPy;
                                tmpPZ[k] = B * (1 - F * A * A) * dPz + B * (F * A * SA2) * dMz;

                                tmpMX[k] = B * (1 - F) * dMx;
                                tmpMY[k] = B * (1 - F) * dMy;
                                tmpMZ[k] = B * F * A * SA2 * dPz + B * (1 - F + F * A * A) * dMz;
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
                    // [kxnynz_kynz + 0]
                    {
                        const Type stencilDPz0 =
                                c8_1 * (- inPZ[kxnynz_kynz + 0] + inPZ[kxnynz_kynz + 1]) +
                                c8_2 * (+ inPZ[kxnynz_kynz + 1] + inPZ[kxnynz_kynz + 2]) +
                                c8_3 * (+ inPZ[kxnynz_kynz + 2] + inPZ[kxnynz_kynz + 3]) +
                                c8_4 * (+ inPZ[kxnynz_kynz + 3] + inPZ[kxnynz_kynz + 4]);

                        const Type dPX = 0;
                        const Type dPY = 0;
                        const Type dPZ = invDz * stencilDPz0;

                        const Type stencilDMz0 =
                                c8_1 * (- inMZ[kxnynz_kynz + 0] + inMZ[kxnynz_kynz + 1]) +
                                c8_2 * (+ inMZ[kxnynz_kynz + 1] + inMZ[kxnynz_kynz + 2]) +
                                c8_3 * (+ inMZ[kxnynz_kynz + 2] + inMZ[kxnynz_kynz + 3]) +
                                c8_4 * (+ inMZ[kxnynz_kynz + 3] + inMZ[kxnynz_kynz + 4]);

                        const Type dMX = 0;
                        const Type dMY = 0;
                        const Type dMZ = invDz * stencilDMz0;

                        const long k = kxnynz_kynz + 0;

                        const Type E   = 1 + 2 * fieldEps[k];
                        const Type A   = fieldEta[k];
                        const Type F   = fieldVsVp[k];
                        const Type B   = fieldBuoy[k];
                        const Type SA2 = sqrt(1 - A * A);

                        tmpPX[k] = B * E * dPX;
                        tmpPY[k] = B * E * dPY;
                        tmpPZ[k] = B * (1 - F * A * A) * dPZ + B * (F * A * SA2) * dMZ;

                        tmpMX[k] = B * (1 - F) * dMX;
                        tmpMY[k] = B * (1 - F) * dMY;
                        tmpMZ[k] = B * F * A * SA2 * dPZ + B * (1 - F + F * A * A) * dMZ;
                    }

                    // kz = 1 -- 1 1/2 cells below free surface for Z derivative, 1 cells below for X/Y derivative
                    // [kxnynz_kynz + 1]
                    {
                        const Type stencilDPx1 =
                                c8_1 * (- inPX[(kx+0) * nynz + kynz + 1] + inPX[(kx+1) * nynz + kynz + 1]) +
                                c8_2 * (- inPX[(kx-1) * nynz + kynz + 1] + inPX[(kx+2) * nynz + kynz + 1]) +
                                c8_3 * (- inPX[(kx-2) * nynz + kynz + 1] + inPX[(kx+3) * nynz + kynz + 1]) +
                                c8_4 * (- inPX[(kx-3) * nynz + kynz + 1] + inPX[(kx+4) * nynz + kynz + 1]);

                        const Type stencilDPy1 =
                                c8_1 * (- inPY[kxnynz + (ky+0) * nz + 1] + inPY[kxnynz + (ky+1) * nz + 1]) +
                                c8_2 * (- inPY[kxnynz + (ky-1) * nz + 1] + inPY[kxnynz + (ky+2) * nz + 1]) +
                                c8_3 * (- inPY[kxnynz + (ky-2) * nz + 1] + inPY[kxnynz + (ky+3) * nz + 1]) +
                                c8_4 * (- inPY[kxnynz + (ky-3) * nz + 1] + inPY[kxnynz + (ky+4) * nz + 1]);

                        const Type stencilDPz1 =
                                c8_1 * (- inPZ[kxnynz_kynz + 1] + inPZ[kxnynz_kynz + 2]) +
                                c8_2 * (- inPZ[kxnynz_kynz + 0] + inPZ[kxnynz_kynz + 3]) +
                                c8_3 * (+ inPZ[kxnynz_kynz + 1] + inPZ[kxnynz_kynz + 4]) +
                                c8_4 * (+ inPZ[kxnynz_kynz + 2] + inPZ[kxnynz_kynz + 5]);

                        const Type dPX = invDx * stencilDPx1;
                        const Type dPY = invDy * stencilDPy1;
                        const Type dPZ = invDz * stencilDPz1;

                        const Type stencilDMx1 =
                                c8_1 * (- inMX[(kx+0) * nynz + kynz + 1] + inMX[(kx+1) * nynz + kynz + 1]) +
                                c8_2 * (- inMX[(kx-1) * nynz + kynz + 1] + inMX[(kx+2) * nynz + kynz + 1]) +
                                c8_3 * (- inMX[(kx-2) * nynz + kynz + 1] + inMX[(kx+3) * nynz + kynz + 1]) +
                                c8_4 * (- inMX[(kx-3) * nynz + kynz + 1] + inMX[(kx+4) * nynz + kynz + 1]);

                        const Type stencilDMy1 =
                                c8_1 * (- inMY[kxnynz + (ky+0) * nz + 1] + inMY[kxnynz + (ky+1) * nz + 1]) +
                                c8_2 * (- inMY[kxnynz + (ky-1) * nz + 1] + inMY[kxnynz + (ky+2) * nz + 1]) +
                                c8_3 * (- inMY[kxnynz + (ky-2) * nz + 1] + inMY[kxnynz + (ky+3) * nz + 1]) +
                                c8_4 * (- inMY[kxnynz + (ky-3) * nz + 1] + inMY[kxnynz + (ky+4) * nz + 1]);

                        const Type stencilDMz1 =
                                c8_1 * (- inMZ[kxnynz_kynz + 1] + inMZ[kxnynz_kynz + 2]) +
                                c8_2 * (- inMZ[kxnynz_kynz + 0] + inMZ[kxnynz_kynz + 3]) +
                                c8_3 * (+ inMZ[kxnynz_kynz + 1] + inMZ[kxnynz_kynz + 4]) +
                                c8_4 * (+ inMZ[kxnynz_kynz + 2] + inMZ[kxnynz_kynz + 5]);

                        const Type dMX = invDx * stencilDMx1;
                        const Type dMY = invDy * stencilDMy1;
                        const Type dMZ = invDz * stencilDMz1;

                        const long k = kxnynz_kynz + 1;

                        const Type E   = 1 + 2 * fieldEps[k];
                        const Type A   = fieldEta[k];
                        const Type F   = fieldVsVp[k];
                        const Type B   = fieldBuoy[k];
                        const Type SA2 = sqrt(1 - A * A);

                        tmpPX[k] = B * E * dPX;
                        tmpPY[k] = B * E * dPY;
                        tmpPZ[k] = B * (1 - F * A * A) * dPZ + B * (F * A * SA2) * dMZ;

                        tmpMX[k] = B * (1 - F) * dMX;
                        tmpMY[k] = B * (1 - F) * dMY;
                        tmpMZ[k] = B * F * A * SA2 * dPZ + B * (1 - F + F * A * A) * dMZ;
                    }

                    // kz = 2 -- 2 1/2 cells below free surface for Z derivative, 2 cells below for X/Y derivative
                    // [kxnynz_kynz + 2]
                    {
                        const Type stencilDPx2 =
                                c8_1 * (- inPX[(kx+0) * nynz + kynz + 2] + inPX[(kx+1) * nynz + kynz + 2]) +
                                c8_2 * (- inPX[(kx-1) * nynz + kynz + 2] + inPX[(kx+2) * nynz + kynz + 2]) +
                                c8_3 * (- inPX[(kx-2) * nynz + kynz + 2] + inPX[(kx+3) * nynz + kynz + 2]) +
                                c8_4 * (- inPX[(kx-3) * nynz + kynz + 2] + inPX[(kx+4) * nynz + kynz + 2]);

                        const Type stencilDPy2 =
                                c8_1 * (- inPY[kxnynz + (ky+0) * nz + 2] + inPY[kxnynz + (ky+1) * nz + 2]) +
                                c8_2 * (- inPY[kxnynz + (ky-1) * nz + 2] + inPY[kxnynz + (ky+2) * nz + 2]) +
                                c8_3 * (- inPY[kxnynz + (ky-2) * nz + 2] + inPY[kxnynz + (ky+3) * nz + 2]) +
                                c8_4 * (- inPY[kxnynz + (ky-3) * nz + 2] + inPY[kxnynz + (ky+4) * nz + 2]);

                        const Type stencilDPz2 =
                                c8_1 * (- inPZ[kxnynz_kynz + 2] + inPZ[kxnynz_kynz + 3]) +
                                c8_2 * (- inPZ[kxnynz_kynz + 1] + inPZ[kxnynz_kynz + 4]) +
                                c8_3 * (- inPZ[kxnynz_kynz + 0] + inPZ[kxnynz_kynz + 5]) +
                                c8_4 * (+ inPZ[kxnynz_kynz + 1] + inPZ[kxnynz_kynz + 6]);

                        const Type dPX = invDx * stencilDPx2;
                        const Type dPY = invDy * stencilDPy2;
                        const Type dPZ = invDz * stencilDPz2;

                        const Type stencilDMx2 =
                                c8_1 * (- inMX[(kx+0) * nynz + kynz + 2] + inMX[(kx+1) * nynz + kynz + 2]) +
                                c8_2 * (- inMX[(kx-1) * nynz + kynz + 2] + inMX[(kx+2) * nynz + kynz + 2]) +
                                c8_3 * (- inMX[(kx-2) * nynz + kynz + 2] + inMX[(kx+3) * nynz + kynz + 2]) +
                                c8_4 * (- inMX[(kx-3) * nynz + kynz + 2] + inMX[(kx+4) * nynz + kynz + 2]);

                        const Type stencilDMy2 =
                                c8_1 * (- inMY[kxnynz + (ky+0) * nz + 2] + inMY[kxnynz + (ky+1) * nz + 2]) +
                                c8_2 * (- inMY[kxnynz + (ky-1) * nz + 2] + inMY[kxnynz + (ky+2) * nz + 2]) +
                                c8_3 * (- inMY[kxnynz + (ky-2) * nz + 2] + inMY[kxnynz + (ky+3) * nz + 2]) +
                                c8_4 * (- inMY[kxnynz + (ky-3) * nz + 2] + inMY[kxnynz + (ky+4) * nz + 2]);

                        const Type stencilDMz2 =
                                c8_1 * (- inMZ[kxnynz_kynz + 2] + inMZ[kxnynz_kynz + 3]) +
                                c8_2 * (- inMZ[kxnynz_kynz + 1] + inMZ[kxnynz_kynz + 4]) +
                                c8_3 * (- inMZ[kxnynz_kynz + 0] + inMZ[kxnynz_kynz + 5]) +
                                c8_4 * (+ inMZ[kxnynz_kynz + 1] + inMZ[kxnynz_kynz + 6]);

                        const Type dMX = invDx * stencilDMx2;
                        const Type dMY = invDy * stencilDMy2;
                        const Type dMZ = invDz * stencilDMz2;

                        const long k = kxnynz_kynz + 2;

                        const Type E   = 1 + 2 * fieldEps[k];
                        const Type A   = fieldEta[k];
                        const Type F   = fieldVsVp[k];
                        const Type B   = fieldBuoy[k];
                        const Type SA2 = sqrt(1 - A * A);

                        tmpPX[k] = B * E * dPX;
                        tmpPY[k] = B * E * dPY;
                        tmpPZ[k] = B * (1 - F * A * A) * dPZ + B * (F * A * SA2) * dMZ;

                        tmpMX[k] = B * (1 - F) * dMX;
                        tmpMY[k] = B * (1 - F) * dMY;
                        tmpMZ[k] = B * F * A * SA2 * dPZ + B * (1 - F + F * A * A) * dMZ;
                    }

                    // kz = 3 -- 3 1/2 cells below free surface for Z derivative, 3 cells below for X/Y derivative
                    // [kxnynz_kynz + 3]
                    {
                        const Type stencilDPx3 =
                                c8_1 * (- inPX[(kx+0) * nynz + kynz + 3] + inPX[(kx+1) * nynz + kynz + 3]) +
                                c8_2 * (- inPX[(kx-1) * nynz + kynz + 3] + inPX[(kx+2) * nynz + kynz + 3]) +
                                c8_3 * (- inPX[(kx-2) * nynz + kynz + 3] + inPX[(kx+3) * nynz + kynz + 3]) +
                                c8_4 * (- inPX[(kx-3) * nynz + kynz + 3] + inPX[(kx+4) * nynz + kynz + 3]);

                        const Type stencilDPy3 =
                                c8_1 * (- inPY[kxnynz + (ky+0) * nz + 3] + inPY[kxnynz + (ky+1) * nz + 3]) +
                                c8_2 * (- inPY[kxnynz + (ky-1) * nz + 3] + inPY[kxnynz + (ky+2) * nz + 3]) +
                                c8_3 * (- inPY[kxnynz + (ky-2) * nz + 3] + inPY[kxnynz + (ky+3) * nz + 3]) +
                                c8_4 * (- inPY[kxnynz + (ky-3) * nz + 3] + inPY[kxnynz + (ky+4) * nz + 3]);

                        const Type stencilDPz3 =
                                c8_1 * (- inPZ[kxnynz_kynz + 3] + inPZ[kxnynz_kynz + 4]) +
                                c8_2 * (- inPZ[kxnynz_kynz + 2] + inPZ[kxnynz_kynz + 5]) +
                                c8_3 * (- inPZ[kxnynz_kynz + 1] + inPZ[kxnynz_kynz + 6]) +
                                c8_4 * (- inPZ[kxnynz_kynz + 0] + inPZ[kxnynz_kynz + 7]);

                        const Type dPX = invDx * stencilDPx3;
                        const Type dPY = invDy * stencilDPy3;
                        const Type dPZ = invDz * stencilDPz3;

                        const Type stencilDMx3 =
                                c8_1 * (- inMX[(kx+0) * nynz + kynz + 3] + inMX[(kx+1) * nynz + kynz + 3]) +
                                c8_2 * (- inMX[(kx-1) * nynz + kynz + 3] + inMX[(kx+2) * nynz + kynz + 3]) +
                                c8_3 * (- inMX[(kx-2) * nynz + kynz + 3] + inMX[(kx+3) * nynz + kynz + 3]) +
                                c8_4 * (- inMX[(kx-3) * nynz + kynz + 3] + inMX[(kx+4) * nynz + kynz + 3]);

                        const Type stencilDMy3 =
                                c8_1 * (- inMY[kxnynz + (ky+0) * nz + 3] + inMY[kxnynz + (ky+1) * nz + 3]) +
                                c8_2 * (- inMY[kxnynz + (ky-1) * nz + 3] + inMY[kxnynz + (ky+2) * nz + 3]) +
                                c8_3 * (- inMY[kxnynz + (ky-2) * nz + 3] + inMY[kxnynz + (ky+3) * nz + 3]) +
                                c8_4 * (- inMY[kxnynz + (ky-3) * nz + 3] + inMY[kxnynz + (ky+4) * nz + 3]);

                        const Type stencilDMz3 =
                                c8_1 * (- inMZ[kxnynz_kynz + 3] + inMZ[kxnynz_kynz + 4]) +
                                c8_2 * (- inMZ[kxnynz_kynz + 2] + inMZ[kxnynz_kynz + 5]) +
                                c8_3 * (- inMZ[kxnynz_kynz + 1] + inMZ[kxnynz_kynz + 6]) +
                                c8_4 * (- inMZ[kxnynz_kynz + 0] + inMZ[kxnynz_kynz + 7]);

                        const Type dMX = invDx * stencilDMx3;
                        const Type dMY = invDy * stencilDMy3;
                        const Type dMZ = invDz * stencilDMz3;

                        const long k = kxnynz_kynz + 3;

                        const Type E   = 1 + 2 * fieldEps[k];
                        const Type A   = fieldEta[k];
                        const Type F   = fieldVsVp[k];
                        const Type B   = fieldBuoy[k];
                        const Type SA2 = sqrt(1 - A * A);

                        tmpPX[k] = B * E * dPX;
                        tmpPY[k] = B * E * dPY;
                        tmpPZ[k] = B * (1 - F * A * A) * dPZ + B * (F * A * SA2) * dMZ;

                        tmpMX[k] = B * (1 - F) * dMX;
                        tmpMY[k] = B * (1 - F) * dMY;
                        tmpMZ[k] = B * F * A * SA2 * dPZ + B * (1 - F + F * A * A) * dMZ;
                    }
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
     * see notes in method secondOrderTimeUpdate_BubeConservation()
     *
     * Nonlinear method: outputs the spatial derivatives for serialization
     * Linear    method: does not output the spatial derivatives
     */
    template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline static void applyFirstDerivatives3D_MinusHalf_TimeUpdate_Nonlinear(
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
            const Type dtMod,
            const Type * __restrict__ const tmpPX,
            const Type * __restrict__ const tmpPY,
            const Type * __restrict__ const tmpPZ,
            const Type * __restrict__ const tmpMX,
            const Type * __restrict__ const tmpMY,
            const Type * __restrict__ const tmpMZ,
            const Type * __restrict__ const fieldVel,
            const Type * __restrict__ const fieldBuoy,
            const Type * __restrict__ const dtOmegaInvQ,
            const Type * __restrict__ const pCur,
            const Type * __restrict__ const mCur,
            Type * __restrict__ pSpace,
            Type * __restrict__ mSpace,
            Type * __restrict__ pOld,
            Type * __restrict__ mOld,
            const long BX_3D,
            const long BY_3D,
            const long BZ_3D) {

        const long nx4 = nx - 4;
        const long ny4 = ny - 4;
        const long nz4 = nz - 4;
        const long nynz = ny * nz;
        const Type dt2 = dtMod * dtMod;

    	// zero output array: note only the annulus that is in the absorbing boundary needs to be zeroed
    	for (long k = 0; k < 4; k++) {

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long kx = 0; kx < nx; kx++) {
#pragma omp simd
                for (long ky = 0; ky < ny; ky++) {
                    const long kindex1 = kx * ny * nz + ky * nz + k;
                    const long kindex2 = kx * ny * nz + ky * nz + (nz - 1 - k);
                    pSpace[kindex1] = pSpace[kindex2] = 0;
                    mSpace[kindex1] = mSpace[kindex2] = 0;
                }
            }

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long kx = 0; kx < nx; kx++) {
#pragma omp simd
                for (long kz = 0; kz < nz; kz++) {
                    const long kindex1 = kx * ny * nz + k * nz + kz;
                    const long kindex2 = kx * ny * nz + (ny - 1 - k) * nz + kz;
                    pSpace[kindex1] = pSpace[kindex2] = 0;
                    mSpace[kindex1] = mSpace[kindex2] = 0;
                }
            }

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long ky = 0; ky < ny; ky++) {
#pragma omp simd
                for (long kz = 0; kz < nz; kz++) {
                    const long kindex1 = k * ny * nz + ky * nz + kz;
                    const long kindex2 = (nx - 1 - k) * ny * nz + ky * nz + kz;
                    pSpace[kindex1] = pSpace[kindex2] = 0;
                    mSpace[kindex1] = mSpace[kindex2] = 0;
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
                                const long k = kxnynz_kynz + kz;
                                const long kynz_kz = + kynz + kz;

                                const Type stencilDPx =
                                        c8_1 * (- tmpPX[(kx-1) * nynz + kynz_kz] + tmpPX[(kx+0) * nynz + kynz_kz]) +
                                        c8_2 * (- tmpPX[(kx-2) * nynz + kynz_kz] + tmpPX[(kx+1) * nynz + kynz_kz]) +
                                        c8_3 * (- tmpPX[(kx-3) * nynz + kynz_kz] + tmpPX[(kx+2) * nynz + kynz_kz]) +
                                        c8_4 * (- tmpPX[(kx-4) * nynz + kynz_kz] + tmpPX[(kx+3) * nynz + kynz_kz]);

                                const Type stencilDPy =
                                        c8_1 * (- tmpPY[kxnynz + (ky-1) * nz + kz] + tmpPY[kxnynz + (ky+0) * nz + kz]) +
                                        c8_2 * (- tmpPY[kxnynz + (ky-2) * nz + kz] + tmpPY[kxnynz + (ky+1) * nz + kz]) +
                                        c8_3 * (- tmpPY[kxnynz + (ky-3) * nz + kz] + tmpPY[kxnynz + (ky+2) * nz + kz]) +
                                        c8_4 * (- tmpPY[kxnynz + (ky-4) * nz + kz] + tmpPY[kxnynz + (ky+3) * nz + kz]);

                                const Type stencilDPz =
                                        c8_1 * (- tmpPZ[kxnynz_kynz + (kz-1)] + tmpPZ[kxnynz_kynz + (kz+0)]) +
                                        c8_2 * (- tmpPZ[kxnynz_kynz + (kz-2)] + tmpPZ[kxnynz_kynz + (kz+1)]) +
                                        c8_3 * (- tmpPZ[kxnynz_kynz + (kz-3)] + tmpPZ[kxnynz_kynz + (kz+2)]) +
                                        c8_4 * (- tmpPZ[kxnynz_kynz + (kz-4)] + tmpPZ[kxnynz_kynz + (kz+3)]);

                                const Type stencilDMx =
                                        c8_1 * (- tmpMX[(kx-1) * nynz + kynz_kz] + tmpMX[(kx+0) * nynz + kynz_kz]) +
                                        c8_2 * (- tmpMX[(kx-2) * nynz + kynz_kz] + tmpMX[(kx+1) * nynz + kynz_kz]) +
                                        c8_3 * (- tmpMX[(kx-3) * nynz + kynz_kz] + tmpMX[(kx+2) * nynz + kynz_kz]) +
                                        c8_4 * (- tmpMX[(kx-4) * nynz + kynz_kz] + tmpMX[(kx+3) * nynz + kynz_kz]);

                                const Type stencilDMy =
                                        c8_1 * (- tmpMY[kxnynz + (ky-1) * nz + kz] + tmpMY[kxnynz + (ky+0) * nz + kz]) +
                                        c8_2 * (- tmpMY[kxnynz + (ky-2) * nz + kz] + tmpMY[kxnynz + (ky+1) * nz + kz]) +
                                        c8_3 * (- tmpMY[kxnynz + (ky-3) * nz + kz] + tmpMY[kxnynz + (ky+2) * nz + kz]) +
                                        c8_4 * (- tmpMY[kxnynz + (ky-4) * nz + kz] + tmpMY[kxnynz + (ky+3) * nz + kz]);

                                const Type stencilDMz =
                                        c8_1 * (- tmpMZ[kxnynz_kynz + (kz-1)] + tmpMZ[kxnynz_kynz + (kz+0)]) +
                                        c8_2 * (- tmpMZ[kxnynz_kynz + (kz-2)] + tmpMZ[kxnynz_kynz + (kz+1)]) +
                                        c8_3 * (- tmpMZ[kxnynz_kynz + (kz-3)] + tmpMZ[kxnynz_kynz + (kz+2)]) +
                                        c8_4 * (- tmpMZ[kxnynz_kynz + (kz-4)] + tmpMZ[kxnynz_kynz + (kz+3)]);

                                const Type dPx = invDx * stencilDPx;
                                const Type dPy = invDy * stencilDPy;
                                const Type dPz = invDz * stencilDPz;
                                const Type dMx = invDx * stencilDMx;
                                const Type dMy = invDy * stencilDMy;
                                const Type dMz = invDz * stencilDMz;

                                const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                                pSpace[k] = dPx + dPy + dPz;
                                mSpace[k] = dMx + dMy + dMz;

                                pOld[k] = dt2V2_B * pSpace[k] - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                                mOld[k] = dt2V2_B * mSpace[k] - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
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
                    // [kxnynz_kynz + 0]
                    {
                        const Type dPx = 0;
                        const Type dPy = 0;
                        const Type dPz = 0;

                        const Type dMx = 0;
                        const Type dMy = 0;
                        const Type dMz = 0;

                        const long k = kxnynz_kynz + 0;
                        const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                        pOld[k] = dt2V2_B * (dPx + dPy + dPz) - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                        mOld[k] = dt2V2_B * (dMx + dMy + dMz) - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];

                        pSpace[k] = dPx + dPy + dPz;
                        mSpace[k] = dMx + dMy + dMz;
                    }

                    // kz = 1 -- one cell below the free surface
                    // [kxnynz_kynz + 1]
                    {
                        const Type stencilDPx1 =
                                c8_1 * (- tmpPX[(kx-1) * nynz + kynz + 1] + tmpPX[(kx+0) * nynz + kynz + 1]) +
                                c8_2 * (- tmpPX[(kx-2) * nynz + kynz + 1] + tmpPX[(kx+1) * nynz + kynz + 1]) +
                                c8_3 * (- tmpPX[(kx-3) * nynz + kynz + 1] + tmpPX[(kx+2) * nynz + kynz + 1]) +
                                c8_4 * (- tmpPX[(kx-4) * nynz + kynz + 1] + tmpPX[(kx+3) * nynz + kynz + 1]);

                        const Type stencilDPy1 =
                                c8_1 * (- tmpPY[kxnynz + (ky-1) * nz + 1] + tmpPY[kxnynz + (ky+0) * nz + 1]) +
                                c8_2 * (- tmpPY[kxnynz + (ky-2) * nz + 1] + tmpPY[kxnynz + (ky+1) * nz + 1]) +
                                c8_3 * (- tmpPY[kxnynz + (ky-3) * nz + 1] + tmpPY[kxnynz + (ky+2) * nz + 1]) +
                                c8_4 * (- tmpPY[kxnynz + (ky-4) * nz + 1] + tmpPY[kxnynz + (ky+3) * nz + 1]);

                        const Type stencilDPz1 =
                                c8_1 * (- tmpPZ[kxnynz_kynz + 0] + tmpPZ[kxnynz_kynz + 1]) +
                                c8_2 * (- tmpPZ[kxnynz_kynz + 0] + tmpPZ[kxnynz_kynz + 2]) +
                                c8_3 * (- tmpPZ[kxnynz_kynz + 1] + tmpPZ[kxnynz_kynz + 3]) +
                                c8_4 * (- tmpPZ[kxnynz_kynz + 2] + tmpPZ[kxnynz_kynz + 4]);

                        const Type stencilDMx1 =
                                c8_1 * (- tmpMX[(kx-1) * nynz + kynz + 1] + tmpMX[(kx+0) * nynz + kynz + 1]) +
                                c8_2 * (- tmpMX[(kx-2) * nynz + kynz + 1] + tmpMX[(kx+1) * nynz + kynz + 1]) +
                                c8_3 * (- tmpMX[(kx-3) * nynz + kynz + 1] + tmpMX[(kx+2) * nynz + kynz + 1]) +
                                c8_4 * (- tmpMX[(kx-4) * nynz + kynz + 1] + tmpMX[(kx+3) * nynz + kynz + 1]);

                        const Type stencilDMy1 =
                                c8_1 * (- tmpMY[kxnynz + (ky-1) * nz + 1] + tmpMY[kxnynz + (ky+0) * nz + 1]) +
                                c8_2 * (- tmpMY[kxnynz + (ky-2) * nz + 1] + tmpMY[kxnynz + (ky+1) * nz + 1]) +
                                c8_3 * (- tmpMY[kxnynz + (ky-3) * nz + 1] + tmpMY[kxnynz + (ky+2) * nz + 1]) +
                                c8_4 * (- tmpMY[kxnynz + (ky-4) * nz + 1] + tmpMY[kxnynz + (ky+3) * nz + 1]);

                        const Type stencilDMz1 =
                                c8_1 * (- tmpMZ[kxnynz_kynz + 0] + tmpMZ[kxnynz_kynz + 1]) +
                                c8_2 * (- tmpMZ[kxnynz_kynz + 0] + tmpMZ[kxnynz_kynz + 2]) +
                                c8_3 * (- tmpMZ[kxnynz_kynz + 1] + tmpMZ[kxnynz_kynz + 3]) +
                                c8_4 * (- tmpMZ[kxnynz_kynz + 2] + tmpMZ[kxnynz_kynz + 4]);

                        const Type dPx = invDx * stencilDPx1;
                        const Type dPy = invDy * stencilDPy1;
                        const Type dPz = invDz * stencilDPz1;
                        const Type dMx = invDx * stencilDMx1;
                        const Type dMy = invDy * stencilDMy1;
                        const Type dMz = invDz * stencilDMz1;

                        const long k = kxnynz_kynz + 1;
                        const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                        pSpace[k] = dPx + dPy + dPz;
                        mSpace[k] = dMx + dMy + dMz;

                        pOld[k] = dt2V2_B * pSpace[k] - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                        mOld[k] = dt2V2_B * mSpace[k] - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
                    }

                    // kz = 2 -- two cells below the free surface
                    // [kxnynz_kynz + 2]
                    {
                        const Type stencilDPx2 =
                                c8_1 * (- tmpPX[(kx-1) * nynz + kynz + 2] + tmpPX[(kx+0) * nynz + kynz + 2]) +
                                c8_2 * (- tmpPX[(kx-2) * nynz + kynz + 2] + tmpPX[(kx+1) * nynz + kynz + 2]) +
                                c8_3 * (- tmpPX[(kx-3) * nynz + kynz + 2] + tmpPX[(kx+2) * nynz + kynz + 2]) +
                                c8_4 * (- tmpPX[(kx-4) * nynz + kynz + 2] + tmpPX[(kx+3) * nynz + kynz + 2]);

                        const Type stencilDPy2 =
                                c8_1 * (- tmpPY[kxnynz + (ky-1) * nz + 2] + tmpPY[kxnynz + (ky+0) * nz + 2]) +
                                c8_2 * (- tmpPY[kxnynz + (ky-2) * nz + 2] + tmpPY[kxnynz + (ky+1) * nz + 2]) +
                                c8_3 * (- tmpPY[kxnynz + (ky-3) * nz + 2] + tmpPY[kxnynz + (ky+2) * nz + 2]) +
                                c8_4 * (- tmpPY[kxnynz + (ky-4) * nz + 2] + tmpPY[kxnynz + (ky+3) * nz + 2]);

                        const Type stencilDPz2 =
                                c8_1 * (- tmpPZ[kxnynz_kynz + 1] + tmpPZ[kxnynz_kynz + 2]) +
                                c8_2 * (- tmpPZ[kxnynz_kynz + 0] + tmpPZ[kxnynz_kynz + 3]) +
                                c8_3 * (- tmpPZ[kxnynz_kynz + 0] + tmpPZ[kxnynz_kynz + 4]) +
                                c8_4 * (- tmpPZ[kxnynz_kynz + 1] + tmpPZ[kxnynz_kynz + 5]);

                        const Type stencilDMx2 =
                                c8_1 * (- tmpMX[(kx-1) * nynz + kynz + 2] + tmpMX[(kx+0) * nynz + kynz + 2]) +
                                c8_2 * (- tmpMX[(kx-2) * nynz + kynz + 2] + tmpMX[(kx+1) * nynz + kynz + 2]) +
                                c8_3 * (- tmpMX[(kx-3) * nynz + kynz + 2] + tmpMX[(kx+2) * nynz + kynz + 2]) +
                                c8_4 * (- tmpMX[(kx-4) * nynz + kynz + 2] + tmpMX[(kx+3) * nynz + kynz + 2]);

                        const Type stencilDMy2 =
                                c8_1 * (- tmpMY[kxnynz + (ky-1) * nz + 2] + tmpMY[kxnynz + (ky+0) * nz + 2]) +
                                c8_2 * (- tmpMY[kxnynz + (ky-2) * nz + 2] + tmpMY[kxnynz + (ky+1) * nz + 2]) +
                                c8_3 * (- tmpMY[kxnynz + (ky-3) * nz + 2] + tmpMY[kxnynz + (ky+2) * nz + 2]) +
                                c8_4 * (- tmpMY[kxnynz + (ky-4) * nz + 2] + tmpMY[kxnynz + (ky+3) * nz + 2]);

                        const Type stencilDMz2 =
                                c8_1 * (- tmpMZ[kxnynz_kynz + 1] + tmpMZ[kxnynz_kynz + 2]) +
                                c8_2 * (- tmpMZ[kxnynz_kynz + 0] + tmpMZ[kxnynz_kynz + 3]) +
                                c8_3 * (- tmpMZ[kxnynz_kynz + 0] + tmpMZ[kxnynz_kynz + 4]) +
                                c8_4 * (- tmpMZ[kxnynz_kynz + 1] + tmpMZ[kxnynz_kynz + 5]);

                        const Type dPx = invDx * stencilDPx2;
                        const Type dPy = invDy * stencilDPy2;
                        const Type dPz = invDz * stencilDPz2;
                        const Type dMx = invDx * stencilDMx2;
                        const Type dMy = invDy * stencilDMy2;
                        const Type dMz = invDz * stencilDMz2;

                        const long k = kxnynz_kynz + 2;
                        const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                        pSpace[k] = dPx + dPy + dPz;
                        mSpace[k] = dMx + dMy + dMz;

                        pOld[k] = dt2V2_B * pSpace[k] - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                        mOld[k] = dt2V2_B * mSpace[k] - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
                    }

                    // kz = 3 -- three cells below the free surface
                    // [kxnynz_kynz + 3]
                    {
                        const Type stencilDPx3 =
                                c8_1 * (- tmpPX[(kx-1) * nynz + kynz + 3] + tmpPX[(kx+0) * nynz + kynz + 3]) +
                                c8_2 * (- tmpPX[(kx-2) * nynz + kynz + 3] + tmpPX[(kx+1) * nynz + kynz + 3]) +
                                c8_3 * (- tmpPX[(kx-3) * nynz + kynz + 3] + tmpPX[(kx+2) * nynz + kynz + 3]) +
                                c8_4 * (- tmpPX[(kx-4) * nynz + kynz + 3] + tmpPX[(kx+3) * nynz + kynz + 3]);

                        const Type stencilDPy3 =
                                c8_1 * (- tmpPY[kxnynz + (ky-1) * nz + 3] + tmpPY[kxnynz + (ky+0) * nz + 3]) +
                                c8_2 * (- tmpPY[kxnynz + (ky-2) * nz + 3] + tmpPY[kxnynz + (ky+1) * nz + 3]) +
                                c8_3 * (- tmpPY[kxnynz + (ky-3) * nz + 3] + tmpPY[kxnynz + (ky+2) * nz + 3]) +
                                c8_4 * (- tmpPY[kxnynz + (ky-4) * nz + 3] + tmpPY[kxnynz + (ky+3) * nz + 3]);

                        const Type stencilDPz3 =
                                c8_1 * (- tmpPZ[kxnynz_kynz + 2] + tmpPZ[kxnynz_kynz + 3]) +
                                c8_2 * (- tmpPZ[kxnynz_kynz + 1] + tmpPZ[kxnynz_kynz + 4]) +
                                c8_3 * (- tmpPZ[kxnynz_kynz + 0] + tmpPZ[kxnynz_kynz + 5]) +
                                c8_4 * (- tmpPZ[kxnynz_kynz + 0] + tmpPZ[kxnynz_kynz + 6]);

                        const Type stencilDMx3 =
                                c8_1 * (- tmpMX[(kx-1) * nynz + kynz + 3] + tmpMX[(kx+0) * nynz + kynz + 3]) +
                                c8_2 * (- tmpMX[(kx-2) * nynz + kynz + 3] + tmpMX[(kx+1) * nynz + kynz + 3]) +
                                c8_3 * (- tmpMX[(kx-3) * nynz + kynz + 3] + tmpMX[(kx+2) * nynz + kynz + 3]) +
                                c8_4 * (- tmpMX[(kx-4) * nynz + kynz + 3] + tmpMX[(kx+3) * nynz + kynz + 3]);

                        const Type stencilDMy3 =
                                c8_1 * (- tmpMY[kxnynz + (ky-1) * nz + 3] + tmpMY[kxnynz + (ky+0) * nz + 3]) +
                                c8_2 * (- tmpMY[kxnynz + (ky-2) * nz + 3] + tmpMY[kxnynz + (ky+1) * nz + 3]) +
                                c8_3 * (- tmpMY[kxnynz + (ky-3) * nz + 3] + tmpMY[kxnynz + (ky+2) * nz + 3]) +
                                c8_4 * (- tmpMY[kxnynz + (ky-4) * nz + 3] + tmpMY[kxnynz + (ky+3) * nz + 3]);

                        const Type stencilDMz3 =
                                c8_1 * (- tmpMZ[kxnynz_kynz + 2] + tmpMZ[kxnynz_kynz + 3]) +
                                c8_2 * (- tmpMZ[kxnynz_kynz + 1] + tmpMZ[kxnynz_kynz + 4]) +
                                c8_3 * (- tmpMZ[kxnynz_kynz + 0] + tmpMZ[kxnynz_kynz + 5]) +
                                c8_4 * (- tmpMZ[kxnynz_kynz + 0] + tmpMZ[kxnynz_kynz + 6]);

                        const Type dPx = invDx * stencilDPx3;
                        const Type dPy = invDy * stencilDPy3;
                        const Type dPz = invDz * stencilDPz3;
                        const Type dMx = invDx * stencilDMx3;
                        const Type dMy = invDy * stencilDMy3;
                        const Type dMz = invDz * stencilDMz3;

                        const long k = kxnynz_kynz + 3;
                        const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                        pSpace[k] = dPx + dPy + dPz;
                        mSpace[k] = dMx + dMy + dMz;

                        pOld[k] = dt2V2_B * pSpace[k] - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                        mOld[k] = dt2V2_B * mSpace[k] - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
                    }
                }
            }
        }
    }

    template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline static void applyFirstDerivatives3D_MinusHalf_TimeUpdate_Linear(
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
            const Type dtMod,
            const Type * __restrict__ const tmpPX,
            const Type * __restrict__ const tmpPY,
            const Type * __restrict__ const tmpPZ,
            const Type * __restrict__ const tmpMX,
            const Type * __restrict__ const tmpMY,
            const Type * __restrict__ const tmpMZ,
            const Type * __restrict__ const fieldVel,
            const Type * __restrict__ const fieldBuoy,
            const Type * __restrict__ const dtOmegaInvQ,
            const Type * __restrict__ const pCur,
            const Type * __restrict__ const mCur,
            Type * __restrict__ pOld,
            Type * __restrict__ mOld,
            const long BX_3D,
            const long BY_3D,
            const long BZ_3D) {

        const long nx4 = nx - 4;
        const long ny4 = ny - 4;
        const long nz4 = nz - 4;
        const long nynz = ny * nz;
        const Type dt2 = dtMod * dtMod;

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
                                const long k = kxnynz_kynz + kz;
                                const long kynz_kz = + kynz + kz;

                                const Type stencilDPx =
                                        c8_1 * (- tmpPX[(kx-1) * nynz + kynz_kz] + tmpPX[(kx+0) * nynz + kynz_kz]) +
                                        c8_2 * (- tmpPX[(kx-2) * nynz + kynz_kz] + tmpPX[(kx+1) * nynz + kynz_kz]) +
                                        c8_3 * (- tmpPX[(kx-3) * nynz + kynz_kz] + tmpPX[(kx+2) * nynz + kynz_kz]) +
                                        c8_4 * (- tmpPX[(kx-4) * nynz + kynz_kz] + tmpPX[(kx+3) * nynz + kynz_kz]);

                                const Type stencilDPy =
                                        c8_1 * (- tmpPY[kxnynz + (ky-1) * nz + kz] + tmpPY[kxnynz + (ky+0) * nz + kz]) +
                                        c8_2 * (- tmpPY[kxnynz + (ky-2) * nz + kz] + tmpPY[kxnynz + (ky+1) * nz + kz]) +
                                        c8_3 * (- tmpPY[kxnynz + (ky-3) * nz + kz] + tmpPY[kxnynz + (ky+2) * nz + kz]) +
                                        c8_4 * (- tmpPY[kxnynz + (ky-4) * nz + kz] + tmpPY[kxnynz + (ky+3) * nz + kz]);

                                const Type stencilDPz =
                                        c8_1 * (- tmpPZ[kxnynz_kynz + (kz-1)] + tmpPZ[kxnynz_kynz + (kz+0)]) +
                                        c8_2 * (- tmpPZ[kxnynz_kynz + (kz-2)] + tmpPZ[kxnynz_kynz + (kz+1)]) +
                                        c8_3 * (- tmpPZ[kxnynz_kynz + (kz-3)] + tmpPZ[kxnynz_kynz + (kz+2)]) +
                                        c8_4 * (- tmpPZ[kxnynz_kynz + (kz-4)] + tmpPZ[kxnynz_kynz + (kz+3)]);

                                const Type stencilDMx =
                                        c8_1 * (- tmpMX[(kx-1) * nynz + kynz_kz] + tmpMX[(kx+0) * nynz + kynz_kz]) +
                                        c8_2 * (- tmpMX[(kx-2) * nynz + kynz_kz] + tmpMX[(kx+1) * nynz + kynz_kz]) +
                                        c8_3 * (- tmpMX[(kx-3) * nynz + kynz_kz] + tmpMX[(kx+2) * nynz + kynz_kz]) +
                                        c8_4 * (- tmpMX[(kx-4) * nynz + kynz_kz] + tmpMX[(kx+3) * nynz + kynz_kz]);

                                const Type stencilDMy =
                                        c8_1 * (- tmpMY[kxnynz + (ky-1) * nz + kz] + tmpMY[kxnynz + (ky+0) * nz + kz]) +
                                        c8_2 * (- tmpMY[kxnynz + (ky-2) * nz + kz] + tmpMY[kxnynz + (ky+1) * nz + kz]) +
                                        c8_3 * (- tmpMY[kxnynz + (ky-3) * nz + kz] + tmpMY[kxnynz + (ky+2) * nz + kz]) +
                                        c8_4 * (- tmpMY[kxnynz + (ky-4) * nz + kz] + tmpMY[kxnynz + (ky+3) * nz + kz]);

                                const Type stencilDMz =
                                        c8_1 * (- tmpMZ[kxnynz_kynz + (kz-1)] + tmpMZ[kxnynz_kynz + (kz+0)]) +
                                        c8_2 * (- tmpMZ[kxnynz_kynz + (kz-2)] + tmpMZ[kxnynz_kynz + (kz+1)]) +
                                        c8_3 * (- tmpMZ[kxnynz_kynz + (kz-3)] + tmpMZ[kxnynz_kynz + (kz+2)]) +
                                        c8_4 * (- tmpMZ[kxnynz_kynz + (kz-4)] + tmpMZ[kxnynz_kynz + (kz+3)]);

                                const Type dPx = invDx * stencilDPx;
                                const Type dPy = invDy * stencilDPy;
                                const Type dPz = invDz * stencilDPz;
                                const Type dMx = invDx * stencilDMx;
                                const Type dMy = invDy * stencilDMy;
                                const Type dMz = invDz * stencilDMz;

                                const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                                pOld[k] = dt2V2_B * (dPx + dPy + dPz) - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                                mOld[k] = dt2V2_B * (dMx + dMy + dMz) - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
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
                    // [kxnynz_kynz + 0]
                    {
                        const Type dPx = 0;
                        const Type dPy = 0;
                        const Type dPz = 0;
                        const Type dMx = 0;
                        const Type dMy = 0;
                        const Type dMz = 0;

                        const long k = kxnynz_kynz + 0;
                        const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                        pOld[k] = dt2V2_B * (dPx + dPy + dPz) - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                        mOld[k] = dt2V2_B * (dMx + dMy + dMz) - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
                    }

                    // kz = 1 -- one cell below the free surface
                    // [kxnynz_kynz + 1]
                    {
                        const Type stencilDPx1 =
                                c8_1 * (- tmpPX[(kx-1) * nynz + kynz + 1] + tmpPX[(kx+0) * nynz + kynz + 1]) +
                                c8_2 * (- tmpPX[(kx-2) * nynz + kynz + 1] + tmpPX[(kx+1) * nynz + kynz + 1]) +
                                c8_3 * (- tmpPX[(kx-3) * nynz + kynz + 1] + tmpPX[(kx+2) * nynz + kynz + 1]) +
                                c8_4 * (- tmpPX[(kx-4) * nynz + kynz + 1] + tmpPX[(kx+3) * nynz + kynz + 1]);

                        const Type stencilDPy1 =
                                c8_1 * (- tmpPY[kxnynz + (ky-1) * nz + 1] + tmpPY[kxnynz + (ky+0) * nz + 1]) +
                                c8_2 * (- tmpPY[kxnynz + (ky-2) * nz + 1] + tmpPY[kxnynz + (ky+1) * nz + 1]) +
                                c8_3 * (- tmpPY[kxnynz + (ky-3) * nz + 1] + tmpPY[kxnynz + (ky+2) * nz + 1]) +
                                c8_4 * (- tmpPY[kxnynz + (ky-4) * nz + 1] + tmpPY[kxnynz + (ky+3) * nz + 1]);

                        const Type stencilDPz1 =
                                c8_1 * (- tmpPZ[kxnynz_kynz + 0] + tmpPZ[kxnynz_kynz + 1]) +
                                c8_2 * (- tmpPZ[kxnynz_kynz + 0] + tmpPZ[kxnynz_kynz + 2]) +
                                c8_3 * (- tmpPZ[kxnynz_kynz + 1] + tmpPZ[kxnynz_kynz + 3]) +
                                c8_4 * (- tmpPZ[kxnynz_kynz + 2] + tmpPZ[kxnynz_kynz + 4]);

                        const Type stencilDMx1 =
                                c8_1 * (- tmpMX[(kx-1) * nynz + kynz + 1] + tmpMX[(kx+0) * nynz + kynz + 1]) +
                                c8_2 * (- tmpMX[(kx-2) * nynz + kynz + 1] + tmpMX[(kx+1) * nynz + kynz + 1]) +
                                c8_3 * (- tmpMX[(kx-3) * nynz + kynz + 1] + tmpMX[(kx+2) * nynz + kynz + 1]) +
                                c8_4 * (- tmpMX[(kx-4) * nynz + kynz + 1] + tmpMX[(kx+3) * nynz + kynz + 1]);

                        const Type stencilDMy1 =
                                c8_1 * (- tmpMY[kxnynz + (ky-1) * nz + 1] + tmpMY[kxnynz + (ky+0) * nz + 1]) +
                                c8_2 * (- tmpMY[kxnynz + (ky-2) * nz + 1] + tmpMY[kxnynz + (ky+1) * nz + 1]) +
                                c8_3 * (- tmpMY[kxnynz + (ky-3) * nz + 1] + tmpMY[kxnynz + (ky+2) * nz + 1]) +
                                c8_4 * (- tmpMY[kxnynz + (ky-4) * nz + 1] + tmpMY[kxnynz + (ky+3) * nz + 1]);

                        const Type stencilDMz1 =
                                c8_1 * (- tmpMZ[kxnynz_kynz + 0] + tmpMZ[kxnynz_kynz + 1]) +
                                c8_2 * (- tmpMZ[kxnynz_kynz + 0] + tmpMZ[kxnynz_kynz + 2]) +
                                c8_3 * (- tmpMZ[kxnynz_kynz + 1] + tmpMZ[kxnynz_kynz + 3]) +
                                c8_4 * (- tmpMZ[kxnynz_kynz + 2] + tmpMZ[kxnynz_kynz + 4]);

                        const Type dPx = invDx * stencilDPx1;
                        const Type dPy = invDy * stencilDPy1;
                        const Type dPz = invDz * stencilDPz1;
                        const Type dMx = invDx * stencilDMx1;
                        const Type dMy = invDy * stencilDMy1;
                        const Type dMz = invDz * stencilDMz1;

                        const long k = kxnynz_kynz + 1;
                        const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                        pOld[k] = dt2V2_B * (dPx + dPy + dPz) - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                        mOld[k] = dt2V2_B * (dMx + dMy + dMz) - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
                    }

                    // kz = 2 -- two cells below the free surface
                    // [kxnynz_kynz + 2]
                    {
                        const Type stencilDPx2 =
                                c8_1 * (- tmpPX[(kx-1) * nynz + kynz + 2] + tmpPX[(kx+0) * nynz + kynz + 2]) +
                                c8_2 * (- tmpPX[(kx-2) * nynz + kynz + 2] + tmpPX[(kx+1) * nynz + kynz + 2]) +
                                c8_3 * (- tmpPX[(kx-3) * nynz + kynz + 2] + tmpPX[(kx+2) * nynz + kynz + 2]) +
                                c8_4 * (- tmpPX[(kx-4) * nynz + kynz + 2] + tmpPX[(kx+3) * nynz + kynz + 2]);

                        const Type stencilDPy2 =
                                c8_1 * (- tmpPY[kxnynz + (ky-1) * nz + 2] + tmpPY[kxnynz + (ky+0) * nz + 2]) +
                                c8_2 * (- tmpPY[kxnynz + (ky-2) * nz + 2] + tmpPY[kxnynz + (ky+1) * nz + 2]) +
                                c8_3 * (- tmpPY[kxnynz + (ky-3) * nz + 2] + tmpPY[kxnynz + (ky+2) * nz + 2]) +
                                c8_4 * (- tmpPY[kxnynz + (ky-4) * nz + 2] + tmpPY[kxnynz + (ky+3) * nz + 2]);

                        const Type stencilDPz2 =
                                c8_1 * (- tmpPZ[kxnynz_kynz + 1] + tmpPZ[kxnynz_kynz + 2]) +
                                c8_2 * (- tmpPZ[kxnynz_kynz + 0] + tmpPZ[kxnynz_kynz + 3]) +
                                c8_3 * (- tmpPZ[kxnynz_kynz + 0] + tmpPZ[kxnynz_kynz + 4]) +
                                c8_4 * (- tmpPZ[kxnynz_kynz + 1] + tmpPZ[kxnynz_kynz + 5]);

                        const Type stencilDMx2 =
                                c8_1 * (- tmpMX[(kx-1) * nynz + kynz + 2] + tmpMX[(kx+0) * nynz + kynz + 2]) +
                                c8_2 * (- tmpMX[(kx-2) * nynz + kynz + 2] + tmpMX[(kx+1) * nynz + kynz + 2]) +
                                c8_3 * (- tmpMX[(kx-3) * nynz + kynz + 2] + tmpMX[(kx+2) * nynz + kynz + 2]) +
                                c8_4 * (- tmpMX[(kx-4) * nynz + kynz + 2] + tmpMX[(kx+3) * nynz + kynz + 2]);

                        const Type stencilDMy2 =
                                c8_1 * (- tmpMY[kxnynz + (ky-1) * nz + 2] + tmpMY[kxnynz + (ky+0) * nz + 2]) +
                                c8_2 * (- tmpMY[kxnynz + (ky-2) * nz + 2] + tmpMY[kxnynz + (ky+1) * nz + 2]) +
                                c8_3 * (- tmpMY[kxnynz + (ky-3) * nz + 2] + tmpMY[kxnynz + (ky+2) * nz + 2]) +
                                c8_4 * (- tmpMY[kxnynz + (ky-4) * nz + 2] + tmpMY[kxnynz + (ky+3) * nz + 2]);

                        const Type stencilDMz2 =
                                c8_1 * (- tmpMZ[kxnynz_kynz + 1] + tmpMZ[kxnynz_kynz + 2]) +
                                c8_2 * (- tmpMZ[kxnynz_kynz + 0] + tmpMZ[kxnynz_kynz + 3]) +
                                c8_3 * (- tmpMZ[kxnynz_kynz + 0] + tmpMZ[kxnynz_kynz + 4]) +
                                c8_4 * (- tmpMZ[kxnynz_kynz + 1] + tmpMZ[kxnynz_kynz + 5]);

                        const Type dPx = invDx * stencilDPx2;
                        const Type dPy = invDy * stencilDPy2;
                        const Type dPz = invDz * stencilDPz2;
                        const Type dMx = invDx * stencilDMx2;
                        const Type dMy = invDy * stencilDMy2;
                        const Type dMz = invDz * stencilDMz2;

                        const long k = kxnynz_kynz + 2;
                        const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                        pOld[k] = dt2V2_B * (dPx + dPy + dPz) - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                        mOld[k] = dt2V2_B * (dMx + dMy + dMz) - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
                    }

                    // kz = 3 -- three cells below the free surface
                    // [kxnynz_kynz + 3]
                    {
                        const Type stencilDPx3 =
                                c8_1 * (- tmpPX[(kx-1) * nynz + kynz + 3] + tmpPX[(kx+0) * nynz + kynz + 3]) +
                                c8_2 * (- tmpPX[(kx-2) * nynz + kynz + 3] + tmpPX[(kx+1) * nynz + kynz + 3]) +
                                c8_3 * (- tmpPX[(kx-3) * nynz + kynz + 3] + tmpPX[(kx+2) * nynz + kynz + 3]) +
                                c8_4 * (- tmpPX[(kx-4) * nynz + kynz + 3] + tmpPX[(kx+3) * nynz + kynz + 3]);

                        const Type stencilDPy3 =
                                c8_1 * (- tmpPY[kxnynz + (ky-1) * nz + 3] + tmpPY[kxnynz + (ky+0) * nz + 3]) +
                                c8_2 * (- tmpPY[kxnynz + (ky-2) * nz + 3] + tmpPY[kxnynz + (ky+1) * nz + 3]) +
                                c8_3 * (- tmpPY[kxnynz + (ky-3) * nz + 3] + tmpPY[kxnynz + (ky+2) * nz + 3]) +
                                c8_4 * (- tmpPY[kxnynz + (ky-4) * nz + 3] + tmpPY[kxnynz + (ky+3) * nz + 3]);

                        const Type stencilDPz3 =
                                c8_1 * (- tmpPZ[kxnynz_kynz + 2] + tmpPZ[kxnynz_kynz + 3]) +
                                c8_2 * (- tmpPZ[kxnynz_kynz + 1] + tmpPZ[kxnynz_kynz + 4]) +
                                c8_3 * (- tmpPZ[kxnynz_kynz + 0] + tmpPZ[kxnynz_kynz + 5]) +
                                c8_4 * (- tmpPZ[kxnynz_kynz + 0] + tmpPZ[kxnynz_kynz + 6]);

                        const Type stencilDMx3 =
                                c8_1 * (- tmpMX[(kx-1) * nynz + kynz + 3] + tmpMX[(kx+0) * nynz + kynz + 3]) +
                                c8_2 * (- tmpMX[(kx-2) * nynz + kynz + 3] + tmpMX[(kx+1) * nynz + kynz + 3]) +
                                c8_3 * (- tmpMX[(kx-3) * nynz + kynz + 3] + tmpMX[(kx+2) * nynz + kynz + 3]) +
                                c8_4 * (- tmpMX[(kx-4) * nynz + kynz + 3] + tmpMX[(kx+3) * nynz + kynz + 3]);

                        const Type stencilDMy3 =
                                c8_1 * (- tmpMY[kxnynz + (ky-1) * nz + 3] + tmpMY[kxnynz + (ky+0) * nz + 3]) +
                                c8_2 * (- tmpMY[kxnynz + (ky-2) * nz + 3] + tmpMY[kxnynz + (ky+1) * nz + 3]) +
                                c8_3 * (- tmpMY[kxnynz + (ky-3) * nz + 3] + tmpMY[kxnynz + (ky+2) * nz + 3]) +
                                c8_4 * (- tmpMY[kxnynz + (ky-4) * nz + 3] + tmpMY[kxnynz + (ky+3) * nz + 3]);

                        const Type stencilDMz3 =
                                c8_1 * (- tmpMZ[kxnynz_kynz + 2] + tmpMZ[kxnynz_kynz + 3]) +
                                c8_2 * (- tmpMZ[kxnynz_kynz + 1] + tmpMZ[kxnynz_kynz + 4]) +
                                c8_3 * (- tmpMZ[kxnynz_kynz + 0] + tmpMZ[kxnynz_kynz + 5]) +
                                c8_4 * (- tmpMZ[kxnynz_kynz + 0] + tmpMZ[kxnynz_kynz + 6]);

                        const Type dPx = invDx * stencilDPx3;
                        const Type dPy = invDy * stencilDPy3;
                        const Type dPz = invDz * stencilDPz3;
                        const Type dMx = invDx * stencilDMx3;
                        const Type dMy = invDy * stencilDMy3;
                        const Type dMz = invDz * stencilDMz3;

                        const long k = kxnynz_kynz + 3;
                        const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                        pOld[k] = dt2V2_B * (dPx + dPy + dPz) - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                        mOld[k] = dt2V2_B * (dMx + dMy + dMz) - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
                    }
                }
            }
        }
    }

    template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline static void applyFirstDerivatives3D_PlusHalf(
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
    inline static void applyFirstDerivatives3D_MinusHalf(
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

};

#endif
