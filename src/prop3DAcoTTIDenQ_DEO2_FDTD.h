#ifndef PROP3DACOTTIDENQ_DEO2_FDTD_H
#define PROP3DACOTTIDENQ_DEO2_FDTD_H

#include <algorithm>
#include <omp.h>
#include <limits>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include "sys/time.h"

class Prop3DAcoTTIDenQ_DEO2_FDTD {

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
    float * __restrict__ _sinTheta = NULL;
    float * __restrict__ _cosTheta = NULL;
    float * __restrict__ _sinPhi = NULL;
    float * __restrict__ _cosPhi = NULL;
    float * __restrict__ _f = NULL;
    float * __restrict__ _dtOmegaInvQ = NULL;
    float * __restrict__ _pSpace = NULL;
    float * __restrict__ _mSpace = NULL;
    float * __restrict__ _tmpPg1a = NULL;
    float * __restrict__ _tmpPg2a = NULL;
    float * __restrict__ _tmpPg3a = NULL;
    float * __restrict__ _tmpMg1a = NULL;
    float * __restrict__ _tmpMg2a = NULL;
    float * __restrict__ _tmpMg3a = NULL;
    float * __restrict__ _tmpPg1b = NULL;
    float * __restrict__ _tmpPg2b = NULL;
    float * __restrict__ _tmpPg3b = NULL;
    float * __restrict__ _tmpMg1b = NULL;
    float * __restrict__ _tmpMg2b = NULL;
    float * __restrict__ _tmpMg3b = NULL;
    float * _pCur = NULL;
    float * _pOld = NULL;
    float * _mCur = NULL;
    float * _mOld = NULL;

    Prop3DAcoTTIDenQ_DEO2_FDTD(
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
                _invDz(1.0 / _dz)  {

        // Allocate arrays
        _v           = new float[_nx * _ny * _nz];
        _eps         = new float[_nx * _ny * _nz];
        _eta         = new float[_nx * _ny * _nz];
        _b           = new float[_nx * _ny * _nz];
        _sinTheta    = new float[_nx * _ny * _nz];
        _cosTheta    = new float[_nx * _ny * _nz];
        _sinPhi      = new float[_nx * _ny * _nz];
        _cosPhi      = new float[_nx * _ny * _nz];
        _f           = new float[_nx * _ny * _nz];
        _dtOmegaInvQ = new float[_nx * _ny * _nz];
        _pSpace      = new float[_nx * _ny * _nz];
        _mSpace      = new float[_nx * _ny * _nz];
        _tmpPg1a     = new float[_nx * _ny * _nz];
        _tmpPg2a     = new float[_nx * _ny * _nz];
        _tmpPg3a     = new float[_nx * _ny * _nz];
        _tmpMg1a     = new float[_nx * _ny * _nz];
        _tmpMg2a     = new float[_nx * _ny * _nz];
        _tmpMg3a     = new float[_nx * _ny * _nz];
        _tmpPg1b     = new float[_nx * _ny * _nz];
        _tmpPg2b     = new float[_nx * _ny * _nz];
        _tmpPg3b     = new float[_nx * _ny * _nz];
        _tmpMg1b     = new float[_nx * _ny * _nz];
        _tmpMg2b     = new float[_nx * _ny * _nz];
        _tmpMg3b     = new float[_nx * _ny * _nz];
        _pOld        = new float[_nx * _ny * _nz];
        _pCur        = new float[_nx * _ny * _nz];
        _mOld        = new float[_nx * _ny * _nz];
        _mCur        = new float[_nx * _ny * _nz];

        numaFirstTouch(_nx, _ny, _nz, _nthread, _v, _eps, _eta, _b,
            _sinTheta, _cosTheta, _sinPhi, _cosPhi, _f, _dtOmegaInvQ, _pSpace, _mSpace,
            _tmpPg1a, _tmpPg2a, _tmpPg3a, _tmpMg1a, _tmpMg2a, _tmpMg3a,
            _tmpPg1b, _tmpPg2b, _tmpPg3b, _tmpMg1b, _tmpMg2b, _tmpMg3b,
            _pOld, _pCur, _mOld, _mCur, _nbx, _nby, _nbz);
    }

    inline void numaFirstTouch(
            const long nx,
            const long ny,
            const long nz,
            const long nthread,
            float * __restrict__ v,
            float * __restrict__ eps,
            float * __restrict__ eta,
            float * __restrict__ b,
            float * __restrict__ sinTheta,
            float * __restrict__ cosTheta,
            float * __restrict__ sinPhi,
            float * __restrict__ cosPhi,
            float * __restrict__ f,
            float * __restrict__ dtOmegaInvQ,
            float * __restrict__ pSpace,
            float * __restrict__ mSpace,
            float * __restrict__ tmpPg1a,
            float * __restrict__ tmpPg2a,
            float * __restrict__ tmpPg3a,
            float * __restrict__ tmpMg1a,
            float * __restrict__ tmpMg2a,
            float * __restrict__ tmpMg3a,
            float * __restrict__ tmpPg1b,
            float * __restrict__ tmpPg2b,
            float * __restrict__ tmpPg3b,
            float * __restrict__ tmpMg1b,
            float * __restrict__ tmpMg2b,
            float * __restrict__ tmpMg3b,
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
                    const long kxmax = std::min(bx + BX_3D, nx4);
                    const long kymax = std::min(by + BY_3D, ny4);
                    const long kzmax = std::min(bz + BZ_3D, nz4);

                    for (long kx = bx; kx < kxmax; kx++) {
                        for (long ky = by; ky < kymax; ky++) {
#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long k = kx * _ny * _nz + ky * _nz + kz;

                                v[k] = 0;
                                eps[k] = 0;
                                eta[k] = 0;
                                b[k] = 0;
                                sinTheta[k] = 0;
                                cosTheta[k] = 0;
                                sinPhi[k] = 0;
                                cosPhi[k] = 0;
                                f[k] = 0;
                                dtOmegaInvQ[k] = 0;
                                pSpace[k] = 0;
                                mSpace[k] = 0;
                                tmpPg1a[k] = 0;
                                tmpPg2a[k] = 0;
                                tmpPg3a[k] = 0;
                                tmpMg1a[k] = 0;
                                tmpMg2a[k] = 0;
                                tmpMg3a[k] = 0;
                                tmpPg1b[k] = 0;
                                tmpPg2b[k] = 0;
                                tmpPg3b[k] = 0;
                                tmpMg1b[k] = 0;
                                tmpMg2b[k] = 0;
                                tmpMg3b[k] = 0;
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
                    v[kindex1] = eps[kindex1] = eta[kindex1] = b[kindex1] = sinTheta[kindex1] =
                        cosTheta[kindex1] = sinPhi[kindex1] = cosPhi[kindex1] = f[kindex1] =
                        dtOmegaInvQ[kindex1] = pSpace[kindex1] = mSpace[kindex1] = tmpPg1a[kindex1] =
                        tmpPg2a[kindex1] = tmpPg3a[kindex1] = tmpMg1a[kindex1] = tmpMg2a[kindex1] =
                        tmpMg3a[kindex1] = tmpPg1b[kindex1] = tmpPg2b[kindex1] = tmpPg3b[kindex1] =
                        tmpMg1b[kindex1] = tmpMg2b[kindex1] = tmpMg3b[kindex1] = pOld[kindex1] =
                        pCur[kindex1] = mOld[kindex1] = mCur[kindex1] = 0;

                    v[kindex2] = eps[kindex2] = eta[kindex2] = b[kindex2] = sinTheta[kindex2] =
                        cosTheta[kindex2] = sinPhi[kindex2] = cosPhi[kindex2] = f[kindex2] =
                        dtOmegaInvQ[kindex2] = pSpace[kindex2] = mSpace[kindex2] = tmpPg1a[kindex2] =
                        tmpPg2a[kindex2] = tmpPg3a[kindex2] = tmpMg1a[kindex2] = tmpMg2a[kindex2] =
                        tmpMg3a[kindex2] = tmpPg1b[kindex2] = tmpPg2b[kindex2] = tmpPg3b[kindex2] =
                        tmpMg1b[kindex2] = tmpMg2b[kindex2] = tmpMg3b[kindex2] = pOld[kindex2] =
                        pCur[kindex2] = mOld[kindex2] = mCur[kindex2] = 0;
                }
            }

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long kx = 0; kx < nx; kx++) {
#pragma omp simd
                for (long kz = 0; kz < nz; kz++) {
                    const long kindex1 = kx * ny * nz + k * nz + kz;
                    const long kindex2 = kx * ny * nz + (ny - 1 - k) * nz + kz;
                    v[kindex1] = eps[kindex1] = eta[kindex1] = b[kindex1] = sinTheta[kindex1] =
                        cosTheta[kindex1] = sinPhi[kindex1] = cosPhi[kindex1] = f[kindex1] =
                        dtOmegaInvQ[kindex1] = pSpace[kindex1] = mSpace[kindex1] = tmpPg1a[kindex1] =
                        tmpPg2a[kindex1] = tmpPg3a[kindex1] = tmpMg1a[kindex1] = tmpMg2a[kindex1] =
                        tmpMg3a[kindex1] = tmpPg1b[kindex1] = tmpPg2b[kindex1] = tmpPg3b[kindex1] =
                        tmpMg1b[kindex1] = tmpMg2b[kindex1] = tmpMg3b[kindex1] = pOld[kindex1] =
                        pCur[kindex1] = mOld[kindex1] = mCur[kindex1] = 0;

                    v[kindex2] = eps[kindex2] = eta[kindex2] = b[kindex2] = sinTheta[kindex2] =
                        cosTheta[kindex2] = sinPhi[kindex2] = cosPhi[kindex2] = f[kindex2] =
                        dtOmegaInvQ[kindex2] = pSpace[kindex2] = mSpace[kindex2] = tmpPg1a[kindex2] =
                        tmpPg2a[kindex2] = tmpPg3a[kindex2] = tmpMg1a[kindex2] = tmpMg2a[kindex2] =
                        tmpMg3a[kindex2] = tmpPg1b[kindex2] = tmpPg2b[kindex2] = tmpPg3b[kindex2] =
                        tmpMg1b[kindex2] = tmpMg2b[kindex2] = tmpMg3b[kindex2] = pOld[kindex2] =
                        pCur[kindex2] = mOld[kindex2] = mCur[kindex2] = 0;
                }
            }

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long ky = 0; ky < ny; ky++) {
#pragma omp simd
                for (long kz = 0; kz < nz; kz++) {
                    const long kindex1 = k * ny * nz + ky * nz + kz;
                    const long kindex2 = (nx - 1 - k) * ny * nz + ky * nz + kz;
                    v[kindex1] = eps[kindex1] = eta[kindex1] = b[kindex1] = sinTheta[kindex1] =
                        cosTheta[kindex1] = sinPhi[kindex1] = cosPhi[kindex1] = f[kindex1] =
                        dtOmegaInvQ[kindex1] = pSpace[kindex1] = mSpace[kindex1] = tmpPg1a[kindex1] =
                        tmpPg2a[kindex1] = tmpPg3a[kindex1] = tmpMg1a[kindex1] = tmpMg2a[kindex1] =
                        tmpMg3a[kindex1] = tmpPg1b[kindex1] = tmpPg2b[kindex1] = tmpPg3b[kindex1] =
                        tmpMg1b[kindex1] = tmpMg2b[kindex1] = tmpMg3b[kindex1] = pOld[kindex1] =
                        pCur[kindex1] = mOld[kindex1] = mCur[kindex1] = 0;

                    v[kindex2] = eps[kindex2] = eta[kindex2] = b[kindex2] = sinTheta[kindex2] =
                        cosTheta[kindex2] = sinPhi[kindex2] = cosPhi[kindex2] = f[kindex2] =
                        dtOmegaInvQ[kindex2] = pSpace[kindex2] = mSpace[kindex2] = tmpPg1a[kindex2] =
                        tmpPg2a[kindex2] = tmpPg3a[kindex2] = tmpMg1a[kindex2] = tmpMg2a[kindex2] =
                        tmpMg3a[kindex2] = tmpPg1b[kindex2] = tmpPg2b[kindex2] = tmpPg3b[kindex2] =
                        tmpMg1b[kindex2] = tmpMg2b[kindex2] = tmpMg3b[kindex2] = pOld[kindex2] =
                        pCur[kindex2] = mOld[kindex2] = mCur[kindex2] = 0;
                }
            }
        }
    }

    ~Prop3DAcoTTIDenQ_DEO2_FDTD() {
        if (_v != NULL) delete [] _v;
        if (_eps != NULL) delete [] _eps;
        if (_eta != NULL) delete [] _eta;
        if (_sinTheta != NULL) delete [] _sinTheta;
        if (_cosTheta != NULL) delete [] _cosTheta;
        if (_sinPhi != NULL) delete [] _sinPhi;
        if (_cosPhi != NULL) delete [] _cosPhi;
        if (_b != NULL) delete [] _b;
        if (_f != NULL) delete [] _f;
        if (_dtOmegaInvQ != NULL) delete [] _dtOmegaInvQ;
        if (_pSpace != NULL) delete [] _pSpace;
        if (_mSpace != NULL) delete [] _mSpace;
        if (_tmpPg1a != NULL) delete [] _tmpPg1a;
        if (_tmpPg2a != NULL) delete [] _tmpPg2a;
        if (_tmpPg3a != NULL) delete [] _tmpPg3a;
        if (_tmpMg1a != NULL) delete [] _tmpMg1a;
        if (_tmpMg2a != NULL) delete [] _tmpMg2a;
        if (_tmpMg3a != NULL) delete [] _tmpMg3a;
        if (_tmpPg1b != NULL) delete [] _tmpPg1b;
        if (_tmpPg2b != NULL) delete [] _tmpPg2b;
        if (_tmpPg3b != NULL) delete [] _tmpPg3b;
        if (_tmpMg1b != NULL) delete [] _tmpMg1b;
        if (_tmpMg2b != NULL) delete [] _tmpMg2b;
        if (_tmpMg3b != NULL) delete [] _tmpMg3b;
        if (_pOld != NULL) delete [] _pOld;
        if (_pCur != NULL) delete [] _pCur;
        if (_mOld != NULL) delete [] _mOld;
        if (_mCur != NULL) delete [] _mCur;
    }

    void info() {
        printf("\n");
        printf("Prop3DAcoTTIDenQ_DEO2_FDTD\n");
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
     * 2918.07.26
     *   - Ken's advice results in 6 derivatives per state variable instead of 11
     *   - Refactoring from [T D- R-] [S R+ D+] to [T D- ] [R- S R+ D+]
     *     T  2nd order time update
     *     D+ forward  staggered spatial derivative
     *     D- backward staggered spatial derivative
     *     S  material parameter sandwich terms
     *     R+ forward  rotation
     *     R- backward rotation
     */
    inline void timeStep() {

        applyRotationSandwichRotation_TTI_FirstDerivatives3D_PlusHalf_TwoFields(
                _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz,
                _pCur, _mCur, _sinTheta, _cosTheta, _sinPhi, _cosPhi, _eps, _eta, _f, _b,
                _tmpPg1a, _tmpPg2a, _tmpPg3a, _tmpMg1a, _tmpMg2a, _tmpMg3a, _nbx, _nby, _nbz);

        applyFirstDerivatives3D_MinusHalf_TimeUpdate_Nonlinear(
                _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz, _dt,
                _tmpPg1a, _tmpPg2a, _tmpPg3a, _tmpMg1a, _tmpMg2a, _tmpMg3a, _v, _b, _dtOmegaInvQ,
                _pCur, _mCur, _pSpace, _mSpace, _pOld, _mOld, _nbx, _nby, _nbz);

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
    inline void scaleSpatialDerivatives() {
#pragma omp parallel for collapse(3) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long by = 0; by < _ny; by += _nby) {
                for (long bz = 0; bz < _nz; bz += _nbz) {
                    const long kxmax = std::min(bx + _nbx, _nx);
                    const long kymax = std::min(by + _nby, _ny);
                    const long kzmax = std::min(bz + _nbz, _nz);

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
    inline void forwardBornInjection_V(
            float *dmodelV,
            float *wavefieldDP, float *wavefieldDM) {
#pragma omp parallel for collapse(3) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long by = 0; by < _ny; by += _nby) {
                for (long bz = 0; bz < _nz; bz += _nbz) {
                    const long kxmax = std::min(bx + _nbx, _nx);
                    const long kymax = std::min(by + _nby, _ny);
                    const long kzmax = std::min(bz + _nbz, _nz);

                    for (long kx = bx; kx < kxmax; kx++) {
                        for (long ky = by; ky < kymax; ky++) {
#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long k = kx * _ny * _nz + ky * _nz + kz;

                                const float V  = _v[k];
                                const float B  = _b[k];
                                const float dV = dmodelV[k];

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

    inline void forwardBornInjection_VEA(
            float *dmodelV, float *dmodelE, float *dmodelA,
            float *wavefieldP, float *wavefieldM, float *wavefieldDP, float *wavefieldDM) {

        // Right side spatial derivatives for the Born source
        applyFirstDerivatives3D_TTI_PlusHalf(
                _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz,
                wavefieldP, wavefieldP, wavefieldP, _sinTheta, _cosTheta, _sinPhi, _cosPhi, _tmpPg1a, _tmpPg2a, _tmpPg3a, _nbx, _nby, _nbz);

        applyFirstDerivatives3D_TTI_PlusHalf(
                _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz,
                wavefieldM, wavefieldM, wavefieldM, _sinTheta, _cosTheta, _sinPhi, _cosPhi, _tmpMg1a, _tmpMg2a, _tmpMg3a, _nbx, _nby, _nbz);

        // Sandwich terms for the Born source
        // note flipped sign for Z derivative term between P and M
#pragma omp parallel for collapse(3) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long by = 0; by < _ny; by += _nby) {
                for (long bz = 0; bz < _nz; bz += _nbz) {
                    const long kxmax = std::min(bx + _nbx, _nx);
                    const long kymax = std::min(by + _nby, _ny);
                    const long kzmax = std::min(bz + _nbz, _nz);

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

                                const float dV = dmodelV[k];
                                const float dE = dmodelE[k];
                                const float dA = dmodelA[k];

                                _tmpPg1b[k] = (+2 * B * dE) *_tmpPg1a[k];
                                _tmpPg2b[k] = (+2 * B * dE) *_tmpPg2a[k];
                                _tmpPg3b[k] = (-2 * B * F * A * dA) *_tmpPg3a[k] +
                                    (dA * B * F * (1 - 2 * A * A) / sqrt(1 - A * A)) * _tmpMg3a[k];

                                _tmpMg1b[k] = 0;
                                _tmpMg2b[k] = 0;
                                _tmpMg3b[k] = (+2 * B * F * A * dA) *_tmpMg3a[k] +
                                    (dA * B * F * (1 - 2 * A * A) / sqrt(1 - A * A)) * _tmpPg3a[k];
                            }
                        }
                    }
                }
            }
        }

        // Left side spatial derivatives for the Born source
        applyFirstDerivatives3D_TTI_MinusHalf(
                _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz,
                _tmpPg1b, _tmpPg2b, _tmpPg3b, _sinTheta, _cosTheta, _sinPhi, _cosPhi, _tmpPg1a, _tmpPg2a, _tmpPg3a, _nbx, _nby, _nbz);

        applyFirstDerivatives3D_TTI_MinusHalf(
                _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz,
                _tmpMg1b, _tmpMg2b, _tmpMg3b, _sinTheta, _cosTheta, _sinPhi, _cosPhi, _tmpMg1a, _tmpMg2a, _tmpMg3a, _nbx, _nby, _nbz);

        // add the born source at the current time
#pragma omp parallel for collapse(3) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long by = 0; by < _ny; by += _nby) {
                for (long bz = 0; bz < _nz; bz += _nbz) {
                    const long kxmax = std::min(bx + _nbx, _nx);
                    const long kymax = std::min(by + _nby, _ny);
                    const long kzmax = std::min(bz + _nbz, _nz);

                    for (long kx = bx; kx < kxmax; kx++) {
                        for (long ky = by; ky < kymax; ky++) {
#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long k = kx * _ny * _nz + ky * _nz + kz;

                                const float V  = _v[k];
                                const float B  = _b[k];
                                const float dV = dmodelV[k];

                                const float dt2v2OverB = _dt * _dt * V * V / B;

                                const float factor = 2 * B * dV / (V * V * V);

                                _pCur[k] += dt2v2OverB * (factor * wavefieldDP[k] + _tmpPg1a[k] + _tmpPg2a[k] + _tmpPg3a[k]);
                                _mCur[k] += dt2v2OverB * (factor * wavefieldDM[k] + _tmpMg1a[k] + _tmpMg2a[k] + _tmpMg3a[k]);
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
     * Note: "dmodelV,dmodelE,dmodelA" are the three components of the model consecutively [vel,eps,eta]

     * Note: "dmodel" is the three components of the model consecutively [vel,eps,eta]
     *
     * User must have:
     *   - called the nonlinear forward
     *   - saved 2nd time derivative of pressure at corresponding time index in array dp2
     *   - Born image term will be accumulated iu the _dm array
     */
    inline void adjointBornAccumulation_V(float *dmodelV,
            float *wavefieldDP, float *wavefieldDM) {
#pragma omp parallel for collapse(3) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long by = 0; by < _ny; by += _nby) {
                for (long bz = 0; bz < _nz; bz += _nbz) {
                    const long kxmax = std::min(bx + _nbx, _nx);
                    const long kymax = std::min(by + _nby, _ny);
                    const long kzmax = std::min(bz + _nbz, _nz);

                    for (long kx = bx; kx < kxmax; kx++) {
                        for (long ky = by; ky < kymax; ky++) {
#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long k = kx * _ny * _nz + ky * _nz + kz;

                                const float V = _v[k];
                                const float B = _b[k];

                                const float factor = 2 * B / (V * V * V);

                                dmodelV[k] += factor * (wavefieldDP[k] * _pOld[k] + wavefieldDM[k] * _mOld[k]);
                            }
                        }
                    }
                }
            }
        }
    }

    inline void adjointBornAccumulation_VEA(float *dmodelV, float *dmodelE, float *dmodelA,
            float *wavefieldP, float *wavefieldM, float *wavefieldDP, float *wavefieldDM) {

        // Right side spatial derivatives for the adjoint accumulation
        applyFirstDerivatives3D_TTI_PlusHalf(
                _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz,
                wavefieldP, wavefieldP, wavefieldP, _sinTheta, _cosTheta, _sinPhi, _cosPhi, _tmpPg1a, _tmpPg2a, _tmpPg3a, _nbx, _nby, _nbz);

        applyFirstDerivatives3D_TTI_PlusHalf(
                _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz,
                wavefieldM, wavefieldM, wavefieldM, _sinTheta, _cosTheta, _sinPhi, _cosPhi, _tmpMg1a, _tmpMg2a, _tmpMg3a, _nbx, _nby, _nbz);

        applyFirstDerivatives3D_TTI_PlusHalf(
                _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz,
                _pOld, _pOld, _pOld, _sinTheta, _cosTheta, _sinPhi, _cosPhi, _tmpPg1b, _tmpPg2b, _tmpPg3b, _nbx, _nby, _nbz);

        applyFirstDerivatives3D_TTI_PlusHalf(
                _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz,
                _mOld, _mOld, _mOld, _sinTheta, _cosTheta, _sinPhi, _cosPhi, _tmpMg1b, _tmpMg2b, _tmpMg3b, _nbx, _nby, _nbz);

        // Sandwich terms for the adjoint accumulation
#pragma omp parallel for collapse(3) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long by = 0; by < _ny; by += _nby) {
                for (long bz = 0; bz < _nz; bz += _nbz) {
                    const long kxmax = std::min(bx + _nbx, _nx);
                    const long kymax = std::min(by + _nby, _ny);
                    const long kzmax = std::min(bz + _nbz, _nz);

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

                                dmodelV[k] += factor * (wavefieldDP[k] * _pOld[k] + wavefieldDM[k] * _mOld[k]);

                                dmodelE[k] += (-2 * B * _tmpPg1a[k] * _tmpPg1b[k] -2 * B * _tmpPg2a[k] * _tmpPg2b[k]);

                                const float partP = 2 * B * F * A * _tmpPg3a[k] - (B * F * (1 - 2 * A * A) / sqrt(1 - A * A)) * _tmpMg3a[k];
                                const float partM = 2 * B * F * A * _tmpMg3a[k] + (B * F * (1 - 2 * A * A) / sqrt(1 - A * A)) * _tmpPg3a[k];

                                dmodelA[k] += (partP * _tmpPg3b[k] - partM * _tmpMg3b[k]);
                            }
                        }
                    }
                }
            }
        }
    }

    template<class Type>
    inline static void applyRotationSandwichRotation_TTI_FirstDerivatives3D_PlusHalf_TwoFields(
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
            Type * __restrict__ inP,
            Type * __restrict__ inM,
            float * __restrict__ sinTheta,
            float * __restrict__ cosTheta,
            float * __restrict__ sinPhi,
            float * __restrict__ cosPhi,
            Type * __restrict__ fieldEps,
            Type * __restrict__ fieldEta,
            Type * __restrict__ fieldVsVp,
            Type * __restrict__ fieldBuoy,
            Type * __restrict__ outPx,
            Type * __restrict__ outPy,
            Type * __restrict__ outPz,
            Type * __restrict__ outMx,
            Type * __restrict__ outMy,
            Type * __restrict__ outMz,
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
                    outPx[kindex1] = outPx[kindex2] = 0;
                    outPy[kindex1] = outPy[kindex2] = 0;
                    outPz[kindex1] = outPz[kindex2] = 0;
                    outMx[kindex1] = outMx[kindex2] = 0;
                    outMy[kindex1] = outMy[kindex2] = 0;
                    outMz[kindex1] = outMz[kindex2] = 0;
                }
            }

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long kx = 0; kx < nx; kx++) {
#pragma omp simd
                for (long kz = 0; kz < nz; kz++) {
                    const long kindex1 = kx * ny * nz + k * nz + kz;
                    const long kindex2 = kx * ny * nz + (ny - 1 - k) * nz + kz;
                    outPx[kindex1] = outPx[kindex2] = 0;
                    outPy[kindex1] = outPy[kindex2] = 0;
                    outPz[kindex1] = outPz[kindex2] = 0;
                    outMx[kindex1] = outMx[kindex2] = 0;
                    outMy[kindex1] = outMy[kindex2] = 0;
                    outMz[kindex1] = outMz[kindex2] = 0;
                }
            }

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long ky = 0; ky < ny; ky++) {
#pragma omp simd
                for (long kz = 0; kz < nz; kz++) {
                    long kindex1 = k * ny * nz + ky * nz + kz;
                    long kindex2 = (nx - 1 - k) * ny * nz + ky * nz + kz;
                    outPx[kindex1] = outPx[kindex2] = 0;
                    outPy[kindex1] = outPy[kindex2] = 0;
                    outPz[kindex1] = outPz[kindex2] = 0;
                    outMx[kindex1] = outMx[kindex2] = 0;
                    outMy[kindex1] = outMy[kindex2] = 0;
                    outMz[kindex1] = outMz[kindex2] = 0;
                }
            }

        }

        // interior
#pragma omp parallel for collapse(3) num_threads(nthread) schedule(static)
        for (long bx = 4; bx < nx4; bx += BX_3D) {
            for (long by = 4; by < ny4; by += BY_3D) {
                for (long bz = 4; bz < nz4; bz += BZ_3D) {
                    const long kxmax = std::min(bx + BX_3D, nx4);
                    const long kymax = std::min(by + BY_3D, ny4);
                    const long kzmax = std::min(bz + BZ_3D, nz4);

                    for (long kx = bx; kx < kxmax; kx++) {
                        const long kxnynz = kx * nynz;

                        for (long ky = by; ky < kymax; ky++) {
                            const long kynz = ky * nz;
                            const long kxnynz_kynz = kxnynz + kynz;
#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long kynz_kz = + kynz + kz;

                                const Type stencilPDx =
                                        c8_1 * (- inP[(kx+0) * nynz + kynz_kz] + inP[(kx+1) * nynz + kynz_kz]) +
                                        c8_2 * (- inP[(kx-1) * nynz + kynz_kz] + inP[(kx+2) * nynz + kynz_kz]) +
                                        c8_3 * (- inP[(kx-2) * nynz + kynz_kz] + inP[(kx+3) * nynz + kynz_kz]) +
                                        c8_4 * (- inP[(kx-3) * nynz + kynz_kz] + inP[(kx+4) * nynz + kynz_kz]);

                                const Type stencilPDy =
                                        c8_1 * (- inP[kxnynz + (ky+0) * nz + kz] + inP[kxnynz + (ky+1) * nz + kz]) +
                                        c8_2 * (- inP[kxnynz + (ky-1) * nz + kz] + inP[kxnynz + (ky+2) * nz + kz]) +
                                        c8_3 * (- inP[kxnynz + (ky-2) * nz + kz] + inP[kxnynz + (ky+3) * nz + kz]) +
                                        c8_4 * (- inP[kxnynz + (ky-3) * nz + kz] + inP[kxnynz + (ky+4) * nz + kz]);

                                const Type stencilPDz =
                                        c8_1 * (- inP[kxnynz_kynz + (kz+0)] + inP[kxnynz_kynz + (kz+1)]) +
                                        c8_2 * (- inP[kxnynz_kynz + (kz-1)] + inP[kxnynz_kynz + (kz+2)]) +
                                        c8_3 * (- inP[kxnynz_kynz + (kz-2)] + inP[kxnynz_kynz + (kz+3)]) +
                                        c8_4 * (- inP[kxnynz_kynz + (kz-3)] + inP[kxnynz_kynz + (kz+4)]);

                                const Type stencilMDx =
                                        c8_1 * (- inM[(kx+0) * nynz + kynz_kz] + inM[(kx+1) * nynz + kynz_kz]) +
                                        c8_2 * (- inM[(kx-1) * nynz + kynz_kz] + inM[(kx+2) * nynz + kynz_kz]) +
                                        c8_3 * (- inM[(kx-2) * nynz + kynz_kz] + inM[(kx+3) * nynz + kynz_kz]) +
                                        c8_4 * (- inM[(kx-3) * nynz + kynz_kz] + inM[(kx+4) * nynz + kynz_kz]);

                                const Type stencilMDy =
                                        c8_1 * (- inM[kxnynz + (ky+0) * nz + kz] + inM[kxnynz + (ky+1) * nz + kz]) +
                                        c8_2 * (- inM[kxnynz + (ky-1) * nz + kz] + inM[kxnynz + (ky+2) * nz + kz]) +
                                        c8_3 * (- inM[kxnynz + (ky-2) * nz + kz] + inM[kxnynz + (ky+3) * nz + kz]) +
                                        c8_4 * (- inM[kxnynz + (ky-3) * nz + kz] + inM[kxnynz + (ky+4) * nz + kz]);

                                const Type stencilMDz =
                                        c8_1 * (- inM[kxnynz_kynz + (kz+0)] + inM[kxnynz_kynz + (kz+1)]) +
                                        c8_2 * (- inM[kxnynz_kynz + (kz-1)] + inM[kxnynz_kynz + (kz+2)]) +
                                        c8_3 * (- inM[kxnynz_kynz + (kz-2)] + inM[kxnynz_kynz + (kz+3)]) +
                                        c8_4 * (- inM[kxnynz_kynz + (kz-3)] + inM[kxnynz_kynz + (kz+4)]);

                                const Type dpdx = invDx * stencilPDx;
                                const Type dpdy = invDy * stencilPDy;
                                const Type dpdz = invDz * stencilPDz;

                                const Type dmdx = invDx * stencilMDx;
                                const Type dmdy = invDy * stencilMDy;
                                const Type dmdz = invDz * stencilMDz;

                                const long k = kx * ny * nz + ky * nz + kz;

                                const float sinThetaCosPhi = sinTheta[k] * cosPhi[k];
                                const float sinThetaSinPhi = sinTheta[k] * sinPhi[k];
                                const Type fieldEta2 = fieldEta[k] * fieldEta[k];
                                const Type fieldBuoyVsVp = fieldBuoy[k] * fieldVsVp[k];

                                const Type g3P = sinThetaCosPhi * dpdx + sinThetaSinPhi * dpdy + cosTheta[k] * dpdz;
                                const Type g3M = sinThetaCosPhi * dmdx + sinThetaSinPhi * dmdy + cosTheta[k] * dmdz;

                                const Type tmpFE = fieldBuoyVsVp * fieldEta[k] * sqrt(1 - fieldEta2);
                                const Type tmpP = - fieldBuoy[k] * (2 * fieldEps[k] + fieldVsVp[k] * fieldEta2) * g3P + tmpFE * g3M;
                                const Type tmpM = tmpFE * g3P + fieldBuoyVsVp * fieldEta2 * g3M;

                                const Type tmpE = fieldBuoy[k] * (1 + 2 * fieldEps[k]);
                                const Type tmpF = fieldBuoy[k] * (1 - fieldVsVp[k]);

                                outPx[k] = tmpE * dpdx + sinThetaCosPhi * tmpP;
                                outPy[k] = tmpE * dpdy + sinThetaSinPhi * tmpP;
                                outPz[k] = tmpE * dpdz + cosTheta[k]    * tmpP;

                                outMx[k] = tmpF * dmdx + sinThetaCosPhi * tmpM;
                                outMy[k] = tmpF * dmdy + sinThetaSinPhi * tmpM;
                                outMz[k] = tmpF * dmdz + cosTheta[k]    * tmpM;
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
                    {
                        const Type stencilPDz0 =
                                c8_1 * (- inP[kxnynz_kynz + 0] + inP[kxnynz_kynz + 1]) +
                                c8_2 * (+ inP[kxnynz_kynz + 1] + inP[kxnynz_kynz + 2]) +
                                c8_3 * (+ inP[kxnynz_kynz + 2] + inP[kxnynz_kynz + 3]) +
                                c8_4 * (+ inP[kxnynz_kynz + 3] + inP[kxnynz_kynz + 4]);

                        const Type stencilMDz0 =
                                c8_1 * (- inM[kxnynz_kynz + 0] + inM[kxnynz_kynz + 1]) +
                                c8_2 * (+ inM[kxnynz_kynz + 1] + inM[kxnynz_kynz + 2]) +
                                c8_3 * (+ inM[kxnynz_kynz + 2] + inM[kxnynz_kynz + 3]) +
                                c8_4 * (+ inM[kxnynz_kynz + 3] + inM[kxnynz_kynz + 4]);

                        const Type dpdx = 0;
                        const Type dpdy = 0;
                        const Type dpdz = invDz * stencilPDz0;

                        const Type dmdx = 0;
                        const Type dmdy = 0;
                        const Type dmdz = invDz * stencilMDz0;

                        const long k = kx * ny * nz + ky * nz + 0;

                        const float sinThetaCosPhi = sinTheta[k] * cosPhi[k];
                        const float sinThetaSinPhi = sinTheta[k] * sinPhi[k];
                        const Type fieldEta2 = fieldEta[k] * fieldEta[k];
                        const Type fieldBuoyVsVp = fieldBuoy[k] * fieldVsVp[k];

                        const Type g3P = sinThetaCosPhi * dpdx + sinThetaSinPhi * dpdy + cosTheta[k] * dpdz;
                        const Type g3M = sinThetaCosPhi * dmdx + sinThetaSinPhi * dmdy + cosTheta[k] * dmdz;

                        const Type tmpFE = fieldBuoyVsVp * fieldEta[k] * sqrt(1 - fieldEta2);
                        const Type tmpP = - fieldBuoy[k] * (2 * fieldEps[k] + fieldVsVp[k] * fieldEta2) * g3P + tmpFE * g3M;
                        const Type tmpM = tmpFE * g3P + fieldBuoyVsVp * fieldEta2 * g3M;

                        const Type tmpE = fieldBuoy[k] * (1 + 2 * fieldEps[k]);
                        const Type tmpF = fieldBuoy[k] * (1 - fieldVsVp[k]);

                        outPx[k] = tmpE * dpdx + sinThetaCosPhi * tmpP;
                        outPy[k] = tmpE * dpdy + sinThetaSinPhi * tmpP;
                        outPz[k] = tmpE * dpdz + cosTheta[k]    * tmpP;

                        outMx[k] = tmpF * dmdx + sinThetaCosPhi * tmpM;
                        outMy[k] = tmpF * dmdy + sinThetaSinPhi * tmpM;
                        outMz[k] = tmpF * dmdz + cosTheta[k]    * tmpM;
                    }

                    // kz = 1 -- 1 1/2 cells below free surface for Z derivative, 1 cells below for X/Y derivative
                    {
                        const Type stencilPDx1 =
                                c8_1 * (- inP[(kx+0) * nynz + kynz + 1] + inP[(kx+1) * nynz + kynz + 1]) +
                                c8_2 * (- inP[(kx-1) * nynz + kynz + 1] + inP[(kx+2) * nynz + kynz + 1]) +
                                c8_3 * (- inP[(kx-2) * nynz + kynz + 1] + inP[(kx+3) * nynz + kynz + 1]) +
                                c8_4 * (- inP[(kx-3) * nynz + kynz + 1] + inP[(kx+4) * nynz + kynz + 1]);

                        const Type stencilPDy1 =
                                c8_1 * (- inP[kxnynz + (ky+0) * nz + 1] + inP[kxnynz + (ky+1) * nz + 1]) +
                                c8_2 * (- inP[kxnynz + (ky-1) * nz + 1] + inP[kxnynz + (ky+2) * nz + 1]) +
                                c8_3 * (- inP[kxnynz + (ky-2) * nz + 1] + inP[kxnynz + (ky+3) * nz + 1]) +
                                c8_4 * (- inP[kxnynz + (ky-3) * nz + 1] + inP[kxnynz + (ky+4) * nz + 1]);

                        const Type stencilPDz1 =
                                c8_1 * (- inP[kxnynz_kynz + 1] + inP[kxnynz_kynz + 2]) +
                                c8_2 * (- inP[kxnynz_kynz + 0] + inP[kxnynz_kynz + 3]) +
                                c8_3 * (+ inP[kxnynz_kynz + 1] + inP[kxnynz_kynz + 4]) +
                                c8_4 * (+ inP[kxnynz_kynz + 2] + inP[kxnynz_kynz + 5]);

                        const Type stencilMDx1 =
                                c8_1 * (- inM[(kx+0) * nynz + kynz + 1] + inM[(kx+1) * nynz + kynz + 1]) +
                                c8_2 * (- inM[(kx-1) * nynz + kynz + 1] + inM[(kx+2) * nynz + kynz + 1]) +
                                c8_3 * (- inM[(kx-2) * nynz + kynz + 1] + inM[(kx+3) * nynz + kynz + 1]) +
                                c8_4 * (- inM[(kx-3) * nynz + kynz + 1] + inM[(kx+4) * nynz + kynz + 1]);

                        const Type stencilMDy1 =
                                c8_1 * (- inM[kxnynz + (ky+0) * nz + 1] + inM[kxnynz + (ky+1) * nz + 1]) +
                                c8_2 * (- inM[kxnynz + (ky-1) * nz + 1] + inM[kxnynz + (ky+2) * nz + 1]) +
                                c8_3 * (- inM[kxnynz + (ky-2) * nz + 1] + inM[kxnynz + (ky+3) * nz + 1]) +
                                c8_4 * (- inM[kxnynz + (ky-3) * nz + 1] + inM[kxnynz + (ky+4) * nz + 1]);

                        const Type stencilMDz1 =
                                c8_1 * (- inM[kxnynz_kynz + 1] + inM[kxnynz_kynz + 2]) +
                                c8_2 * (- inM[kxnynz_kynz + 0] + inM[kxnynz_kynz + 3]) +
                                c8_3 * (+ inM[kxnynz_kynz + 1] + inM[kxnynz_kynz + 4]) +
                                c8_4 * (+ inM[kxnynz_kynz + 2] + inM[kxnynz_kynz + 5]);

                        const Type dpdx = invDx * stencilPDx1;
                        const Type dpdy = invDy * stencilPDy1;
                        const Type dpdz = invDz * stencilPDz1;

                        const Type dmdx = invDx * stencilMDx1;
                        const Type dmdy = invDy * stencilMDy1;
                        const Type dmdz = invDz * stencilMDz1;

                        const long k = kx * ny * nz + ky * nz + 1;

                        const float sinThetaCosPhi = sinTheta[k] * cosPhi[k];
                        const float sinThetaSinPhi = sinTheta[k] * sinPhi[k];
                        const Type fieldEta2 = fieldEta[k] * fieldEta[k];
                        const Type fieldBuoyVsVp = fieldBuoy[k] * fieldVsVp[k];

                        const Type g3P = sinThetaCosPhi * dpdx + sinThetaSinPhi * dpdy + cosTheta[k] * dpdz;
                        const Type g3M = sinThetaCosPhi * dmdx + sinThetaSinPhi * dmdy + cosTheta[k] * dmdz;

                        const Type tmpFE = fieldBuoyVsVp * fieldEta[k] * sqrt(1 - fieldEta2);
                        const Type tmpP = - fieldBuoy[k] * (2 * fieldEps[k] + fieldVsVp[k] * fieldEta2) * g3P + tmpFE * g3M;
                        const Type tmpM = tmpFE * g3P + fieldBuoyVsVp * fieldEta2 * g3M;

                        const Type tmpE = fieldBuoy[k] * (1 + 2 * fieldEps[k]);
                        const Type tmpF = fieldBuoy[k] * (1 - fieldVsVp[k]);

                        outPx[k] = tmpE * dpdx + sinThetaCosPhi * tmpP;
                        outPy[k] = tmpE * dpdy + sinThetaSinPhi * tmpP;
                        outPz[k] = tmpE * dpdz + cosTheta[k]    * tmpP;

                        outMx[k] = tmpF * dmdx + sinThetaCosPhi * tmpM;
                        outMy[k] = tmpF * dmdy + sinThetaSinPhi * tmpM;
                        outMz[k] = tmpF * dmdz + cosTheta[k]    * tmpM;
                    }

                    // kz = 2 -- 2 1/2 cells below free surface for Z derivative, 2 cells below for X/Y derivative
                    {
                        const Type stencilPDx2 =
                                c8_1 * (- inP[(kx+0) * nynz + kynz + 2] + inP[(kx+1) * nynz + kynz + 2]) +
                                c8_2 * (- inP[(kx-1) * nynz + kynz + 2] + inP[(kx+2) * nynz + kynz + 2]) +
                                c8_3 * (- inP[(kx-2) * nynz + kynz + 2] + inP[(kx+3) * nynz + kynz + 2]) +
                                c8_4 * (- inP[(kx-3) * nynz + kynz + 2] + inP[(kx+4) * nynz + kynz + 2]);

                        const Type stencilPDy2 =
                                c8_1 * (- inP[kxnynz + (ky+0) * nz + 2] + inP[kxnynz + (ky+1) * nz + 2]) +
                                c8_2 * (- inP[kxnynz + (ky-1) * nz + 2] + inP[kxnynz + (ky+2) * nz + 2]) +
                                c8_3 * (- inP[kxnynz + (ky-2) * nz + 2] + inP[kxnynz + (ky+3) * nz + 2]) +
                                c8_4 * (- inP[kxnynz + (ky-3) * nz + 2] + inP[kxnynz + (ky+4) * nz + 2]);

                        const Type stencilPDz2 =
                                c8_1 * (- inP[kxnynz_kynz + 2] + inP[kxnynz_kynz + 3]) +
                                c8_2 * (- inP[kxnynz_kynz + 1] + inP[kxnynz_kynz + 4]) +
                                c8_3 * (- inP[kxnynz_kynz + 0] + inP[kxnynz_kynz + 5]) +
                                c8_4 * (+ inP[kxnynz_kynz + 1] + inP[kxnynz_kynz + 6]);

                        const Type stencilMDx2 =
                                c8_1 * (- inM[(kx+0) * nynz + kynz + 2] + inM[(kx+1) * nynz + kynz + 2]) +
                                c8_2 * (- inM[(kx-1) * nynz + kynz + 2] + inM[(kx+2) * nynz + kynz + 2]) +
                                c8_3 * (- inM[(kx-2) * nynz + kynz + 2] + inM[(kx+3) * nynz + kynz + 2]) +
                                c8_4 * (- inM[(kx-3) * nynz + kynz + 2] + inM[(kx+4) * nynz + kynz + 2]);

                        const Type stencilMDy2 =
                                c8_1 * (- inM[kxnynz + (ky+0) * nz + 2] + inM[kxnynz + (ky+1) * nz + 2]) +
                                c8_2 * (- inM[kxnynz + (ky-1) * nz + 2] + inM[kxnynz + (ky+2) * nz + 2]) +
                                c8_3 * (- inM[kxnynz + (ky-2) * nz + 2] + inM[kxnynz + (ky+3) * nz + 2]) +
                                c8_4 * (- inM[kxnynz + (ky-3) * nz + 2] + inM[kxnynz + (ky+4) * nz + 2]);

                        const Type stencilMDz2 =
                                c8_1 * (- inM[kxnynz_kynz + 2] + inM[kxnynz_kynz + 3]) +
                                c8_2 * (- inM[kxnynz_kynz + 1] + inM[kxnynz_kynz + 4]) +
                                c8_3 * (- inM[kxnynz_kynz + 0] + inM[kxnynz_kynz + 5]) +
                                c8_4 * (+ inM[kxnynz_kynz + 1] + inM[kxnynz_kynz + 6]);

                        const Type dpdx = invDx * stencilPDx2;
                        const Type dpdy = invDy * stencilPDy2;
                        const Type dpdz = invDz * stencilPDz2;

                        const Type dmdx = invDx * stencilMDx2;
                        const Type dmdy = invDy * stencilMDy2;
                        const Type dmdz = invDz * stencilMDz2;

                        const long k = kx * ny * nz + ky * nz + 2;

                        const float sinThetaCosPhi = sinTheta[k] * cosPhi[k];
                        const float sinThetaSinPhi = sinTheta[k] * sinPhi[k];
                        const Type fieldEta2 = fieldEta[k] * fieldEta[k];
                        const Type fieldBuoyVsVp = fieldBuoy[k] * fieldVsVp[k];

                        const Type g3P = sinThetaCosPhi * dpdx + sinThetaSinPhi * dpdy + cosTheta[k] * dpdz;
                        const Type g3M = sinThetaCosPhi * dmdx + sinThetaSinPhi * dmdy + cosTheta[k] * dmdz;

                        const Type tmpFE = fieldBuoyVsVp * fieldEta[k] * sqrt(1 - fieldEta2);
                        const Type tmpP = - fieldBuoy[k] * (2 * fieldEps[k] + fieldVsVp[k] * fieldEta2) * g3P + tmpFE * g3M;
                        const Type tmpM = tmpFE * g3P + fieldBuoyVsVp * fieldEta2 * g3M;

                        const Type tmpE = fieldBuoy[k] * (1 + 2 * fieldEps[k]);
                        const Type tmpF = fieldBuoy[k] * (1 - fieldVsVp[k]);

                        outPx[k] = tmpE * dpdx + sinThetaCosPhi * tmpP;
                        outPy[k] = tmpE * dpdy + sinThetaSinPhi * tmpP;
                        outPz[k] = tmpE * dpdz + cosTheta[k]    * tmpP;

                        outMx[k] = tmpF * dmdx + sinThetaCosPhi * tmpM;
                        outMy[k] = tmpF * dmdy + sinThetaSinPhi * tmpM;
                        outMz[k] = tmpF * dmdz + cosTheta[k]    * tmpM;
                    }

                    // kz = 3 -- 3 1/2 cells below free surface for Z derivative, 3 cells below for X/Y derivative
                    {
                        const Type stencilPDx3 =
                                c8_1 * (- inP[(kx+0) * nynz + kynz + 3] + inP[(kx+1) * nynz + kynz + 3]) +
                                c8_2 * (- inP[(kx-1) * nynz + kynz + 3] + inP[(kx+2) * nynz + kynz + 3]) +
                                c8_3 * (- inP[(kx-2) * nynz + kynz + 3] + inP[(kx+3) * nynz + kynz + 3]) +
                                c8_4 * (- inP[(kx-3) * nynz + kynz + 3] + inP[(kx+4) * nynz + kynz + 3]);

                        const Type stencilPDy3 =
                                c8_1 * (- inP[kxnynz + (ky+0) * nz + 3] + inP[kxnynz + (ky+1) * nz + 3]) +
                                c8_2 * (- inP[kxnynz + (ky-1) * nz + 3] + inP[kxnynz + (ky+2) * nz + 3]) +
                                c8_3 * (- inP[kxnynz + (ky-2) * nz + 3] + inP[kxnynz + (ky+3) * nz + 3]) +
                                c8_4 * (- inP[kxnynz + (ky-3) * nz + 3] + inP[kxnynz + (ky+4) * nz + 3]);

                        const Type stencilPDz3 =
                                c8_1 * (- inP[kxnynz_kynz + 3] + inP[kxnynz_kynz + 4]) +
                                c8_2 * (- inP[kxnynz_kynz + 2] + inP[kxnynz_kynz + 5]) +
                                c8_3 * (- inP[kxnynz_kynz + 1] + inP[kxnynz_kynz + 6]) +
                                c8_4 * (- inP[kxnynz_kynz + 0] + inP[kxnynz_kynz + 7]);

                        const Type stencilMDx3 =
                                c8_1 * (- inM[(kx+0) * nynz + kynz + 3] + inM[(kx+1) * nynz + kynz + 3]) +
                                c8_2 * (- inM[(kx-1) * nynz + kynz + 3] + inM[(kx+2) * nynz + kynz + 3]) +
                                c8_3 * (- inM[(kx-2) * nynz + kynz + 3] + inM[(kx+3) * nynz + kynz + 3]) +
                                c8_4 * (- inM[(kx-3) * nynz + kynz + 3] + inM[(kx+4) * nynz + kynz + 3]);

                        const Type stencilMDy3 =
                                c8_1 * (- inM[kxnynz + (ky+0) * nz + 3] + inM[kxnynz + (ky+1) * nz + 3]) +
                                c8_2 * (- inM[kxnynz + (ky-1) * nz + 3] + inM[kxnynz + (ky+2) * nz + 3]) +
                                c8_3 * (- inM[kxnynz + (ky-2) * nz + 3] + inM[kxnynz + (ky+3) * nz + 3]) +
                                c8_4 * (- inM[kxnynz + (ky-3) * nz + 3] + inM[kxnynz + (ky+4) * nz + 3]);

                        const Type stencilMDz3 =
                                c8_1 * (- inM[kxnynz_kynz + 3] + inM[kxnynz_kynz + 4]) +
                                c8_2 * (- inM[kxnynz_kynz + 2] + inM[kxnynz_kynz + 5]) +
                                c8_3 * (- inM[kxnynz_kynz + 1] + inM[kxnynz_kynz + 6]) +
                                c8_4 * (- inM[kxnynz_kynz + 0] + inM[kxnynz_kynz + 7]);

                        const Type dpdx = invDx * stencilPDx3;
                        const Type dpdy = invDy * stencilPDy3;
                        const Type dpdz = invDz * stencilPDz3;

                        const Type dmdx = invDx * stencilMDx3;
                        const Type dmdy = invDy * stencilMDy3;
                        const Type dmdz = invDz * stencilMDz3;

                        const long k = kx * ny * nz + ky * nz + 3;

                        const float sinThetaCosPhi = sinTheta[k] * cosPhi[k];
                        const float sinThetaSinPhi = sinTheta[k] * sinPhi[k];
                        const Type fieldEta2 = fieldEta[k] * fieldEta[k];
                        const Type fieldBuoyVsVp = fieldBuoy[k] * fieldVsVp[k];

                        const Type g3P = sinThetaCosPhi * dpdx + sinThetaSinPhi * dpdy + cosTheta[k] * dpdz;
                        const Type g3M = sinThetaCosPhi * dmdx + sinThetaSinPhi * dmdy + cosTheta[k] * dmdz;

                        const Type tmpFE = fieldBuoyVsVp * fieldEta[k] * sqrt(1 - fieldEta2);
                        const Type tmpP = - fieldBuoy[k] * (2 * fieldEps[k] + fieldVsVp[k] * fieldEta2) * g3P + tmpFE * g3M;
                        const Type tmpM = tmpFE * g3P + fieldBuoyVsVp * fieldEta2 * g3M;

                        const Type tmpE = fieldBuoy[k] * (1 + 2 * fieldEps[k]);
                        const Type tmpF = fieldBuoy[k] * (1 - fieldVsVp[k]);

                        outPx[k] = tmpE * dpdx + sinThetaCosPhi * tmpP;
                        outPy[k] = tmpE * dpdy + sinThetaSinPhi * tmpP;
                        outPz[k] = tmpE * dpdz + cosTheta[k]    * tmpP;

                        outMx[k] = tmpF * dmdx + sinThetaCosPhi * tmpM;
                        outMy[k] = tmpF * dmdy + sinThetaSinPhi * tmpM;
                        outMz[k] = tmpF * dmdz + cosTheta[k]    * tmpM;
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
                    const long kxmax = std::min(bx + BX_3D, nx4);
                    const long kymax = std::min(by + BY_3D, ny4);
                    const long kzmax = std::min(bz + BZ_3D, nz4);

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
                        const float dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

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
    inline static void applyFirstDerivatives3D_TTI_PlusHalf_Sandwich(
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
            Type * __restrict__ inP_G1,
            Type * __restrict__ inP_G2,
            Type * __restrict__ inP_G3,
            Type * __restrict__ inM_G1,
            Type * __restrict__ inM_G2,
            Type * __restrict__ inM_G3,
            Type * __restrict__ fieldEps,
            Type * __restrict__ fieldEta,
            Type * __restrict__ fieldVsVp,
            Type * __restrict__ fieldBuoy,
            float * __restrict__ sinTheta,
            float * __restrict__ cosTheta,
            float * __restrict__ sinPhi,
            float * __restrict__ cosPhi,
            Type * __restrict__ outP_G1,
            Type * __restrict__ outP_G2,
            Type * __restrict__ outP_G3,
            Type * __restrict__ outM_G1,
            Type * __restrict__ outM_G2,
            Type * __restrict__ outM_G3,
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
                    outP_G1[kindex1] = outP_G1[kindex2] = 0;
                    outP_G2[kindex1] = outP_G2[kindex2] = 0;
                    outP_G3[kindex1] = outP_G3[kindex2] = 0;
                    outM_G1[kindex1] = outM_G1[kindex2] = 0;
                    outM_G2[kindex1] = outM_G2[kindex2] = 0;
                    outM_G3[kindex1] = outM_G3[kindex2] = 0;
                }
            }

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long kx = 0; kx < nx; kx++) {
#pragma omp simd
                for (long kz = 0; kz < nz; kz++) {
                    long kindex1 = kx * ny * nz + k * nz + kz;
                    long kindex2 = kx * ny * nz + (ny - 1 - k) * nz + kz;
                    outP_G1[kindex1] = outP_G1[kindex2] = 0;
                    outP_G2[kindex1] = outP_G2[kindex2] = 0;
                    outP_G3[kindex1] = outP_G3[kindex2] = 0;
                    outM_G1[kindex1] = outM_G1[kindex2] = 0;
                    outM_G2[kindex1] = outM_G2[kindex2] = 0;
                    outM_G3[kindex1] = outM_G3[kindex2] = 0;
                }
            }

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long ky = 0; ky < ny; ky++) {
#pragma omp simd
                for (long kz = 0; kz < nz; kz++) {
                    long kindex1 = k * ny * nz + ky * nz + kz;
                    long kindex2 = (nx - 1 - k) * ny * nz + ky * nz + kz;
                    outP_G1[kindex1] = outP_G1[kindex2] = 0;
                    outP_G2[kindex1] = outP_G2[kindex2] = 0;
                    outP_G3[kindex1] = outP_G3[kindex2] = 0;
                    outM_G1[kindex1] = outM_G1[kindex2] = 0;
                    outM_G2[kindex1] = outM_G2[kindex2] = 0;
                    outM_G3[kindex1] = outM_G3[kindex2] = 0;
                }
            }

        }

        // interior
#pragma omp parallel for collapse(3) num_threads(nthread) schedule(static)
        for (long bx = 4; bx < nx4; bx += BX_3D) {
            for (long by = 4; by < ny4; by += BY_3D) {
                for (long bz = 4; bz < nz4; bz += BZ_3D) {
                    const long kxmax = std::min(bx + BX_3D, nx4);
                    const long kymax = std::min(by + BY_3D, ny4);
                    const long kzmax = std::min(bz + BZ_3D, nz4);

                    for (long kx = bx; kx < kxmax; kx++) {
                        const long kxnynz = kx * nynz;

                        for (long ky = by; ky < kymax; ky++) {
                            const long kynz = ky * nz;
                            const long kxnynz_kynz = kxnynz + kynz;

#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long kynz_kz = + kynz + kz;

                                const Type stencilP_G1 =
                                        c8_1 * (- inP_G1[(kx+0) * nynz + kynz_kz] + inP_G1[(kx+1) * nynz + kynz_kz]) +
                                        c8_2 * (- inP_G1[(kx-1) * nynz + kynz_kz] + inP_G1[(kx+2) * nynz + kynz_kz]) +
                                        c8_3 * (- inP_G1[(kx-2) * nynz + kynz_kz] + inP_G1[(kx+3) * nynz + kynz_kz]) +
                                        c8_4 * (- inP_G1[(kx-3) * nynz + kynz_kz] + inP_G1[(kx+4) * nynz + kynz_kz]);

                                const Type stencilP_G2 =
                                        c8_1 * (- inP_G2[kxnynz + (ky+0) * nz + kz] + inP_G2[kxnynz + (ky+1) * nz + kz]) +
                                        c8_2 * (- inP_G2[kxnynz + (ky-1) * nz + kz] + inP_G2[kxnynz + (ky+2) * nz + kz]) +
                                        c8_3 * (- inP_G2[kxnynz + (ky-2) * nz + kz] + inP_G2[kxnynz + (ky+3) * nz + kz]) +
                                        c8_4 * (- inP_G2[kxnynz + (ky-3) * nz + kz] + inP_G2[kxnynz + (ky+4) * nz + kz]);

                                const Type stencilP_G3 =
                                        c8_1 * (- inP_G3[kxnynz_kynz + (kz+0)] + inP_G3[kxnynz_kynz + (kz+1)]) +
                                        c8_2 * (- inP_G3[kxnynz_kynz + (kz-1)] + inP_G3[kxnynz_kynz + (kz+2)]) +
                                        c8_3 * (- inP_G3[kxnynz_kynz + (kz-2)] + inP_G3[kxnynz_kynz + (kz+3)]) +
                                        c8_4 * (- inP_G3[kxnynz_kynz + (kz-3)] + inP_G3[kxnynz_kynz + (kz+4)]);

                                const Type stencilM_G1 =
                                        c8_1 * (- inM_G1[(kx+0) * nynz + kynz_kz] + inM_G1[(kx+1) * nynz + kynz_kz]) +
                                        c8_2 * (- inM_G1[(kx-1) * nynz + kynz_kz] + inM_G1[(kx+2) * nynz + kynz_kz]) +
                                        c8_3 * (- inM_G1[(kx-2) * nynz + kynz_kz] + inM_G1[(kx+3) * nynz + kynz_kz]) +
                                        c8_4 * (- inM_G1[(kx-3) * nynz + kynz_kz] + inM_G1[(kx+4) * nynz + kynz_kz]);

                                const Type stencilM_G2 =
                                        c8_1 * (- inM_G2[kxnynz + (ky+0) * nz + kz] + inM_G2[kxnynz + (ky+1) * nz + kz]) +
                                        c8_2 * (- inM_G2[kxnynz + (ky-1) * nz + kz] + inM_G2[kxnynz + (ky+2) * nz + kz]) +
                                        c8_3 * (- inM_G2[kxnynz + (ky-2) * nz + kz] + inM_G2[kxnynz + (ky+3) * nz + kz]) +
                                        c8_4 * (- inM_G2[kxnynz + (ky-3) * nz + kz] + inM_G2[kxnynz + (ky+4) * nz + kz]);

                                const Type stencilM_G3 =
                                        c8_1 * (- inM_G3[kxnynz_kynz + (kz+0)] + inM_G3[kxnynz_kynz + (kz+1)]) +
                                        c8_2 * (- inM_G3[kxnynz_kynz + (kz-1)] + inM_G3[kxnynz_kynz + (kz+2)]) +
                                        c8_3 * (- inM_G3[kxnynz_kynz + (kz-2)] + inM_G3[kxnynz_kynz + (kz+3)]) +
                                        c8_4 * (- inM_G3[kxnynz_kynz + (kz-3)] + inM_G3[kxnynz_kynz + (kz+4)]);

                                const Type dpx = invDx * stencilP_G1;
                                const Type dpy = invDy * stencilP_G2;
                                const Type dpz = invDz * stencilP_G3;

                                const Type dmx = invDx * stencilM_G1;
                                const Type dmy = invDy * stencilM_G2;
                                const Type dmz = invDz * stencilM_G3;

                                long k = kxnynz_kynz + kz;

                                const Type E = 1 + 2 * fieldEps[k];
                                const Type A = fieldEta[k];
                                const Type F = fieldVsVp[k];
                                const Type B = fieldBuoy[k];
                                const Type SA2 = sqrt(1 - A * A);

                                const float cosThetaCosPhi = cosTheta[k] * cosPhi[k];
                                const float cosThetaSinPhi = cosTheta[k] * sinPhi[k];
                                const float sinThetaCosPhi = sinTheta[k] * cosPhi[k];

                                Type dPg1 = cosThetaCosPhi * dpx + cosThetaSinPhi * dpy - sinTheta[k] * dpz;
                                Type dPg2 = - sinPhi[k] * dpx + cosPhi[k] * dpy;
                                Type dPg3 = sinThetaCosPhi * dpx + sinTheta[k] * sinPhi[k] * dpy + cosTheta[k] * dpz;

                                Type dMg1 = cosThetaCosPhi * dmx + cosThetaSinPhi * dmy - sinTheta[k] * dmz;
                                Type dMg2 = - sinPhi[k] * dmx + cosPhi[k] * dmy;
                                Type dMg3 = sinThetaCosPhi * dmx + sinTheta[k] * sinPhi[k] * dmy + cosTheta[k] * dmz;

                                // combine terms for application of adjoint g3
                                outP_G1[k] = B * E * dPg1;
                                outP_G2[k] = B * E * dPg2;
                                outP_G3[k] = B * (1 - F * A * A) * dPg3 + B * F * A * SA2 * dMg3;

                                outM_G1[k] = B * (1 - F) * dMg1;
                                outM_G2[k] = B * (1 - F) * dMg2;
                                outM_G3[k] = B * F * A * SA2 * dPg3 + B * (1 - F + F * A * A) * dMg3;
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
                    {
                        const Type stencilP_G3 =
                                c8_1 * (- inP_G3[kxnynz_kynz + 0] + inP_G3[kxnynz_kynz + 1]) +
                                c8_2 * (+ inP_G3[kxnynz_kynz + 1] + inP_G3[kxnynz_kynz + 2]) +
                                c8_3 * (+ inP_G3[kxnynz_kynz + 2] + inP_G3[kxnynz_kynz + 3]) +
                                c8_4 * (+ inP_G3[kxnynz_kynz + 3] + inP_G3[kxnynz_kynz + 4]);

                        const Type stencilM_G3 =
                                c8_1 * (- inM_G3[kxnynz_kynz + 0] + inM_G3[kxnynz_kynz + 1]) +
                                c8_2 * (+ inM_G3[kxnynz_kynz + 1] + inM_G3[kxnynz_kynz + 2]) +
                                c8_3 * (+ inM_G3[kxnynz_kynz + 2] + inM_G3[kxnynz_kynz + 3]) +
                                c8_4 * (+ inM_G3[kxnynz_kynz + 3] + inM_G3[kxnynz_kynz + 4]);

                        const Type dpx = 0;
                        const Type dpy = 0;
                        const Type dpz = invDz * stencilP_G3;

                        const Type dmx = 0;
                        const Type dmy = 0;
                        const Type dmz = invDz * stencilM_G3;

                        const long k = kxnynz_kynz + 0;

                        const Type E = 1 + 2 * fieldEps[k];
                        const Type A = fieldEta[k];
                        const Type F = fieldVsVp[k];
                        const Type B = fieldBuoy[k];
                        const Type SA2 = sqrt(1 - A * A);

                        const float cosThetaCosPhi = cosTheta[k] * cosPhi[k];
                        const float cosThetaSinPhi = cosTheta[k] * sinPhi[k];
                        const float sinThetaCosPhi = sinTheta[k] * cosPhi[k];

                        Type dPg1 = cosThetaCosPhi * dpx + cosThetaSinPhi * dpy - sinTheta[k] * dpz;
                        Type dPg2 = - sinPhi[k] * dpx + cosPhi[k] * dpy;
                        Type dPg3 = sinThetaCosPhi * dpx + sinTheta[k] * sinPhi[k] * dpy + cosTheta[k] * dpz;

                        Type dMg1 = cosThetaCosPhi * dmx + cosThetaSinPhi * dmy - sinTheta[k] * dmz;
                        Type dMg2 = - sinPhi[k] * dmx + cosPhi[k] * dmy;
                        Type dMg3 = sinThetaCosPhi * dmx + sinTheta[k] * sinPhi[k] * dmy + cosTheta[k] * dmz;

                        // combine terms for application of adjoint g3
                        outP_G1[k] = B * E * dPg1;
                        outP_G2[k] = B * E * dPg2;
                        outP_G3[k] = B * (1 - F * A * A) * dPg3 + B * F * A * SA2 * dMg3;

                        outM_G1[k] = B * (1 - F) * dMg1;
                        outM_G2[k] = B * (1 - F) * dMg2;
                        outM_G3[k] = B * F * A * SA2 * dPg3 + B * (1 - F + F * A * A) * dMg3;
                    }

                    // kz = 1 -- 1 1/2 cells below free surface for Z derivative, 1 cells below for X/Y derivative
                    {
                        const Type stencilP_G11 =
                                c8_1 * (- inP_G1[(kx+0) * nynz + kynz + 1] + inP_G1[(kx+1) * nynz + kynz + 1]) +
                                c8_2 * (- inP_G1[(kx-1) * nynz + kynz + 1] + inP_G1[(kx+2) * nynz + kynz + 1]) +
                                c8_3 * (- inP_G1[(kx-2) * nynz + kynz + 1] + inP_G1[(kx+3) * nynz + kynz + 1]) +
                                c8_4 * (- inP_G1[(kx-3) * nynz + kynz + 1] + inP_G1[(kx+4) * nynz + kynz + 1]);

                        const Type stencilP_G21 =
                                c8_1 * (- inP_G2[kxnynz + (ky+0) * nz + 1] + inP_G2[kxnynz + (ky+1) * nz + 1]) +
                                c8_2 * (- inP_G2[kxnynz + (ky-1) * nz + 1] + inP_G2[kxnynz + (ky+2) * nz + 1]) +
                                c8_3 * (- inP_G2[kxnynz + (ky-2) * nz + 1] + inP_G2[kxnynz + (ky+3) * nz + 1]) +
                                c8_4 * (- inP_G2[kxnynz + (ky-3) * nz + 1] + inP_G2[kxnynz + (ky+4) * nz + 1]);

                        const Type stencilP_G31 =
                                c8_1 * (- inP_G3[kxnynz_kynz + 1] + inP_G3[kxnynz_kynz + 2]) +
                                c8_2 * (- inP_G3[kxnynz_kynz + 0] + inP_G3[kxnynz_kynz + 3]) +
                                c8_3 * (+ inP_G3[kxnynz_kynz + 1] + inP_G3[kxnynz_kynz + 4]) +
                                c8_4 * (+ inP_G3[kxnynz_kynz + 2] + inP_G3[kxnynz_kynz + 5]);

                        const Type stencilM_G11 =
                                c8_1 * (- inM_G1[(kx+0) * nynz + kynz + 1] + inM_G1[(kx+1) * nynz + kynz + 1]) +
                                c8_2 * (- inM_G1[(kx-1) * nynz + kynz + 1] + inM_G1[(kx+2) * nynz + kynz + 1]) +
                                c8_3 * (- inM_G1[(kx-2) * nynz + kynz + 1] + inM_G1[(kx+3) * nynz + kynz + 1]) +
                                c8_4 * (- inM_G1[(kx-3) * nynz + kynz + 1] + inM_G1[(kx+4) * nynz + kynz + 1]);

                        const Type stencilM_G21 =
                                c8_1 * (- inM_G2[kxnynz + (ky+0) * nz + 1] + inM_G2[kxnynz + (ky+1) * nz + 1]) +
                                c8_2 * (- inM_G2[kxnynz + (ky-1) * nz + 1] + inM_G2[kxnynz + (ky+2) * nz + 1]) +
                                c8_3 * (- inM_G2[kxnynz + (ky-2) * nz + 1] + inM_G2[kxnynz + (ky+3) * nz + 1]) +
                                c8_4 * (- inM_G2[kxnynz + (ky-3) * nz + 1] + inM_G2[kxnynz + (ky+4) * nz + 1]);

                        const Type stencilM_G31 =
                                c8_1 * (- inM_G3[kxnynz_kynz + 1] + inM_G3[kxnynz_kynz + 2]) +
                                c8_2 * (- inM_G3[kxnynz_kynz + 0] + inM_G3[kxnynz_kynz + 3]) +
                                c8_3 * (+ inM_G3[kxnynz_kynz + 1] + inM_G3[kxnynz_kynz + 4]) +
                                c8_4 * (+ inM_G3[kxnynz_kynz + 2] + inM_G3[kxnynz_kynz + 5]);

                        const Type dpx = invDx * stencilP_G11;
                        const Type dpy = invDy * stencilP_G21;
                        const Type dpz = invDz * stencilP_G31;

                        const Type dmx = invDx * stencilM_G11;
                        const Type dmy = invDy * stencilM_G21;
                        const Type dmz = invDz * stencilM_G31;

                        const long k = kxnynz_kynz + 1;

                        const Type E = 1 + 2 * fieldEps[k];
                        const Type A = fieldEta[k];
                        const Type F = fieldVsVp[k];
                        const Type B = fieldBuoy[k];
                        const Type SA2 = sqrt(1 - A * A);

                        const float cosThetaCosPhi = cosTheta[k] * cosPhi[k];
                        const float cosThetaSinPhi = cosTheta[k] * sinPhi[k];
                        const float sinThetaCosPhi = sinTheta[k] * cosPhi[k];

                        Type dPg1 = cosThetaCosPhi * dpx + cosThetaSinPhi * dpy - sinTheta[k] * dpz;
                        Type dPg2 = - sinPhi[k] * dpx + cosPhi[k] * dpy;
                        Type dPg3 = sinThetaCosPhi * dpx + sinTheta[k] * sinPhi[k] * dpy + cosTheta[k] * dpz;

                        Type dMg1 = cosThetaCosPhi * dmx + cosThetaSinPhi * dmy - sinTheta[k] * dmz;
                        Type dMg2 = - sinPhi[k] * dmx + cosPhi[k] * dmy;
                        Type dMg3 = sinThetaCosPhi * dmx + sinTheta[k] * sinPhi[k] * dmy + cosTheta[k] * dmz;

                        // combine terms for application of adjoint g3
                        outP_G1[k] = B * E * dPg1;
                        outP_G2[k] = B * E * dPg2;
                        outP_G3[k] = B * (1 - F * A * A) * dPg3 + B * F * A * SA2 * dMg3;

                        outM_G1[k] = B * (1 - F) * dMg1;
                        outM_G2[k] = B * (1 - F) * dMg2;
                        outM_G3[k] = B * F * A * SA2 * dPg3 + B * (1 - F + F * A * A) * dMg3;
                    }

                    // kz = 2 -- 2 1/2 cells below free surface for Z derivative, 2 cells below for X/Y derivative
                    {
                        const Type stencilP_G12 =
                                c8_1 * (- inP_G1[(kx+0) * nynz + kynz + 2] + inP_G1[(kx+1) * nynz + kynz + 2]) +
                                c8_2 * (- inP_G1[(kx-1) * nynz + kynz + 2] + inP_G1[(kx+2) * nynz + kynz + 2]) +
                                c8_3 * (- inP_G1[(kx-2) * nynz + kynz + 2] + inP_G1[(kx+3) * nynz + kynz + 2]) +
                                c8_4 * (- inP_G1[(kx-3) * nynz + kynz + 2] + inP_G1[(kx+4) * nynz + kynz + 2]);

                        const Type stencilP_G22 =
                                c8_1 * (- inP_G2[kxnynz + (ky+0) * nz + 2] + inP_G2[kxnynz + (ky+1) * nz + 2]) +
                                c8_2 * (- inP_G2[kxnynz + (ky-1) * nz + 2] + inP_G2[kxnynz + (ky+2) * nz + 2]) +
                                c8_3 * (- inP_G2[kxnynz + (ky-2) * nz + 2] + inP_G2[kxnynz + (ky+3) * nz + 2]) +
                                c8_4 * (- inP_G2[kxnynz + (ky-3) * nz + 2] + inP_G2[kxnynz + (ky+4) * nz + 2]);

                        const Type stencilP_G32 =
                                c8_1 * (- inP_G3[kxnynz_kynz + 2] + inP_G3[kxnynz_kynz + 3]) +
                                c8_2 * (- inP_G3[kxnynz_kynz + 1] + inP_G3[kxnynz_kynz + 4]) +
                                c8_3 * (- inP_G3[kxnynz_kynz + 0] + inP_G3[kxnynz_kynz + 5]) +
                                c8_4 * (+ inP_G3[kxnynz_kynz + 1] + inP_G3[kxnynz_kynz + 6]);

                        const Type stencilM_G12 =
                                c8_1 * (- inM_G1[(kx+0) * nynz + kynz + 2] + inM_G1[(kx+1) * nynz + kynz + 2]) +
                                c8_2 * (- inM_G1[(kx-1) * nynz + kynz + 2] + inM_G1[(kx+2) * nynz + kynz + 2]) +
                                c8_3 * (- inM_G1[(kx-2) * nynz + kynz + 2] + inM_G1[(kx+3) * nynz + kynz + 2]) +
                                c8_4 * (- inM_G1[(kx-3) * nynz + kynz + 2] + inM_G1[(kx+4) * nynz + kynz + 2]);

                        const Type stencilM_G22 =
                                c8_1 * (- inM_G2[kxnynz + (ky+0) * nz + 2] + inM_G2[kxnynz + (ky+1) * nz + 2]) +
                                c8_2 * (- inM_G2[kxnynz + (ky-1) * nz + 2] + inM_G2[kxnynz + (ky+2) * nz + 2]) +
                                c8_3 * (- inM_G2[kxnynz + (ky-2) * nz + 2] + inM_G2[kxnynz + (ky+3) * nz + 2]) +
                                c8_4 * (- inM_G2[kxnynz + (ky-3) * nz + 2] + inM_G2[kxnynz + (ky+4) * nz + 2]);

                        const Type stencilM_G32 =
                                c8_1 * (- inM_G3[kxnynz_kynz + 2] + inM_G3[kxnynz_kynz + 3]) +
                                c8_2 * (- inM_G3[kxnynz_kynz + 1] + inM_G3[kxnynz_kynz + 4]) +
                                c8_3 * (- inM_G3[kxnynz_kynz + 0] + inM_G3[kxnynz_kynz + 5]) +
                                c8_4 * (+ inM_G3[kxnynz_kynz + 1] + inM_G3[kxnynz_kynz + 6]);

                        const Type dpx = invDx * stencilP_G12;
                        const Type dpy = invDy * stencilP_G22;
                        const Type dpz = invDz * stencilP_G32;

                        const Type dmx = invDx * stencilM_G12;
                        const Type dmy = invDy * stencilM_G22;
                        const Type dmz = invDz * stencilM_G32;

                        const long k = kxnynz_kynz + 2;

                        const Type E = 1 + 2 * fieldEps[k];
                        const Type A = fieldEta[k];
                        const Type F = fieldVsVp[k];
                        const Type B = fieldBuoy[k];
                        const Type SA2 = sqrt(1 - A * A);

                        const float cosThetaCosPhi = cosTheta[k] * cosPhi[k];
                        const float cosThetaSinPhi = cosTheta[k] * sinPhi[k];
                        const float sinThetaCosPhi = sinTheta[k] * cosPhi[k];

                        Type dPg1 = cosThetaCosPhi * dpx + cosThetaSinPhi * dpy - sinTheta[k] * dpz;
                        Type dPg2 = - sinPhi[k] * dpx + cosPhi[k] * dpy;
                        Type dPg3 = sinThetaCosPhi * dpx + sinTheta[k] * sinPhi[k] * dpy + cosTheta[k] * dpz;

                        Type dMg1 = cosThetaCosPhi * dmx + cosThetaSinPhi * dmy - sinTheta[k] * dmz;
                        Type dMg2 = - sinPhi[k] * dmx + cosPhi[k] * dmy;
                        Type dMg3 = sinThetaCosPhi * dmx + sinTheta[k] * sinPhi[k] * dmy + cosTheta[k] * dmz;

                        // combine terms for application of adjoint g3
                        outP_G1[k] = B * E * dPg1;
                        outP_G2[k] = B * E * dPg2;
                        outP_G3[k] = B * (1 - F * A * A) * dPg3 + B * F * A * SA2 * dMg3;

                        outM_G1[k] = B * (1 - F) * dMg1;
                        outM_G2[k] = B * (1 - F) * dMg2;
                        outM_G3[k] = B * F * A * SA2 * dPg3 + B * (1 - F + F * A * A) * dMg3;
                    }

                    // kz = 3 -- 3 1/2 cells below free surface for Z derivative, 3 cells below for X/Y derivative
                    {
                        const Type stencilP_G13 =
                                c8_1 * (- inP_G1[(kx+0) * nynz + kynz + 3] + inP_G1[(kx+1) * nynz + kynz + 3]) +
                                c8_2 * (- inP_G1[(kx-1) * nynz + kynz + 3] + inP_G1[(kx+2) * nynz + kynz + 3]) +
                                c8_3 * (- inP_G1[(kx-2) * nynz + kynz + 3] + inP_G1[(kx+3) * nynz + kynz + 3]) +
                                c8_4 * (- inP_G1[(kx-3) * nynz + kynz + 3] + inP_G1[(kx+4) * nynz + kynz + 3]);

                        const Type stencilP_G23 =
                                c8_1 * (- inP_G2[kxnynz + (ky+0) * nz + 3] + inP_G2[kxnynz + (ky+1) * nz + 3]) +
                                c8_2 * (- inP_G2[kxnynz + (ky-1) * nz + 3] + inP_G2[kxnynz + (ky+2) * nz + 3]) +
                                c8_3 * (- inP_G2[kxnynz + (ky-2) * nz + 3] + inP_G2[kxnynz + (ky+3) * nz + 3]) +
                                c8_4 * (- inP_G2[kxnynz + (ky-3) * nz + 3] + inP_G2[kxnynz + (ky+4) * nz + 3]);

                        const Type stencilP_G33 =
                                c8_1 * (- inP_G3[kxnynz_kynz + 3] + inP_G3[kxnynz_kynz + 4]) +
                                c8_2 * (- inP_G3[kxnynz_kynz + 2] + inP_G3[kxnynz_kynz + 5]) +
                                c8_3 * (- inP_G3[kxnynz_kynz + 1] + inP_G3[kxnynz_kynz + 6]) +
                                c8_4 * (- inP_G3[kxnynz_kynz + 0] + inP_G3[kxnynz_kynz + 7]);

                        const Type stencilM_G13 =
                                c8_1 * (- inM_G1[(kx+0) * nynz + kynz + 3] + inM_G1[(kx+1) * nynz + kynz + 3]) +
                                c8_2 * (- inM_G1[(kx-1) * nynz + kynz + 3] + inM_G1[(kx+2) * nynz + kynz + 3]) +
                                c8_3 * (- inM_G1[(kx-2) * nynz + kynz + 3] + inM_G1[(kx+3) * nynz + kynz + 3]) +
                                c8_4 * (- inM_G1[(kx-3) * nynz + kynz + 3] + inM_G1[(kx+4) * nynz + kynz + 3]);

                        const Type stencilM_G23 =
                                c8_1 * (- inM_G2[kxnynz + (ky+0) * nz + 3] + inM_G2[kxnynz + (ky+1) * nz + 3]) +
                                c8_2 * (- inM_G2[kxnynz + (ky-1) * nz + 3] + inM_G2[kxnynz + (ky+2) * nz + 3]) +
                                c8_3 * (- inM_G2[kxnynz + (ky-2) * nz + 3] + inM_G2[kxnynz + (ky+3) * nz + 3]) +
                                c8_4 * (- inM_G2[kxnynz + (ky-3) * nz + 3] + inM_G2[kxnynz + (ky+4) * nz + 3]);

                        const Type stencilM_G33 =
                                c8_1 * (- inM_G3[kxnynz_kynz + 3] + inM_G3[kxnynz_kynz + 4]) +
                                c8_2 * (- inM_G3[kxnynz_kynz + 2] + inM_G3[kxnynz_kynz + 5]) +
                                c8_3 * (- inM_G3[kxnynz_kynz + 1] + inM_G3[kxnynz_kynz + 6]) +
                                c8_4 * (- inM_G3[kxnynz_kynz + 0] + inM_G3[kxnynz_kynz + 7]);

                        const Type dpx = invDx * stencilP_G13;
                        const Type dpy = invDy * stencilP_G23;
                        const Type dpz = invDz * stencilP_G33;

                        const Type dmx = invDx * stencilM_G13;
                        const Type dmy = invDy * stencilM_G23;
                        const Type dmz = invDz * stencilM_G33;

                        const long k = kxnynz_kynz + 3;

                        const Type E = 1 + 2 * fieldEps[k];
                        const Type A = fieldEta[k];
                        const Type F = fieldVsVp[k];
                        const Type B = fieldBuoy[k];
                        const Type SA2 = sqrt(1 - A * A);

                        const float cosThetaCosPhi = cosTheta[k] * cosPhi[k];
                        const float cosThetaSinPhi = cosTheta[k] * sinPhi[k];
                        const float sinThetaCosPhi = sinTheta[k] * cosPhi[k];

                        Type dPg1 = cosThetaCosPhi * dpx + cosThetaSinPhi * dpy - sinTheta[k] * dpz;
                        Type dPg2 = - sinPhi[k] * dpx + cosPhi[k] * dpy;
                        Type dPg3 = sinThetaCosPhi * dpx + sinTheta[k] * sinPhi[k] * dpy + cosTheta[k] * dpz;

                        Type dMg1 = cosThetaCosPhi * dmx + cosThetaSinPhi * dmy - sinTheta[k] * dmz;
                        Type dMg2 = - sinPhi[k] * dmx + cosPhi[k] * dmy;
                        Type dMg3 = sinThetaCosPhi * dmx + sinTheta[k] * sinPhi[k] * dmy + cosTheta[k] * dmz;

                        // combine terms for application of adjoint g3
                        outP_G1[k] = B * E * dPg1;
                        outP_G2[k] = B * E * dPg2;
                        outP_G3[k] = B * (1 - F * A * A) * dPg3 + B * F * A * SA2 * dMg3;

                        outM_G1[k] = B * (1 - F) * dMg1;
                        outM_G2[k] = B * (1 - F) * dMg2;
                        outM_G3[k] = B * F * A * SA2 * dPg3 + B * (1 - F + F * A * A) * dMg3;
                    }
                }
            }

        }
    }

    template<class Type>
    inline static void applyFirstDerivatives3D_TTI_MinusHalf_TimeUpdate_Nonlinear(
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
            const Type dt,
            Type * __restrict__ inP_G1,
            Type * __restrict__ inP_G2,
            Type * __restrict__ inP_G3,
            Type * __restrict__ inM_G1,
            Type * __restrict__ inM_G2,
            Type * __restrict__ inM_G3,
            Type * __restrict__ fieldVel,
            Type * __restrict__ fieldBuoy,
            Type * __restrict__ dtOmegaInvQ,
            float * __restrict__ sinTheta,
            float * __restrict__ cosTheta,
            float * __restrict__ sinPhi,
            float * __restrict__ cosPhi,
            Type * __restrict__ pCur,
            Type * __restrict__ mCur,
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
        const Type dt2 = dt * dt;

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
        // TODO -- this does significantly more compute than what John has in his latest stuff.
#pragma omp parallel for collapse(3) num_threads(nthread) schedule(static)
        for (long bx = 4; bx < nx4; bx += BX_3D) {
            for (long by = 4; by < ny4; by += BY_3D) {
                for (long bz = 4; bz < nz4; bz += BZ_3D) {
                    const long kxmax = std::min(bx + BX_3D, nx4);
                    const long kymax = std::min(by + BY_3D, ny4);
                    const long kzmax = std::min(bz + BZ_3D, nz4);

                    for (long kx = bx; kx < kxmax; kx++) {
                        const long kxnynz = kx * nynz;

                        for (long ky = by; ky < kymax; ky++) {
                            const long kynz = ky * nz;
                            const long kxnynz_kynz = kxnynz + kynz;

#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long kynz_kz = + kynz + kz;

                                const long kxm4 = (kx-4) * nynz + kynz_kz;
                                const long kxm3 = (kx-3) * nynz + kynz_kz;
                                const long kxm2 = (kx-2) * nynz + kynz_kz;
                                const long kxm1 = (kx-1) * nynz + kynz_kz;
                                const long kxp0 = (kx+0) * nynz + kynz_kz;
                                const long kxp1 = (kx+1) * nynz + kynz_kz;
                                const long kxp2 = (kx+2) * nynz + kynz_kz;
                                const long kxp3 = (kx+3) * nynz + kynz_kz;

                                const long kym4 = kxnynz + (ky-4) * nz + kz;
                                const long kym3 = kxnynz + (ky-3) * nz + kz;
                                const long kym2 = kxnynz + (ky-2) * nz + kz;
                                const long kym1 = kxnynz + (ky-1) * nz + kz;
                                const long kyp0 = kxnynz + (ky+0) * nz + kz;
                                const long kyp1 = kxnynz + (ky+1) * nz + kz;
                                const long kyp2 = kxnynz + (ky+2) * nz + kz;
                                const long kyp3 = kxnynz + (ky+3) * nz + kz;

                                const long kzm4 = kxnynz_kynz + (kz-4);
                                const long kzm3 = kxnynz_kynz + (kz-3);
                                const long kzm2 = kxnynz_kynz + (kz-2);
                                const long kzm1 = kxnynz_kynz + (kz-1);
                                const long kzp0 = kxnynz_kynz + (kz+0);
                                const long kzp1 = kxnynz_kynz + (kz+1);
                                const long kzp2 = kxnynz_kynz + (kz+2);
                                const long kzp3 = kxnynz_kynz + (kz+3);


                                // ........................ G1 ........................
                                const Type stencilP_G1A =
                                        c8_1 * (- cosTheta[kxm1] * cosPhi[kxm1] * inP_G1[kxm1] + cosTheta[kxp0] * cosPhi[kxp0] * inP_G1[kxp0]) +
                                        c8_2 * (- cosTheta[kxm2] * cosPhi[kxm2] * inP_G1[kxm2] + cosTheta[kxp1] * cosPhi[kxp1] * inP_G1[kxp1]) +
                                        c8_3 * (- cosTheta[kxm3] * cosPhi[kxm3] * inP_G1[kxm3] + cosTheta[kxp2] * cosPhi[kxp2] * inP_G1[kxp2]) +
                                        c8_4 * (- cosTheta[kxm4] * cosPhi[kxm4] * inP_G1[kxm4] + cosTheta[kxp3] * cosPhi[kxp3] * inP_G1[kxp3]);

                                const Type stencilP_G1B =
                                        c8_1 * (- cosTheta[kym1] * sinPhi[kym1] * inP_G1[kym1] + cosTheta[kyp0] * sinPhi[kyp0] * inP_G1[kyp0]) +
                                        c8_2 * (- cosTheta[kym2] * sinPhi[kym2] * inP_G1[kym2] + cosTheta[kyp1] * sinPhi[kyp1] * inP_G1[kyp1]) +
                                        c8_3 * (- cosTheta[kym3] * sinPhi[kym3] * inP_G1[kym3] + cosTheta[kyp2] * sinPhi[kyp2] * inP_G1[kyp2]) +
                                        c8_4 * (- cosTheta[kym4] * sinPhi[kym4] * inP_G1[kym4] + cosTheta[kyp3] * sinPhi[kyp3] * inP_G1[kyp3]);

                                const Type stencilP_G1C =
                                        c8_1 * (- sinTheta[kzm1] * inP_G1[kzm1] + sinTheta[kzp0] * inP_G1[kzp0]) +
                                        c8_2 * (- sinTheta[kzm2] * inP_G1[kzm2] + sinTheta[kzp1] * inP_G1[kzp1]) +
                                        c8_3 * (- sinTheta[kzm3] * inP_G1[kzm3] + sinTheta[kzp2] * inP_G1[kzp2]) +
                                        c8_4 * (- sinTheta[kzm4] * inP_G1[kzm4] + sinTheta[kzp3] * inP_G1[kzp3]);


                                // ........................ G2 ........................
                                const Type stencilP_G2A =
                                        c8_1 * (- sinPhi[kxm1] * inP_G2[kxm1] + sinPhi[kxp0] * inP_G2[kxp0]) +
                                        c8_2 * (- sinPhi[kxm2] * inP_G2[kxm2] + sinPhi[kxp1] * inP_G2[kxp1]) +
                                        c8_3 * (- sinPhi[kxm3] * inP_G2[kxm3] + sinPhi[kxp2] * inP_G2[kxp2]) +
                                        c8_4 * (- sinPhi[kxm4] * inP_G2[kxm4] + sinPhi[kxp3] * inP_G2[kxp3]);

                                const Type stencilP_G2B =
                                        c8_1 * (- cosPhi[kym1] * inP_G2[kym1] + cosPhi[kyp0] * inP_G2[kyp0]) +
                                        c8_2 * (- cosPhi[kym2] * inP_G2[kym2] + cosPhi[kyp1] * inP_G2[kyp1]) +
                                        c8_3 * (- cosPhi[kym3] * inP_G2[kym3] + cosPhi[kyp2] * inP_G2[kyp2]) +
                                        c8_4 * (- cosPhi[kym4] * inP_G2[kym4] + cosPhi[kyp3] * inP_G2[kyp3]);


                                // ........................ G3 ........................
                                const Type stencilP_G3A =
                                        c8_1 * (- sinTheta[kxm1] * cosPhi[kxm1] * inP_G3[kxm1] + sinTheta[kxp0] * cosPhi[kxp0] * inP_G3[kxp0]) +
                                        c8_2 * (- sinTheta[kxm2] * cosPhi[kxm2] * inP_G3[kxm2] + sinTheta[kxp1] * cosPhi[kxp1] * inP_G3[kxp1]) +
                                        c8_3 * (- sinTheta[kxm3] * cosPhi[kxm3] * inP_G3[kxm3] + sinTheta[kxp2] * cosPhi[kxp2] * inP_G3[kxp2]) +
                                        c8_4 * (- sinTheta[kxm4] * cosPhi[kxm4] * inP_G3[kxm4] + sinTheta[kxp3] * cosPhi[kxp3] * inP_G3[kxp3]);

                                const Type stencilP_G3B =
                                        c8_1 * (- sinTheta[kym1] * sinPhi[kym1] * inP_G3[kym1] + sinTheta[kyp0] * sinPhi[kyp0] * inP_G3[kyp0]) +
                                        c8_2 * (- sinTheta[kym2] * sinPhi[kym2] * inP_G3[kym2] + sinTheta[kyp1] * sinPhi[kyp1] * inP_G3[kyp1]) +
                                        c8_3 * (- sinTheta[kym3] * sinPhi[kym3] * inP_G3[kym3] + sinTheta[kyp2] * sinPhi[kyp2] * inP_G3[kyp2]) +
                                        c8_4 * (- sinTheta[kym4] * sinPhi[kym4] * inP_G3[kym4] + sinTheta[kyp3] * sinPhi[kyp3] * inP_G3[kyp3]);

                                const Type stencilP_G3C =
                                        c8_1 * (- cosTheta[kzm1] * inP_G3[kzm1] + cosTheta[kzp0] * inP_G3[kzp0]) +
                                        c8_2 * (- cosTheta[kzm2] * inP_G3[kzm2] + cosTheta[kzp1] * inP_G3[kzp1]) +
                                        c8_3 * (- cosTheta[kzm3] * inP_G3[kzm3] + cosTheta[kzp2] * inP_G3[kzp2]) +
                                        c8_4 * (- cosTheta[kzm4] * inP_G3[kzm4] + cosTheta[kzp3] * inP_G3[kzp3]);


                                // ........................ G1 ........................
                                const Type stencilM_G1A =
                                        c8_1 * (- cosTheta[kxm1] * cosPhi[kxm1] * inM_G1[kxm1] + cosTheta[kxp0] * cosPhi[kxp0] * inM_G1[kxp0]) +
                                        c8_2 * (- cosTheta[kxm2] * cosPhi[kxm2] * inM_G1[kxm2] + cosTheta[kxp1] * cosPhi[kxp1] * inM_G1[kxp1]) +
                                        c8_3 * (- cosTheta[kxm3] * cosPhi[kxm3] * inM_G1[kxm3] + cosTheta[kxp2] * cosPhi[kxp2] * inM_G1[kxp2]) +
                                        c8_4 * (- cosTheta[kxm4] * cosPhi[kxm4] * inM_G1[kxm4] + cosTheta[kxp3] * cosPhi[kxp3] * inM_G1[kxp3]);

                                const Type stencilM_G1B =
                                        c8_1 * (- cosTheta[kym1] * sinPhi[kym1] * inM_G1[kym1] + cosTheta[kyp0] * sinPhi[kyp0] * inM_G1[kyp0]) +
                                        c8_2 * (- cosTheta[kym2] * sinPhi[kym2] * inM_G1[kym2] + cosTheta[kyp1] * sinPhi[kyp1] * inM_G1[kyp1]) +
                                        c8_3 * (- cosTheta[kym3] * sinPhi[kym3] * inM_G1[kym3] + cosTheta[kyp2] * sinPhi[kyp2] * inM_G1[kyp2]) +
                                        c8_4 * (- cosTheta[kym4] * sinPhi[kym4] * inM_G1[kym4] + cosTheta[kyp3] * sinPhi[kyp3] * inM_G1[kyp3]);

                                const Type stencilM_G1C =
                                        c8_1 * (- sinTheta[kzm1] * inM_G1[kzm1] + sinTheta[kzp0] * inM_G1[kzp0]) +
                                        c8_2 * (- sinTheta[kzm2] * inM_G1[kzm2] + sinTheta[kzp1] * inM_G1[kzp1]) +
                                        c8_3 * (- sinTheta[kzm3] * inM_G1[kzm3] + sinTheta[kzp2] * inM_G1[kzp2]) +
                                        c8_4 * (- sinTheta[kzm4] * inM_G1[kzm4] + sinTheta[kzp3] * inM_G1[kzp3]);


                                // ........................ G2 ........................
                                const Type stencilM_G2A =
                                        c8_1 * (- sinPhi[kxm1] * inM_G2[kxm1] + sinPhi[kxp0] * inM_G2[kxp0]) +
                                        c8_2 * (- sinPhi[kxm2] * inM_G2[kxm2] + sinPhi[kxp1] * inM_G2[kxp1]) +
                                        c8_3 * (- sinPhi[kxm3] * inM_G2[kxm3] + sinPhi[kxp2] * inM_G2[kxp2]) +
                                        c8_4 * (- sinPhi[kxm4] * inM_G2[kxm4] + sinPhi[kxp3] * inM_G2[kxp3]);

                                const Type stencilM_G2B =
                                        c8_1 * (- cosPhi[kym1] * inM_G2[kym1] + cosPhi[kyp0] * inM_G2[kyp0]) +
                                        c8_2 * (- cosPhi[kym2] * inM_G2[kym2] + cosPhi[kyp1] * inM_G2[kyp1]) +
                                        c8_3 * (- cosPhi[kym3] * inM_G2[kym3] + cosPhi[kyp2] * inM_G2[kyp2]) +
                                        c8_4 * (- cosPhi[kym4] * inM_G2[kym4] + cosPhi[kyp3] * inM_G2[kyp3]);


                                // ........................ G3 ........................
                                const Type stencilM_G3A =
                                        c8_1 * (- sinTheta[kxm1] * cosPhi[kxm1] * inM_G3[kxm1] + sinTheta[kxp0] * cosPhi[kxp0] * inM_G3[kxp0]) +
                                        c8_2 * (- sinTheta[kxm2] * cosPhi[kxm2] * inM_G3[kxm2] + sinTheta[kxp1] * cosPhi[kxp1] * inM_G3[kxp1]) +
                                        c8_3 * (- sinTheta[kxm3] * cosPhi[kxm3] * inM_G3[kxm3] + sinTheta[kxp2] * cosPhi[kxp2] * inM_G3[kxp2]) +
                                        c8_4 * (- sinTheta[kxm4] * cosPhi[kxm4] * inM_G3[kxm4] + sinTheta[kxp3] * cosPhi[kxp3] * inM_G3[kxp3]);

                                const Type stencilM_G3B =
                                        c8_1 * (- sinTheta[kym1] * sinPhi[kym1] * inM_G3[kym1] + sinTheta[kyp0] * sinPhi[kyp0] * inM_G3[kyp0]) +
                                        c8_2 * (- sinTheta[kym2] * sinPhi[kym2] * inM_G3[kym2] + sinTheta[kyp1] * sinPhi[kyp1] * inM_G3[kyp1]) +
                                        c8_3 * (- sinTheta[kym3] * sinPhi[kym3] * inM_G3[kym3] + sinTheta[kyp2] * sinPhi[kyp2] * inM_G3[kyp2]) +
                                        c8_4 * (- sinTheta[kym4] * sinPhi[kym4] * inM_G3[kym4] + sinTheta[kyp3] * sinPhi[kyp3] * inM_G3[kyp3]);

                                const Type stencilM_G3C =
                                        c8_1 * (- cosTheta[kzm1] * inM_G3[kzm1] + cosTheta[kzp0] * inM_G3[kzp0]) +
                                        c8_2 * (- cosTheta[kzm2] * inM_G3[kzm2] + cosTheta[kzp1] * inM_G3[kzp1]) +
                                        c8_3 * (- cosTheta[kzm3] * inM_G3[kzm3] + cosTheta[kzp2] * inM_G3[kzp2]) +
                                        c8_4 * (- cosTheta[kzm4] * inM_G3[kzm4] + cosTheta[kzp3] * inM_G3[kzp3]);

                                const long k = kxnynz_kynz + kz;

                                const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                                const Type dpg1 = invDx * stencilP_G1A + invDy * stencilP_G1B - invDz * stencilP_G1C;
                                const Type dpg2 = - invDx * stencilP_G2A + invDy * stencilP_G2B;
                                const Type dpg3 = invDx * stencilP_G3A + invDy * stencilP_G3B + invDz * stencilP_G3C;

                                const Type dmg1 = invDx * stencilM_G1A + invDy * stencilM_G1B - invDz * stencilM_G1C;
                                const Type dmg2 = - invDx * stencilM_G2A + invDy * stencilM_G2B;
                                const Type dmg3 = invDx * stencilM_G3A + invDy * stencilM_G3B + invDz * stencilM_G3C;

                                pSpace[k] = dpg1 + dpg2 + dpg3;
                                mSpace[k] = dmg1 + dmg2 + dmg3;

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

                    // kz = 0 -- at the free surface -- p = 0, dp = 0
                    {
                        const Type dpg1 = 0;
                        const Type dpg2 = 0;
                        const Type dpg3 = 0;

                        const Type dmg1 = 0;
                        const Type dmg2 = 0;
                        const Type dmg3 = 0;

                        const long k = kxnynz_kynz + 0;

                        const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                        pSpace[k] = dpg1 + dpg2 + dpg3;
                        mSpace[k] = dmg1 + dmg2 + dmg3;

                        pOld[k] = dt2V2_B * pSpace[k] - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                        mOld[k] = dt2V2_B * mSpace[k] - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
                    }

                    // kz = 1 -- one cell below the free surface
                    {
                        const long kz = 1;
                        const long kynz_kz = + kynz + kz;

                        const long kxm4 = (kx-4) * nynz + kynz_kz;
                        const long kxm3 = (kx-3) * nynz + kynz_kz;
                        const long kxm2 = (kx-2) * nynz + kynz_kz;
                        const long kxm1 = (kx-1) * nynz + kynz_kz;
                        const long kxp0 = (kx+0) * nynz + kynz_kz;
                        const long kxp1 = (kx+1) * nynz + kynz_kz;
                        const long kxp2 = (kx+2) * nynz + kynz_kz;
                        const long kxp3 = (kx+3) * nynz + kynz_kz;

                        const long kym4 = kxnynz + (ky-4) * nz + kz;
                        const long kym3 = kxnynz + (ky-3) * nz + kz;
                        const long kym2 = kxnynz + (ky-2) * nz + kz;
                        const long kym1 = kxnynz + (ky-1) * nz + kz;
                        const long kyp0 = kxnynz + (ky+0) * nz + kz;
                        const long kyp1 = kxnynz + (ky+1) * nz + kz;
                        const long kyp2 = kxnynz + (ky+2) * nz + kz;
                        const long kyp3 = kxnynz + (ky+3) * nz + kz;

                        const long kzm4 = kxnynz_kynz + 2;
                        const long kzm3 = kxnynz_kynz + 1;
                        const long kzm2 = kxnynz_kynz + 0;
                        const long kzm1 = kxnynz_kynz + 0;
                        const long kzp0 = kxnynz_kynz + 1;
                        const long kzp1 = kxnynz_kynz + 2;
                        const long kzp2 = kxnynz_kynz + 3;
                        const long kzp3 = kxnynz_kynz + 4;


                        // ........................ G1 ........................
                        const Type stencilP_G1A =
                                c8_1 * (- cosTheta[kxm1] * cosPhi[kxm1] * inP_G1[kxm1] + cosTheta[kxp0] * cosPhi[kxp0] * inP_G1[kxp0]) +
                                c8_2 * (- cosTheta[kxm2] * cosPhi[kxm2] * inP_G1[kxm2] + cosTheta[kxp1] * cosPhi[kxp1] * inP_G1[kxp1]) +
                                c8_3 * (- cosTheta[kxm3] * cosPhi[kxm3] * inP_G1[kxm3] + cosTheta[kxp2] * cosPhi[kxp2] * inP_G1[kxp2]) +
                                c8_4 * (- cosTheta[kxm4] * cosPhi[kxm4] * inP_G1[kxm4] + cosTheta[kxp3] * cosPhi[kxp3] * inP_G1[kxp3]);

                        const Type stencilP_G1B =
                                c8_1 * (- cosTheta[kym1] * sinPhi[kym1] * inP_G1[kym1] + cosTheta[kyp0] * sinPhi[kyp0] * inP_G1[kyp0]) +
                                c8_2 * (- cosTheta[kym2] * sinPhi[kym2] * inP_G1[kym2] + cosTheta[kyp1] * sinPhi[kyp1] * inP_G1[kyp1]) +
                                c8_3 * (- cosTheta[kym3] * sinPhi[kym3] * inP_G1[kym3] + cosTheta[kyp2] * sinPhi[kyp2] * inP_G1[kyp2]) +
                                c8_4 * (- cosTheta[kym4] * sinPhi[kym4] * inP_G1[kym4] + cosTheta[kyp3] * sinPhi[kyp3] * inP_G1[kyp3]);

                        const Type stencilP_G1C =
                                c8_1 * (- sinTheta[kzm1] * inP_G1[kzm1] + sinTheta[kzp0] * inP_G1[kzp0]) +
                                c8_2 * (- sinTheta[kzm2] * inP_G1[kzm2] + sinTheta[kzp1] * inP_G1[kzp1]) +
                                c8_3 * (- sinTheta[kzm3] * inP_G1[kzm3] + sinTheta[kzp2] * inP_G1[kzp2]) +
                                c8_4 * (- sinTheta[kzm4] * inP_G1[kzm4] + sinTheta[kzp3] * inP_G1[kzp3]);


                        // ........................ G2 ........................
                        const Type stencilP_G2A =
                                c8_1 * (- sinPhi[kxm1] * inP_G2[kxm1] + sinPhi[kxp0] * inP_G2[kxp0]) +
                                c8_2 * (- sinPhi[kxm2] * inP_G2[kxm2] + sinPhi[kxp1] * inP_G2[kxp1]) +
                                c8_3 * (- sinPhi[kxm3] * inP_G2[kxm3] + sinPhi[kxp2] * inP_G2[kxp2]) +
                                c8_4 * (- sinPhi[kxm4] * inP_G2[kxm4] + sinPhi[kxp3] * inP_G2[kxp3]);

                        const Type stencilP_G2B =
                                c8_1 * (- cosPhi[kym1] * inP_G2[kym1] + cosPhi[kyp0] * inP_G2[kyp0]) +
                                c8_2 * (- cosPhi[kym2] * inP_G2[kym2] + cosPhi[kyp1] * inP_G2[kyp1]) +
                                c8_3 * (- cosPhi[kym3] * inP_G2[kym3] + cosPhi[kyp2] * inP_G2[kyp2]) +
                                c8_4 * (- cosPhi[kym4] * inP_G2[kym4] + cosPhi[kyp3] * inP_G2[kyp3]);


                        // ........................ G3 ........................
                        const Type stencilP_G3A =
                                c8_1 * (- sinTheta[kxm1] * cosPhi[kxm1] * inP_G3[kxm1] + sinTheta[kxp0] * cosPhi[kxp0] * inP_G3[kxp0]) +
                                c8_2 * (- sinTheta[kxm2] * cosPhi[kxm2] * inP_G3[kxm2] + sinTheta[kxp1] * cosPhi[kxp1] * inP_G3[kxp1]) +
                                c8_3 * (- sinTheta[kxm3] * cosPhi[kxm3] * inP_G3[kxm3] + sinTheta[kxp2] * cosPhi[kxp2] * inP_G3[kxp2]) +
                                c8_4 * (- sinTheta[kxm4] * cosPhi[kxm4] * inP_G3[kxm4] + sinTheta[kxp3] * cosPhi[kxp3] * inP_G3[kxp3]);

                        const Type stencilP_G3B =
                                c8_1 * (- sinTheta[kym1] * sinPhi[kym1] * inP_G3[kym1] + sinTheta[kyp0] * sinPhi[kyp0] * inP_G3[kyp0]) +
                                c8_2 * (- sinTheta[kym2] * sinPhi[kym2] * inP_G3[kym2] + sinTheta[kyp1] * sinPhi[kyp1] * inP_G3[kyp1]) +
                                c8_3 * (- sinTheta[kym3] * sinPhi[kym3] * inP_G3[kym3] + sinTheta[kyp2] * sinPhi[kyp2] * inP_G3[kyp2]) +
                                c8_4 * (- sinTheta[kym4] * sinPhi[kym4] * inP_G3[kym4] + sinTheta[kyp3] * sinPhi[kyp3] * inP_G3[kyp3]);

                        const Type stencilP_G3C =
                                c8_1 * (- cosTheta[kzm1] * inP_G3[kzm1] + cosTheta[kzp0] * inP_G3[kzp0]) +
                                c8_2 * (- cosTheta[kzm2] * inP_G3[kzm2] + cosTheta[kzp1] * inP_G3[kzp1]) +
                                c8_3 * (- cosTheta[kzm3] * inP_G3[kzm3] + cosTheta[kzp2] * inP_G3[kzp2]) +
                                c8_4 * (- cosTheta[kzm4] * inP_G3[kzm4] + cosTheta[kzp3] * inP_G3[kzp3]);


                        // ........................ G1 ........................
                        const Type stencilM_G1A =
                                c8_1 * (- cosTheta[kxm1] * cosPhi[kxm1] * inM_G1[kxm1] + cosTheta[kxp0] * cosPhi[kxp0] * inM_G1[kxp0]) +
                                c8_2 * (- cosTheta[kxm2] * cosPhi[kxm2] * inM_G1[kxm2] + cosTheta[kxp1] * cosPhi[kxp1] * inM_G1[kxp1]) +
                                c8_3 * (- cosTheta[kxm3] * cosPhi[kxm3] * inM_G1[kxm3] + cosTheta[kxp2] * cosPhi[kxp2] * inM_G1[kxp2]) +
                                c8_4 * (- cosTheta[kxm4] * cosPhi[kxm4] * inM_G1[kxm4] + cosTheta[kxp3] * cosPhi[kxp3] * inM_G1[kxp3]);

                        const Type stencilM_G1B =
                                c8_1 * (- cosTheta[kym1] * sinPhi[kym1] * inM_G1[kym1] + cosTheta[kyp0] * sinPhi[kyp0] * inM_G1[kyp0]) +
                                c8_2 * (- cosTheta[kym2] * sinPhi[kym2] * inM_G1[kym2] + cosTheta[kyp1] * sinPhi[kyp1] * inM_G1[kyp1]) +
                                c8_3 * (- cosTheta[kym3] * sinPhi[kym3] * inM_G1[kym3] + cosTheta[kyp2] * sinPhi[kyp2] * inM_G1[kyp2]) +
                                c8_4 * (- cosTheta[kym4] * sinPhi[kym4] * inM_G1[kym4] + cosTheta[kyp3] * sinPhi[kyp3] * inM_G1[kyp3]);

                        const Type stencilM_G1C =
                                c8_1 * (- sinTheta[kzm1] * inM_G1[kzm1] + sinTheta[kzp0] * inM_G1[kzp0]) +
                                c8_2 * (- sinTheta[kzm2] * inM_G1[kzm2] + sinTheta[kzp1] * inM_G1[kzp1]) +
                                c8_3 * (- sinTheta[kzm3] * inM_G1[kzm3] + sinTheta[kzp2] * inM_G1[kzp2]) +
                                c8_4 * (- sinTheta[kzm4] * inM_G1[kzm4] + sinTheta[kzp3] * inM_G1[kzp3]);


                        // ........................ G2 ........................
                        const Type stencilM_G2A =
                                c8_1 * (- sinPhi[kxm1] * inM_G2[kxm1] + sinPhi[kxp0] * inM_G2[kxp0]) +
                                c8_2 * (- sinPhi[kxm2] * inM_G2[kxm2] + sinPhi[kxp1] * inM_G2[kxp1]) +
                                c8_3 * (- sinPhi[kxm3] * inM_G2[kxm3] + sinPhi[kxp2] * inM_G2[kxp2]) +
                                c8_4 * (- sinPhi[kxm4] * inM_G2[kxm4] + sinPhi[kxp3] * inM_G2[kxp3]);

                        const Type stencilM_G2B =
                                c8_1 * (- cosPhi[kym1] * inM_G2[kym1] + cosPhi[kyp0] * inM_G2[kyp0]) +
                                c8_2 * (- cosPhi[kym2] * inM_G2[kym2] + cosPhi[kyp1] * inM_G2[kyp1]) +
                                c8_3 * (- cosPhi[kym3] * inM_G2[kym3] + cosPhi[kyp2] * inM_G2[kyp2]) +
                                c8_4 * (- cosPhi[kym4] * inM_G2[kym4] + cosPhi[kyp3] * inM_G2[kyp3]);


                        // ........................ G3 ........................
                        const Type stencilM_G3A =
                                c8_1 * (- sinTheta[kxm1] * cosPhi[kxm1] * inM_G3[kxm1] + sinTheta[kxp0] * cosPhi[kxp0] * inM_G3[kxp0]) +
                                c8_2 * (- sinTheta[kxm2] * cosPhi[kxm2] * inM_G3[kxm2] + sinTheta[kxp1] * cosPhi[kxp1] * inM_G3[kxp1]) +
                                c8_3 * (- sinTheta[kxm3] * cosPhi[kxm3] * inM_G3[kxm3] + sinTheta[kxp2] * cosPhi[kxp2] * inM_G3[kxp2]) +
                                c8_4 * (- sinTheta[kxm4] * cosPhi[kxm4] * inM_G3[kxm4] + sinTheta[kxp3] * cosPhi[kxp3] * inM_G3[kxp3]);

                        const Type stencilM_G3B =
                                c8_1 * (- sinTheta[kym1] * sinPhi[kym1] * inM_G3[kym1] + sinTheta[kyp0] * sinPhi[kyp0] * inM_G3[kyp0]) +
                                c8_2 * (- sinTheta[kym2] * sinPhi[kym2] * inM_G3[kym2] + sinTheta[kyp1] * sinPhi[kyp1] * inM_G3[kyp1]) +
                                c8_3 * (- sinTheta[kym3] * sinPhi[kym3] * inM_G3[kym3] + sinTheta[kyp2] * sinPhi[kyp2] * inM_G3[kyp2]) +
                                c8_4 * (- sinTheta[kym4] * sinPhi[kym4] * inM_G3[kym4] + sinTheta[kyp3] * sinPhi[kyp3] * inM_G3[kyp3]);

                        const Type stencilM_G3C =
                                c8_1 * (- cosTheta[kzm1] * inM_G3[kzm1] + cosTheta[kzp0] * inM_G3[kzp0]) +
                                c8_2 * (- cosTheta[kzm2] * inM_G3[kzm2] + cosTheta[kzp1] * inM_G3[kzp1]) +
                                c8_3 * (- cosTheta[kzm3] * inM_G3[kzm3] + cosTheta[kzp2] * inM_G3[kzp2]) +
                                c8_4 * (- cosTheta[kzm4] * inM_G3[kzm4] + cosTheta[kzp3] * inM_G3[kzp3]);

                        const long k = kxnynz_kynz + 1;

                        const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                        const Type dpg1 = invDx * stencilP_G1A + invDy * stencilP_G1B - invDz * stencilP_G1C;
                        const Type dpg2 = - invDx * stencilP_G2A + invDy * stencilP_G2B;
                        const Type dpg3 = invDx * stencilP_G3A + invDy * stencilP_G3B + invDz * stencilP_G3C;

                        const Type dmg1 = invDx * stencilM_G1A + invDy * stencilM_G1B - invDz * stencilM_G1C;
                        const Type dmg2 = - invDx * stencilM_G2A + invDy * stencilM_G2B;
                        const Type dmg3 = invDx * stencilM_G3A + invDy * stencilM_G3B + invDz * stencilM_G3C;

                        pSpace[k] = dpg1 + dpg2 + dpg3;
                        mSpace[k] = dmg1 + dmg2 + dmg3;

                        pOld[k] = dt2V2_B * pSpace[k] - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                        mOld[k] = dt2V2_B * mSpace[k] - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
                    }

                    // kz = 2 -- two cells below the free surface
                    {
                        const long kz = 2;
                        const long kynz_kz = + kynz + kz;

                        const long kxm4 = (kx-4) * nynz + kynz_kz;
                        const long kxm3 = (kx-3) * nynz + kynz_kz;
                        const long kxm2 = (kx-2) * nynz + kynz_kz;
                        const long kxm1 = (kx-1) * nynz + kynz_kz;
                        const long kxp0 = (kx+0) * nynz + kynz_kz;
                        const long kxp1 = (kx+1) * nynz + kynz_kz;
                        const long kxp2 = (kx+2) * nynz + kynz_kz;
                        const long kxp3 = (kx+3) * nynz + kynz_kz;

                        const long kym4 = kxnynz + (ky-4) * nz + kz;
                        const long kym3 = kxnynz + (ky-3) * nz + kz;
                        const long kym2 = kxnynz + (ky-2) * nz + kz;
                        const long kym1 = kxnynz + (ky-1) * nz + kz;
                        const long kyp0 = kxnynz + (ky+0) * nz + kz;
                        const long kyp1 = kxnynz + (ky+1) * nz + kz;
                        const long kyp2 = kxnynz + (ky+2) * nz + kz;
                        const long kyp3 = kxnynz + (ky+3) * nz + kz;

                        const long kzm4 = kxnynz_kynz + 1;
                        const long kzm3 = kxnynz_kynz + 0;
                        const long kzm2 = kxnynz_kynz + 0;
                        const long kzm1 = kxnynz_kynz + 1;
                        const long kzp0 = kxnynz_kynz + 2;
                        const long kzp1 = kxnynz_kynz + 3;
                        const long kzp2 = kxnynz_kynz + 4;
                        const long kzp3 = kxnynz_kynz + 5;


                        // ........................ G1 ........................
                        const Type stencilP_G1A =
                                c8_1 * (- cosTheta[kxm1] * cosPhi[kxm1] * inP_G1[kxm1] + cosTheta[kxp0] * cosPhi[kxp0] * inP_G1[kxp0]) +
                                c8_2 * (- cosTheta[kxm2] * cosPhi[kxm2] * inP_G1[kxm2] + cosTheta[kxp1] * cosPhi[kxp1] * inP_G1[kxp1]) +
                                c8_3 * (- cosTheta[kxm3] * cosPhi[kxm3] * inP_G1[kxm3] + cosTheta[kxp2] * cosPhi[kxp2] * inP_G1[kxp2]) +
                                c8_4 * (- cosTheta[kxm4] * cosPhi[kxm4] * inP_G1[kxm4] + cosTheta[kxp3] * cosPhi[kxp3] * inP_G1[kxp3]);

                        const Type stencilP_G1B =
                                c8_1 * (- cosTheta[kym1] * sinPhi[kym1] * inP_G1[kym1] + cosTheta[kyp0] * sinPhi[kyp0] * inP_G1[kyp0]) +
                                c8_2 * (- cosTheta[kym2] * sinPhi[kym2] * inP_G1[kym2] + cosTheta[kyp1] * sinPhi[kyp1] * inP_G1[kyp1]) +
                                c8_3 * (- cosTheta[kym3] * sinPhi[kym3] * inP_G1[kym3] + cosTheta[kyp2] * sinPhi[kyp2] * inP_G1[kyp2]) +
                                c8_4 * (- cosTheta[kym4] * sinPhi[kym4] * inP_G1[kym4] + cosTheta[kyp3] * sinPhi[kyp3] * inP_G1[kyp3]);

                        const Type stencilP_G1C =
                                c8_1 * (- sinTheta[kzm1] * inP_G1[kzm1] + sinTheta[kzp0] * inP_G1[kzp0]) +
                                c8_2 * (- sinTheta[kzm2] * inP_G1[kzm2] + sinTheta[kzp1] * inP_G1[kzp1]) +
                                c8_3 * (- sinTheta[kzm3] * inP_G1[kzm3] + sinTheta[kzp2] * inP_G1[kzp2]) +
                                c8_4 * (- sinTheta[kzm4] * inP_G1[kzm4] + sinTheta[kzp3] * inP_G1[kzp3]);


                        // ........................ G2 ........................
                        const Type stencilP_G2A =
                                c8_1 * (- sinPhi[kxm1] * inP_G2[kxm1] + sinPhi[kxp0] * inP_G2[kxp0]) +
                                c8_2 * (- sinPhi[kxm2] * inP_G2[kxm2] + sinPhi[kxp1] * inP_G2[kxp1]) +
                                c8_3 * (- sinPhi[kxm3] * inP_G2[kxm3] + sinPhi[kxp2] * inP_G2[kxp2]) +
                                c8_4 * (- sinPhi[kxm4] * inP_G2[kxm4] + sinPhi[kxp3] * inP_G2[kxp3]);

                        const Type stencilP_G2B =
                                c8_1 * (- cosPhi[kym1] * inP_G2[kym1] + cosPhi[kyp0] * inP_G2[kyp0]) +
                                c8_2 * (- cosPhi[kym2] * inP_G2[kym2] + cosPhi[kyp1] * inP_G2[kyp1]) +
                                c8_3 * (- cosPhi[kym3] * inP_G2[kym3] + cosPhi[kyp2] * inP_G2[kyp2]) +
                                c8_4 * (- cosPhi[kym4] * inP_G2[kym4] + cosPhi[kyp3] * inP_G2[kyp3]);


                        // ........................ G3 ........................
                        const Type stencilP_G3A =
                                c8_1 * (- sinTheta[kxm1] * cosPhi[kxm1] * inP_G3[kxm1] + sinTheta[kxp0] * cosPhi[kxp0] * inP_G3[kxp0]) +
                                c8_2 * (- sinTheta[kxm2] * cosPhi[kxm2] * inP_G3[kxm2] + sinTheta[kxp1] * cosPhi[kxp1] * inP_G3[kxp1]) +
                                c8_3 * (- sinTheta[kxm3] * cosPhi[kxm3] * inP_G3[kxm3] + sinTheta[kxp2] * cosPhi[kxp2] * inP_G3[kxp2]) +
                                c8_4 * (- sinTheta[kxm4] * cosPhi[kxm4] * inP_G3[kxm4] + sinTheta[kxp3] * cosPhi[kxp3] * inP_G3[kxp3]);

                        const Type stencilP_G3B =
                                c8_1 * (- sinTheta[kym1] * sinPhi[kym1] * inP_G3[kym1] + sinTheta[kyp0] * sinPhi[kyp0] * inP_G3[kyp0]) +
                                c8_2 * (- sinTheta[kym2] * sinPhi[kym2] * inP_G3[kym2] + sinTheta[kyp1] * sinPhi[kyp1] * inP_G3[kyp1]) +
                                c8_3 * (- sinTheta[kym3] * sinPhi[kym3] * inP_G3[kym3] + sinTheta[kyp2] * sinPhi[kyp2] * inP_G3[kyp2]) +
                                c8_4 * (- sinTheta[kym4] * sinPhi[kym4] * inP_G3[kym4] + sinTheta[kyp3] * sinPhi[kyp3] * inP_G3[kyp3]);

                        const Type stencilP_G3C =
                                c8_1 * (- cosTheta[kzm1] * inP_G3[kzm1] + cosTheta[kzp0] * inP_G3[kzp0]) +
                                c8_2 * (- cosTheta[kzm2] * inP_G3[kzm2] + cosTheta[kzp1] * inP_G3[kzp1]) +
                                c8_3 * (- cosTheta[kzm3] * inP_G3[kzm3] + cosTheta[kzp2] * inP_G3[kzp2]) +
                                c8_4 * (- cosTheta[kzm4] * inP_G3[kzm4] + cosTheta[kzp3] * inP_G3[kzp3]);


                        // ........................ G1 ........................
                        const Type stencilM_G1A =
                                c8_1 * (- cosTheta[kxm1] * cosPhi[kxm1] * inM_G1[kxm1] + cosTheta[kxp0] * cosPhi[kxp0] * inM_G1[kxp0]) +
                                c8_2 * (- cosTheta[kxm2] * cosPhi[kxm2] * inM_G1[kxm2] + cosTheta[kxp1] * cosPhi[kxp1] * inM_G1[kxp1]) +
                                c8_3 * (- cosTheta[kxm3] * cosPhi[kxm3] * inM_G1[kxm3] + cosTheta[kxp2] * cosPhi[kxp2] * inM_G1[kxp2]) +
                                c8_4 * (- cosTheta[kxm4] * cosPhi[kxm4] * inM_G1[kxm4] + cosTheta[kxp3] * cosPhi[kxp3] * inM_G1[kxp3]);

                        const Type stencilM_G1B =
                                c8_1 * (- cosTheta[kym1] * sinPhi[kym1] * inM_G1[kym1] + cosTheta[kyp0] * sinPhi[kyp0] * inM_G1[kyp0]) +
                                c8_2 * (- cosTheta[kym2] * sinPhi[kym2] * inM_G1[kym2] + cosTheta[kyp1] * sinPhi[kyp1] * inM_G1[kyp1]) +
                                c8_3 * (- cosTheta[kym3] * sinPhi[kym3] * inM_G1[kym3] + cosTheta[kyp2] * sinPhi[kyp2] * inM_G1[kyp2]) +
                                c8_4 * (- cosTheta[kym4] * sinPhi[kym4] * inM_G1[kym4] + cosTheta[kyp3] * sinPhi[kyp3] * inM_G1[kyp3]);

                        const Type stencilM_G1C =
                                c8_1 * (- sinTheta[kzm1] * inM_G1[kzm1] + sinTheta[kzp0] * inM_G1[kzp0]) +
                                c8_2 * (- sinTheta[kzm2] * inM_G1[kzm2] + sinTheta[kzp1] * inM_G1[kzp1]) +
                                c8_3 * (- sinTheta[kzm3] * inM_G1[kzm3] + sinTheta[kzp2] * inM_G1[kzp2]) +
                                c8_4 * (- sinTheta[kzm4] * inM_G1[kzm4] + sinTheta[kzp3] * inM_G1[kzp3]);


                        // ........................ G2 ........................
                        const Type stencilM_G2A =
                                c8_1 * (- sinPhi[kxm1] * inM_G2[kxm1] + sinPhi[kxp0] * inM_G2[kxp0]) +
                                c8_2 * (- sinPhi[kxm2] * inM_G2[kxm2] + sinPhi[kxp1] * inM_G2[kxp1]) +
                                c8_3 * (- sinPhi[kxm3] * inM_G2[kxm3] + sinPhi[kxp2] * inM_G2[kxp2]) +
                                c8_4 * (- sinPhi[kxm4] * inM_G2[kxm4] + sinPhi[kxp3] * inM_G2[kxp3]);

                        const Type stencilM_G2B =
                                c8_1 * (- cosPhi[kym1] * inM_G2[kym1] + cosPhi[kyp0] * inM_G2[kyp0]) +
                                c8_2 * (- cosPhi[kym2] * inM_G2[kym2] + cosPhi[kyp1] * inM_G2[kyp1]) +
                                c8_3 * (- cosPhi[kym3] * inM_G2[kym3] + cosPhi[kyp2] * inM_G2[kyp2]) +
                                c8_4 * (- cosPhi[kym4] * inM_G2[kym4] + cosPhi[kyp3] * inM_G2[kyp3]);


                        // ........................ G3 ........................
                        const Type stencilM_G3A =
                                c8_1 * (- sinTheta[kxm1] * cosPhi[kxm1] * inM_G3[kxm1] + sinTheta[kxp0] * cosPhi[kxp0] * inM_G3[kxp0]) +
                                c8_2 * (- sinTheta[kxm2] * cosPhi[kxm2] * inM_G3[kxm2] + sinTheta[kxp1] * cosPhi[kxp1] * inM_G3[kxp1]) +
                                c8_3 * (- sinTheta[kxm3] * cosPhi[kxm3] * inM_G3[kxm3] + sinTheta[kxp2] * cosPhi[kxp2] * inM_G3[kxp2]) +
                                c8_4 * (- sinTheta[kxm4] * cosPhi[kxm4] * inM_G3[kxm4] + sinTheta[kxp3] * cosPhi[kxp3] * inM_G3[kxp3]);

                        const Type stencilM_G3B =
                                c8_1 * (- sinTheta[kym1] * sinPhi[kym1] * inM_G3[kym1] + sinTheta[kyp0] * sinPhi[kyp0] * inM_G3[kyp0]) +
                                c8_2 * (- sinTheta[kym2] * sinPhi[kym2] * inM_G3[kym2] + sinTheta[kyp1] * sinPhi[kyp1] * inM_G3[kyp1]) +
                                c8_3 * (- sinTheta[kym3] * sinPhi[kym3] * inM_G3[kym3] + sinTheta[kyp2] * sinPhi[kyp2] * inM_G3[kyp2]) +
                                c8_4 * (- sinTheta[kym4] * sinPhi[kym4] * inM_G3[kym4] + sinTheta[kyp3] * sinPhi[kyp3] * inM_G3[kyp3]);

                        const Type stencilM_G3C =
                                c8_1 * (- cosTheta[kzm1] * inM_G3[kzm1] + cosTheta[kzp0] * inM_G3[kzp0]) +
                                c8_2 * (- cosTheta[kzm2] * inM_G3[kzm2] + cosTheta[kzp1] * inM_G3[kzp1]) +
                                c8_3 * (- cosTheta[kzm3] * inM_G3[kzm3] + cosTheta[kzp2] * inM_G3[kzp2]) +
                                c8_4 * (- cosTheta[kzm4] * inM_G3[kzm4] + cosTheta[kzp3] * inM_G3[kzp3]);

                        const long k = kxnynz_kynz + 2;

                        const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                        const Type dpg1 = invDx * stencilP_G1A + invDy * stencilP_G1B - invDz * stencilP_G1C;
                        const Type dpg2 = - invDx * stencilP_G2A + invDy * stencilP_G2B;
                        const Type dpg3 = invDx * stencilP_G3A + invDy * stencilP_G3B + invDz * stencilP_G3C;

                        const Type dmg1 = invDx * stencilM_G1A + invDy * stencilM_G1B - invDz * stencilM_G1C;
                        const Type dmg2 = - invDx * stencilM_G2A + invDy * stencilM_G2B;
                        const Type dmg3 = invDx * stencilM_G3A + invDy * stencilM_G3B + invDz * stencilM_G3C;

                        pSpace[k] = dpg1 + dpg2 + dpg3;
                        mSpace[k] = dmg1 + dmg2 + dmg3;

                        pOld[k] = dt2V2_B * pSpace[k] - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                        mOld[k] = dt2V2_B * mSpace[k] - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
                    }

                    // kz = 3 -- three cells below the free surface
                    {
                        const long kz = 3;
                        const long kynz_kz = + kynz + kz;

                        const long kxm4 = (kx-4) * nynz + kynz_kz;
                        const long kxm3 = (kx-3) * nynz + kynz_kz;
                        const long kxm2 = (kx-2) * nynz + kynz_kz;
                        const long kxm1 = (kx-1) * nynz + kynz_kz;
                        const long kxp0 = (kx+0) * nynz + kynz_kz;
                        const long kxp1 = (kx+1) * nynz + kynz_kz;
                        const long kxp2 = (kx+2) * nynz + kynz_kz;
                        const long kxp3 = (kx+3) * nynz + kynz_kz;

                        const long kym4 = kxnynz + (ky-4) * nz + kz;
                        const long kym3 = kxnynz + (ky-3) * nz + kz;
                        const long kym2 = kxnynz + (ky-2) * nz + kz;
                        const long kym1 = kxnynz + (ky-1) * nz + kz;
                        const long kyp0 = kxnynz + (ky+0) * nz + kz;
                        const long kyp1 = kxnynz + (ky+1) * nz + kz;
                        const long kyp2 = kxnynz + (ky+2) * nz + kz;
                        const long kyp3 = kxnynz + (ky+3) * nz + kz;

                        const long kzm4 = kxnynz_kynz + 0;
                        const long kzm3 = kxnynz_kynz + 0;
                        const long kzm2 = kxnynz_kynz + 1;
                        const long kzm1 = kxnynz_kynz + 2;
                        const long kzp0 = kxnynz_kynz + 3;
                        const long kzp1 = kxnynz_kynz + 4;
                        const long kzp2 = kxnynz_kynz + 5;
                        const long kzp3 = kxnynz_kynz + 6;


                        // ........................ G1 ........................
                        const Type stencilP_G1A =
                                c8_1 * (- cosTheta[kxm1] * cosPhi[kxm1] * inP_G1[kxm1] + cosTheta[kxp0] * cosPhi[kxp0] * inP_G1[kxp0]) +
                                c8_2 * (- cosTheta[kxm2] * cosPhi[kxm2] * inP_G1[kxm2] + cosTheta[kxp1] * cosPhi[kxp1] * inP_G1[kxp1]) +
                                c8_3 * (- cosTheta[kxm3] * cosPhi[kxm3] * inP_G1[kxm3] + cosTheta[kxp2] * cosPhi[kxp2] * inP_G1[kxp2]) +
                                c8_4 * (- cosTheta[kxm4] * cosPhi[kxm4] * inP_G1[kxm4] + cosTheta[kxp3] * cosPhi[kxp3] * inP_G1[kxp3]);

                        const Type stencilP_G1B =
                                c8_1 * (- cosTheta[kym1] * sinPhi[kym1] * inP_G1[kym1] + cosTheta[kyp0] * sinPhi[kyp0] * inP_G1[kyp0]) +
                                c8_2 * (- cosTheta[kym2] * sinPhi[kym2] * inP_G1[kym2] + cosTheta[kyp1] * sinPhi[kyp1] * inP_G1[kyp1]) +
                                c8_3 * (- cosTheta[kym3] * sinPhi[kym3] * inP_G1[kym3] + cosTheta[kyp2] * sinPhi[kyp2] * inP_G1[kyp2]) +
                                c8_4 * (- cosTheta[kym4] * sinPhi[kym4] * inP_G1[kym4] + cosTheta[kyp3] * sinPhi[kyp3] * inP_G1[kyp3]);

                        const Type stencilP_G1C =
                                c8_1 * (- sinTheta[kzm1] * inP_G1[kzm1] + sinTheta[kzp0] * inP_G1[kzp0]) +
                                c8_2 * (- sinTheta[kzm2] * inP_G1[kzm2] + sinTheta[kzp1] * inP_G1[kzp1]) +
                                c8_3 * (- sinTheta[kzm3] * inP_G1[kzm3] + sinTheta[kzp2] * inP_G1[kzp2]) +
                                c8_4 * (- sinTheta[kzm4] * inP_G1[kzm4] + sinTheta[kzp3] * inP_G1[kzp3]);


                        // ........................ G2 ........................
                        const Type stencilP_G2A =
                                c8_1 * (- sinPhi[kxm1] * inP_G2[kxm1] + sinPhi[kxp0] * inP_G2[kxp0]) +
                                c8_2 * (- sinPhi[kxm2] * inP_G2[kxm2] + sinPhi[kxp1] * inP_G2[kxp1]) +
                                c8_3 * (- sinPhi[kxm3] * inP_G2[kxm3] + sinPhi[kxp2] * inP_G2[kxp2]) +
                                c8_4 * (- sinPhi[kxm4] * inP_G2[kxm4] + sinPhi[kxp3] * inP_G2[kxp3]);

                        const Type stencilP_G2B =
                                c8_1 * (- cosPhi[kym1] * inP_G2[kym1] + cosPhi[kyp0] * inP_G2[kyp0]) +
                                c8_2 * (- cosPhi[kym2] * inP_G2[kym2] + cosPhi[kyp1] * inP_G2[kyp1]) +
                                c8_3 * (- cosPhi[kym3] * inP_G2[kym3] + cosPhi[kyp2] * inP_G2[kyp2]) +
                                c8_4 * (- cosPhi[kym4] * inP_G2[kym4] + cosPhi[kyp3] * inP_G2[kyp3]);


                        // ........................ G3 ........................
                        const Type stencilP_G3A =
                                c8_1 * (- sinTheta[kxm1] * cosPhi[kxm1] * inP_G3[kxm1] + sinTheta[kxp0] * cosPhi[kxp0] * inP_G3[kxp0]) +
                                c8_2 * (- sinTheta[kxm2] * cosPhi[kxm2] * inP_G3[kxm2] + sinTheta[kxp1] * cosPhi[kxp1] * inP_G3[kxp1]) +
                                c8_3 * (- sinTheta[kxm3] * cosPhi[kxm3] * inP_G3[kxm3] + sinTheta[kxp2] * cosPhi[kxp2] * inP_G3[kxp2]) +
                                c8_4 * (- sinTheta[kxm4] * cosPhi[kxm4] * inP_G3[kxm4] + sinTheta[kxp3] * cosPhi[kxp3] * inP_G3[kxp3]);

                        const Type stencilP_G3B =
                                c8_1 * (- sinTheta[kym1] * sinPhi[kym1] * inP_G3[kym1] + sinTheta[kyp0] * sinPhi[kyp0] * inP_G3[kyp0]) +
                                c8_2 * (- sinTheta[kym2] * sinPhi[kym2] * inP_G3[kym2] + sinTheta[kyp1] * sinPhi[kyp1] * inP_G3[kyp1]) +
                                c8_3 * (- sinTheta[kym3] * sinPhi[kym3] * inP_G3[kym3] + sinTheta[kyp2] * sinPhi[kyp2] * inP_G3[kyp2]) +
                                c8_4 * (- sinTheta[kym4] * sinPhi[kym4] * inP_G3[kym4] + sinTheta[kyp3] * sinPhi[kyp3] * inP_G3[kyp3]);

                        const Type stencilP_G3C =
                                c8_1 * (- cosTheta[kzm1] * inP_G3[kzm1] + cosTheta[kzp0] * inP_G3[kzp0]) +
                                c8_2 * (- cosTheta[kzm2] * inP_G3[kzm2] + cosTheta[kzp1] * inP_G3[kzp1]) +
                                c8_3 * (- cosTheta[kzm3] * inP_G3[kzm3] + cosTheta[kzp2] * inP_G3[kzp2]) +
                                c8_4 * (- cosTheta[kzm4] * inP_G3[kzm4] + cosTheta[kzp3] * inP_G3[kzp3]);


                        // ........................ G1 ........................
                        const Type stencilM_G1A =
                                c8_1 * (- cosTheta[kxm1] * cosPhi[kxm1] * inM_G1[kxm1] + cosTheta[kxp0] * cosPhi[kxp0] * inM_G1[kxp0]) +
                                c8_2 * (- cosTheta[kxm2] * cosPhi[kxm2] * inM_G1[kxm2] + cosTheta[kxp1] * cosPhi[kxp1] * inM_G1[kxp1]) +
                                c8_3 * (- cosTheta[kxm3] * cosPhi[kxm3] * inM_G1[kxm3] + cosTheta[kxp2] * cosPhi[kxp2] * inM_G1[kxp2]) +
                                c8_4 * (- cosTheta[kxm4] * cosPhi[kxm4] * inM_G1[kxm4] + cosTheta[kxp3] * cosPhi[kxp3] * inM_G1[kxp3]);

                        const Type stencilM_G1B =
                                c8_1 * (- cosTheta[kym1] * sinPhi[kym1] * inM_G1[kym1] + cosTheta[kyp0] * sinPhi[kyp0] * inM_G1[kyp0]) +
                                c8_2 * (- cosTheta[kym2] * sinPhi[kym2] * inM_G1[kym2] + cosTheta[kyp1] * sinPhi[kyp1] * inM_G1[kyp1]) +
                                c8_3 * (- cosTheta[kym3] * sinPhi[kym3] * inM_G1[kym3] + cosTheta[kyp2] * sinPhi[kyp2] * inM_G1[kyp2]) +
                                c8_4 * (- cosTheta[kym4] * sinPhi[kym4] * inM_G1[kym4] + cosTheta[kyp3] * sinPhi[kyp3] * inM_G1[kyp3]);

                        const Type stencilM_G1C =
                                c8_1 * (- sinTheta[kzm1] * inM_G1[kzm1] + sinTheta[kzp0] * inM_G1[kzp0]) +
                                c8_2 * (- sinTheta[kzm2] * inM_G1[kzm2] + sinTheta[kzp1] * inM_G1[kzp1]) +
                                c8_3 * (- sinTheta[kzm3] * inM_G1[kzm3] + sinTheta[kzp2] * inM_G1[kzp2]) +
                                c8_4 * (- sinTheta[kzm4] * inM_G1[kzm4] + sinTheta[kzp3] * inM_G1[kzp3]);


                        // ........................ G2 ........................
                        const Type stencilM_G2A =
                                c8_1 * (- sinPhi[kxm1] * inM_G2[kxm1] + sinPhi[kxp0] * inM_G2[kxp0]) +
                                c8_2 * (- sinPhi[kxm2] * inM_G2[kxm2] + sinPhi[kxp1] * inM_G2[kxp1]) +
                                c8_3 * (- sinPhi[kxm3] * inM_G2[kxm3] + sinPhi[kxp2] * inM_G2[kxp2]) +
                                c8_4 * (- sinPhi[kxm4] * inM_G2[kxm4] + sinPhi[kxp3] * inM_G2[kxp3]);

                        const Type stencilM_G2B =
                                c8_1 * (- cosPhi[kym1] * inM_G2[kym1] + cosPhi[kyp0] * inM_G2[kyp0]) +
                                c8_2 * (- cosPhi[kym2] * inM_G2[kym2] + cosPhi[kyp1] * inM_G2[kyp1]) +
                                c8_3 * (- cosPhi[kym3] * inM_G2[kym3] + cosPhi[kyp2] * inM_G2[kyp2]) +
                                c8_4 * (- cosPhi[kym4] * inM_G2[kym4] + cosPhi[kyp3] * inM_G2[kyp3]);


                        // ........................ G3 ........................
                        const Type stencilM_G3A =
                                c8_1 * (- sinTheta[kxm1] * cosPhi[kxm1] * inM_G3[kxm1] + sinTheta[kxp0] * cosPhi[kxp0] * inM_G3[kxp0]) +
                                c8_2 * (- sinTheta[kxm2] * cosPhi[kxm2] * inM_G3[kxm2] + sinTheta[kxp1] * cosPhi[kxp1] * inM_G3[kxp1]) +
                                c8_3 * (- sinTheta[kxm3] * cosPhi[kxm3] * inM_G3[kxm3] + sinTheta[kxp2] * cosPhi[kxp2] * inM_G3[kxp2]) +
                                c8_4 * (- sinTheta[kxm4] * cosPhi[kxm4] * inM_G3[kxm4] + sinTheta[kxp3] * cosPhi[kxp3] * inM_G3[kxp3]);

                        const Type stencilM_G3B =
                                c8_1 * (- sinTheta[kym1] * sinPhi[kym1] * inM_G3[kym1] + sinTheta[kyp0] * sinPhi[kyp0] * inM_G3[kyp0]) +
                                c8_2 * (- sinTheta[kym2] * sinPhi[kym2] * inM_G3[kym2] + sinTheta[kyp1] * sinPhi[kyp1] * inM_G3[kyp1]) +
                                c8_3 * (- sinTheta[kym3] * sinPhi[kym3] * inM_G3[kym3] + sinTheta[kyp2] * sinPhi[kyp2] * inM_G3[kyp2]) +
                                c8_4 * (- sinTheta[kym4] * sinPhi[kym4] * inM_G3[kym4] + sinTheta[kyp3] * sinPhi[kyp3] * inM_G3[kyp3]);

                        const Type stencilM_G3C =
                                c8_1 * (- cosTheta[kzm1] * inM_G3[kzm1] + cosTheta[kzp0] * inM_G3[kzp0]) +
                                c8_2 * (- cosTheta[kzm2] * inM_G3[kzm2] + cosTheta[kzp1] * inM_G3[kzp1]) +
                                c8_3 * (- cosTheta[kzm3] * inM_G3[kzm3] + cosTheta[kzp2] * inM_G3[kzp2]) +
                                c8_4 * (- cosTheta[kzm4] * inM_G3[kzm4] + cosTheta[kzp3] * inM_G3[kzp3]);

                        const long k = kxnynz_kynz + 3;

                        const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                        const Type dpg1 = invDx * stencilP_G1A + invDy * stencilP_G1B - invDz * stencilP_G1C;
                        const Type dpg2 = - invDx * stencilP_G2A + invDy * stencilP_G2B;
                        const Type dpg3 = invDx * stencilP_G3A + invDy * stencilP_G3B + invDz * stencilP_G3C;

                        const Type dmg1 = invDx * stencilM_G1A + invDy * stencilM_G1B - invDz * stencilM_G1C;
                        const Type dmg2 = - invDx * stencilM_G2A + invDy * stencilM_G2B;
                        const Type dmg3 = invDx * stencilM_G3A + invDy * stencilM_G3B + invDz * stencilM_G3C;

                        pSpace[k] = dpg1 + dpg2 + dpg3;
                        mSpace[k] = dmg1 + dmg2 + dmg3;

                        pOld[k] = dt2V2_B * pSpace[k] - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                        mOld[k] = dt2V2_B * mSpace[k] - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
                    }
                }
            }

        }
    }

    template<class Type>
    inline static void applyFirstDerivatives3D_TTI_PlusHalf(
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
            Type * __restrict__ inG1,
            Type * __restrict__ inG2,
            Type * __restrict__ inG3,
            float * __restrict__ sinTheta,
            float * __restrict__ cosTheta,
            float * __restrict__ sinPhi,
            float * __restrict__ cosPhi,
            Type * __restrict__ outG1,
            Type * __restrict__ outG2,
            Type * __restrict__ outG3,
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
                    outG1[kindex1] = outG1[kindex2] = 0;
                    outG2[kindex1] = outG2[kindex2] = 0;
                    outG3[kindex1] = outG3[kindex2] = 0;
                }
            }

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long kx = 0; kx < nx; kx++) {
#pragma omp simd
                for (long kz = 0; kz < nz; kz++) {
                    long kindex1 = kx * ny * nz + k * nz + kz;
                    long kindex2 = kx * ny * nz + (ny - 1 - k) * nz + kz;
                    outG1[kindex1] = outG1[kindex2] = 0;
                    outG2[kindex1] = outG2[kindex2] = 0;
                    outG3[kindex1] = outG3[kindex2] = 0;
                }
            }

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long ky = 0; ky < ny; ky++) {
#pragma omp simd
                for (long kz = 0; kz < nz; kz++) {
                    long kindex1 = k * ny * nz + ky * nz + kz;
                    long kindex2 = (nx - 1 - k) * ny * nz + ky * nz + kz;
                    outG1[kindex1] = outG1[kindex2] = 0;
                    outG2[kindex1] = outG2[kindex2] = 0;
                    outG3[kindex1] = outG3[kindex2] = 0;
                }
            }

        }

        // interior
#pragma omp parallel for collapse(3) num_threads(nthread) schedule(static)
        for (long bx = 4; bx < nx4; bx += BX_3D) {
            for (long by = 4; by < ny4; by += BY_3D) {
                for (long bz = 4; bz < nz4; bz += BZ_3D) {
                    const long kxmax = std::min(bx + BX_3D, nx4);
                    const long kymax = std::min(by + BY_3D, ny4);
                    const long kzmax = std::min(bz + BZ_3D, nz4);

                    for (long kx = bx; kx < kxmax; kx++) {
                        const long kxnynz = kx * nynz;

                        for (long ky = by; ky < kymax; ky++) {
                            const long kynz = ky * nz;
                            const long kxnynz_kynz = kxnynz + kynz;

#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long kynz_kz = + kynz + kz;

                                const Type stencilG1 =
                                        c8_1 * (- inG1[(kx+0) * nynz + kynz_kz] + inG1[(kx+1) * nynz + kynz_kz]) +
                                        c8_2 * (- inG1[(kx-1) * nynz + kynz_kz] + inG1[(kx+2) * nynz + kynz_kz]) +
                                        c8_3 * (- inG1[(kx-2) * nynz + kynz_kz] + inG1[(kx+3) * nynz + kynz_kz]) +
                                        c8_4 * (- inG1[(kx-3) * nynz + kynz_kz] + inG1[(kx+4) * nynz + kynz_kz]);

                                const Type stencilG2 =
                                        c8_1 * (- inG2[kxnynz + (ky+0) * nz + kz] + inG2[kxnynz + (ky+1) * nz + kz]) +
                                        c8_2 * (- inG2[kxnynz + (ky-1) * nz + kz] + inG2[kxnynz + (ky+2) * nz + kz]) +
                                        c8_3 * (- inG2[kxnynz + (ky-2) * nz + kz] + inG2[kxnynz + (ky+3) * nz + kz]) +
                                        c8_4 * (- inG2[kxnynz + (ky-3) * nz + kz] + inG2[kxnynz + (ky+4) * nz + kz]);

                                const Type stencilG3 =
                                        c8_1 * (- inG3[kxnynz_kynz + (kz+0)] + inG3[kxnynz_kynz + (kz+1)]) +
                                        c8_2 * (- inG3[kxnynz_kynz + (kz-1)] + inG3[kxnynz_kynz + (kz+2)]) +
                                        c8_3 * (- inG3[kxnynz_kynz + (kz-2)] + inG3[kxnynz_kynz + (kz+3)]) +
                                        c8_4 * (- inG3[kxnynz_kynz + (kz-3)] + inG3[kxnynz_kynz + (kz+4)]);

                                long k = kxnynz_kynz + kz;

                                const Type dx = invDx * stencilG1;
                                const Type dy = invDy * stencilG2;
                                const Type dz = invDz * stencilG3;

                                const float cosThetaCosPhi = cosTheta[k] * cosPhi[k];
                                const float cosThetaSinPhi = cosTheta[k] * sinPhi[k];
                                const float sinThetaCosPhi = sinTheta[k] * cosPhi[k];

                                outG1[k] = cosThetaCosPhi * dx + cosThetaSinPhi * dy - sinTheta[k] * dz;
                                outG2[k] = - sinPhi[k] * dx + cosPhi[k] * dy;
                                outG3[k] = sinThetaCosPhi * dx + sinTheta[k] * sinPhi[k] * dy + cosTheta[k] * dz;
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
                    {
                        const Type stencilG30 =
                                c8_1 * (- inG3[kxnynz_kynz + 0] + inG3[kxnynz_kynz + 1]) +
                                c8_2 * (+ inG3[kxnynz_kynz + 1] + inG3[kxnynz_kynz + 2]) +
                                c8_3 * (+ inG3[kxnynz_kynz + 2] + inG3[kxnynz_kynz + 3]) +
                                c8_4 * (+ inG3[kxnynz_kynz + 3] + inG3[kxnynz_kynz + 4]);

                        const long k0 = kxnynz_kynz + 0;

                        const Type dz0 = invDz * stencilG30;

                        outG1[k0] = -sinTheta[k0] * dz0;
                        outG2[k0] = 0;
                        outG3[k0] = cosTheta[k0] * dz0;
                    }

                    // kz = 1 -- 1 1/2 cells below free surface for Z derivative, 1 cells below for X/Y derivative
                    {
                        const Type stencilG11 =
                                c8_1 * (- inG1[(kx+0) * nynz + kynz + 1] + inG1[(kx+1) * nynz + kynz + 1]) +
                                c8_2 * (- inG1[(kx-1) * nynz + kynz + 1] + inG1[(kx+2) * nynz + kynz + 1]) +
                                c8_3 * (- inG1[(kx-2) * nynz + kynz + 1] + inG1[(kx+3) * nynz + kynz + 1]) +
                                c8_4 * (- inG1[(kx-3) * nynz + kynz + 1] + inG1[(kx+4) * nynz + kynz + 1]);

                        const Type stencilG21 =
                                c8_1 * (- inG2[kxnynz + (ky+0) * nz + 1] + inG2[kxnynz + (ky+1) * nz + 1]) +
                                c8_2 * (- inG2[kxnynz + (ky-1) * nz + 1] + inG2[kxnynz + (ky+2) * nz + 1]) +
                                c8_3 * (- inG2[kxnynz + (ky-2) * nz + 1] + inG2[kxnynz + (ky+3) * nz + 1]) +
                                c8_4 * (- inG2[kxnynz + (ky-3) * nz + 1] + inG2[kxnynz + (ky+4) * nz + 1]);

                        const Type stencilG31 =
                                c8_1 * (- inG3[kxnynz_kynz + 1] + inG3[kxnynz_kynz + 2]) +
                                c8_2 * (- inG3[kxnynz_kynz + 0] + inG3[kxnynz_kynz + 3]) +
                                c8_3 * (+ inG3[kxnynz_kynz + 1] + inG3[kxnynz_kynz + 4]) +
                                c8_4 * (+ inG3[kxnynz_kynz + 2] + inG3[kxnynz_kynz + 5]);

                        const long k1 = kxnynz_kynz + 1;

                        const Type dx1 = invDx * stencilG11;
                        const Type dy1 = invDy * stencilG21;
                        const Type dz1 = invDz * stencilG31;

                        outG1[k1] = cosTheta[k1] * cosPhi[k1] * dx1 + cosTheta[k1] * sinPhi[k1] * dy1 - sinTheta[k1] * dz1;
                        outG2[k1] = - sinPhi[k1] * dx1 + cosPhi[k1] * dy1;
                        outG3[k1] = sinTheta[k1] * cosPhi[k1] * dx1 + sinTheta[k1] * sinPhi[k1] * dy1 + cosTheta[k1] * dz1;
                    }

                    // kz = 2 -- 2 1/2 cells below free surface for Z derivative, 2 cells below for X/Y derivative
                    {
                        const Type stencilG12 =
                                c8_1 * (- inG1[(kx+0) * nynz + kynz + 2] + inG1[(kx+1) * nynz + kynz + 2]) +
                                c8_2 * (- inG1[(kx-1) * nynz + kynz + 2] + inG1[(kx+2) * nynz + kynz + 2]) +
                                c8_3 * (- inG1[(kx-2) * nynz + kynz + 2] + inG1[(kx+3) * nynz + kynz + 2]) +
                                c8_4 * (- inG1[(kx-3) * nynz + kynz + 2] + inG1[(kx+4) * nynz + kynz + 2]);

                        const Type stencilG22 =
                                c8_1 * (- inG2[kxnynz + (ky+0) * nz + 2] + inG2[kxnynz + (ky+1) * nz + 2]) +
                                c8_2 * (- inG2[kxnynz + (ky-1) * nz + 2] + inG2[kxnynz + (ky+2) * nz + 2]) +
                                c8_3 * (- inG2[kxnynz + (ky-2) * nz + 2] + inG2[kxnynz + (ky+3) * nz + 2]) +
                                c8_4 * (- inG2[kxnynz + (ky-3) * nz + 2] + inG2[kxnynz + (ky+4) * nz + 2]);

                        const Type stencilG32 =
                                c8_1 * (- inG3[kxnynz_kynz + 2] + inG3[kxnynz_kynz + 3]) +
                                c8_2 * (- inG3[kxnynz_kynz + 1] + inG3[kxnynz_kynz + 4]) +
                                c8_3 * (- inG3[kxnynz_kynz + 0] + inG3[kxnynz_kynz + 5]) +
                                c8_4 * (+ inG3[kxnynz_kynz + 1] + inG3[kxnynz_kynz + 6]);

                        const long k2 = kxnynz_kynz + 2;

                        const Type dx2 = invDx * stencilG12;
                        const Type dy2 = invDy * stencilG22;
                        const Type dz2 = invDz * stencilG32;

                        outG1[k2] = cosTheta[k2] * cosPhi[k2] * dx2 + cosTheta[k2] * sinPhi[k2] * dy2 - sinTheta[k2] * dz2;
                        outG2[k2] = - sinPhi[k2] * dx2 + cosPhi[k2] * dy2;
                        outG3[k2] = sinTheta[k2] * cosPhi[k2] * dx2 + sinTheta[k2] * sinPhi[k2] * dy2 + cosTheta[k2] * dz2;
                    }

                    // kz = 3 -- 3 1/2 cells below free surface for Z derivative, 3 cells below for X/Y derivative
                    {
                        const Type stencilG13 =
                                c8_1 * (- inG1[(kx+0) * nynz + kynz + 3] + inG1[(kx+1) * nynz + kynz + 3]) +
                                c8_2 * (- inG1[(kx-1) * nynz + kynz + 3] + inG1[(kx+2) * nynz + kynz + 3]) +
                                c8_3 * (- inG1[(kx-2) * nynz + kynz + 3] + inG1[(kx+3) * nynz + kynz + 3]) +
                                c8_4 * (- inG1[(kx-3) * nynz + kynz + 3] + inG1[(kx+4) * nynz + kynz + 3]);

                        const Type stencilG23 =
                                c8_1 * (- inG2[kxnynz + (ky+0) * nz + 3] + inG2[kxnynz + (ky+1) * nz + 3]) +
                                c8_2 * (- inG2[kxnynz + (ky-1) * nz + 3] + inG2[kxnynz + (ky+2) * nz + 3]) +
                                c8_3 * (- inG2[kxnynz + (ky-2) * nz + 3] + inG2[kxnynz + (ky+3) * nz + 3]) +
                                c8_4 * (- inG2[kxnynz + (ky-3) * nz + 3] + inG2[kxnynz + (ky+4) * nz + 3]);

                        const Type stencilG33 =
                                c8_1 * (- inG3[kxnynz_kynz + 3] + inG3[kxnynz_kynz + 4]) +
                                c8_2 * (- inG3[kxnynz_kynz + 2] + inG3[kxnynz_kynz + 5]) +
                                c8_3 * (- inG3[kxnynz_kynz + 1] + inG3[kxnynz_kynz + 6]) +
                                c8_4 * (- inG3[kxnynz_kynz + 0] + inG3[kxnynz_kynz + 7]);

                        const long k3 = kxnynz_kynz + 3;

                        const Type dx3 = invDx * stencilG13;
                        const Type dy3 = invDy * stencilG23;
                        const Type dz3 = invDz * stencilG33;

                        outG1[k3] = cosTheta[k3] * cosPhi[k3] * dx3 + cosTheta[k3] * sinPhi[k3] * dy3 - sinTheta[k3] * dz3;
                        outG2[k3] = - sinPhi[k3] * dx3 + cosPhi[k3] * dy3;
                        outG3[k3] = sinTheta[k3] * cosPhi[k3] * dx3 + sinTheta[k3] * sinPhi[k3] * dy3 + cosTheta[k3] * dz3;
                    }
                }
            }
        }
    }

    template<class Type>
    inline static void applyFirstDerivatives3D_TTI_MinusHalf(
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
            Type * __restrict__ inG1,
            Type * __restrict__ inG2,
            Type * __restrict__ inG3,
            float * __restrict__ sinTheta,
            float * __restrict__ cosTheta,
            float * __restrict__ sinPhi,
            float * __restrict__ cosPhi,
            Type * __restrict__ outG1,
            Type * __restrict__ outG2,
            Type * __restrict__ outG3,
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
                    outG1[kindex1] = outG1[kindex2] = 0;
                    outG2[kindex1] = outG2[kindex2] = 0;
                    outG3[kindex1] = outG3[kindex2] = 0;
                }
            }

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long kx = 0; kx < nx; kx++) {
#pragma omp simd
                for (long kz = 0; kz < nz; kz++) {
                    const long kindex1 = kx * ny * nz + k * nz + kz;
                    const long kindex2 = kx * ny * nz + (ny - 1 - k) * nz + kz;
                    outG1[kindex1] = outG1[kindex2] = 0;
                    outG2[kindex1] = outG2[kindex2] = 0;
                    outG3[kindex1] = outG3[kindex2] = 0;
                }
            }

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long ky = 0; ky < ny; ky++) {
#pragma omp simd
                for (long kz = 0; kz < nz; kz++) {
                    const long kindex1 = k * ny * nz + ky * nz + kz;
                    const long kindex2 = (nx - 1 - k) * ny * nz + ky * nz + kz;
                    outG1[kindex1] = outG1[kindex2] = 0;
                    outG2[kindex1] = outG2[kindex2] = 0;
                    outG3[kindex1] = outG3[kindex2] = 0;
                }
            }

        }

        // interior
#pragma omp parallel for collapse(3) num_threads(nthread) schedule(static)
        for (long bx = 4; bx < nx4; bx += BX_3D) {
            for (long by = 4; by < ny4; by += BY_3D) {
                for (long bz = 4; bz < nz4; bz += BZ_3D) {
                    const long kxmax = std::min(bx + BX_3D, nx4);
                    const long kymax = std::min(by + BY_3D, ny4);
                    const long kzmax = std::min(bz + BZ_3D, nz4);

                    for (long kx = bx; kx < kxmax; kx++) {
                        const long kxnynz = kx * nynz;

                        for (long ky = by; ky < kymax; ky++) {
                            const long kynz = ky * nz;
                            const long kxnynz_kynz = kxnynz + kynz;

#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long kynz_kz = + kynz + kz;

                                const long kxm4 = (kx-4) * nynz + kynz_kz;
                                const long kxm3 = (kx-3) * nynz + kynz_kz;
                                const long kxm2 = (kx-2) * nynz + kynz_kz;
                                const long kxm1 = (kx-1) * nynz + kynz_kz;
                                const long kxp0 = (kx+0) * nynz + kynz_kz;
                                const long kxp1 = (kx+1) * nynz + kynz_kz;
                                const long kxp2 = (kx+2) * nynz + kynz_kz;
                                const long kxp3 = (kx+3) * nynz + kynz_kz;

                                const long kym4 = kxnynz + (ky-4) * nz + kz;
                                const long kym3 = kxnynz + (ky-3) * nz + kz;
                                const long kym2 = kxnynz + (ky-2) * nz + kz;
                                const long kym1 = kxnynz + (ky-1) * nz + kz;
                                const long kyp0 = kxnynz + (ky+0) * nz + kz;
                                const long kyp1 = kxnynz + (ky+1) * nz + kz;
                                const long kyp2 = kxnynz + (ky+2) * nz + kz;
                                const long kyp3 = kxnynz + (ky+3) * nz + kz;

                                const long kzm4 = kxnynz_kynz + (kz-4);
                                const long kzm3 = kxnynz_kynz + (kz-3);
                                const long kzm2 = kxnynz_kynz + (kz-2);
                                const long kzm1 = kxnynz_kynz + (kz-1);
                                const long kzp0 = kxnynz_kynz + (kz+0);
                                const long kzp1 = kxnynz_kynz + (kz+1);
                                const long kzp2 = kxnynz_kynz + (kz+2);
                                const long kzp3 = kxnynz_kynz + (kz+3);


                                // ........................ G1 ........................
                                const Type stencilG1A =
                                        c8_1 * (- cosTheta[kxm1] * cosPhi[kxm1] * inG1[kxm1] + cosTheta[kxp0] * cosPhi[kxp0] * inG1[kxp0]) +
                                        c8_2 * (- cosTheta[kxm2] * cosPhi[kxm2] * inG1[kxm2] + cosTheta[kxp1] * cosPhi[kxp1] * inG1[kxp1]) +
                                        c8_3 * (- cosTheta[kxm3] * cosPhi[kxm3] * inG1[kxm3] + cosTheta[kxp2] * cosPhi[kxp2] * inG1[kxp2]) +
                                        c8_4 * (- cosTheta[kxm4] * cosPhi[kxm4] * inG1[kxm4] + cosTheta[kxp3] * cosPhi[kxp3] * inG1[kxp3]);

                                const Type stencilG1B =
                                        c8_1 * (- cosTheta[kym1] * sinPhi[kym1] * inG1[kym1] + cosTheta[kyp0] * sinPhi[kyp0] * inG1[kyp0]) +
                                        c8_2 * (- cosTheta[kym2] * sinPhi[kym2] * inG1[kym2] + cosTheta[kyp1] * sinPhi[kyp1] * inG1[kyp1]) +
                                        c8_3 * (- cosTheta[kym3] * sinPhi[kym3] * inG1[kym3] + cosTheta[kyp2] * sinPhi[kyp2] * inG1[kyp2]) +
                                        c8_4 * (- cosTheta[kym4] * sinPhi[kym4] * inG1[kym4] + cosTheta[kyp3] * sinPhi[kyp3] * inG1[kyp3]);

                                const Type stencilG1C =
                                        c8_1 * (- sinTheta[kzm1] * inG1[kzm1] + sinTheta[kzp0] * inG1[kzp0]) +
                                        c8_2 * (- sinTheta[kzm2] * inG1[kzm2] + sinTheta[kzp1] * inG1[kzp1]) +
                                        c8_3 * (- sinTheta[kzm3] * inG1[kzm3] + sinTheta[kzp2] * inG1[kzp2]) +
                                        c8_4 * (- sinTheta[kzm4] * inG1[kzm4] + sinTheta[kzp3] * inG1[kzp3]);


                                // ........................ G2 ........................
                                const Type stencilG2A =
                                        c8_1 * (- sinPhi[kxm1] * inG2[kxm1] + sinPhi[kxp0] * inG2[kxp0]) +
                                        c8_2 * (- sinPhi[kxm2] * inG2[kxm2] + sinPhi[kxp1] * inG2[kxp1]) +
                                        c8_3 * (- sinPhi[kxm3] * inG2[kxm3] + sinPhi[kxp2] * inG2[kxp2]) +
                                        c8_4 * (- sinPhi[kxm4] * inG2[kxm4] + sinPhi[kxp3] * inG2[kxp3]);

                                const Type stencilG2B =
                                        c8_1 * (- cosPhi[kym1] * inG2[kym1] + cosPhi[kyp0] * inG2[kyp0]) +
                                        c8_2 * (- cosPhi[kym2] * inG2[kym2] + cosPhi[kyp1] * inG2[kyp1]) +
                                        c8_3 * (- cosPhi[kym3] * inG2[kym3] + cosPhi[kyp2] * inG2[kyp2]) +
                                        c8_4 * (- cosPhi[kym4] * inG2[kym4] + cosPhi[kyp3] * inG2[kyp3]);


                                // ........................ G3 ........................
                                const Type stencilG3A =
                                        c8_1 * (- sinTheta[kxm1] * cosPhi[kxm1] * inG3[kxm1] + sinTheta[kxp0] * cosPhi[kxp0] * inG3[kxp0]) +
                                        c8_2 * (- sinTheta[kxm2] * cosPhi[kxm2] * inG3[kxm2] + sinTheta[kxp1] * cosPhi[kxp1] * inG3[kxp1]) +
                                        c8_3 * (- sinTheta[kxm3] * cosPhi[kxm3] * inG3[kxm3] + sinTheta[kxp2] * cosPhi[kxp2] * inG3[kxp2]) +
                                        c8_4 * (- sinTheta[kxm4] * cosPhi[kxm4] * inG3[kxm4] + sinTheta[kxp3] * cosPhi[kxp3] * inG3[kxp3]);

                                const Type stencilG3B =
                                        c8_1 * (- sinTheta[kym1] * sinPhi[kym1] * inG3[kym1] + sinTheta[kyp0] * sinPhi[kyp0] * inG3[kyp0]) +
                                        c8_2 * (- sinTheta[kym2] * sinPhi[kym2] * inG3[kym2] + sinTheta[kyp1] * sinPhi[kyp1] * inG3[kyp1]) +
                                        c8_3 * (- sinTheta[kym3] * sinPhi[kym3] * inG3[kym3] + sinTheta[kyp2] * sinPhi[kyp2] * inG3[kyp2]) +
                                        c8_4 * (- sinTheta[kym4] * sinPhi[kym4] * inG3[kym4] + sinTheta[kyp3] * sinPhi[kyp3] * inG3[kyp3]);

                                const Type stencilG3C =
                                        c8_1 * (- cosTheta[kzm1] * inG3[kzm1] + cosTheta[kzp0] * inG3[kzp0]) +
                                        c8_2 * (- cosTheta[kzm2] * inG3[kzm2] + cosTheta[kzp1] * inG3[kzp1]) +
                                        c8_3 * (- cosTheta[kzm3] * inG3[kzm3] + cosTheta[kzp2] * inG3[kzp2]) +
                                        c8_4 * (- cosTheta[kzm4] * inG3[kzm4] + cosTheta[kzp3] * inG3[kzp3]);

                                const long k = kxnynz_kynz + kz;

                                outG1[k] = invDx * stencilG1A + invDy * stencilG1B - invDz * stencilG1C;
                                outG2[k] = - invDx * stencilG2A + invDy * stencilG2B;
                                outG3[k] = invDx * stencilG3A + invDy * stencilG3B + invDz * stencilG3C;
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

                    {
                        // kz = 0 -- at the free surface -- p = 0, dp = 0
                        const long k = kxnynz_kynz + 0;
                        outG1[k] = 0;
                        outG2[k] = 0;
                        outG3[k] = 0;
                    }

                    // kz = 1 -- one cell below the free surface
                    {
                        const long kz = 1;
                        const long kynz_kz = + kynz + kz;

                        const long kxm4 = (kx-4) * nynz + kynz_kz;
                        const long kxm3 = (kx-3) * nynz + kynz_kz;
                        const long kxm2 = (kx-2) * nynz + kynz_kz;
                        const long kxm1 = (kx-1) * nynz + kynz_kz;
                        const long kxp0 = (kx+0) * nynz + kynz_kz;
                        const long kxp1 = (kx+1) * nynz + kynz_kz;
                        const long kxp2 = (kx+2) * nynz + kynz_kz;
                        const long kxp3 = (kx+3) * nynz + kynz_kz;

                        const long kym4 = kxnynz + (ky-4) * nz + kz;
                        const long kym3 = kxnynz + (ky-3) * nz + kz;
                        const long kym2 = kxnynz + (ky-2) * nz + kz;
                        const long kym1 = kxnynz + (ky-1) * nz + kz;
                        const long kyp0 = kxnynz + (ky+0) * nz + kz;
                        const long kyp1 = kxnynz + (ky+1) * nz + kz;
                        const long kyp2 = kxnynz + (ky+2) * nz + kz;
                        const long kyp3 = kxnynz + (ky+3) * nz + kz;

                        const long kzm4 = kxnynz_kynz + 2;
                        const long kzm3 = kxnynz_kynz + 1;
                        const long kzm2 = kxnynz_kynz + 0;
                        const long kzm1 = kxnynz_kynz + 0;
                        const long kzp0 = kxnynz_kynz + 1;
                        const long kzp1 = kxnynz_kynz + 2;
                        const long kzp2 = kxnynz_kynz + 3;
                        const long kzp3 = kxnynz_kynz + 4;


                        // ........................ G1 ........................
                        const Type stencilG1A =
                                c8_1 * (- cosTheta[kxm1] * cosPhi[kxm1] * inG1[kxm1] + cosTheta[kxp0] * cosPhi[kxp0] * inG1[kxp0]) +
                                c8_2 * (- cosTheta[kxm2] * cosPhi[kxm2] * inG1[kxm2] + cosTheta[kxp1] * cosPhi[kxp1] * inG1[kxp1]) +
                                c8_3 * (- cosTheta[kxm3] * cosPhi[kxm3] * inG1[kxm3] + cosTheta[kxp2] * cosPhi[kxp2] * inG1[kxp2]) +
                                c8_4 * (- cosTheta[kxm4] * cosPhi[kxm4] * inG1[kxm4] + cosTheta[kxp3] * cosPhi[kxp3] * inG1[kxp3]);

                        const Type stencilG1B =
                                c8_1 * (- cosTheta[kym1] * sinPhi[kym1] * inG1[kym1] + cosTheta[kyp0] * sinPhi[kyp0] * inG1[kyp0]) +
                                c8_2 * (- cosTheta[kym2] * sinPhi[kym2] * inG1[kym2] + cosTheta[kyp1] * sinPhi[kyp1] * inG1[kyp1]) +
                                c8_3 * (- cosTheta[kym3] * sinPhi[kym3] * inG1[kym3] + cosTheta[kyp2] * sinPhi[kyp2] * inG1[kyp2]) +
                                c8_4 * (- cosTheta[kym4] * sinPhi[kym4] * inG1[kym4] + cosTheta[kyp3] * sinPhi[kyp3] * inG1[kyp3]);

                        const Type stencilG1C =
                                c8_1 * (- sinTheta[kzm1] * inG1[kzm1] + sinTheta[kzp0] * inG1[kzp0]) +
                                c8_2 * (- sinTheta[kzm2] * inG1[kzm2] + sinTheta[kzp1] * inG1[kzp1]) +
                                c8_3 * (- sinTheta[kzm3] * inG1[kzm3] + sinTheta[kzp2] * inG1[kzp2]) +
                                c8_4 * (- sinTheta[kzm4] * inG1[kzm4] + sinTheta[kzp3] * inG1[kzp3]);


                        // ........................ G2 ........................
                        const Type stencilG2A =
                                c8_1 * (- sinPhi[kxm1] * inG2[kxm1] + sinPhi[kxp0] * inG2[kxp0]) +
                                c8_2 * (- sinPhi[kxm2] * inG2[kxm2] + sinPhi[kxp1] * inG2[kxp1]) +
                                c8_3 * (- sinPhi[kxm3] * inG2[kxm3] + sinPhi[kxp2] * inG2[kxp2]) +
                                c8_4 * (- sinPhi[kxm4] * inG2[kxm4] + sinPhi[kxp3] * inG2[kxp3]);

                        const Type stencilG2B =
                                c8_1 * (- cosPhi[kym1] * inG2[kym1] + cosPhi[kyp0] * inG2[kyp0]) +
                                c8_2 * (- cosPhi[kym2] * inG2[kym2] + cosPhi[kyp1] * inG2[kyp1]) +
                                c8_3 * (- cosPhi[kym3] * inG2[kym3] + cosPhi[kyp2] * inG2[kyp2]) +
                                c8_4 * (- cosPhi[kym4] * inG2[kym4] + cosPhi[kyp3] * inG2[kyp3]);


                        // ........................ G3 ........................
                        const Type stencilG3A =
                                c8_1 * (- sinTheta[kxm1] * cosPhi[kxm1] * inG3[kxm1] + sinTheta[kxp0] * cosPhi[kxp0] * inG3[kxp0]) +
                                c8_2 * (- sinTheta[kxm2] * cosPhi[kxm2] * inG3[kxm2] + sinTheta[kxp1] * cosPhi[kxp1] * inG3[kxp1]) +
                                c8_3 * (- sinTheta[kxm3] * cosPhi[kxm3] * inG3[kxm3] + sinTheta[kxp2] * cosPhi[kxp2] * inG3[kxp2]) +
                                c8_4 * (- sinTheta[kxm4] * cosPhi[kxm4] * inG3[kxm4] + sinTheta[kxp3] * cosPhi[kxp3] * inG3[kxp3]);

                        const Type stencilG3B =
                                c8_1 * (- sinTheta[kym1] * sinPhi[kym1] * inG3[kym1] + sinTheta[kyp0] * sinPhi[kyp0] * inG3[kyp0]) +
                                c8_2 * (- sinTheta[kym2] * sinPhi[kym2] * inG3[kym2] + sinTheta[kyp1] * sinPhi[kyp1] * inG3[kyp1]) +
                                c8_3 * (- sinTheta[kym3] * sinPhi[kym3] * inG3[kym3] + sinTheta[kyp2] * sinPhi[kyp2] * inG3[kyp2]) +
                                c8_4 * (- sinTheta[kym4] * sinPhi[kym4] * inG3[kym4] + sinTheta[kyp3] * sinPhi[kyp3] * inG3[kyp3]);

                        const Type stencilG3C =
                                c8_1 * (- cosTheta[kzm1] * inG3[kzm1] + cosTheta[kzp0] * inG3[kzp0]) +
                                c8_2 * (- cosTheta[kzm2] * inG3[kzm2] + cosTheta[kzp1] * inG3[kzp1]) +
                                c8_3 * (- cosTheta[kzm3] * inG3[kzm3] + cosTheta[kzp2] * inG3[kzp2]) +
                                c8_4 * (- cosTheta[kzm4] * inG3[kzm4] + cosTheta[kzp3] * inG3[kzp3]);

                        const long k = kxnynz_kynz + kz;

                        outG1[k] = invDx * stencilG1A + invDy * stencilG1B - invDz * stencilG1C;
                        outG2[k] = - invDx * stencilG2A + invDy * stencilG2B;
                        outG3[k] = invDx * stencilG3A + invDy * stencilG3B + invDz * stencilG3C;
                    }

                    // kz = 2 -- two cells below the free surface
                    {
                        const long kz = 2;
                        const long kynz_kz = + kynz + kz;

                        const long kxm4 = (kx-4) * nynz + kynz_kz;
                        const long kxm3 = (kx-3) * nynz + kynz_kz;
                        const long kxm2 = (kx-2) * nynz + kynz_kz;
                        const long kxm1 = (kx-1) * nynz + kynz_kz;
                        const long kxp0 = (kx+0) * nynz + kynz_kz;
                        const long kxp1 = (kx+1) * nynz + kynz_kz;
                        const long kxp2 = (kx+2) * nynz + kynz_kz;
                        const long kxp3 = (kx+3) * nynz + kynz_kz;

                        const long kym4 = kxnynz + (ky-4) * nz + kz;
                        const long kym3 = kxnynz + (ky-3) * nz + kz;
                        const long kym2 = kxnynz + (ky-2) * nz + kz;
                        const long kym1 = kxnynz + (ky-1) * nz + kz;
                        const long kyp0 = kxnynz + (ky+0) * nz + kz;
                        const long kyp1 = kxnynz + (ky+1) * nz + kz;
                        const long kyp2 = kxnynz + (ky+2) * nz + kz;
                        const long kyp3 = kxnynz + (ky+3) * nz + kz;

                        const long kzm4 = kxnynz_kynz + 1;
                        const long kzm3 = kxnynz_kynz + 0;
                        const long kzm2 = kxnynz_kynz + 0;
                        const long kzm1 = kxnynz_kynz + 1;
                        const long kzp0 = kxnynz_kynz + 2;
                        const long kzp1 = kxnynz_kynz + 3;
                        const long kzp2 = kxnynz_kynz + 4;
                        const long kzp3 = kxnynz_kynz + 5;


                        // ........................ G1 ........................
                        const Type stencilG1A =
                                c8_1 * (- cosTheta[kxm1] * cosPhi[kxm1] * inG1[kxm1] + cosTheta[kxp0] * cosPhi[kxp0] * inG1[kxp0]) +
                                c8_2 * (- cosTheta[kxm2] * cosPhi[kxm2] * inG1[kxm2] + cosTheta[kxp1] * cosPhi[kxp1] * inG1[kxp1]) +
                                c8_3 * (- cosTheta[kxm3] * cosPhi[kxm3] * inG1[kxm3] + cosTheta[kxp2] * cosPhi[kxp2] * inG1[kxp2]) +
                                c8_4 * (- cosTheta[kxm4] * cosPhi[kxm4] * inG1[kxm4] + cosTheta[kxp3] * cosPhi[kxp3] * inG1[kxp3]);

                        const Type stencilG1B =
                                c8_1 * (- cosTheta[kym1] * sinPhi[kym1] * inG1[kym1] + cosTheta[kyp0] * sinPhi[kyp0] * inG1[kyp0]) +
                                c8_2 * (- cosTheta[kym2] * sinPhi[kym2] * inG1[kym2] + cosTheta[kyp1] * sinPhi[kyp1] * inG1[kyp1]) +
                                c8_3 * (- cosTheta[kym3] * sinPhi[kym3] * inG1[kym3] + cosTheta[kyp2] * sinPhi[kyp2] * inG1[kyp2]) +
                                c8_4 * (- cosTheta[kym4] * sinPhi[kym4] * inG1[kym4] + cosTheta[kyp3] * sinPhi[kyp3] * inG1[kyp3]);

                        const Type stencilG1C =
                                c8_1 * (- sinTheta[kzm1] * inG1[kzm1] + sinTheta[kzp0] * inG1[kzp0]) +
                                c8_2 * (- sinTheta[kzm2] * inG1[kzm2] + sinTheta[kzp1] * inG1[kzp1]) +
                                c8_3 * (- sinTheta[kzm3] * inG1[kzm3] + sinTheta[kzp2] * inG1[kzp2]) +
                                c8_4 * (- sinTheta[kzm4] * inG1[kzm4] + sinTheta[kzp3] * inG1[kzp3]);


                        // ........................ G2 ........................
                        const Type stencilG2A =
                                c8_1 * (- sinPhi[kxm1] * inG2[kxm1] + sinPhi[kxp0] * inG2[kxp0]) +
                                c8_2 * (- sinPhi[kxm2] * inG2[kxm2] + sinPhi[kxp1] * inG2[kxp1]) +
                                c8_3 * (- sinPhi[kxm3] * inG2[kxm3] + sinPhi[kxp2] * inG2[kxp2]) +
                                c8_4 * (- sinPhi[kxm4] * inG2[kxm4] + sinPhi[kxp3] * inG2[kxp3]);

                        const Type stencilG2B =
                                c8_1 * (- cosPhi[kym1] * inG2[kym1] + cosPhi[kyp0] * inG2[kyp0]) +
                                c8_2 * (- cosPhi[kym2] * inG2[kym2] + cosPhi[kyp1] * inG2[kyp1]) +
                                c8_3 * (- cosPhi[kym3] * inG2[kym3] + cosPhi[kyp2] * inG2[kyp2]) +
                                c8_4 * (- cosPhi[kym4] * inG2[kym4] + cosPhi[kyp3] * inG2[kyp3]);


                        // ........................ G3 ........................
                        const Type stencilG3A =
                                c8_1 * (- sinTheta[kxm1] * cosPhi[kxm1] * inG3[kxm1] + sinTheta[kxp0] * cosPhi[kxp0] * inG3[kxp0]) +
                                c8_2 * (- sinTheta[kxm2] * cosPhi[kxm2] * inG3[kxm2] + sinTheta[kxp1] * cosPhi[kxp1] * inG3[kxp1]) +
                                c8_3 * (- sinTheta[kxm3] * cosPhi[kxm3] * inG3[kxm3] + sinTheta[kxp2] * cosPhi[kxp2] * inG3[kxp2]) +
                                c8_4 * (- sinTheta[kxm4] * cosPhi[kxm4] * inG3[kxm4] + sinTheta[kxp3] * cosPhi[kxp3] * inG3[kxp3]);

                        const Type stencilG3B =
                                c8_1 * (- sinTheta[kym1] * sinPhi[kym1] * inG3[kym1] + sinTheta[kyp0] * sinPhi[kyp0] * inG3[kyp0]) +
                                c8_2 * (- sinTheta[kym2] * sinPhi[kym2] * inG3[kym2] + sinTheta[kyp1] * sinPhi[kyp1] * inG3[kyp1]) +
                                c8_3 * (- sinTheta[kym3] * sinPhi[kym3] * inG3[kym3] + sinTheta[kyp2] * sinPhi[kyp2] * inG3[kyp2]) +
                                c8_4 * (- sinTheta[kym4] * sinPhi[kym4] * inG3[kym4] + sinTheta[kyp3] * sinPhi[kyp3] * inG3[kyp3]);

                        const Type stencilG3C =
                                c8_1 * (- cosTheta[kzm1] * inG3[kzm1] + cosTheta[kzp0] * inG3[kzp0]) +
                                c8_2 * (- cosTheta[kzm2] * inG3[kzm2] + cosTheta[kzp1] * inG3[kzp1]) +
                                c8_3 * (- cosTheta[kzm3] * inG3[kzm3] + cosTheta[kzp2] * inG3[kzp2]) +
                                c8_4 * (- cosTheta[kzm4] * inG3[kzm4] + cosTheta[kzp3] * inG3[kzp3]);

                        const long k = kxnynz_kynz + kz;

                        outG1[k] = invDx * stencilG1A + invDy * stencilG1B - invDz * stencilG1C;
                        outG2[k] = - invDx * stencilG2A + invDy * stencilG2B;
                        outG3[k] = invDx * stencilG3A + invDy * stencilG3B + invDz * stencilG3C;
                    }

                    // kz = 3 -- three cells below the free surface
                    {
                        const long kz = 3;
                        const long kynz_kz = + kynz + kz;

                        const long kxm4 = (kx-4) * nynz + kynz_kz;
                        const long kxm3 = (kx-3) * nynz + kynz_kz;
                        const long kxm2 = (kx-2) * nynz + kynz_kz;
                        const long kxm1 = (kx-1) * nynz + kynz_kz;
                        const long kxp0 = (kx+0) * nynz + kynz_kz;
                        const long kxp1 = (kx+1) * nynz + kynz_kz;
                        const long kxp2 = (kx+2) * nynz + kynz_kz;
                        const long kxp3 = (kx+3) * nynz + kynz_kz;

                        const long kym4 = kxnynz + (ky-4) * nz + kz;
                        const long kym3 = kxnynz + (ky-3) * nz + kz;
                        const long kym2 = kxnynz + (ky-2) * nz + kz;
                        const long kym1 = kxnynz + (ky-1) * nz + kz;
                        const long kyp0 = kxnynz + (ky+0) * nz + kz;
                        const long kyp1 = kxnynz + (ky+1) * nz + kz;
                        const long kyp2 = kxnynz + (ky+2) * nz + kz;
                        const long kyp3 = kxnynz + (ky+3) * nz + kz;

                        const long kzm4 = kxnynz_kynz + 0;
                        const long kzm3 = kxnynz_kynz + 0;
                        const long kzm2 = kxnynz_kynz + 1;
                        const long kzm1 = kxnynz_kynz + 2;
                        const long kzp0 = kxnynz_kynz + 3;
                        const long kzp1 = kxnynz_kynz + 4;
                        const long kzp2 = kxnynz_kynz + 5;
                        const long kzp3 = kxnynz_kynz + 6;


                        // ........................ G1 ........................
                        const Type stencilG1A =
                                c8_1 * (- cosTheta[kxm1] * cosPhi[kxm1] * inG1[kxm1] + cosTheta[kxp0] * cosPhi[kxp0] * inG1[kxp0]) +
                                c8_2 * (- cosTheta[kxm2] * cosPhi[kxm2] * inG1[kxm2] + cosTheta[kxp1] * cosPhi[kxp1] * inG1[kxp1]) +
                                c8_3 * (- cosTheta[kxm3] * cosPhi[kxm3] * inG1[kxm3] + cosTheta[kxp2] * cosPhi[kxp2] * inG1[kxp2]) +
                                c8_4 * (- cosTheta[kxm4] * cosPhi[kxm4] * inG1[kxm4] + cosTheta[kxp3] * cosPhi[kxp3] * inG1[kxp3]);

                        const Type stencilG1B =
                                c8_1 * (- cosTheta[kym1] * sinPhi[kym1] * inG1[kym1] + cosTheta[kyp0] * sinPhi[kyp0] * inG1[kyp0]) +
                                c8_2 * (- cosTheta[kym2] * sinPhi[kym2] * inG1[kym2] + cosTheta[kyp1] * sinPhi[kyp1] * inG1[kyp1]) +
                                c8_3 * (- cosTheta[kym3] * sinPhi[kym3] * inG1[kym3] + cosTheta[kyp2] * sinPhi[kyp2] * inG1[kyp2]) +
                                c8_4 * (- cosTheta[kym4] * sinPhi[kym4] * inG1[kym4] + cosTheta[kyp3] * sinPhi[kyp3] * inG1[kyp3]);

                        const Type stencilG1C =
                                c8_1 * (- sinTheta[kzm1] * inG1[kzm1] + sinTheta[kzp0] * inG1[kzp0]) +
                                c8_2 * (- sinTheta[kzm2] * inG1[kzm2] + sinTheta[kzp1] * inG1[kzp1]) +
                                c8_3 * (- sinTheta[kzm3] * inG1[kzm3] + sinTheta[kzp2] * inG1[kzp2]) +
                                c8_4 * (- sinTheta[kzm4] * inG1[kzm4] + sinTheta[kzp3] * inG1[kzp3]);


                        // ........................ G2 ........................
                        const Type stencilG2A =
                                c8_1 * (- sinPhi[kxm1] * inG2[kxm1] + sinPhi[kxp0] * inG2[kxp0]) +
                                c8_2 * (- sinPhi[kxm2] * inG2[kxm2] + sinPhi[kxp1] * inG2[kxp1]) +
                                c8_3 * (- sinPhi[kxm3] * inG2[kxm3] + sinPhi[kxp2] * inG2[kxp2]) +
                                c8_4 * (- sinPhi[kxm4] * inG2[kxm4] + sinPhi[kxp3] * inG2[kxp3]);

                        const Type stencilG2B =
                                c8_1 * (- cosPhi[kym1] * inG2[kym1] + cosPhi[kyp0] * inG2[kyp0]) +
                                c8_2 * (- cosPhi[kym2] * inG2[kym2] + cosPhi[kyp1] * inG2[kyp1]) +
                                c8_3 * (- cosPhi[kym3] * inG2[kym3] + cosPhi[kyp2] * inG2[kyp2]) +
                                c8_4 * (- cosPhi[kym4] * inG2[kym4] + cosPhi[kyp3] * inG2[kyp3]);


                        // ........................ G3 ........................
                        const Type stencilG3A =
                                c8_1 * (- sinTheta[kxm1] * cosPhi[kxm1] * inG3[kxm1] + sinTheta[kxp0] * cosPhi[kxp0] * inG3[kxp0]) +
                                c8_2 * (- sinTheta[kxm2] * cosPhi[kxm2] * inG3[kxm2] + sinTheta[kxp1] * cosPhi[kxp1] * inG3[kxp1]) +
                                c8_3 * (- sinTheta[kxm3] * cosPhi[kxm3] * inG3[kxm3] + sinTheta[kxp2] * cosPhi[kxp2] * inG3[kxp2]) +
                                c8_4 * (- sinTheta[kxm4] * cosPhi[kxm4] * inG3[kxm4] + sinTheta[kxp3] * cosPhi[kxp3] * inG3[kxp3]);

                        const Type stencilG3B =
                                c8_1 * (- sinTheta[kym1] * sinPhi[kym1] * inG3[kym1] + sinTheta[kyp0] * sinPhi[kyp0] * inG3[kyp0]) +
                                c8_2 * (- sinTheta[kym2] * sinPhi[kym2] * inG3[kym2] + sinTheta[kyp1] * sinPhi[kyp1] * inG3[kyp1]) +
                                c8_3 * (- sinTheta[kym3] * sinPhi[kym3] * inG3[kym3] + sinTheta[kyp2] * sinPhi[kyp2] * inG3[kyp2]) +
                                c8_4 * (- sinTheta[kym4] * sinPhi[kym4] * inG3[kym4] + sinTheta[kyp3] * sinPhi[kyp3] * inG3[kyp3]);

                        const Type stencilG3C =
                                c8_1 * (- cosTheta[kzm1] * inG3[kzm1] + cosTheta[kzp0] * inG3[kzp0]) +
                                c8_2 * (- cosTheta[kzm2] * inG3[kzm2] + cosTheta[kzp1] * inG3[kzp1]) +
                                c8_3 * (- cosTheta[kzm3] * inG3[kzm3] + cosTheta[kzp2] * inG3[kzp2]) +
                                c8_4 * (- cosTheta[kzm4] * inG3[kzm4] + cosTheta[kzp3] * inG3[kzp3]);

                        const long k = kxnynz_kynz + kz;

                        outG1[k] = invDx * stencilG1A + invDy * stencilG1B - invDz * stencilG1C;
                        outG2[k] = - invDx * stencilG2A + invDy * stencilG2B;
                        outG3[k] = invDx * stencilG3A + invDy * stencilG3B + invDz * stencilG3C;
                    }
                }
            }
        }
    }

};

#endif
