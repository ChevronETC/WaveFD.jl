#ifndef PROP2DACOTTIDENQ_DEO2_FDTD_H
#define PROP2DACOTTIDENQ_DEO2_FDTD_H

#include <omp.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <fftw3.h>
#include <complex>

#define MIN(x,y) ((x)<(y)?(x):(y))

class Prop2DAcoTTIDenQ_DEO2_FDTD {

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
    float * __restrict__ _sinTheta = NULL;
    float * __restrict__ _cosTheta = NULL;
    float * __restrict__ _f = NULL;
    float * __restrict__ _dtOmegaInvQ = NULL;
    float * __restrict__ _pSpace = NULL;
    float * __restrict__ _mSpace = NULL;
    float * __restrict__ _tmpPg1a = NULL;
    float * __restrict__ _tmpPg3a = NULL;
    float * __restrict__ _tmpMg1a = NULL;
    float * __restrict__ _tmpMg3a = NULL;
    float * __restrict__ _tmpPg1b = NULL;
    float * __restrict__ _tmpPg3b = NULL;
    float * __restrict__ _tmpMg1b = NULL;
    float * __restrict__ _tmpMg3b = NULL;
    float * _pOld = NULL;
    float * _pCur = NULL;
    float * _mOld = NULL;
    float * _mCur = NULL;

    Prop2DAcoTTIDenQ_DEO2_FDTD(
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
        _sinTheta    = new float[_nx * _nz];
        _cosTheta    = new float[_nx * _nz];
        _f           = new float[_nx * _nz];
        _dtOmegaInvQ = new float[_nx * _nz];
        _pSpace      = new float[_nx * _nz];
        _mSpace      = new float[_nx * _nz];
        _tmpPg1a     = new float[_nx * _nz];
        _tmpPg3a     = new float[_nx * _nz];
        _tmpMg1a     = new float[_nx * _nz];
        _tmpMg3a     = new float[_nx * _nz];
        _tmpPg1b     = new float[_nx * _nz];
        _tmpPg3b     = new float[_nx * _nz];
        _tmpMg1b     = new float[_nx * _nz];
        _tmpMg3b     = new float[_nx * _nz];
        _pOld        = new float[_nx * _nz];
        _pCur        = new float[_nx * _nz];
        _mOld        = new float[_nx * _nz];
        _mCur        = new float[_nx * _nz];

        numaFirstTouch(_nx, _nz, _nthread, _v, _eps, _eta, _b,
            _sinTheta, _cosTheta, _f, _dtOmegaInvQ, _pSpace, _mSpace,
            _tmpPg1a, _tmpPg3a, _tmpMg1a, _tmpMg3a,
            _tmpPg1b, _tmpPg3b, _tmpMg1b, _tmpMg3b,
            _pOld, _pCur, _mOld, _mCur, _nbx, _nbz);
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
            float * __restrict__ sinTheta,
            float * __restrict__ cosTheta,
            float * __restrict__ f,
            float * __restrict__ dtOmegaInvQ,
            float * __restrict__ pSpace,
            float * __restrict__ mSpace,
            float * __restrict__ tmpPg1a,
            float * __restrict__ tmpPg3a,
            float * __restrict__ tmpMg1a,
            float * __restrict__ tmpMg3a,
            float * __restrict__ tmpPg1b,
            float * __restrict__ tmpPg3b,
            float * __restrict__ tmpMg1b,
            float * __restrict__ tmpMg3b,
            float * __restrict__ pOld,
            float * __restrict__ pCur,
            float * __restrict__ mOld,
            float * __restrict__ mCur,
            const long BX_2D,
            const long BZ_2D) {

        const long nx4 = _nx - 4;
        const long nz4 = _nz - 4;

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
                        eps[k] = 0;
                        eta[k] = 0;
                        b[k] = 0;
                        sinTheta[k] = 0;
                        cosTheta[k] = 0;
                        f[k] = 0;
                        dtOmegaInvQ[k] = 0;
                        pSpace[k] = 0;
                        mSpace[k] = 0;
                        tmpPg1a[k] = 0;
                        tmpPg3a[k] = 0;
                        tmpMg1a[k] = 0;
                        tmpMg3a[k] = 0;
                        tmpPg1b[k] = 0;
                        tmpPg3b[k] = 0;
                        tmpMg1b[k] = 0;
                        tmpMg3b[k] = 0;
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
                v[k] = eps[k] = eta[k] = b[k] = sinTheta[k] = cosTheta[k] = f[k] = dtOmegaInvQ[k] = pSpace[k] =
                    mSpace[k] = tmpPg1a[k] = tmpPg3a[k] = tmpMg1a[k] = tmpMg3a[k] = tmpPg1b[k] = tmpPg3b[k] =
                    tmpMg1b[k] = tmpMg3b[k] = pOld[k] = pCur[k] = mOld[k] = mCur[k] = 0;
            }
        }
#pragma omp parallel for num_threads(nthread) schedule(static)
        for (long kz = nz4; kz < nz; kz++) {
#pragma omp simd
            for (long kx = 0; kx < nx; kx++) {
                const long k = kx * _nz + kz;
                v[k] = eps[k] = eta[k] = b[k] = sinTheta[k] = cosTheta[k] = f[k] = dtOmegaInvQ[k] = pSpace[k] =
                    mSpace[k] = tmpPg1a[k] = tmpPg3a[k] = tmpMg1a[k] = tmpMg3a[k] = tmpPg1b[k] = tmpPg3b[k] =
                    tmpMg1b[k] = tmpMg3b[k] = pOld[k] = pCur[k] = mOld[k] = mCur[k] = 0;
            }
        }

#pragma omp parallel for num_threads(nthread) schedule(static)
        for (long kx = 0; kx < 4; kx++) {
#pragma omp simd
            for (long kz = 0; kz < nz; kz++) {
                const long k = kx * _nz + kz;
                v[k] = eps[k] = eta[k] = b[k] = sinTheta[k] = cosTheta[k] = f[k] = dtOmegaInvQ[k] = pSpace[k] =
                    mSpace[k] = tmpPg1a[k] = tmpPg3a[k] = tmpMg1a[k] = tmpMg3a[k] = tmpPg1b[k] = tmpPg3b[k] =
                    tmpMg1b[k] = tmpMg3b[k] = pOld[k] = pCur[k] = mOld[k] = mCur[k] = 0;
            }
        }
#pragma omp parallel for num_threads(nthread) schedule(static)
        for (long kx = nx4; kx < nx; kx++) {
#pragma omp simd
            for (long kz = 0; kz < nz; kz++) {
                const long k = kx * _nz + kz;
                v[k] = eps[k] = eta[k] = b[k] = sinTheta[k] = cosTheta[k] = f[k] = dtOmegaInvQ[k] = pSpace[k] =
                    mSpace[k] = tmpPg1a[k] = tmpPg3a[k] = tmpMg1a[k] = tmpMg3a[k] = tmpPg1b[k] = tmpPg3b[k] =
                    tmpMg1b[k] = tmpMg3b[k] = pOld[k] = pCur[k] = mOld[k] = mCur[k] = 0;
            }
        }
    }

    ~Prop2DAcoTTIDenQ_DEO2_FDTD() {
        if (_v != NULL) delete [] _v;
        if (_eps != NULL) delete [] _eps;
        if (_eta != NULL) delete [] _eta;
        if (_sinTheta != NULL) delete [] _sinTheta;
        if (_cosTheta != NULL) delete [] _cosTheta;
        if (_b != NULL) delete [] _b;
        if (_f != NULL) delete [] _f;
        if (_dtOmegaInvQ != NULL) delete [] _dtOmegaInvQ;
        if (_pSpace != NULL) delete [] _pSpace;
        if (_mSpace != NULL) delete [] _mSpace;
        if (_tmpPg1a != NULL) delete [] _tmpPg1a;
        if (_tmpPg3a != NULL) delete [] _tmpPg3a;
        if (_tmpMg1a != NULL) delete [] _tmpMg1a;
        if (_tmpMg3a != NULL) delete [] _tmpMg3a;
        if (_tmpPg1b != NULL) delete [] _tmpPg1b;
        if (_tmpPg3b != NULL) delete [] _tmpPg3b;
        if (_tmpMg1b != NULL) delete [] _tmpMg1b;
        if (_tmpMg3b != NULL) delete [] _tmpMg3b;
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
        printf("Prop2DAcoTTIDenQ_DEO2_FDTD\n");
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

        applyFirstDerivatives2D_TTI_PlusHalf_Sandwich(
            _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
            _pCur, _pCur, _mCur, _mCur, _sinTheta, _cosTheta, _eps, _eta, _f, _b,
            _tmpPg1a, _tmpPg3a, _tmpMg1a, _tmpMg3a, _nbx, _nbz);

        applyFirstDerivatives2D_TTI_MinusHalf_TimeUpdate_Nonlinear(
            _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz, _dt,
            _tmpPg1a, _tmpPg3a, _tmpMg1a, _tmpMg3a, _sinTheta, _cosTheta, _v, _b, _dtOmegaInvQ,
            _pCur, _mCur, _pSpace, _mSpace, _pOld, _mOld, _nbx, _nbz);

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
    * Note: "dVel" is the three components of the model consecutively [vel,eps,eta]
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

#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void forwardBornInjection_VEA(float *dVel, float *dEps, float *dEta,
        float *wavefieldP, float *wavefieldM, float *wavefieldDP, float *wavefieldDM) {

        // Right side spatial derivatives for the Born source
        applyFirstDerivatives2D_TTI_PlusHalf(
            _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
            wavefieldP, wavefieldP, _sinTheta, _cosTheta, _tmpPg1a, _tmpPg3a, _nbx, _nbz);

        applyFirstDerivatives2D_TTI_PlusHalf(
            _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
            wavefieldM, wavefieldM, _sinTheta, _cosTheta, _tmpMg1a, _tmpMg3a, _nbx, _nbz);

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
                        const float F  = _f[k];
                        const float dE = dEps[k];
                        const float dA = dEta[k];

                        _tmpPg1b[k] = (+2 * B * dE) *_tmpPg1a[k];
                        _tmpPg3b[k] = (-2 * B * F * A * dA) *_tmpPg3a[k] +
                            (dA * B * F * (1 - 2 * A * A) / sqrt(1 - A * A)) * _tmpMg3a[k];

                        _tmpMg1b[k] = 0;
                        _tmpMg3b[k] = (+2 * B * F * A * dA) *_tmpMg3a[k] +
                            (dA * B * F * (1 - 2 * A * A) / sqrt(1 - A * A)) * _tmpPg3a[k];
                    }
                }
            }
        }

        // Left side spatial derivatives for the Born source
        applyFirstDerivatives2D_TTI_MinusHalf(
            _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
            _tmpPg1b, _tmpPg3b, _sinTheta, _cosTheta, _tmpPg1a, _tmpPg3a, _nbx, _nbz);

        applyFirstDerivatives2D_TTI_MinusHalf(
            _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
            _tmpMg1b, _tmpMg3b, _sinTheta, _cosTheta, _tmpMg1a, _tmpMg3a, _nbx, _nbz);

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
                        const float dV = dVel[k];
                        const float dt2v2OverB = _dt * _dt * V * V / B;
                        const float factor = 2 * B * dV / (V * V * V);

                        _pCur[k] += dt2v2OverB * (factor * wavefieldDP[k] + _tmpPg1a[k] + _tmpPg3a[k]);
                        _mCur[k] += dt2v2OverB * (factor * wavefieldDM[k] + _tmpMg1a[k] + _tmpMg3a[k]);
                    }
                }
            }
        }
    }

    /**
    * Accumulate the Born image term at the current time
    *
    * Note: "dVel" is the three components of the model consecutively [vel,eps,eta]
    * 
    * User must have:
    *   - called the nonlinear forward
    *   - saved 2nd time derivative of pressure at corresponding time index in array dp2
    *   - Born image term will be accumulated iu the _dm array
    */
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void adjointBornAccumulation_V(float *dVel, float *wavefieldDP, float *wavefieldDM) {
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
                        dVel[k] += factor * (wavefieldDP[k] * _pOld[k] + wavefieldDM[k] * _mOld[k]);
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
     inline void adjointBornAccumulation_wavefieldsep_V(float *dVel, float *wavefieldDP, float *wavefieldDM, const long isFWI) {
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

#pragma omp for schedule(static)
            for (long bx = 0; bx < _nx; bx += _nbx) {
                const long kxmax = MIN(bx + _nbx, _nx);
                for (long kx = bx; kx < kxmax; kx++) {

#pragma omp simd
                    for (long kfft = 0; kfft < nfft; kfft++) {
                        tmp_nlf_p[kfft] = 0;
                        tmp_adj_p[kfft] = 0;
                        tmp_nlf_m[kfft] = 0;
                        tmp_adj_m[kfft] = 0;
                    }  

#pragma omp simd
                    for (long kz = 0; kz < _nz; kz++) {
                        const long k = kx * _nz + kz;
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
                        const long k = kx * _nz + kz;
                        const float V = _v[k];
                        const float B = _b[k];
                        const float factor = 2 * B / (V * V * V);
                        dVel[k] += factor * (real(tmp_nlf_p[kz] * tmp_adj_p[kz]) + real(tmp_nlf_m[kz] * tmp_adj_m[kz]));
                    }

                } // end loop over kx
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
        applyFirstDerivatives2D_TTI_PlusHalf(
            _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
            wavefieldP, wavefieldP, _sinTheta, _cosTheta, _tmpPg1a, _tmpPg3a, _nbx, _nbz);

        // Right hand derivatives for M
        applyFirstDerivatives2D_TTI_PlusHalf(
            _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
            wavefieldM, wavefieldM, _sinTheta, _cosTheta, _tmpMg1a, _tmpMg3a, _nbx, _nbz);

        applyFirstDerivatives2D_TTI_PlusHalf(
            _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
            _pOld, _pOld, _sinTheta, _cosTheta, _tmpPg1b, _tmpPg3b, _nbx, _nbz);

        // Right hand derivatives for M
        applyFirstDerivatives2D_TTI_PlusHalf(
            _freeSurface, _nx, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDz,
            _mOld, _mOld, _sinTheta, _cosTheta, _tmpMg1b, _tmpMg3b, _nbx, _nbz);

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

                        dVel[k] +=
                            factor * wavefieldDP[k] * _pOld[k] +
                            factor * wavefieldDM[k] * _mOld[k];

                        dEps[k] += -2 * B * _tmpPg1a[k] * _tmpPg1b[k];

                        const float partP =
                            2 * B * F * A * _tmpPg3a[k] - (B * F * (1 - 2 * A * A) / sqrt(1 - A * A)) * _tmpMg3a[k];

                        const float partM =
                            2 * B * F * A * _tmpMg3a[k] + (B * F * (1 - 2 * A * A) / sqrt(1 - A * A)) * _tmpPg3a[k];

                        dEta[k] += partP * _tmpPg3b[k] - partM * _tmpMg3b[k];
                    }
                }
            }
        }
    }

    template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline static void applyFirstDerivatives2D_TTI_PlusHalf_Sandwich(
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
            const Type * __restrict__ const inPG1,
            const Type * __restrict__ const inPG3,
            const Type * __restrict__ const inMG1,
            const Type * __restrict__ const inMG3,
            const float * __restrict__ const sinTheta,
            const float * __restrict__ const cosTheta,
            const Type * __restrict__ const fieldEps,
            const Type * __restrict__ const fieldEta,
            const Type * __restrict__ const fieldVsVp,
            const Type * __restrict__ const fieldBuoy,
            Type * __restrict__ outPG1,
            Type * __restrict__ outPG3,
            Type * __restrict__ outMG1,
            Type * __restrict__ outMG3,
            const long BX_2D,
            const long BZ_2D) {

        const long nx4 = nx - 4;
        const long nz4 = nz - 4;

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
                        outPG1[k] = 0;
                        outPG3[k] = 0;
                        outMG1[k] = 0;
                        outMG3[k] = 0;
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

                        const Type stencilPG1 =
                                c8_1 * (- inPG1[(kx+0) * nz + kz] + inPG1[(kx+1) * nz + kz]) +
                                c8_2 * (- inPG1[(kx-1) * nz + kz] + inPG1[(kx+2) * nz + kz]) +
                                c8_3 * (- inPG1[(kx-2) * nz + kz] + inPG1[(kx+3) * nz + kz]) +
                                c8_4 * (- inPG1[(kx-3) * nz + kz] + inPG1[(kx+4) * nz + kz]);

                        const Type stencilPG3 =
                                c8_1 * (- inPG3[kx * nz + (kz+0)] + inPG3[kx * nz + (kz+1)]) +
                                c8_2 * (- inPG3[kx * nz + (kz-1)] + inPG3[kx * nz + (kz+2)]) +
                                c8_3 * (- inPG3[kx * nz + (kz-2)] + inPG3[kx * nz + (kz+3)]) +
                                c8_4 * (- inPG3[kx * nz + (kz-3)] + inPG3[kx * nz + (kz+4)]);

                        const Type dpx = invDx * stencilPG1;
                        const Type dpz = invDz * stencilPG3;

                        const Type dPG1 = cosTheta[kx * nz + kz] * dpx - sinTheta[kx * nz + kz] * dpz;
                        const Type dPG3 = sinTheta[kx * nz + kz] * dpx + cosTheta[kx * nz + kz] * dpz;

                        const Type stencilMG1 =
                                c8_1 * (- inMG1[(kx+0) * nz + kz] + inMG1[(kx+1) * nz + kz]) +
                                c8_2 * (- inMG1[(kx-1) * nz + kz] + inMG1[(kx+2) * nz + kz]) +
                                c8_3 * (- inMG1[(kx-2) * nz + kz] + inMG1[(kx+3) * nz + kz]) +
                                c8_4 * (- inMG1[(kx-3) * nz + kz] + inMG1[(kx+4) * nz + kz]);

                        const Type stencilMG3 =
                                c8_1 * (- inMG3[kx * nz + (kz+0)] + inMG3[kx * nz + (kz+1)]) +
                                c8_2 * (- inMG3[kx * nz + (kz-1)] + inMG3[kx * nz + (kz+2)]) +
                                c8_3 * (- inMG3[kx * nz + (kz-2)] + inMG3[kx * nz + (kz+3)]) +
                                c8_4 * (- inMG3[kx * nz + (kz-3)] + inMG3[kx * nz + (kz+4)]);

                        const Type dmx = invDx * stencilMG1;
                        const Type dmz = invDz * stencilMG3;

                        const Type dMG1 = cosTheta[kx * nz + kz] * dmx - sinTheta[kx * nz + kz] * dmz;
                        const Type dMG3 = sinTheta[kx * nz + kz] * dmx + cosTheta[kx * nz + kz] * dmz;

                        // assemble the sandwich
                        const long k = kx * nz + kz;

                        const Type E = 1 + 2 * fieldEps[k];
                        const Type A = fieldEta[k];
                        const Type F = fieldVsVp[k];
                        const Type B = fieldBuoy[k];

                        outPG1[k] = B * E * dPG1;
                        outPG3[k] = B * (1 - F * A * A) * dPG3 + B * F * A * sqrt(1 - A * A) * dMG3;
                        outMG1[k] = B * (1 - F) * dMG1;
                        outMG3[k] = B * F * A * sqrt(1 - A * A) * dPG3 + B * (1 - F + F * A * A) * dMG3;
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
                {
                    const Type stencilPG30 =
                            c8_1 * (- inPG3[kx * nz + 0] + inPG3[kx * nz + 1]) +
                            c8_2 * (+ inPG3[kx * nz + 1] + inPG3[kx * nz + 2]) +
                            c8_3 * (+ inPG3[kx * nz + 2] + inPG3[kx * nz + 3]) +
                            c8_4 * (+ inPG3[kx * nz + 3] + inPG3[kx * nz + 4]);

                    const Type dpx0 = 0;
                    const Type dpz0 = invDz * stencilPG30;

                    const Type dPG1 = cosTheta[kx * nz + 0] * dpx0 - sinTheta[kx * nz + 0] * dpz0;
                    const Type dPG3 = sinTheta[kx * nz + 0] * dpx0 + cosTheta[kx * nz + 0] * dpz0;

                    const Type stencilMG30 =
                            c8_1 * (- inMG3[kx * nz + 0] + inMG3[kx * nz + 1]) +
                            c8_2 * (+ inMG3[kx * nz + 1] + inMG3[kx * nz + 2]) +
                            c8_3 * (+ inMG3[kx * nz + 2] + inMG3[kx * nz + 3]) +
                            c8_4 * (+ inMG3[kx * nz + 3] + inMG3[kx * nz + 4]);

                    const Type dmx0 = 0;
                    const Type dmz0 = invDz * stencilMG30;

                    const Type dMG1 = cosTheta[kx * nz + 0] * dmx0 - sinTheta[kx * nz + 0] * dmz0;
                    const Type dMG3 = sinTheta[kx * nz + 0] * dmx0 + cosTheta[kx * nz + 0] * dmz0;

                    // assemble the sandwich
                    const long k = kx * nz + 0;

                    const Type E = 1 + 2 * fieldEps[k];
                    const Type A = fieldEta[k];
                    const Type F = fieldVsVp[k];
                    const Type B = fieldBuoy[k];

                    outPG1[k] = B * E * dPG1;
                    outPG3[k] = B * (1 - F * A * A) * dPG3 + B * F * A * sqrt(1 - A * A) * dMG3;
                    outMG1[k] = B * (1 - F) * dMG1;
                    outMG3[k] = B * F * A * sqrt(1 - A * A) * dPG3 + B * (1 - F + F * A * A) * dMG3;
                }

                // kz = 1 -- 1 1/2 cells below free surface for Z derivative, 1 cells below for X/Y derivative
                {
                    const Type stencilPG11 =
                            c8_1 * (- inPG1[(kx+0) * nz + 1] + inPG1[(kx+1) * nz + 1]) +
                            c8_2 * (- inPG1[(kx-1) * nz + 1] + inPG1[(kx+2) * nz + 1]) +
                            c8_3 * (- inPG1[(kx-2) * nz + 1] + inPG1[(kx+3) * nz + 1]) +
                            c8_4 * (- inPG1[(kx-3) * nz + 1] + inPG1[(kx+4) * nz + 1]);

                    const Type stencilPG31 =
                            c8_1 * (- inPG3[kx * nz + 1] + inPG3[kx * nz + 2]) +
                            c8_2 * (- inPG3[kx * nz + 0] + inPG3[kx * nz + 3]) +
                            c8_3 * (+ inPG3[kx * nz + 1] + inPG3[kx * nz + 4]) +
                            c8_4 * (+ inPG3[kx * nz + 2] + inPG3[kx * nz + 5]);

                    const Type dpx1 = invDx * stencilPG11;
                    const Type dpz1 = invDz * stencilPG31;

                    const Type dPG1 = cosTheta[kx * nz + 1] * dpx1 - sinTheta[kx * nz + 1] * dpz1;
                    const Type dPG3 = sinTheta[kx * nz + 1] * dpx1 + cosTheta[kx * nz + 1] * dpz1;

                    const Type stencilMG11 =
                            c8_1 * (- inMG1[(kx+0) * nz + 1] + inMG1[(kx+1) * nz + 1]) +
                            c8_2 * (- inMG1[(kx-1) * nz + 1] + inMG1[(kx+2) * nz + 1]) +
                            c8_3 * (- inMG1[(kx-2) * nz + 1] + inMG1[(kx+3) * nz + 1]) +
                            c8_4 * (- inMG1[(kx-3) * nz + 1] + inMG1[(kx+4) * nz + 1]);

                    const Type stencilMG31 =
                            c8_1 * (- inMG3[kx * nz + 1] + inMG3[kx * nz + 2]) +
                            c8_2 * (- inMG3[kx * nz + 0] + inMG3[kx * nz + 3]) +
                            c8_3 * (+ inMG3[kx * nz + 1] + inMG3[kx * nz + 4]) +
                            c8_4 * (+ inMG3[kx * nz + 2] + inMG3[kx * nz + 5]);

                    const Type dmx1 = invDx * stencilMG11;
                    const Type dmz1 = invDz * stencilMG31;

                    const Type dMG1 = cosTheta[kx * nz + 1] * dmx1 - sinTheta[kx * nz + 1] * dmz1;
                    const Type dMG3 = sinTheta[kx * nz + 1] * dmx1 + cosTheta[kx * nz + 1] * dmz1;

                    // assemble the sandwich
                    const long k = kx * nz + 1;

                    const Type E = 1 + 2 * fieldEps[k];
                    const Type A = fieldEta[k];
                    const Type F = fieldVsVp[k];
                    const Type B = fieldBuoy[k];

                    outPG1[k] = B * E * dPG1;
                    outPG3[k] = B * (1 - F * A * A) * dPG3 + B * F * A * sqrt(1 - A * A) * dMG3;
                    outMG1[k] = B * (1 - F) * dMG1;
                    outMG3[k] = B * F * A * sqrt(1 - A * A) * dPG3 + B * (1 - F + F * A * A) * dMG3;
                }

                // kz = 2 -- 2 1/2 cells below free surface for Z derivative, 2 cells below for X/Y derivative
                {
                    const Type stencilPG12 =
                            c8_1 * (- inPG1[(kx+0) * nz + 2] + inPG1[(kx+1) * nz + 2]) +
                            c8_2 * (- inPG1[(kx-1) * nz + 2] + inPG1[(kx+2) * nz + 2]) +
                            c8_3 * (- inPG1[(kx-2) * nz + 2] + inPG1[(kx+3) * nz + 2]) +
                            c8_4 * (- inPG1[(kx-3) * nz + 2] + inPG1[(kx+4) * nz + 2]);

                    const Type stencilPG32 =
                            c8_1 * (- inPG3[kx * nz + 2] + inPG3[kx * nz + 3]) +
                            c8_2 * (- inPG3[kx * nz + 1] + inPG3[kx * nz + 4]) +
                            c8_3 * (- inPG3[kx * nz + 0] + inPG3[kx * nz + 5]) +
                            c8_4 * (+ inPG3[kx * nz + 1] + inPG3[kx * nz + 6]);

                    const Type dpx2 = invDx * stencilPG12;
                    const Type dpz2 = invDz * stencilPG32;

                    const Type dPG1 = cosTheta[kx * nz + 2] * dpx2 - sinTheta[kx * nz + 2] * dpz2;
                    const Type dPG3 = sinTheta[kx * nz + 2] * dpx2 + cosTheta[kx * nz + 2] * dpz2;

                    const Type stencilMG12 =
                            c8_1 * (- inMG1[(kx+0) * nz + 2] + inMG1[(kx+1) * nz + 2]) +
                            c8_2 * (- inMG1[(kx-1) * nz + 2] + inMG1[(kx+2) * nz + 2]) +
                            c8_3 * (- inMG1[(kx-2) * nz + 2] + inMG1[(kx+3) * nz + 2]) +
                            c8_4 * (- inMG1[(kx-3) * nz + 2] + inMG1[(kx+4) * nz + 2]);

                    const Type stencilMG32 =
                            c8_1 * (- inMG3[kx * nz + 2] + inMG3[kx * nz + 3]) +
                            c8_2 * (- inMG3[kx * nz + 1] + inMG3[kx * nz + 4]) +
                            c8_3 * (- inMG3[kx * nz + 0] + inMG3[kx * nz + 5]) +
                            c8_4 * (+ inMG3[kx * nz + 1] + inMG3[kx * nz + 6]);

                    const Type dmx2 = invDx * stencilMG12;
                    const Type dmz2 = invDz * stencilMG32;

                    const Type dMG1 = cosTheta[kx * nz + 2] * dmx2 - sinTheta[kx * nz + 2] * dmz2;
                    const Type dMG3 = sinTheta[kx * nz + 2] * dmx2 + cosTheta[kx * nz + 2] * dmz2;

                    // assemble the sandwich
                    const long k = kx * nz + 2;

                    const Type E = 1 + 2 * fieldEps[k];
                    const Type A = fieldEta[k];
                    const Type F = fieldVsVp[k];
                    const Type B = fieldBuoy[k];

                    outPG1[k] = B * E * dPG1;
                    outPG3[k] = B * (1 - F * A * A) * dPG3 + B * F * A * sqrt(1 - A * A) * dMG3;
                    outMG1[k] = B * (1 - F) * dMG1;
                    outMG3[k] = B * F * A * sqrt(1 - A * A) * dPG3 + B * (1 - F + F * A * A) * dMG3;
                }

                // kz = 3 -- 3 1/2 cells below free surface for Z derivative, 3 cells below for X/Y derivative
                {
                    const Type stencilPG13 =
                            c8_1 * (- inPG1[(kx+0) * nz + 3] + inPG1[(kx+1) * nz + 3]) +
                            c8_2 * (- inPG1[(kx-1) * nz + 3] + inPG1[(kx+2) * nz + 3]) +
                            c8_3 * (- inPG1[(kx-2) * nz + 3] + inPG1[(kx+3) * nz + 3]) +
                            c8_4 * (- inPG1[(kx-3) * nz + 3] + inPG1[(kx+4) * nz + 3]);

                    const Type stencilPG33 =
                            c8_1 * (- inPG3[kx * nz + 3] + inPG3[kx * nz + 4]) +
                            c8_2 * (- inPG3[kx * nz + 2] + inPG3[kx * nz + 5]) +
                            c8_3 * (- inPG3[kx * nz + 1] + inPG3[kx * nz + 6]) +
                            c8_4 * (- inPG3[kx * nz + 0] + inPG3[kx * nz + 7]);

                    const Type dpx3 = invDx * stencilPG13;
                    const Type dpz3 = invDz * stencilPG33;

                    const Type dPG1 = cosTheta[kx * nz + 3] * dpx3 - sinTheta[kx * nz + 3] * dpz3;
                    const Type dPG3 = sinTheta[kx * nz + 3] * dpx3 + cosTheta[kx * nz + 3] * dpz3;

                    const Type stencilMG13 =
                            c8_1 * (- inMG1[(kx+0) * nz + 3] + inMG1[(kx+1) * nz + 3]) +
                            c8_2 * (- inMG1[(kx-1) * nz + 3] + inMG1[(kx+2) * nz + 3]) +
                            c8_3 * (- inMG1[(kx-2) * nz + 3] + inMG1[(kx+3) * nz + 3]) +
                            c8_4 * (- inMG1[(kx-3) * nz + 3] + inMG1[(kx+4) * nz + 3]);

                    const Type stencilMG33 =
                            c8_1 * (- inMG3[kx * nz + 3] + inMG3[kx * nz + 4]) +
                            c8_2 * (- inMG3[kx * nz + 2] + inMG3[kx * nz + 5]) +
                            c8_3 * (- inMG3[kx * nz + 1] + inMG3[kx * nz + 6]) +
                            c8_4 * (- inMG3[kx * nz + 0] + inMG3[kx * nz + 7]);

                    const Type dmx3 = invDx * stencilMG13;
                    const Type dmz3 = invDz * stencilMG33;

                    const Type dMG1 = cosTheta[kx * nz + 3] * dmx3 - sinTheta[kx * nz + 3] * dmz3;
                    const Type dMG3 = sinTheta[kx * nz + 3] * dmx3 + cosTheta[kx * nz + 3] * dmz3;

                    // assemble the sandwich
                    const long k = kx * nz + 3;

                    const Type E = 1 + 2 * fieldEps[k];
                    const Type A = fieldEta[k];
                    const Type F = fieldVsVp[k];
                    const Type B = fieldBuoy[k];

                    outPG1[k] = B * E * dPG1;
                    outPG3[k] = B * (1 - F * A * A) * dPG3 + B * F * A * sqrt(1 - A * A) * dMG3;
                    outMG1[k] = B * (1 - F) * dMG1;
                    outMG3[k] = B * F * A * sqrt(1 - A * A) * dPG3 + B * (1 - F + F * A * A) * dMG3;
                }
            }
        }
    }

    template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline static void applyFirstDerivatives2D_TTI_MinusHalf_TimeUpdate_Nonlinear(
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
            const Type * __restrict__ const inPG1,
            const Type * __restrict__ const inPG3,
            const Type * __restrict__ const inMG1,
            const Type * __restrict__ const inMG3,
            const float * __restrict__ const sinTheta,
            const float * __restrict__ const cosTheta,
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

        // zero output array
#pragma omp parallel for collapse(2) num_threads(nthread) schedule(static)
        for (long bx = 0; bx < nx; bx += BX_2D) {
            for (long bz = 0; bz < nz; bz += BZ_2D) { /* cache blocking */

                const long kxmax = MIN(bx + BX_2D, nx);
                const long kzmax = MIN(bz + BZ_2D, nz);

                for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                    for (long kz = bz; kz < kzmax; kz++) {
                        pSpace[kx * nz + kz] = 0;
                        mSpace[kx * nz + kz] = 0;
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

                        const Type stencilPG1A =
                                c8_1 * (- cosTheta[(kx-1) * nz + kz] * inPG1[(kx-1) * nz + kz] + cosTheta[(kx+0) * nz + kz] * inPG1[(kx+0) * nz + kz]) +
                                c8_2 * (- cosTheta[(kx-2) * nz + kz] * inPG1[(kx-2) * nz + kz] + cosTheta[(kx+1) * nz + kz] * inPG1[(kx+1) * nz + kz]) +
                                c8_3 * (- cosTheta[(kx-3) * nz + kz] * inPG1[(kx-3) * nz + kz] + cosTheta[(kx+2) * nz + kz] * inPG1[(kx+2) * nz + kz]) +
                                c8_4 * (- cosTheta[(kx-4) * nz + kz] * inPG1[(kx-4) * nz + kz] + cosTheta[(kx+3) * nz + kz] * inPG1[(kx+3) * nz + kz]);

                        const Type stencilPG1B =
                                c8_1 * (- sinTheta[kx * nz + (kz-1)] * inPG1[kx * nz + (kz-1)] + sinTheta[kx * nz + (kz+0)] * inPG1[kx * nz + (kz+0)]) +
                                c8_2 * (- sinTheta[kx * nz + (kz-2)] * inPG1[kx * nz + (kz-2)] + sinTheta[kx * nz + (kz+1)] * inPG1[kx * nz + (kz+1)]) +
                                c8_3 * (- sinTheta[kx * nz + (kz-3)] * inPG1[kx * nz + (kz-3)] + sinTheta[kx * nz + (kz+2)] * inPG1[kx * nz + (kz+2)]) +
                                c8_4 * (- sinTheta[kx * nz + (kz-4)] * inPG1[kx * nz + (kz-4)] + sinTheta[kx * nz + (kz+3)] * inPG1[kx * nz + (kz+3)]);

                        const Type stencilPG3A =
                                c8_1 * (- sinTheta[(kx-1) * nz + kz] * inPG3[(kx-1) * nz + kz] + sinTheta[(kx+0) * nz + kz] * inPG3[(kx+0) * nz + kz]) +
                                c8_2 * (- sinTheta[(kx-2) * nz + kz] * inPG3[(kx-2) * nz + kz] + sinTheta[(kx+1) * nz + kz] * inPG3[(kx+1) * nz + kz]) +
                                c8_3 * (- sinTheta[(kx-3) * nz + kz] * inPG3[(kx-3) * nz + kz] + sinTheta[(kx+2) * nz + kz] * inPG3[(kx+2) * nz + kz]) +
                                c8_4 * (- sinTheta[(kx-4) * nz + kz] * inPG3[(kx-4) * nz + kz] + sinTheta[(kx+3) * nz + kz] * inPG3[(kx+3) * nz + kz]);

                        const Type stencilPG3B =
                                c8_1 * (- cosTheta[kx * nz + (kz-1)] * inPG3[kx * nz + (kz-1)] + cosTheta[kx * nz + (kz+0)] * inPG3[kx * nz + (kz+0)]) +
                                c8_2 * (- cosTheta[kx * nz + (kz-2)] * inPG3[kx * nz + (kz-2)] + cosTheta[kx * nz + (kz+1)] * inPG3[kx * nz + (kz+1)]) +
                                c8_3 * (- cosTheta[kx * nz + (kz-3)] * inPG3[kx * nz + (kz-3)] + cosTheta[kx * nz + (kz+2)] * inPG3[kx * nz + (kz+2)]) +
                                c8_4 * (- cosTheta[kx * nz + (kz-4)] * inPG3[kx * nz + (kz-4)] + cosTheta[kx * nz + (kz+3)] * inPG3[kx * nz + (kz+3)]);

                        const Type dPG1 = invDx * stencilPG1A - invDz * stencilPG1B;
                        const Type dPG3 = invDx * stencilPG3A + invDz * stencilPG3B;

                        const Type stencilMG1A =
                                c8_1 * (- cosTheta[(kx-1) * nz + kz] * inMG1[(kx-1) * nz + kz] + cosTheta[(kx+0) * nz + kz] * inMG1[(kx+0) * nz + kz]) +
                                c8_2 * (- cosTheta[(kx-2) * nz + kz] * inMG1[(kx-2) * nz + kz] + cosTheta[(kx+1) * nz + kz] * inMG1[(kx+1) * nz + kz]) +
                                c8_3 * (- cosTheta[(kx-3) * nz + kz] * inMG1[(kx-3) * nz + kz] + cosTheta[(kx+2) * nz + kz] * inMG1[(kx+2) * nz + kz]) +
                                c8_4 * (- cosTheta[(kx-4) * nz + kz] * inMG1[(kx-4) * nz + kz] + cosTheta[(kx+3) * nz + kz] * inMG1[(kx+3) * nz + kz]);

                        const Type stencilMG1B =
                                c8_1 * (- sinTheta[kx * nz + (kz-1)] * inMG1[kx * nz + (kz-1)] + sinTheta[kx * nz + (kz+0)] * inMG1[kx * nz + (kz+0)]) +
                                c8_2 * (- sinTheta[kx * nz + (kz-2)] * inMG1[kx * nz + (kz-2)] + sinTheta[kx * nz + (kz+1)] * inMG1[kx * nz + (kz+1)]) +
                                c8_3 * (- sinTheta[kx * nz + (kz-3)] * inMG1[kx * nz + (kz-3)] + sinTheta[kx * nz + (kz+2)] * inMG1[kx * nz + (kz+2)]) +
                                c8_4 * (- sinTheta[kx * nz + (kz-4)] * inMG1[kx * nz + (kz-4)] + sinTheta[kx * nz + (kz+3)] * inMG1[kx * nz + (kz+3)]);

                        const Type stencilMG3A =
                                c8_1 * (- sinTheta[(kx-1) * nz + kz] * inMG3[(kx-1) * nz + kz] + sinTheta[(kx+0) * nz + kz] * inMG3[(kx+0) * nz + kz]) +
                                c8_2 * (- sinTheta[(kx-2) * nz + kz] * inMG3[(kx-2) * nz + kz] + sinTheta[(kx+1) * nz + kz] * inMG3[(kx+1) * nz + kz]) +
                                c8_3 * (- sinTheta[(kx-3) * nz + kz] * inMG3[(kx-3) * nz + kz] + sinTheta[(kx+2) * nz + kz] * inMG3[(kx+2) * nz + kz]) +
                                c8_4 * (- sinTheta[(kx-4) * nz + kz] * inMG3[(kx-4) * nz + kz] + sinTheta[(kx+3) * nz + kz] * inMG3[(kx+3) * nz + kz]);

                        const Type stencilMG3B =
                                c8_1 * (- cosTheta[kx * nz + (kz-1)] * inMG3[kx * nz + (kz-1)] + cosTheta[kx * nz + (kz+0)] * inMG3[kx * nz + (kz+0)]) +
                                c8_2 * (- cosTheta[kx * nz + (kz-2)] * inMG3[kx * nz + (kz-2)] + cosTheta[kx * nz + (kz+1)] * inMG3[kx * nz + (kz+1)]) +
                                c8_3 * (- cosTheta[kx * nz + (kz-3)] * inMG3[kx * nz + (kz-3)] + cosTheta[kx * nz + (kz+2)] * inMG3[kx * nz + (kz+2)]) +
                                c8_4 * (- cosTheta[kx * nz + (kz-4)] * inMG3[kx * nz + (kz-4)] + cosTheta[kx * nz + (kz+3)] * inMG3[kx * nz + (kz+3)]);

                        const Type dMG1 = invDx * stencilMG1A - invDz * stencilMG1B;
                        const Type dMG3 = invDx * stencilMG3A + invDz * stencilMG3B;

                        // apply the 2nd order time update
                        const long k = kx * nz + kz;

                        const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                        pSpace[k] = dPG1 + dPG3;
                        mSpace[k] = dMG1 + dMG3;

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

                // kz = 0 -- at the free surface -- p = 0, dp = 0
                {
                    const Type dPG1 = 0;
                    const Type dPG3 = 0;

                    const Type dMG1 = 0;
                    const Type dMG3 = 0;

                    // apply the 2nd order time update
                    const long k = kx * nz + 0;

                    const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                    pSpace[k] = dPG1 + dPG3;
                    mSpace[k] = dMG1 + dMG3;

                    pOld[k] = dt2V2_B * pSpace[k] - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                    mOld[k] = dt2V2_B * mSpace[k] - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
                }

                // kz = 1 -- one cell below the free surface
                {
                    const Type stencilPG1A1 =
                            c8_1 * (- cosTheta[(kx-1) * nz + 1] * inPG1[(kx-1) * nz + 1] + cosTheta[(kx+0) * nz + 1] * inPG1[(kx+0) * nz + 1]) +
                            c8_2 * (- cosTheta[(kx-2) * nz + 1] * inPG1[(kx-2) * nz + 1] + cosTheta[(kx+1) * nz + 1] * inPG1[(kx+1) * nz + 1]) +
                            c8_3 * (- cosTheta[(kx-3) * nz + 1] * inPG1[(kx-3) * nz + 1] + cosTheta[(kx+2) * nz + 1] * inPG1[(kx+2) * nz + 1]) +
                            c8_4 * (- cosTheta[(kx-4) * nz + 1] * inPG1[(kx-4) * nz + 1] + cosTheta[(kx+3) * nz + 1] * inPG1[(kx+3) * nz + 1]);

                    const Type stencilPG1B1 =
                            c8_1 * (- sinTheta[kx * nz + 0] * inPG1[kx * nz + 0] + sinTheta[kx * nz + 1] * inPG1[kx * nz + 1]) +
                            c8_2 * (- sinTheta[kx * nz + 0] * inPG1[kx * nz + 0] + sinTheta[kx * nz + 2] * inPG1[kx * nz + 2]) +
                            c8_3 * (- sinTheta[kx * nz + 1] * inPG1[kx * nz + 1] + sinTheta[kx * nz + 3] * inPG1[kx * nz + 3]) +
                            c8_4 * (- sinTheta[kx * nz + 2] * inPG1[kx * nz + 2] + sinTheta[kx * nz + 4] * inPG1[kx * nz + 4]);

                    const Type stencilPG3A1 =
                            c8_1 * (- sinTheta[(kx-1) * nz + 1] * inPG3[(kx-1) * nz + 1] + sinTheta[(kx+0) * nz + 1] * inPG3[(kx+0) * nz + 1]) +
                            c8_2 * (- sinTheta[(kx-2) * nz + 1] * inPG3[(kx-2) * nz + 1] + sinTheta[(kx+1) * nz + 1] * inPG3[(kx+1) * nz + 1]) +
                            c8_3 * (- sinTheta[(kx-3) * nz + 1] * inPG3[(kx-3) * nz + 1] + sinTheta[(kx+2) * nz + 1] * inPG3[(kx+2) * nz + 1]) +
                            c8_4 * (- sinTheta[(kx-4) * nz + 1] * inPG3[(kx-4) * nz + 1] + sinTheta[(kx+3) * nz + 1] * inPG3[(kx+3) * nz + 1]);

                    const Type stencilPG3B1 =
                            c8_1 * (- cosTheta[kx * nz + 0] * inPG3[kx * nz + 0] + cosTheta[kx * nz + 1] * inPG3[kx * nz + 1]) +
                            c8_2 * (- cosTheta[kx * nz + 0] * inPG3[kx * nz + 0] + cosTheta[kx * nz + 2] * inPG3[kx * nz + 2]) +
                            c8_3 * (- cosTheta[kx * nz + 1] * inPG3[kx * nz + 1] + cosTheta[kx * nz + 3] * inPG3[kx * nz + 3]) +
                            c8_4 * (- cosTheta[kx * nz + 2] * inPG3[kx * nz + 2] + cosTheta[kx * nz + 4] * inPG3[kx * nz + 4]);

                    const Type dPG1 = invDx * stencilPG1A1 - invDz * stencilPG1B1;
                    const Type dPG3 = invDx * stencilPG3A1 + invDz * stencilPG3B1;

                    const Type stencilMG1A1 =
                            c8_1 * (- cosTheta[(kx-1) * nz + 1] * inMG1[(kx-1) * nz + 1] + cosTheta[(kx+0) * nz + 1] * inMG1[(kx+0) * nz + 1]) +
                            c8_2 * (- cosTheta[(kx-2) * nz + 1] * inMG1[(kx-2) * nz + 1] + cosTheta[(kx+1) * nz + 1] * inMG1[(kx+1) * nz + 1]) +
                            c8_3 * (- cosTheta[(kx-3) * nz + 1] * inMG1[(kx-3) * nz + 1] + cosTheta[(kx+2) * nz + 1] * inMG1[(kx+2) * nz + 1]) +
                            c8_4 * (- cosTheta[(kx-4) * nz + 1] * inMG1[(kx-4) * nz + 1] + cosTheta[(kx+3) * nz + 1] * inMG1[(kx+3) * nz + 1]);

                    const Type stencilMG1B1 =
                            c8_1 * (- sinTheta[kx * nz + 0] * inMG1[kx * nz + 0] + sinTheta[kx * nz + 1] * inMG1[kx * nz + 1]) +
                            c8_2 * (- sinTheta[kx * nz + 0] * inMG1[kx * nz + 0] + sinTheta[kx * nz + 2] * inMG1[kx * nz + 2]) +
                            c8_3 * (- sinTheta[kx * nz + 1] * inMG1[kx * nz + 1] + sinTheta[kx * nz + 3] * inMG1[kx * nz + 3]) +
                            c8_4 * (- sinTheta[kx * nz + 2] * inMG1[kx * nz + 2] + sinTheta[kx * nz + 4] * inMG1[kx * nz + 4]);

                    const Type stencilMG3A1 =
                            c8_1 * (- sinTheta[(kx-1) * nz + 1] * inMG3[(kx-1) * nz + 1] + sinTheta[(kx+0) * nz + 1] * inMG3[(kx+0) * nz + 1]) +
                            c8_2 * (- sinTheta[(kx-2) * nz + 1] * inMG3[(kx-2) * nz + 1] + sinTheta[(kx+1) * nz + 1] * inMG3[(kx+1) * nz + 1]) +
                            c8_3 * (- sinTheta[(kx-3) * nz + 1] * inMG3[(kx-3) * nz + 1] + sinTheta[(kx+2) * nz + 1] * inMG3[(kx+2) * nz + 1]) +
                            c8_4 * (- sinTheta[(kx-4) * nz + 1] * inMG3[(kx-4) * nz + 1] + sinTheta[(kx+3) * nz + 1] * inMG3[(kx+3) * nz + 1]);

                    const Type stencilMG3B1 =
                            c8_1 * (- cosTheta[kx * nz + 0] * inMG3[kx * nz + 0] + cosTheta[kx * nz + 1] * inMG3[kx * nz + 1]) +
                            c8_2 * (- cosTheta[kx * nz + 0] * inMG3[kx * nz + 0] + cosTheta[kx * nz + 2] * inMG3[kx * nz + 2]) +
                            c8_3 * (- cosTheta[kx * nz + 1] * inMG3[kx * nz + 1] + cosTheta[kx * nz + 3] * inMG3[kx * nz + 3]) +
                            c8_4 * (- cosTheta[kx * nz + 2] * inMG3[kx * nz + 2] + cosTheta[kx * nz + 4] * inMG3[kx * nz + 4]);

                    const Type dMG1 = invDx * stencilMG1A1 - invDz * stencilMG1B1;
                    const Type dMG3 = invDx * stencilMG3A1 + invDz * stencilMG3B1;

                    // apply the 2nd order time update
                    const long k = kx * nz + 1;

                    const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                    pSpace[k] = dPG1 + dPG3;
                    mSpace[k] = dMG1 + dMG3;

                    pOld[k] = dt2V2_B * pSpace[k] - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                    mOld[k] = dt2V2_B * mSpace[k] - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
                }

                // kz = 2 -- two cells below the free surface
                {
                    const Type stencilPG1A2 =
                            c8_1 * (- cosTheta[(kx-1) * nz + 2] * inPG1[(kx-1) * nz + 2] + cosTheta[(kx+0) * nz + 2] * inPG1[(kx+0) * nz + 2]) +
                            c8_2 * (- cosTheta[(kx-2) * nz + 2] * inPG1[(kx-2) * nz + 2] + cosTheta[(kx+1) * nz + 2] * inPG1[(kx+1) * nz + 2]) +
                            c8_3 * (- cosTheta[(kx-3) * nz + 2] * inPG1[(kx-3) * nz + 2] + cosTheta[(kx+2) * nz + 2] * inPG1[(kx+2) * nz + 2]) +
                            c8_4 * (- cosTheta[(kx-4) * nz + 2] * inPG1[(kx-4) * nz + 2] + cosTheta[(kx+3) * nz + 2] * inPG1[(kx+3) * nz + 2]);

                    const Type stencilPG1B2 =
                            c8_1 * (- sinTheta[kx * nz + 1] * inPG1[kx * nz + 1] + sinTheta[kx * nz + 2] * inPG1[kx * nz + 2]) +
                            c8_2 * (- sinTheta[kx * nz + 0] * inPG1[kx * nz + 0] + sinTheta[kx * nz + 3] * inPG1[kx * nz + 3]) +
                            c8_3 * (- sinTheta[kx * nz + 0] * inPG1[kx * nz + 0] + sinTheta[kx * nz + 4] * inPG1[kx * nz + 4]) +
                            c8_4 * (- sinTheta[kx * nz + 1] * inPG1[kx * nz + 1] + sinTheta[kx * nz + 5] * inPG1[kx * nz + 5]);

                    const Type stencilPG3A2 =
                            c8_1 * (- sinTheta[(kx-1) * nz + 2] * inPG3[(kx-1) * nz + 2] + sinTheta[(kx+0) * nz + 2] * inPG3[(kx+0) * nz + 2]) +
                            c8_2 * (- sinTheta[(kx-2) * nz + 2] * inPG3[(kx-2) * nz + 2] + sinTheta[(kx+1) * nz + 2] * inPG3[(kx+1) * nz + 2]) +
                            c8_3 * (- sinTheta[(kx-3) * nz + 2] * inPG3[(kx-3) * nz + 2] + sinTheta[(kx+2) * nz + 2] * inPG3[(kx+2) * nz + 2]) +
                            c8_4 * (- sinTheta[(kx-4) * nz + 2] * inPG3[(kx-4) * nz + 2] + sinTheta[(kx+3) * nz + 2] * inPG3[(kx+3) * nz + 2]);

                    const Type stencilPG3B2 =
                            c8_1 * (- cosTheta[kx * nz + 1] * inPG3[kx * nz + 1] + cosTheta[kx * nz + 2] * inPG3[kx * nz + 2]) +
                            c8_2 * (- cosTheta[kx * nz + 0] * inPG3[kx * nz + 0] + cosTheta[kx * nz + 3] * inPG3[kx * nz + 3]) +
                            c8_3 * (- cosTheta[kx * nz + 0] * inPG3[kx * nz + 0] + cosTheta[kx * nz + 4] * inPG3[kx * nz + 4]) +
                            c8_4 * (- cosTheta[kx * nz + 1] * inPG3[kx * nz + 1] + cosTheta[kx * nz + 5] * inPG3[kx * nz + 5]);

                    const Type dPG1 = invDx * stencilPG1A2 - invDz * stencilPG1B2;
                    const Type dPG3 = invDx * stencilPG3A2 + invDz * stencilPG3B2;

                    const Type stencilMG1A2 =
                            c8_1 * (- cosTheta[(kx-1) * nz + 2] * inMG1[(kx-1) * nz + 2] + cosTheta[(kx+0) * nz + 2] * inMG1[(kx+0) * nz + 2]) +
                            c8_2 * (- cosTheta[(kx-2) * nz + 2] * inMG1[(kx-2) * nz + 2] + cosTheta[(kx+1) * nz + 2] * inMG1[(kx+1) * nz + 2]) +
                            c8_3 * (- cosTheta[(kx-3) * nz + 2] * inMG1[(kx-3) * nz + 2] + cosTheta[(kx+2) * nz + 2] * inMG1[(kx+2) * nz + 2]) +
                            c8_4 * (- cosTheta[(kx-4) * nz + 2] * inMG1[(kx-4) * nz + 2] + cosTheta[(kx+3) * nz + 2] * inMG1[(kx+3) * nz + 2]);

                    const Type stencilMG1B2 =
                            c8_1 * (- sinTheta[kx * nz + 1] * inMG1[kx * nz + 1] + sinTheta[kx * nz + 2] * inMG1[kx * nz + 2]) +
                            c8_2 * (- sinTheta[kx * nz + 0] * inMG1[kx * nz + 0] + sinTheta[kx * nz + 3] * inMG1[kx * nz + 3]) +
                            c8_3 * (- sinTheta[kx * nz + 0] * inMG1[kx * nz + 0] + sinTheta[kx * nz + 4] * inMG1[kx * nz + 4]) +
                            c8_4 * (- sinTheta[kx * nz + 1] * inMG1[kx * nz + 1] + sinTheta[kx * nz + 5] * inMG1[kx * nz + 5]);

                    const Type stencilMG3A2 =
                            c8_1 * (- sinTheta[(kx-1) * nz + 2] * inMG3[(kx-1) * nz + 2] + sinTheta[(kx+0) * nz + 2] * inMG3[(kx+0) * nz + 2]) +
                            c8_2 * (- sinTheta[(kx-2) * nz + 2] * inMG3[(kx-2) * nz + 2] + sinTheta[(kx+1) * nz + 2] * inMG3[(kx+1) * nz + 2]) +
                            c8_3 * (- sinTheta[(kx-3) * nz + 2] * inMG3[(kx-3) * nz + 2] + sinTheta[(kx+2) * nz + 2] * inMG3[(kx+2) * nz + 2]) +
                            c8_4 * (- sinTheta[(kx-4) * nz + 2] * inMG3[(kx-4) * nz + 2] + sinTheta[(kx+3) * nz + 2] * inMG3[(kx+3) * nz + 2]);

                    const Type stencilMG3B2 =
                            c8_1 * (- cosTheta[kx * nz + 1] * inMG3[kx * nz + 1] + cosTheta[kx * nz + 2] * inMG3[kx * nz + 2]) +
                            c8_2 * (- cosTheta[kx * nz + 0] * inMG3[kx * nz + 0] + cosTheta[kx * nz + 3] * inMG3[kx * nz + 3]) +
                            c8_3 * (- cosTheta[kx * nz + 0] * inMG3[kx * nz + 0] + cosTheta[kx * nz + 4] * inMG3[kx * nz + 4]) +
                            c8_4 * (- cosTheta[kx * nz + 1] * inMG3[kx * nz + 1] + cosTheta[kx * nz + 5] * inMG3[kx * nz + 5]);

                    const Type dMG1 = invDx * stencilMG1A2 - invDz * stencilMG1B2;
                    const Type dMG3 = invDx * stencilMG3A2 + invDz * stencilMG3B2;

                    // apply the 2nd order time update
                    const long k = kx * nz + 2;

                    const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                    pSpace[k] = dPG1 + dPG3;
                    mSpace[k] = dMG1 + dMG3;

                    pOld[k] = dt2V2_B * pSpace[k] - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                    mOld[k] = dt2V2_B * mSpace[k] - dtOmegaInvQ[k] * (mCur[k] - mOld[k]) - mOld[k] + 2 * mCur[k];
                }

                // kz = 3 -- three cells below the free surface
                {
                    const Type stencilPG1A3 =
                            c8_1 * (- cosTheta[(kx-1) * nz + 3] * inPG1[(kx-1) * nz + 3] + cosTheta[(kx+0) * nz + 3] * inPG1[(kx+0) * nz + 3]) +
                            c8_2 * (- cosTheta[(kx-2) * nz + 3] * inPG1[(kx-2) * nz + 3] + cosTheta[(kx+1) * nz + 3] * inPG1[(kx+1) * nz + 3]) +
                            c8_3 * (- cosTheta[(kx-3) * nz + 3] * inPG1[(kx-3) * nz + 3] + cosTheta[(kx+2) * nz + 3] * inPG1[(kx+2) * nz + 3]) +
                            c8_4 * (- cosTheta[(kx-4) * nz + 3] * inPG1[(kx-4) * nz + 3] + cosTheta[(kx+3) * nz + 3] * inPG1[(kx+3) * nz + 3]);

                    const Type stencilPG1B3 =
                            c8_1 * (- sinTheta[kx * nz + 2] * inPG1[kx * nz + 2] + sinTheta[kx * nz + 3] * inPG1[kx * nz + 3]) +
                            c8_2 * (- sinTheta[kx * nz + 1] * inPG1[kx * nz + 1] + sinTheta[kx * nz + 4] * inPG1[kx * nz + 4]) +
                            c8_3 * (- sinTheta[kx * nz + 0] * inPG1[kx * nz + 0] + sinTheta[kx * nz + 5] * inPG1[kx * nz + 5]) +
                            c8_4 * (- sinTheta[kx * nz + 0] * inPG1[kx * nz + 0] + sinTheta[kx * nz + 6] * inPG1[kx * nz + 6]);

                    const Type stencilPG3A3 =
                            c8_1 * (- sinTheta[(kx-1) * nz + 3] * inPG3[(kx-1) * nz + 3] + sinTheta[(kx+0) * nz + 3] * inPG3[(kx+0) * nz + 3]) +
                            c8_2 * (- sinTheta[(kx-2) * nz + 3] * inPG3[(kx-2) * nz + 3] + sinTheta[(kx+1) * nz + 3] * inPG3[(kx+1) * nz + 3]) +
                            c8_3 * (- sinTheta[(kx-3) * nz + 3] * inPG3[(kx-3) * nz + 3] + sinTheta[(kx+2) * nz + 3] * inPG3[(kx+2) * nz + 3]) +
                            c8_4 * (- sinTheta[(kx-4) * nz + 3] * inPG3[(kx-4) * nz + 3] + sinTheta[(kx+3) * nz + 3] * inPG3[(kx+3) * nz + 3]);

                    const Type stencilPG3B3 =
                            c8_1 * (- cosTheta[kx * nz + 2] * inPG3[kx * nz + 2] + cosTheta[kx * nz + 3] * inPG3[kx * nz + 3]) +
                            c8_2 * (- cosTheta[kx * nz + 1] * inPG3[kx * nz + 1] + cosTheta[kx * nz + 4] * inPG3[kx * nz + 4]) +
                            c8_3 * (- cosTheta[kx * nz + 0] * inPG3[kx * nz + 0] + cosTheta[kx * nz + 5] * inPG3[kx * nz + 5]) +
                            c8_4 * (- cosTheta[kx * nz + 0] * inPG3[kx * nz + 0] + cosTheta[kx * nz + 6] * inPG3[kx * nz + 6]);

                    const Type dPG1 = invDx * stencilPG1A3 - invDz * stencilPG1B3;
                    const Type dPG3 = invDx * stencilPG3A3 + invDz * stencilPG3B3;

                    const Type stencilMG1A3 =
                            c8_1 * (- cosTheta[(kx-1) * nz + 3] * inMG1[(kx-1) * nz + 3] + cosTheta[(kx+0) * nz + 3] * inMG1[(kx+0) * nz + 3]) +
                            c8_2 * (- cosTheta[(kx-2) * nz + 3] * inMG1[(kx-2) * nz + 3] + cosTheta[(kx+1) * nz + 3] * inMG1[(kx+1) * nz + 3]) +
                            c8_3 * (- cosTheta[(kx-3) * nz + 3] * inMG1[(kx-3) * nz + 3] + cosTheta[(kx+2) * nz + 3] * inMG1[(kx+2) * nz + 3]) +
                            c8_4 * (- cosTheta[(kx-4) * nz + 3] * inMG1[(kx-4) * nz + 3] + cosTheta[(kx+3) * nz + 3] * inMG1[(kx+3) * nz + 3]);

                    const Type stencilMG1B3 =
                            c8_1 * (- sinTheta[kx * nz + 2] * inMG1[kx * nz + 2] + sinTheta[kx * nz + 3] * inMG1[kx * nz + 3]) +
                            c8_2 * (- sinTheta[kx * nz + 1] * inMG1[kx * nz + 1] + sinTheta[kx * nz + 4] * inMG1[kx * nz + 4]) +
                            c8_3 * (- sinTheta[kx * nz + 0] * inMG1[kx * nz + 0] + sinTheta[kx * nz + 5] * inMG1[kx * nz + 5]) +
                            c8_4 * (- sinTheta[kx * nz + 0] * inMG1[kx * nz + 0] + sinTheta[kx * nz + 6] * inMG1[kx * nz + 6]);

                    const Type stencilMG3A3 =
                            c8_1 * (- sinTheta[(kx-1) * nz + 3] * inMG3[(kx-1) * nz + 3] + sinTheta[(kx+0) * nz + 3] * inMG3[(kx+0) * nz + 3]) +
                            c8_2 * (- sinTheta[(kx-2) * nz + 3] * inMG3[(kx-2) * nz + 3] + sinTheta[(kx+1) * nz + 3] * inMG3[(kx+1) * nz + 3]) +
                            c8_3 * (- sinTheta[(kx-3) * nz + 3] * inMG3[(kx-3) * nz + 3] + sinTheta[(kx+2) * nz + 3] * inMG3[(kx+2) * nz + 3]) +
                            c8_4 * (- sinTheta[(kx-4) * nz + 3] * inMG3[(kx-4) * nz + 3] + sinTheta[(kx+3) * nz + 3] * inMG3[(kx+3) * nz + 3]);

                    const Type stencilMG3B3 =
                            c8_1 * (- cosTheta[kx * nz + 2] * inMG3[kx * nz + 2] + cosTheta[kx * nz + 3] * inMG3[kx * nz + 3]) +
                            c8_2 * (- cosTheta[kx * nz + 1] * inMG3[kx * nz + 1] + cosTheta[kx * nz + 4] * inMG3[kx * nz + 4]) +
                            c8_3 * (- cosTheta[kx * nz + 0] * inMG3[kx * nz + 0] + cosTheta[kx * nz + 5] * inMG3[kx * nz + 5]) +
                            c8_4 * (- cosTheta[kx * nz + 0] * inMG3[kx * nz + 0] + cosTheta[kx * nz + 6] * inMG3[kx * nz + 6]);

                    const Type dMG1 = invDx * stencilMG1A3 - invDz * stencilMG1B3;
                    const Type dMG3 = invDx * stencilMG3A3 + invDz * stencilMG3B3;

                    // apply the 2nd order time update
                    const long k = kx * nz + 3;

                    const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                    pSpace[k] = dPG1 + dPG3;
                    mSpace[k] = dMG1 + dMG3;

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
    inline static void applyFirstDerivatives2D_TTI_PlusHalf(
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
            const Type * __restrict__ const inG1,
            const Type * __restrict__ const inG3,
            const float * __restrict__ const sinTheta,
            const float * __restrict__ const cosTheta,
            Type * __restrict__ outG1,
            Type * __restrict__ outG3,
            const long BX_2D,
            const long BZ_2D) {

        const long nx4 = nx - 4;
        const long nz4 = nz - 4;

        // zero output array
#pragma omp parallel for collapse(2) num_threads(nthread) schedule(static)
        for (long bx = 0; bx < nx; bx += BX_2D) {
            for (long bz = 0; bz < nz; bz += BZ_2D) { /* cache blocking */

                const long kxmax = MIN(bx + BX_2D, nx);
                const long kzmax = MIN(bz + BZ_2D, nz);

                for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                    for (long kz = bz; kz < kzmax; kz++) {
                        outG1[kx * nz + kz] = 0;
                        outG3[kx * nz + kz] = 0;
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

                        const Type stencilG1 =
                                c8_1 * (- inG1[(kx+0) * nz + kz] + inG1[(kx+1) * nz + kz]) +
                                c8_2 * (- inG1[(kx-1) * nz + kz] + inG1[(kx+2) * nz + kz]) +
                                c8_3 * (- inG1[(kx-2) * nz + kz] + inG1[(kx+3) * nz + kz]) +
                                c8_4 * (- inG1[(kx-3) * nz + kz] + inG1[(kx+4) * nz + kz]);

                        const Type stencilG3 =
                                c8_1 * (- inG3[kx * nz + (kz+0)] + inG3[kx * nz + (kz+1)]) +
                                c8_2 * (- inG3[kx * nz + (kz-1)] + inG3[kx * nz + (kz+2)]) +
                                c8_3 * (- inG3[kx * nz + (kz-2)] + inG3[kx * nz + (kz+3)]) +
                                c8_4 * (- inG3[kx * nz + (kz-3)] + inG3[kx * nz + (kz+4)]);

                        const Type dx = invDx * stencilG1;
                        const Type dz = invDz * stencilG3;

                        outG1[kx * nz + kz] = cosTheta[kx * nz + kz] * dx - sinTheta[kx * nz + kz] * dz;
                        outG3[kx * nz + kz] = sinTheta[kx * nz + kz] * dx + cosTheta[kx * nz + kz] * dz;
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
                {
                    const Type stencilG30 =
                            c8_1 * (- inG3[kx * nz + 0] + inG3[kx * nz + 1]) +
                            c8_2 * (+ inG3[kx * nz + 1] + inG3[kx * nz + 2]) +
                            c8_3 * (+ inG3[kx * nz + 2] + inG3[kx * nz + 3]) +
                            c8_4 * (+ inG3[kx * nz + 3] + inG3[kx * nz + 4]);

                    const Type dz0 = invDz * stencilG30;

                    outG1[kx * nz + 0] = - sinTheta[kx * nz + 0] * dz0;
                    outG3[kx * nz + 0] = + cosTheta[kx * nz + 0] * dz0;
                }

                // kz = 1 -- 1 1/2 cells below free surface for Z derivative, 1 cells below for X/Y derivative
                {
                    const Type stencilG11 =
                            c8_1 * (- inG1[(kx+0) * nz + 1] + inG1[(kx+1) * nz + 1]) +
                            c8_2 * (- inG1[(kx-1) * nz + 1] + inG1[(kx+2) * nz + 1]) +
                            c8_3 * (- inG1[(kx-2) * nz + 1] + inG1[(kx+3) * nz + 1]) +
                            c8_4 * (- inG1[(kx-3) * nz + 1] + inG1[(kx+4) * nz + 1]);

                    const Type stencilG31 =
                            c8_1 * (- inG3[kx * nz + 1] + inG3[kx * nz + 2]) +
                            c8_2 * (- inG3[kx * nz + 0] + inG3[kx * nz + 3]) +
                            c8_3 * (+ inG3[kx * nz + 1] + inG3[kx * nz + 4]) +
                            c8_4 * (+ inG3[kx * nz + 2] + inG3[kx * nz + 5]);

                    const Type dx1 = invDx * stencilG11;
                    const Type dz1 = invDz * stencilG31;

                    outG1[kx * nz + 1] = cosTheta[kx * nz + 1] * dx1 - sinTheta[kx * nz + 1] * dz1;
                    outG3[kx * nz + 1] = sinTheta[kx * nz + 1] * dx1 + cosTheta[kx * nz + 1] * dz1;
                }

                // kz = 2 -- 2 1/2 cells below free surface for Z derivative, 2 cells below for X/Y derivative
                {
                    const Type stencilG12 =
                            c8_1 * (- inG1[(kx+0) * nz + 2] + inG1[(kx+1) * nz + 2]) +
                            c8_2 * (- inG1[(kx-1) * nz + 2] + inG1[(kx+2) * nz + 2]) +
                            c8_3 * (- inG1[(kx-2) * nz + 2] + inG1[(kx+3) * nz + 2]) +
                            c8_4 * (- inG1[(kx-3) * nz + 2] + inG1[(kx+4) * nz + 2]);

                    const Type stencilG32 =
                            c8_1 * (- inG3[kx * nz + 2] + inG3[kx * nz + 3]) +
                            c8_2 * (- inG3[kx * nz + 1] + inG3[kx * nz + 4]) +
                            c8_3 * (- inG3[kx * nz + 0] + inG3[kx * nz + 5]) +
                            c8_4 * (+ inG3[kx * nz + 1] + inG3[kx * nz + 6]);

                    const Type dx2 = invDx * stencilG12;
                    const Type dz2 = invDz * stencilG32;

                    outG1[kx * nz + 2] = cosTheta[kx * nz + 2] * dx2 - sinTheta[kx * nz + 2] * dz2;
                    outG3[kx * nz + 2] = sinTheta[kx * nz + 2] * dx2 + cosTheta[kx * nz + 2] * dz2;
                }

                // kz = 3 -- 3 1/2 cells below free surface for Z derivative, 3 cells below for X/Y derivative
                {
                    const Type stencilG13 =
                            c8_1 * (- inG1[(kx+0) * nz + 3] + inG1[(kx+1) * nz + 3]) +
                            c8_2 * (- inG1[(kx-1) * nz + 3] + inG1[(kx+2) * nz + 3]) +
                            c8_3 * (- inG1[(kx-2) * nz + 3] + inG1[(kx+3) * nz + 3]) +
                            c8_4 * (- inG1[(kx-3) * nz + 3] + inG1[(kx+4) * nz + 3]);

                    const Type stencilG33 =
                            c8_1 * (- inG3[kx * nz + 3] + inG3[kx * nz + 4]) +
                            c8_2 * (- inG3[kx * nz + 2] + inG3[kx * nz + 5]) +
                            c8_3 * (- inG3[kx * nz + 1] + inG3[kx * nz + 6]) +
                            c8_4 * (- inG3[kx * nz + 0] + inG3[kx * nz + 7]);

                    const Type dx3 = invDx * stencilG13;
                    const Type dz3 = invDz * stencilG33;

                    outG1[kx * nz + 3] = cosTheta[kx * nz + 3] * dx3 - sinTheta[kx * nz + 3] * dz3;
                    outG3[kx * nz + 3] = sinTheta[kx * nz + 3] * dx3 + cosTheta[kx * nz + 3] * dz3;
                }
            }
        }
    }

    template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline static void applyFirstDerivatives2D_TTI_MinusHalf(
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
            const Type * __restrict__ const inG1,
            const Type * __restrict__ const inG3,
            const float * __restrict__ const sinTheta,
            const float * __restrict__ const cosTheta,
            Type * __restrict__ outG1,
            Type * __restrict__ outG3,
            const long BX_2D,
            const long BZ_2D) {

        const long nx4 = nx - 4;
        const long nz4 = nz - 4;

        // zero output array
#pragma omp parallel for collapse(2) num_threads(nthread) schedule(static)
        for (long bx = 0; bx < nx; bx += BX_2D) {
            for (long bz = 0; bz < nz; bz += BZ_2D) { /* cache blocking */

                const long kxmax = MIN(bx + BX_2D, nx);
                const long kzmax = MIN(bz + BZ_2D, nz);

                for (long kx = bx; kx < kxmax; kx++) {
#pragma omp simd
                    for (long kz = bz; kz < kzmax; kz++) {
                        outG1[kx * nz + kz] = 0;
                        outG3[kx * nz + kz] = 0;
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

                        const Type stencilG1A =
                                c8_1 * (- cosTheta[(kx-1) * nz + kz] * inG1[(kx-1) * nz + kz] + cosTheta[(kx+0) * nz + kz] * inG1[(kx+0) * nz + kz]) +
                                c8_2 * (- cosTheta[(kx-2) * nz + kz] * inG1[(kx-2) * nz + kz] + cosTheta[(kx+1) * nz + kz] * inG1[(kx+1) * nz + kz]) +
                                c8_3 * (- cosTheta[(kx-3) * nz + kz] * inG1[(kx-3) * nz + kz] + cosTheta[(kx+2) * nz + kz] * inG1[(kx+2) * nz + kz]) +
                                c8_4 * (- cosTheta[(kx-4) * nz + kz] * inG1[(kx-4) * nz + kz] + cosTheta[(kx+3) * nz + kz] * inG1[(kx+3) * nz + kz]);

                        const Type stencilG1B =
                                c8_1 * (- sinTheta[kx * nz + (kz-1)] * inG1[kx * nz + (kz-1)] + sinTheta[kx * nz + (kz+0)] * inG1[kx * nz + (kz+0)]) +
                                c8_2 * (- sinTheta[kx * nz + (kz-2)] * inG1[kx * nz + (kz-2)] + sinTheta[kx * nz + (kz+1)] * inG1[kx * nz + (kz+1)]) +
                                c8_3 * (- sinTheta[kx * nz + (kz-3)] * inG1[kx * nz + (kz-3)] + sinTheta[kx * nz + (kz+2)] * inG1[kx * nz + (kz+2)]) +
                                c8_4 * (- sinTheta[kx * nz + (kz-4)] * inG1[kx * nz + (kz-4)] + sinTheta[kx * nz + (kz+3)] * inG1[kx * nz + (kz+3)]);

                        const Type stencilG3A =
                                c8_1 * (- sinTheta[(kx-1) * nz + kz] * inG3[(kx-1) * nz + kz] + sinTheta[(kx+0) * nz + kz] * inG3[(kx+0) * nz + kz]) +
                                c8_2 * (- sinTheta[(kx-2) * nz + kz] * inG3[(kx-2) * nz + kz] + sinTheta[(kx+1) * nz + kz] * inG3[(kx+1) * nz + kz]) +
                                c8_3 * (- sinTheta[(kx-3) * nz + kz] * inG3[(kx-3) * nz + kz] + sinTheta[(kx+2) * nz + kz] * inG3[(kx+2) * nz + kz]) +
                                c8_4 * (- sinTheta[(kx-4) * nz + kz] * inG3[(kx-4) * nz + kz] + sinTheta[(kx+3) * nz + kz] * inG3[(kx+3) * nz + kz]);

                        const Type stencilG3B =
                                c8_1 * (- cosTheta[kx * nz + (kz-1)] * inG3[kx * nz + (kz-1)] + cosTheta[kx * nz + (kz+0)] * inG3[kx * nz + (kz+0)]) +
                                c8_2 * (- cosTheta[kx * nz + (kz-2)] * inG3[kx * nz + (kz-2)] + cosTheta[kx * nz + (kz+1)] * inG3[kx * nz + (kz+1)]) +
                                c8_3 * (- cosTheta[kx * nz + (kz-3)] * inG3[kx * nz + (kz-3)] + cosTheta[kx * nz + (kz+2)] * inG3[kx * nz + (kz+2)]) +
                                c8_4 * (- cosTheta[kx * nz + (kz-4)] * inG3[kx * nz + (kz-4)] + cosTheta[kx * nz + (kz+3)] * inG3[kx * nz + (kz+3)]);

                        outG1[kx * nz + kz] = invDx * stencilG1A - invDz * stencilG1B;
                        outG3[kx * nz + kz] = invDx * stencilG3A + invDz * stencilG3B;
                    }
                }
            }
        }

        // roll on free surface
        if (freeSurface) {
#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long kx = 4; kx < nx4; kx++) {

                // kz = 0 -- at the free surface -- p = 0, dp = 0
                outG1[kx * nz + 0] = 0;
                outG3[kx * nz + 0] = 0;

                // kz = 1 -- one cell below the free surface
                {
                    const Type stencilG1A1 =
                            c8_1 * (- cosTheta[(kx-1) * nz + 1] * inG1[(kx-1) * nz + 1] + cosTheta[(kx+0) * nz + 1] * inG1[(kx+0) * nz + 1]) +
                            c8_2 * (- cosTheta[(kx-2) * nz + 1] * inG1[(kx-2) * nz + 1] + cosTheta[(kx+1) * nz + 1] * inG1[(kx+1) * nz + 1]) +
                            c8_3 * (- cosTheta[(kx-3) * nz + 1] * inG1[(kx-3) * nz + 1] + cosTheta[(kx+2) * nz + 1] * inG1[(kx+2) * nz + 1]) +
                            c8_4 * (- cosTheta[(kx-4) * nz + 1] * inG1[(kx-4) * nz + 1] + cosTheta[(kx+3) * nz + 1] * inG1[(kx+3) * nz + 1]);

                    const Type stencilG1B1 =
                            c8_1 * (- sinTheta[kx * nz + 0] * inG1[kx * nz + 0] + sinTheta[kx * nz + 1] * inG1[kx * nz + 1]) +
                            c8_2 * (- sinTheta[kx * nz + 0] * inG1[kx * nz + 0] + sinTheta[kx * nz + 2] * inG1[kx * nz + 2]) +
                            c8_3 * (- sinTheta[kx * nz + 1] * inG1[kx * nz + 1] + sinTheta[kx * nz + 3] * inG1[kx * nz + 3]) +
                            c8_4 * (- sinTheta[kx * nz + 2] * inG1[kx * nz + 2] + sinTheta[kx * nz + 4] * inG1[kx * nz + 4]);

                    const Type stencilG3A1 =
                            c8_1 * (- sinTheta[(kx-1) * nz + 1] * inG3[(kx-1) * nz + 1] + sinTheta[(kx+0) * nz + 1] * inG3[(kx+0) * nz + 1]) +
                            c8_2 * (- sinTheta[(kx-2) * nz + 1] * inG3[(kx-2) * nz + 1] + sinTheta[(kx+1) * nz + 1] * inG3[(kx+1) * nz + 1]) +
                            c8_3 * (- sinTheta[(kx-3) * nz + 1] * inG3[(kx-3) * nz + 1] + sinTheta[(kx+2) * nz + 1] * inG3[(kx+2) * nz + 1]) +
                            c8_4 * (- sinTheta[(kx-4) * nz + 1] * inG3[(kx-4) * nz + 1] + sinTheta[(kx+3) * nz + 1] * inG3[(kx+3) * nz + 1]);

                    const Type stencilG3B1 =
                            c8_1 * (- cosTheta[kx * nz + 0] * inG3[kx * nz + 0] + cosTheta[kx * nz + 1] * inG3[kx * nz + 1]) +
                            c8_2 * (- cosTheta[kx * nz + 0] * inG3[kx * nz + 0] + cosTheta[kx * nz + 2] * inG3[kx * nz + 2]) +
                            c8_3 * (- cosTheta[kx * nz + 1] * inG3[kx * nz + 1] + cosTheta[kx * nz + 3] * inG3[kx * nz + 3]) +
                            c8_4 * (- cosTheta[kx * nz + 2] * inG3[kx * nz + 2] + cosTheta[kx * nz + 4] * inG3[kx * nz + 4]);

                    outG1[kx * nz + 1] = invDx * stencilG1A1 - invDz * stencilG1B1;
                    outG3[kx * nz + 1] = invDx * stencilG3A1 + invDz * stencilG3B1;
                }

                // kz = 2 -- two cells below the free surface
                {
                    const Type stencilG1A2 =
                            c8_1 * (- cosTheta[(kx-1) * nz + 2] * inG1[(kx-1) * nz + 2] + cosTheta[(kx+0) * nz + 2] * inG1[(kx+0) * nz + 2]) +
                            c8_2 * (- cosTheta[(kx-2) * nz + 2] * inG1[(kx-2) * nz + 2] + cosTheta[(kx+1) * nz + 2] * inG1[(kx+1) * nz + 2]) +
                            c8_3 * (- cosTheta[(kx-3) * nz + 2] * inG1[(kx-3) * nz + 2] + cosTheta[(kx+2) * nz + 2] * inG1[(kx+2) * nz + 2]) +
                            c8_4 * (- cosTheta[(kx-4) * nz + 2] * inG1[(kx-4) * nz + 2] + cosTheta[(kx+3) * nz + 2] * inG1[(kx+3) * nz + 2]);

                    const Type stencilG1B2 =
                            c8_1 * (- sinTheta[kx * nz + 1] * inG1[kx * nz + 1] + sinTheta[kx * nz + 2] * inG1[kx * nz + 2]) +
                            c8_2 * (- sinTheta[kx * nz + 0] * inG1[kx * nz + 0] + sinTheta[kx * nz + 3] * inG1[kx * nz + 3]) +
                            c8_3 * (- sinTheta[kx * nz + 0] * inG1[kx * nz + 0] + sinTheta[kx * nz + 4] * inG1[kx * nz + 4]) +
                            c8_4 * (- sinTheta[kx * nz + 1] * inG1[kx * nz + 1] + sinTheta[kx * nz + 5] * inG1[kx * nz + 5]);

                    const Type stencilG3A2 =
                            c8_1 * (- sinTheta[(kx-1) * nz + 2] * inG3[(kx-1) * nz + 2] + sinTheta[(kx+0) * nz + 2] * inG3[(kx+0) * nz + 2]) +
                            c8_2 * (- sinTheta[(kx-2) * nz + 2] * inG3[(kx-2) * nz + 2] + sinTheta[(kx+1) * nz + 2] * inG3[(kx+1) * nz + 2]) +
                            c8_3 * (- sinTheta[(kx-3) * nz + 2] * inG3[(kx-3) * nz + 2] + sinTheta[(kx+2) * nz + 2] * inG3[(kx+2) * nz + 2]) +
                            c8_4 * (- sinTheta[(kx-4) * nz + 2] * inG3[(kx-4) * nz + 2] + sinTheta[(kx+3) * nz + 2] * inG3[(kx+3) * nz + 2]);

                    const Type stencilG3B2 =
                            c8_1 * (- cosTheta[kx * nz + 1] * inG3[kx * nz + 1] + cosTheta[kx * nz + 2] * inG3[kx * nz + 2]) +
                            c8_2 * (- cosTheta[kx * nz + 0] * inG3[kx * nz + 0] + cosTheta[kx * nz + 3] * inG3[kx * nz + 3]) +
                            c8_3 * (- cosTheta[kx * nz + 0] * inG3[kx * nz + 0] + cosTheta[kx * nz + 4] * inG3[kx * nz + 4]) +
                            c8_4 * (- cosTheta[kx * nz + 1] * inG3[kx * nz + 1] + cosTheta[kx * nz + 5] * inG3[kx * nz + 5]);

                    outG1[kx * nz + 2] = invDx * stencilG1A2 - invDz * stencilG1B2;
                    outG3[kx * nz + 2] = invDx * stencilG3A2 + invDz * stencilG3B2;
                }

                // kz = 3 -- three cells below the free surface
                {
                    const Type stencilG1A3 =
                            c8_1 * (- cosTheta[(kx-1) * nz + 3] * inG1[(kx-1) * nz + 3] + cosTheta[(kx+0) * nz + 3] * inG1[(kx+0) * nz + 3]) +
                            c8_2 * (- cosTheta[(kx-2) * nz + 3] * inG1[(kx-2) * nz + 3] + cosTheta[(kx+1) * nz + 3] * inG1[(kx+1) * nz + 3]) +
                            c8_3 * (- cosTheta[(kx-3) * nz + 3] * inG1[(kx-3) * nz + 3] + cosTheta[(kx+2) * nz + 3] * inG1[(kx+2) * nz + 3]) +
                            c8_4 * (- cosTheta[(kx-4) * nz + 3] * inG1[(kx-4) * nz + 3] + cosTheta[(kx+3) * nz + 3] * inG1[(kx+3) * nz + 3]);

                    const Type stencilG1B3 =
                            c8_1 * (- sinTheta[kx * nz + 2] * inG1[kx * nz + 2] + sinTheta[kx * nz + 3] * inG1[kx * nz + 3]) +
                            c8_2 * (- sinTheta[kx * nz + 1] * inG1[kx * nz + 1] + sinTheta[kx * nz + 4] * inG1[kx * nz + 4]) +
                            c8_3 * (- sinTheta[kx * nz + 0] * inG1[kx * nz + 0] + sinTheta[kx * nz + 5] * inG1[kx * nz + 5]) +
                            c8_4 * (- sinTheta[kx * nz + 0] * inG1[kx * nz + 0] + sinTheta[kx * nz + 6] * inG1[kx * nz + 6]);

                    const Type stencilG3A3 =
                            c8_1 * (- sinTheta[(kx-1) * nz + 3] * inG3[(kx-1) * nz + 3] + sinTheta[(kx+0) * nz + 3] * inG3[(kx+0) * nz + 3]) +
                            c8_2 * (- sinTheta[(kx-2) * nz + 3] * inG3[(kx-2) * nz + 3] + sinTheta[(kx+1) * nz + 3] * inG3[(kx+1) * nz + 3]) +
                            c8_3 * (- sinTheta[(kx-3) * nz + 3] * inG3[(kx-3) * nz + 3] + sinTheta[(kx+2) * nz + 3] * inG3[(kx+2) * nz + 3]) +
                            c8_4 * (- sinTheta[(kx-4) * nz + 3] * inG3[(kx-4) * nz + 3] + sinTheta[(kx+3) * nz + 3] * inG3[(kx+3) * nz + 3]);

                    const Type stencilG3B3 =
                            c8_1 * (- cosTheta[kx * nz + 2] * inG3[kx * nz + 2] + cosTheta[kx * nz + 3] * inG3[kx * nz + 3]) +
                            c8_2 * (- cosTheta[kx * nz + 1] * inG3[kx * nz + 1] + cosTheta[kx * nz + 4] * inG3[kx * nz + 4]) +
                            c8_3 * (- cosTheta[kx * nz + 0] * inG3[kx * nz + 0] + cosTheta[kx * nz + 5] * inG3[kx * nz + 5]) +
                            c8_4 * (- cosTheta[kx * nz + 0] * inG3[kx * nz + 0] + cosTheta[kx * nz + 6] * inG3[kx * nz + 6]);

                    outG1[kx * nz + 3] = invDx * stencilG1A3 - invDz * stencilG1B3;
                    outG3[kx * nz + 3] = invDx * stencilG3A3 + invDz * stencilG3B3;
                }
            }
        }
    }

};

#endif
