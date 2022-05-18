#ifndef ABSORB_H
#define ABSORB_H

#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.1415926535897
#define MIN(x,y) ((x)<(y)?(x):(y))

/**
 * Ported from ModelingQuasiPeriodic2D 2017.09.19
 * Log decay of Q from qInterior to qmin with approach to boundary
 *
 * 2018.05.24
 *  - dt is now rolled up in this field to save compute in the fused derivs + time update method
 *    invQ ==> dtOmegaInvQ
 */
void setupDtOmegaInvQ_2D(
        const long freeSurface,
        const long nx,
        const long nz,
        const long nsponge,
        const long nthread,
        const float dt,
        const float freqQ,
        const float qMin,
        const float qInterior,
        float *dtOmegaInvQ) {

    if (freqQ < FLT_EPSILON) {
        char msg[1000];
        sprintf(msg, "Error -- freqQ [%f] is too small!\n", freqQ);
        perror(msg);
        exit(EXIT_FAILURE);
    }
    // check for unphysical q values
    if ((qMin < FLT_EPSILON) && (qInterior < FLT_EPSILON)) {
        printf("Warning -- qMin and qMax unphysical, dtOmegaInvQ set to zero!\n");
        memset(dtOmegaInvQ, 0, sizeof(dtOmegaInvQ));
        return;
    }
    float *qprof = new float[nsponge];

    const double lqmin = log(qMin);
    const double lqmax = log(qInterior);

    for (long ksponge = 0; ksponge < nsponge; ksponge++) {
        const double dk = (double) (ksponge) / (double) (nsponge - 1);
        const double lq = lqmin + dk * (lqmax - lqmin);
        qprof[ksponge] = expf(lq);
    }

#pragma omp parallel for num_threads(nthread) schedule(guided)
    for (long kx = 0; kx < nx; kx++) {
#pragma omp simd
        for (long kz = 0; kz < nz; kz++) {
            const long ksx = MIN(kx, (nx - 1 - kx));
            const long ksz = (freeSurface) ? (nz - 1 - kz) : MIN(kz, (nz - 1 - kz));
            const long ksponge = MIN(ksx, ksz);

            dtOmegaInvQ[kx * nz + kz] = dt * 2.0 * PI * freqQ / qInterior;

            if (ksponge < nsponge) {
                dtOmegaInvQ[kx * nz + kz] = dt * 2.0 * PI * freqQ / qprof[ksponge];
            }
        }
    }

    delete[] qprof;
}

/**
 * Ported from ModelingQuasiPeriodic2D 2017.09.19
 * Log decay of Q from qInterior to qmin with approach to boundary
 *
 * 2018.05.24
 *  - dt is now rolled up in this field to save compute in the fused derivs + time update method
 *    invQ ==> dtOmegaInvQ
 */
void setupDtOmegaInvQ_3D(
        const long freeSurface,
        const long nx,
        const long ny,
        const long nz,
        const long nsponge,
        const long nthread,
        const float dt,
        const float freqQ,
        const float qMin,
        const float qInterior,
        float *dtOmegaInvQ) {

    if (freqQ < FLT_EPSILON) {
        char msg[1000];
        sprintf(msg, "Error -- freqQ [%f] is too small!\n", freqQ);
        perror(msg);
        exit(EXIT_FAILURE);
    }
    // check for unphysical q values
    if ((qMin < FLT_EPSILON) && (qInterior < FLT_EPSILON)) {
        printf("Warning -- qMin and qMax unphysical, dtOmegaInvQ set to zero!\n");
        memset(dtOmegaInvQ, 0, sizeof(dtOmegaInvQ));
        return;
    }
    const long nynz = ny * nz;

    float *qprof = new float[nsponge];

    const float qmin = qMin;
    const float qmax = qInterior;

    const float lqmin = log(qmin);
    const float lqmax = log(qmax);

    for (long ksponge = 0; ksponge < nsponge; ksponge++){
        const float dk = (float)(ksponge) / (float)(nsponge - 1);
        const float lq = lqmin + dk * (lqmax - lqmin);
        qprof[ksponge] = exp(lq);
    }

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (long kz = 0; kz < nz; kz++) {
        for (long kx = 0; kx < nx; kx++) {
            const long kxnynz = kx * nynz;

#pragma omp simd
            for (long ky = 0; ky < ny; ky++) {
                const long ksx = MIN(kx, (nx - 1 - kx));
                const long ksy = MIN(ky, (ny - 1 - ky));
                const long ksz = (freeSurface) ? (nz - 1 - kz) : MIN(kz, (nz - 1 - kz));
                const long ksponge = MIN(ksx, MIN(ksy, ksz));

                const long kynz = ky * nz;
                const long kxnynz_kynz = kxnynz + kynz;

                dtOmegaInvQ[kxnynz_kynz + kz] = dt * 2 * PI * freqQ / qInterior;

                if (ksponge < nsponge) {
                    dtOmegaInvQ[kxnynz_kynz + kz] = dt * 2 * PI * freqQ / qprof[ksponge];
                }
            }
        }
    }

    delete [] qprof;
}

#endif
