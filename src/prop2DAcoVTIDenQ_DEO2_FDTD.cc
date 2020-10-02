#include "absorb.h"
#include "prop2DAcoVTIDenQ_DEO2_FDTD.h"

extern "C" {

void *Prop2DAcoVTIDenQ_DEO2_FDTD_alloc(
        long fs,
        long nthread,
        long nx,
        long nz,
        long nsponge,
        float dx,
        float dz,
        float dt,
        long nbx,
        long nbz) {
    bool freeSurface = (fs > 0) ? true : false;

    Prop2DAcoVTIDenQ_DEO2_FDTD *p = new Prop2DAcoVTIDenQ_DEO2_FDTD(
            freeSurface, nthread, nx, nz, nsponge, dx, dz, dt, nbx, nbz);

    return (void*) p;
}

void Prop2DAcoVTIDenQ_DEO2_FDTD_free(void* p) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    delete pc;
}

void Prop2DAcoVTIDenQ_DEO2_FDTD_SetupDtOmegaInvQ(void *p, float freqQ, float qMin, float qInterior) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    long nx = pc->_nx;
    long nz = pc->_nz;
    long nsponge = pc->_nsponge;
    long nthread = pc->_nthread;
    long fs = (pc->_freeSurface) ? 1 : 0;
    float dt = pc->_dt;
    setupDtOmegaInvQ_2D(fs, nx, nz, nsponge, nthread, dt, freqQ, qMin, qInterior, pc->_dtOmegaInvQ);
}

void Prop2DAcoVTIDenQ_DEO2_FDTD_TimeStep(void *p) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    pc->timeStep();
}

void Prop2DAcoVTIDenQ_DEO2_FDTD_TimeStepLinear(void *p) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    pc->timeStepLinear();
}

void Prop2DAcoVTIDenQ_DEO2_FDTD_ForwardBornInjection_VEA(
        void *p, float *dV, float *dE, float *dA,
        float *wavefieldP, float *wavefieldM, float *wavefieldDP, float *wavefieldDM) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    pc->forwardBornInjection_VEA(dV, dE, dA, wavefieldP, wavefieldM, wavefieldDP, wavefieldDM);
}

void Prop2DAcoVTIDenQ_DEO2_FDTD_AdjointBornAccumulation_VEA(
        void *p, float *dV, float *dE, float *dA,
        float *wavefieldP, float *wavefieldM, float *wavefieldDP, float *wavefieldDM) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    pc->adjointBornAccumulation_VEA(dV, dE, dA, wavefieldP, wavefieldM, wavefieldDP, wavefieldDM);
}

void Prop2DAcoVTIDenQ_DEO2_FDTD_ForwardBornInjection_V(
        void *p, float *dV,
        float *wavefieldDP, float *wavefieldDM) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    pc->forwardBornInjection_V(dV, wavefieldDP, wavefieldDM);
}

void Prop2DAcoVTIDenQ_DEO2_FDTD_AdjointBornAccumulation_V(
        void *p, float *dV,
        float *wavefieldDP, float *wavefieldDM) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    pc->adjointBornAccumulation_V(dV, wavefieldDP, wavefieldDM);
}

void Prop2DAcoVTIDenQ_DEO2_FDTD_ScaleSpatialDerivatives(void *p) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    pc->scaleSpatialDerivatives();
}

long Prop2DAcoVTIDenQ_DEO2_FDTD_getNx(void *p) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    return pc->_nx;
}

long Prop2DAcoVTIDenQ_DEO2_FDTD_getNz(void *p) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    return pc->_nz;
}

float * Prop2DAcoVTIDenQ_DEO2_FDTD_getV(void *p) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    return pc->_v;
}

float * Prop2DAcoVTIDenQ_DEO2_FDTD_getEps(void *p) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    return pc->_eps;
}

float * Prop2DAcoVTIDenQ_DEO2_FDTD_getEta(void *p) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    return pc->_eta;
}

float * Prop2DAcoVTIDenQ_DEO2_FDTD_getB(void *p) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    return pc->_b;
}

float * Prop2DAcoVTIDenQ_DEO2_FDTD_getF(void *p) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    return pc->_f;
}

float * Prop2DAcoVTIDenQ_DEO2_FDTD_getDtOmegaInvQ(void *p) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    return pc->_dtOmegaInvQ;
}

float * Prop2DAcoVTIDenQ_DEO2_FDTD_getPSpace(void *p) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    return pc->_pSpace;
}

float * Prop2DAcoVTIDenQ_DEO2_FDTD_getMSpace(void *p) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    return pc->_mSpace;
}

float * Prop2DAcoVTIDenQ_DEO2_FDTD_getPCur(void *p) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    return pc->_pCur;
}

float * Prop2DAcoVTIDenQ_DEO2_FDTD_getPOld(void *p) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    return pc->_pOld;
}

float * Prop2DAcoVTIDenQ_DEO2_FDTD_getMCur(void *p) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    return pc->_mCur;
}

float * Prop2DAcoVTIDenQ_DEO2_FDTD_getMOld(void *p) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    return pc->_mOld;
}

float * Prop2DAcoVTIDenQ_DEO2_FDTD_getTmpPx1(void *p) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    return pc->_tmpPx1;
}

float * Prop2DAcoVTIDenQ_DEO2_FDTD_getTmpPz1(void *p) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    return pc->_tmpPz1;
}

float * Prop2DAcoVTIDenQ_DEO2_FDTD_getTmpMx1(void *p) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    return pc->_tmpMx1;
}

float * Prop2DAcoVTIDenQ_DEO2_FDTD_getTmpMz1(void *p) {
    Prop2DAcoVTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoVTIDenQ_DEO2_FDTD *>(p);
    return pc->_tmpMz1;
}

}
