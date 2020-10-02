#include "absorb.h"
#include "prop2DAcoTTIDenQ_DEO2_FDTD.h"

extern "C" {

void *Prop2DAcoTTIDenQ_DEO2_FDTD_alloc(
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

    Prop2DAcoTTIDenQ_DEO2_FDTD *p = new Prop2DAcoTTIDenQ_DEO2_FDTD(
        freeSurface, nthread, nx, nz, nsponge, dx, dz, dt, nbx, nbz);

    return (void*) p;
}

void Prop2DAcoTTIDenQ_DEO2_FDTD_free(void* p) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    delete pc;
}

void Prop2DAcoTTIDenQ_DEO2_FDTD_SetupDtOmegaInvQ(void *p, float freqQ, float qMin, float qInterior) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    long nx = pc->_nx;
    long nz = pc->_nz;
    long nsponge = pc->_nsponge;
    long nthread = pc->_nthread;
    long fs = (pc->_freeSurface) ? 1 : 0;
    float dt = pc->_dt;
    setupDtOmegaInvQ_2D(fs, nx, nz, nsponge, nthread, dt, freqQ, qMin, qInterior, pc->_dtOmegaInvQ);
}

void Prop2DAcoTTIDenQ_DEO2_FDTD_TimeStep(void *p) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    pc->timeStep();
}

void Prop2DAcoTTIDenQ_DEO2_FDTD_ScaleSpatialDerivatives(void *p) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    pc->scaleSpatialDerivatives();
}

void Prop2DAcoTTIDenQ_DEO2_FDTD_ForwardBornInjection_V(void *p,
        float *dmodelV,
        float *wavefieldDP, float *wavefieldDM) {
	Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
	pc->forwardBornInjection_V(dmodelV, wavefieldDP, wavefieldDM);
}

void Prop2DAcoTTIDenQ_DEO2_FDTD_ForwardBornInjection_VEA(void *p,
        float *dmodelV, float *dmodelE, float *dmodelA,
        float *wavefieldP, float *wavefieldM, float *wavefieldDP, float *wavefieldDM) {
	Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
	pc->forwardBornInjection_VEA(dmodelV, dmodelE, dmodelA, wavefieldP, wavefieldM, wavefieldDP, wavefieldDM);
}

void Prop2DAcoTTIDenQ_DEO2_FDTD_AdjointBornAccumulation_V(void *p,
        float *dmodelV,
        float *wavefieldDP, float *wavefieldDM) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    pc->adjointBornAccumulation_V(dmodelV, wavefieldDP, wavefieldDM);
}

void Prop2DAcoTTIDenQ_DEO2_FDTD_AdjointBornAccumulation_VEA(void *p,
        float *dmodelV, float *dmodelE, float *dmodelA,
        float *wavefieldP, float *wavefieldM, float *wavefieldDP, float *wavefieldDM) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    pc->adjointBornAccumulation_VEA(dmodelV, dmodelE, dmodelA, wavefieldP, wavefieldM, wavefieldDP, wavefieldDM);
}

long Prop2DAcoTTIDenQ_DEO2_FDTD_getNx(void *p) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_nx;
}

long Prop2DAcoTTIDenQ_DEO2_FDTD_getNz(void *p) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_nz;
}

float * Prop2DAcoTTIDenQ_DEO2_FDTD_getV(void *p) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_v;
}

float * Prop2DAcoTTIDenQ_DEO2_FDTD_getEps(void *p) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_eps;
}

float * Prop2DAcoTTIDenQ_DEO2_FDTD_getEta(void *p) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_eta;
}

float * Prop2DAcoTTIDenQ_DEO2_FDTD_getSinTheta(void *p) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_sinTheta;
}

float * Prop2DAcoTTIDenQ_DEO2_FDTD_getCosTheta(void *p) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_cosTheta;
}

float * Prop2DAcoTTIDenQ_DEO2_FDTD_getB(void *p) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_b;
}

float * Prop2DAcoTTIDenQ_DEO2_FDTD_getF(void *p) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_f;
}

float * Prop2DAcoTTIDenQ_DEO2_FDTD_getDtOmegaInvQ(void *p) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_dtOmegaInvQ;
}

float * Prop2DAcoTTIDenQ_DEO2_FDTD_getPSpace(void *p) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_pSpace;
}

float * Prop2DAcoTTIDenQ_DEO2_FDTD_getMSpace(void *p) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_mSpace;
}

float * Prop2DAcoTTIDenQ_DEO2_FDTD_getPCur(void *p) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_pCur;
}

float * Prop2DAcoTTIDenQ_DEO2_FDTD_getPOld(void *p) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_pOld;
}

float * Prop2DAcoTTIDenQ_DEO2_FDTD_getMCur(void *p) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_mCur;
}

float * Prop2DAcoTTIDenQ_DEO2_FDTD_getMOld(void *p) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_mOld;
}

float * Prop2DAcoTTIDenQ_DEO2_FDTD_getTmpPg1(void *p) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_tmpPg1a;
}

float * Prop2DAcoTTIDenQ_DEO2_FDTD_getTmpPg3(void *p) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_tmpPg3a;
}

float * Prop2DAcoTTIDenQ_DEO2_FDTD_getTmpMg1(void *p) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_tmpMg1a;
}

float * Prop2DAcoTTIDenQ_DEO2_FDTD_getTmpMg3(void *p) {
    Prop2DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_tmpMg3a;
}

}
