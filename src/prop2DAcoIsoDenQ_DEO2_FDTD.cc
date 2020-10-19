#include "absorb.h"
#include "prop2DAcoIsoDenQ_DEO2_FDTD.h"

extern "C" {

void *Prop2DAcoIsoDenQ_DEO2_FDTD_alloc(
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

    Prop2DAcoIsoDenQ_DEO2_FDTD *p = new Prop2DAcoIsoDenQ_DEO2_FDTD(
        freeSurface, nthread, nx, nz, nsponge, dx, dz, dt, nbx, nbz);

    return (void*) p;
}

void Prop2DAcoIsoDenQ_DEO2_FDTD_free(void* p) {
    Prop2DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoIsoDenQ_DEO2_FDTD *>(p);
    delete pc;
}

void Prop2DAcoIsoDenQ_DEO2_FDTD_SetupDtOmegaInvQ(void *p, float freqQ, float qMin, float qInterior) {
    Prop2DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoIsoDenQ_DEO2_FDTD *>(p);
    long nx = pc->_nx;
    long nz = pc->_nz;
    long nsponge = pc->_nsponge;
    long nthread = pc->_nthread;
    long fs = (pc->_freeSurface) ? 1 : 0;
    float dt = pc->_dt;
    setupDtOmegaInvQ_2D(fs, nx, nz, nsponge, nthread, dt, freqQ, qMin, qInterior, pc->_dtOmegaInvQ);
}

void Prop2DAcoIsoDenQ_DEO2_FDTD_TimeStep(void *p) {
    Prop2DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoIsoDenQ_DEO2_FDTD *>(p);
    pc->timeStep();
}

void Prop2DAcoIsoDenQ_DEO2_FDTD_ScaleSpatialDerivatives(void *p) {
    Prop2DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoIsoDenQ_DEO2_FDTD *>(p);
    pc->scaleSpatialDerivatives();
}

void Prop2DAcoIsoDenQ_DEO2_FDTD_ForwardBornInjection(
        void *p, float *dVel, float *wavefieldDP) {
    Prop2DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoIsoDenQ_DEO2_FDTD *>(p);
    pc->forwardBornInjection(dVel, wavefieldDP);
}

void Prop2DAcoIsoDenQ_DEO2_FDTD_AdjointBornAccumulation(
        void *p, float *dVel, float *wavefieldDP) {
    Prop2DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoIsoDenQ_DEO2_FDTD *>(p);
    pc->adjointBornAccumulation(dVel, wavefieldDP);
}

void Prop2DAcoIsoDenQ_DEO2_FDTD_AdjointBornAccumulation_wavefieldsep(
        void *p, float *dVel, float *wavefieldDP, const long isFWI) {
    Prop2DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoIsoDenQ_DEO2_FDTD *>(p);
    pc->adjointBornAccumulation_wavefieldsep(dVel, wavefieldDP, isFWI);
}

long Prop2DAcoIsoDenQ_DEO2_FDTD_getNx(void *p) {
    Prop2DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoIsoDenQ_DEO2_FDTD *>(p);
    return pc->_nx;
}

long Prop2DAcoIsoDenQ_DEO2_FDTD_getNz(void *p) {
    Prop2DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoIsoDenQ_DEO2_FDTD *>(p);
    return pc->_nz;
}

float * Prop2DAcoIsoDenQ_DEO2_FDTD_getV(void *p) {
    Prop2DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoIsoDenQ_DEO2_FDTD *>(p);
    return pc->_v;
}

float * Prop2DAcoIsoDenQ_DEO2_FDTD_getB(void *p) {
    Prop2DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoIsoDenQ_DEO2_FDTD *>(p);
    return pc->_b;
}

float * Prop2DAcoIsoDenQ_DEO2_FDTD_getDtOmegaInvQ(void *p) {
    Prop2DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoIsoDenQ_DEO2_FDTD *>(p);
    return pc->_dtOmegaInvQ;
}

float * Prop2DAcoIsoDenQ_DEO2_FDTD_getPSpace(void *p) {
    Prop2DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoIsoDenQ_DEO2_FDTD *>(p);
    return pc->_pSpace;
}

float * Prop2DAcoIsoDenQ_DEO2_FDTD_getPCur(void *p) {
    Prop2DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoIsoDenQ_DEO2_FDTD *>(p);
    return pc->_pCur;
}

float * Prop2DAcoIsoDenQ_DEO2_FDTD_getPOld(void *p) {
    Prop2DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop2DAcoIsoDenQ_DEO2_FDTD *>(p);
    return pc->_pOld;
}

}
