#include "absorb.h"
#include "prop3DAcoIsoDenQ_DEO2_FDTD.h"

extern "C" {

void *Prop3DAcoIsoDenQ_DEO2_FDTD_alloc(
        long fs,
        long nthread,
        long nx,
        long ny,
        long nz,
        long nsponge,
        float dx,
        float dy,
        float dz,
        float dt,
        long nbx,
        long nby,
        long nbz) {
    bool freeSurface = (fs > 0) ? true : false;

    Prop3DAcoIsoDenQ_DEO2_FDTD *p = new Prop3DAcoIsoDenQ_DEO2_FDTD(
        freeSurface, nthread, nx, ny, nz, nsponge, dx, dy, dz, dt, nbx, nby, nbz);

    return (void*) p;
}

void Prop3DAcoIsoDenQ_DEO2_FDTD_free(void* p) {
    Prop3DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoIsoDenQ_DEO2_FDTD *>(p);
    delete pc;
}

void Prop3DAcoIsoDenQ_DEO2_FDTD_SetupDtOmegaInvQ(void *p, float freqQ, float qMin, float qInterior) {
    Prop3DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoIsoDenQ_DEO2_FDTD *>(p);
    long nx = pc->_nx;
    long ny = pc->_ny;
    long nz = pc->_nz;
    long nsponge = pc->_nsponge;
    long nthread = pc->_nthread;
    long fs = (pc->_freeSurface) ? 1 : 0;
    float dt = pc->_dt;
    setupDtOmegaInvQ_3D(fs, nx, ny, nz, nsponge, nthread, dt, freqQ, qMin, qInterior, pc->_dtOmegaInvQ);
}

void Prop3DAcoIsoDenQ_DEO2_FDTD_TimeStep(void *p) {
    Prop3DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoIsoDenQ_DEO2_FDTD *>(p);
    pc->timeStep();
}

void Prop3DAcoIsoDenQ_DEO2_FDTD_ScaleSpatialDerivatives(void *p) {
    Prop3DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoIsoDenQ_DEO2_FDTD *>(p);
    pc->scaleSpatialDerivatives();
}

void Prop3DAcoIsoDenQ_DEO2_FDTD_ForwardBornInjection_V(
        void *p, float *dVel, float *wavefieldDP) {
    Prop3DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoIsoDenQ_DEO2_FDTD *>(p);
    pc->forwardBornInjection_V(dVel, wavefieldDP);
}

void Prop3DAcoIsoDenQ_DEO2_FDTD_ForwardBornInjection_VB(
        void *p, float *dVel, float *dBuoy, float *wavefieldP, float *wavefieldDP) {
    Prop3DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoIsoDenQ_DEO2_FDTD *>(p);
    pc->forwardBornInjection_VB(dVel, dBuoy, wavefieldP, wavefieldDP);
}

void Prop3DAcoIsoDenQ_DEO2_FDTD_ForwardBornInjection_B(
        void *p, float *dBuoy, float *wavefieldP, float *wavefieldDP) {
    Prop3DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoIsoDenQ_DEO2_FDTD *>(p);
    pc->forwardBornInjection_B(dBuoy, wavefieldP, wavefieldDP);
}

void Prop3DAcoIsoDenQ_DEO2_FDTD_AdjointBornAccumulation_V(
        void *p, float *dVel, float *wavefieldDP) {
    Prop3DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoIsoDenQ_DEO2_FDTD *>(p);
    pc->adjointBornAccumulation_V(dVel, wavefieldDP);
}

void Prop3DAcoIsoDenQ_DEO2_FDTD_AdjointBornAccumulation_VB(
        void *p, float *dVel, float *dBuoy, float *wavefieldP, float *wavefieldDP) {
    Prop3DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoIsoDenQ_DEO2_FDTD *>(p);
    pc->adjointBornAccumulation_VB(dVel, dBuoy, wavefieldP, wavefieldDP);
}

void Prop3DAcoIsoDenQ_DEO2_FDTD_AdjointBornAccumulation_B(
        void *p, float *dBuoy, float *wavefieldP, float *wavefieldDP) {
    Prop3DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoIsoDenQ_DEO2_FDTD *>(p);
    pc->adjointBornAccumulation_B(dBuoy, wavefieldP, wavefieldDP);
}

void Prop3DAcoIsoDenQ_DEO2_FDTD_AdjointBornAccumulation_wavefieldsep(
        void *p, float *dVel, float *wavefieldDP, const long isFWI) {
    Prop3DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoIsoDenQ_DEO2_FDTD *>(p);
    pc->adjointBornAccumulation_wavefieldsep(dVel, wavefieldDP, isFWI);
}

long Prop3DAcoIsoDenQ_DEO2_FDTD_getNx(void *p) {
    Prop3DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoIsoDenQ_DEO2_FDTD *>(p);
    return pc->_nx;
}

long Prop3DAcoIsoDenQ_DEO2_FDTD_getNy(void *p) {
    Prop3DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoIsoDenQ_DEO2_FDTD *>(p);
    return pc->_ny;
}

long Prop3DAcoIsoDenQ_DEO2_FDTD_getNz(void *p) {
    Prop3DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoIsoDenQ_DEO2_FDTD *>(p);
    return pc->_nz;
}

float * Prop3DAcoIsoDenQ_DEO2_FDTD_getV(void *p) {
    Prop3DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoIsoDenQ_DEO2_FDTD *>(p);
    return pc->_v;
}

float * Prop3DAcoIsoDenQ_DEO2_FDTD_getB(void *p) {
    Prop3DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoIsoDenQ_DEO2_FDTD *>(p);
    return pc->_b;
}

float * Prop3DAcoIsoDenQ_DEO2_FDTD_getDtOmegaInvQ(void *p) {
    Prop3DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoIsoDenQ_DEO2_FDTD *>(p);
    return pc->_dtOmegaInvQ;
}

float * Prop3DAcoIsoDenQ_DEO2_FDTD_getPSpace(void *p) {
    Prop3DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoIsoDenQ_DEO2_FDTD *>(p);
    return pc->_pSpace;
}

float * Prop3DAcoIsoDenQ_DEO2_FDTD_getPCur(void *p) {
    Prop3DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoIsoDenQ_DEO2_FDTD *>(p);
    return pc->_pCur;
}

float * Prop3DAcoIsoDenQ_DEO2_FDTD_getPOld(void *p) {
    Prop3DAcoIsoDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoIsoDenQ_DEO2_FDTD *>(p);
    return pc->_pOld;
}

}
