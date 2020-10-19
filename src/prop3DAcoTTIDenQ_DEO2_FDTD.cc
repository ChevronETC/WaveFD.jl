#include "prop3DAcoTTIDenQ_DEO2_FDTD.h"
#include "absorb.h"

extern "C" {

void *Prop3DAcoTTIDenQ_DEO2_FDTD_alloc(
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

    Prop3DAcoTTIDenQ_DEO2_FDTD *p = new Prop3DAcoTTIDenQ_DEO2_FDTD(
        freeSurface, nthread, nx, ny, nz, nsponge, dx, dy, dz, dt, nbx, nby, nbz);

    return (void*) p;
}

void Prop3DAcoTTIDenQ_DEO2_FDTD_free(void* p) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    delete pc;
}

void Prop3DAcoTTIDenQ_DEO2_FDTD_SetupDtOmegaInvQ(void *p, float freqQ, float qMin, float qInterior) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    long nx = pc->_nx;
    long ny = pc->_ny;
    long nz = pc->_nz;
    long nsponge = pc->_nsponge;
    long nthread = pc->_nthread;
    long fs = (pc->_freeSurface) ? 1 : 0;
    float dt = pc->_dt;
    setupDtOmegaInvQ_3D(fs, nx, ny, nz, nsponge, nthread, dt, freqQ, qMin, qInterior, pc->_dtOmegaInvQ);
}

void Prop3DAcoTTIDenQ_DEO2_FDTD_TimeStep(void *p) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    pc->timeStep();
}

void Prop3DAcoTTIDenQ_DEO2_FDTD_ScaleSpatialDerivatives(void *p) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    pc->scaleSpatialDerivatives();
}

void Prop3DAcoTTIDenQ_DEO2_FDTD_ForwardBornInjection_V(
        void *p, float *dVel, float *wavefieldDP, float *wavefieldDM) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    pc->forwardBornInjection_V(dVel, wavefieldDP, wavefieldDM);
}

void Prop3DAcoTTIDenQ_DEO2_FDTD_ForwardBornInjection_VEA(
        void *p, float *dVel, float *dEps, float *dEta,
        float *wavefieldP, float *wavefieldM, float *wavefieldDP, float *wavefieldDM) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    pc->forwardBornInjection_VEA(dVel, dEps, dEta, wavefieldP, wavefieldM, wavefieldDP, wavefieldDM);
}

void Prop3DAcoTTIDenQ_DEO2_FDTD_AdjointBornAccumulation_V(
        void *p, float *dVel, float *wavefieldDP, float *wavefieldDM) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    pc->adjointBornAccumulation_V(dVel, wavefieldDP, wavefieldDM);
}

void Prop3DAcoTTIDenQ_DEO2_FDTD_AdjointBornAccumulation_wavefieldsep_V(
        void *p, float *dVel, float *wavefieldDP, float *wavefieldDM, const long isFWI) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    pc->adjointBornAccumulation_wavefieldsep_V(dVel, wavefieldDP, wavefieldDM, isFWI);
}

void Prop3DAcoTTIDenQ_DEO2_FDTD_AdjointBornAccumulation_VEA(
        void *p, float *dVel, float *dEps, float *dEta,
        float *wavefieldP, float *wavefieldM, float *wavefieldDP, float *wavefieldDM) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    pc->adjointBornAccumulation_VEA(dVel, dEps, dEta, wavefieldP, wavefieldM, wavefieldDP, wavefieldDM);
}

long Prop3DAcoTTIDenQ_DEO2_FDTD_getNx(void *p) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_nx;
}

long Prop3DAcoTTIDenQ_DEO2_FDTD_getNy(void *p) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_ny;
}

long Prop3DAcoTTIDenQ_DEO2_FDTD_getNz(void *p) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_nz;
}

float * Prop3DAcoTTIDenQ_DEO2_FDTD_getV(void *p) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_v;
}

float * Prop3DAcoTTIDenQ_DEO2_FDTD_getEps(void *p) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_eps;
}

float * Prop3DAcoTTIDenQ_DEO2_FDTD_getEta(void *p) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_eta;
}

float * Prop3DAcoTTIDenQ_DEO2_FDTD_getSinTheta(void *p) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_sinTheta;
}

float * Prop3DAcoTTIDenQ_DEO2_FDTD_getCosTheta(void *p) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_cosTheta;
}

float * Prop3DAcoTTIDenQ_DEO2_FDTD_getSinPhi(void *p) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_sinPhi;
}

float * Prop3DAcoTTIDenQ_DEO2_FDTD_getCosPhi(void *p) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_cosPhi;
}

float * Prop3DAcoTTIDenQ_DEO2_FDTD_getB(void *p) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_b;
}

float * Prop3DAcoTTIDenQ_DEO2_FDTD_getF(void *p) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_f;
}

float * Prop3DAcoTTIDenQ_DEO2_FDTD_getDtOmegaInvQ(void *p) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_dtOmegaInvQ;
}

float * Prop3DAcoTTIDenQ_DEO2_FDTD_getPSpace(void *p) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_pSpace;
}

float * Prop3DAcoTTIDenQ_DEO2_FDTD_getMSpace(void *p) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_mSpace;
}

float * Prop3DAcoTTIDenQ_DEO2_FDTD_getPCur(void *p) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_pCur;
}

float * Prop3DAcoTTIDenQ_DEO2_FDTD_getPOld(void *p) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_pOld;
}

float * Prop3DAcoTTIDenQ_DEO2_FDTD_getMCur(void *p) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_mCur;
}

float * Prop3DAcoTTIDenQ_DEO2_FDTD_getMOld(void *p) {
    Prop3DAcoTTIDenQ_DEO2_FDTD *pc = reinterpret_cast<Prop3DAcoTTIDenQ_DEO2_FDTD *>(p);
    return pc->_mOld;
}

}
