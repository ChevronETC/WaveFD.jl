using  Random, Test, WaveFD, LinearAlgebra

nthreads = 12
nz,ny,nx = 101, 200, 301
dz,dy,dx = 25.0f0,25.0f0, 25.0f0
nsponge = 40
freqQ = 5.0f0
qMin = 0.1f0
qInterior = 100.0f0
dtmod = 0.002f0

@testset "2D, serial, freesurface=$(freesurface)" for freesurface in (false,true)
    w1 = zeros(Float32,nz,nx)
    w2 = zeros(Float32,nz,nx)
    WaveFD.setup_q_profile_2D_serial!(w1, freesurface, nsponge, dtmod, freqQ, qMin, qInterior)
    w2 .= WaveFD.DtOmegaInvQ(Prop2DAcoIsoDenQ_DEO2_FDTD(;
        freesurface=freesurface, nz=nz, nx=nx, dz=dz, dx=dx, nsponge=nsponge, nthreads=nthreads,
        dt=dtmod, freqQ=freqQ, qMin=qMin, qInterior=qInterior))
    @test w1 ≈ w2
end

@testset "2D, threaded, freesurface=$(freesurface)" for freesurface in (false,true)
    w1 = zeros(Float32,nz,nx)
    w2 = zeros(Float32,nz,nx)
    WaveFD.setup_q_profile_2D_threaded!(w1, freesurface, nsponge, dtmod, freqQ, qMin, qInterior)
    w2 .= WaveFD.DtOmegaInvQ(Prop2DAcoIsoDenQ_DEO2_FDTD(;
        freesurface=freesurface, nz=nz, nx=nx, dz=dz, dx=dx, nsponge=nsponge, nthreads=nthreads,
        dt=dtmod, freqQ=freqQ, qMin=qMin, qInterior=qInterior))
    @test w1 ≈ w2
end

@testset "2D, serial == threaded, freesurface=$(freesurface)" for freesurface in (false,true)
    w1 = zeros(Float32,nz,nx)
    w2 = zeros(Float32,nz,nx)
    WaveFD.setup_q_profile_2D_serial!(w1, freesurface, nsponge, dtmod, freqQ, qMin, qInterior)
    WaveFD.setup_q_profile_2D_threaded!(w2, freesurface, nsponge, dtmod, freqQ, qMin, qInterior)
    @test w1 ≈ w2
end

@testset "3D, serial, freesurface=$(freesurface)" for freesurface in (false,true)
    w1 = zeros(Float32,nz,ny,nx)
    w2 = zeros(Float32,nz,ny,nx)
    WaveFD.setup_q_profile_3D_serial!(w1, freesurface, nsponge, dtmod, freqQ, qMin, qInterior)
    w2 .= WaveFD.DtOmegaInvQ(Prop3DAcoIsoDenQ_DEO2_FDTD(;
        freesurface=freesurface, nz=nz, ny=ny, nx=nx, dz=dz, dy=dy, dx=dx, nsponge=nsponge, nthreads=nthreads,
        dt=dtmod, freqQ=freqQ, qMin=qMin, qInterior=qInterior))
    @test w1 ≈ w2
end

@testset "3D, threaded, freesurface=$(freesurface)" for freesurface in (false,true)
    w1 = zeros(Float32,nz,ny,nx)
    w2 = zeros(Float32,nz,ny,nx)
    WaveFD.setup_q_profile_3D_threaded!(w1, freesurface, nsponge, dtmod, freqQ, qMin, qInterior)
    w2 .= WaveFD.DtOmegaInvQ(Prop3DAcoIsoDenQ_DEO2_FDTD(;
        freesurface=freesurface, nz=nz, ny=ny, nx=nx, dz=dz, dy=dy, dx=dx, nsponge=nsponge, nthreads=nthreads,
        dt=dtmod, freqQ=freqQ, qMin=qMin, qInterior=qInterior))
    @test w1 ≈ w2
end

@testset "3D, serial == threaded, freesurface=$(freesurface)" for freesurface in (false,true)
    w1 = zeros(Float32,nz,ny,nx)
    w2 = zeros(Float32,nz,ny,nx)
    WaveFD.setup_q_profile_3D_serial!(w1, freesurface, nsponge, dtmod, freqQ, qMin, qInterior)
    WaveFD.setup_q_profile_3D_threaded!(w2, freesurface, nsponge, dtmod, freqQ, qMin, qInterior)
    @test w1 ≈ w2
end

nothing