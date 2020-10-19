using LinearAlgebra, Test, WaveFD

# Unit tests for the 3 types of imaging condition are a little bit heuristic. We construct 
# the nonlinear and adjoint wavefields with horizontal bars, and test the result.
#
# The FWI imaging condition result will be longer spatial wavelength and extend the region 
# of positive values in the correlation, while the RTM imaging condition result will be 
# shorter spatial wavelength and introduce negative values.
#
# Note these imaging conditions are tested under propagation in JetPackWaveFD.
dt = 0.001
dz,dy,dx = 10.0,10.0,10.0
nz,ny,nx = 51,51,51
nthreads = Sys.CPU_THREADS
n = 10
z0 = div(nz,2)
z1 = z0 - div(n,2)
z2 = z0 + div(n,2)
x0 = div(nx,2)

@testset "ImgCondition" begin

    @testset "Imaging Condition 2D tests, $(physics)" for physics in ("ISO", "VTI", "TTI")
        if physics == "ISO"
            piso = WaveFD.Prop2DAcoIsoDenQ_DEO2_FDTD(freesurface=false, nz=nz, nx=nx, dz=dz, dx=dx, dt=dt, nthreads=nthreads)
            WaveFD.B(piso) .= 1
            WaveFD.V(piso) .= 1
            WaveFD.POld(piso)[z1:z2,:] .= 1
        elseif physics == "VTI"
            pvti = WaveFD.Prop2DAcoVTIDenQ_DEO2_FDTD(freesurface=false, nz=nz, nx=nx, dz=dz, dx=dx, dt=dt, nthreads=nthreads)
            WaveFD.B(pvti) .= 1
            WaveFD.V(pvti) .= 1
            WaveFD.POld(pvti)[z1:z2,:] .= 1
            WaveFD.MOld(pvti)[z1:z2,:] .= 2
        elseif physics == "TTI"
            ptti = WaveFD.Prop2DAcoTTIDenQ_DEO2_FDTD(freesurface=false, nz=nz, nx=nx, dz=dz, dx=dx, dt=dt, nthreads=nthreads)
            WaveFD.B(ptti) .= 1
            WaveFD.V(ptti) .= 1
            WaveFD.POld(ptti)[z1:z2,:] .= 1
            WaveFD.MOld(ptti)[z1:z2,:] .= 2
        end
        wp = zeros(Float32,nz,nx)
        wm = zeros(Float32,nz,nx)
        wp[z1:z2,:] .= 1
        wm[z1:z2,:] .= 2

        gstd = zeros(Float32,nz,nx)
        gfwi = zeros(Float32,nz,nx)
        grtm = zeros(Float32,nz,nx)

        if physics == "ISO"
            WaveFD.adjointBornAccumulation!(piso, WaveFD.ImagingConditionStandard(), gstd, wp)
            WaveFD.adjointBornAccumulation!(piso, WaveFD.ImagingConditionWaveFieldSeparationFWI(), gfwi, wp)
            WaveFD.adjointBornAccumulation!(piso, WaveFD.ImagingConditionWaveFieldSeparationRTM(), grtm, wp)
        elseif physics == "VTI"
            w_dict = Dict("pspace" => wp, "mspace" => wm)
            WaveFD.adjointBornAccumulation!(pvti, WaveFD.Prop2DAcoVTIDenQ_DEO2_FDTD_Model_V(), WaveFD.ImagingConditionStandard(), Dict("v" => gstd), w_dict)
            WaveFD.adjointBornAccumulation!(pvti, WaveFD.Prop2DAcoVTIDenQ_DEO2_FDTD_Model_V(), WaveFD.ImagingConditionWaveFieldSeparationFWI(), Dict("v" => gfwi), w_dict)
            WaveFD.adjointBornAccumulation!(pvti, WaveFD.Prop2DAcoVTIDenQ_DEO2_FDTD_Model_V(), WaveFD.ImagingConditionWaveFieldSeparationRTM(), Dict("v" => grtm), w_dict)
        elseif physics == "TTI"
            w_dict = Dict("pspace" => wp, "mspace" => wm)
            WaveFD.adjointBornAccumulation!(ptti, WaveFD.Prop2DAcoTTIDenQ_DEO2_FDTD_Model_V(), WaveFD.ImagingConditionStandard(), Dict("v" => gstd), w_dict)
            WaveFD.adjointBornAccumulation!(ptti, WaveFD.Prop2DAcoTTIDenQ_DEO2_FDTD_Model_V(), WaveFD.ImagingConditionWaveFieldSeparationFWI(), Dict("v" => gfwi), w_dict)
            WaveFD.adjointBornAccumulation!(ptti, WaveFD.Prop2DAcoTTIDenQ_DEO2_FDTD_Model_V(), WaveFD.ImagingConditionWaveFieldSeparationRTM(), Dict("v" => grtm), w_dict)
        end

        @test minimum(gstd) ≥ 0
        @test maximum(gstd) ≥ 0
        @test minimum(gfwi) > 0
        @test maximum(gfwi) > 0
        @test minimum(grtm) < 0
        @test maximum(grtm) > 0
        @test sum(sign.(gfwi)) > sum(sign.(gstd))
        @test dot(gstd,gfwi) > dot(gstd,grtm)
    end

    @testset "Imaging Condition 3D tests, $(physics)" for physics in ("ISO", "VTI", "TTI")
        if physics == "ISO"
            piso = WaveFD.Prop3DAcoIsoDenQ_DEO2_FDTD(freesurface=false, nz=nz, ny=ny, nx=nx, dz=dz, dy=dy, dx=dx, dt=dt, nthreads=nthreads)
            WaveFD.B(piso) .= 1
            WaveFD.V(piso) .= 1
            WaveFD.POld(piso)[z1:z2,:,:] .= 1
        elseif physics == "VTI"
            pvti = WaveFD.Prop3DAcoVTIDenQ_DEO2_FDTD(freesurface=false, nz=nz, ny=ny, nx=nx, dz=dz, dy=dy, dx=dx, dt=dt, nthreads=nthreads)
            WaveFD.B(pvti) .= 1
            WaveFD.V(pvti) .= 1
            WaveFD.POld(pvti)[z1:z2,:,:] .= 1
            WaveFD.MOld(pvti)[z1:z2,:,:] .= 2
        elseif physics == "TTI"
            ptti = WaveFD.Prop3DAcoTTIDenQ_DEO2_FDTD(freesurface=false, nz=nz, ny=ny, nx=nx, dz=dz, dy=dy, dx=dx, dt=dt, nthreads=nthreads)
            WaveFD.B(ptti) .= 1
            WaveFD.V(ptti) .= 1
            WaveFD.POld(ptti)[z1:z2,:,:] .= 1
            WaveFD.MOld(ptti)[z1:z2,:,:] .= 2
        end

        wp = zeros(Float32,nz,ny,nx)
        wm = zeros(Float32,nz,ny,nx)
        wp[z1:z2,:,:] .= 1
        wm[z1:z2,:,:] .= 2

        gstd = zeros(Float32,nz,ny,nx)
        gfwi = zeros(Float32,nz,ny,nx)
        grtm = zeros(Float32,nz,ny,nx)

        if physics == "ISO"
            WaveFD.adjointBornAccumulation!(piso, WaveFD.ImagingConditionStandard(), gstd, wp)
            WaveFD.adjointBornAccumulation!(piso, WaveFD.ImagingConditionWaveFieldSeparationFWI(), gfwi, wp)
            WaveFD.adjointBornAccumulation!(piso, WaveFD.ImagingConditionWaveFieldSeparationRTM(), grtm, wp)
        elseif physics == "VTI"
            w_dict = Dict("pspace" => wp, "mspace" => wm)
            WaveFD.adjointBornAccumulation!(pvti, WaveFD.Prop3DAcoVTIDenQ_DEO2_FDTD_Model_V(), WaveFD.ImagingConditionStandard(), Dict("v" => gstd), w_dict)
            WaveFD.adjointBornAccumulation!(pvti, WaveFD.Prop3DAcoVTIDenQ_DEO2_FDTD_Model_V(), WaveFD.ImagingConditionWaveFieldSeparationFWI(), Dict("v" => gfwi), w_dict)
            WaveFD.adjointBornAccumulation!(pvti, WaveFD.Prop3DAcoVTIDenQ_DEO2_FDTD_Model_V(), WaveFD.ImagingConditionWaveFieldSeparationRTM(), Dict("v" => grtm), w_dict)
        elseif physics == "TTI"
            w_dict = Dict("pspace" => wp, "mspace" => wm)
            WaveFD.adjointBornAccumulation!(ptti, WaveFD.Prop3DAcoTTIDenQ_DEO2_FDTD_Model_V(), WaveFD.ImagingConditionStandard(), Dict("v" => gstd), w_dict)
            WaveFD.adjointBornAccumulation!(ptti, WaveFD.Prop3DAcoTTIDenQ_DEO2_FDTD_Model_V(), WaveFD.ImagingConditionWaveFieldSeparationFWI(), Dict("v" => gfwi), w_dict)
            WaveFD.adjointBornAccumulation!(ptti, WaveFD.Prop3DAcoTTIDenQ_DEO2_FDTD_Model_V(), WaveFD.ImagingConditionWaveFieldSeparationRTM(), Dict("v" => grtm), w_dict)
        end

        @test minimum(gstd) ≥ 0
        @test maximum(gstd) ≥ 0
        @test minimum(gfwi) > 0
        @test maximum(gfwi) > 0
        @test minimum(grtm) < 0
        @test maximum(grtm) > 0
        @test sum(sign.(gfwi)) > sum(sign.(gstd))
        @test dot(gstd,gfwi) > dot(gstd,grtm)
    end
end

nothing
