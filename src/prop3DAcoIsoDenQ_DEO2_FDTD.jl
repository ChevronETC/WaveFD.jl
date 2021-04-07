struct Prop3DAcoIsoDenQ_DEO2_FDTD
    p::Ptr{Cvoid}
end

function Prop3DAcoIsoDenQ_DEO2_FDTD(;
        nz=0,
        ny=0,
        nx=0,
        nsponge=60,
        nbz=512,
        nby=8,
        nbx=8,
        dz=5.0,
        dy=8.0,
        dx=5.0,
        dt=0.001,
        nthreads=Sys.CPU_THREADS,
        freesurface=true,
        freqQ=5.0,
        qMin=0.1,
        qInterior=100.0)
    nz,ny,nx,nsponge,nbz,nby,nbx,nthreads = map(x->round(Int,x), (nz,ny,nx,nsponge,nbz,nby,nbx,nthreads))
    dz,dy,dx,dt = map(x->Float32(x), (dz,dy,dx,dt))

    @assert nx > 0
    @assert ny > 0
    @assert nz > 0
    @assert nsponge > 0
    @assert nthreads > 0
    @assert nbx > 0
    @assert nby > 0
    @assert nbz > 0

    fs = freesurface ? 1 : 0

    p = ccall((:Prop3DAcoIsoDenQ_DEO2_FDTD_alloc, libprop3DAcoIsoDenQ_DEO2_FDTD),
        Ptr{Cvoid},
        (Clong, Clong,    Clong, Clong, Clong, Clong,   Cfloat, Cfloat, Cfloat, Cfloat, Clong, Clong, Clong),
         fs,    nthreads, nx,    ny,    nz,    nsponge, dx,     dy,     dz,     dt,     nbx,   nby,   nbz)

    ccall((:Prop3DAcoIsoDenQ_DEO2_FDTD_SetupDtOmegaInvQ, libprop3DAcoIsoDenQ_DEO2_FDTD),
        Cvoid,
        (Ptr{Cvoid}, Cfloat, Cfloat, Cfloat),
         p,          freqQ,  qMin,   qInterior)

    Prop3DAcoIsoDenQ_DEO2_FDTD(p)
end

free(prop::Prop3DAcoIsoDenQ_DEO2_FDTD) = ccall((:Prop3DAcoIsoDenQ_DEO2_FDTD_free, libprop3DAcoIsoDenQ_DEO2_FDTD), Cvoid, (Ptr{Cvoid},), prop.p)

function size(prop::Prop3DAcoIsoDenQ_DEO2_FDTD)
    nz = ccall((:Prop3DAcoIsoDenQ_DEO2_FDTD_getNz, libprop3DAcoIsoDenQ_DEO2_FDTD), (Clong), (Ptr{Cvoid},), prop.p)
    ny = ccall((:Prop3DAcoIsoDenQ_DEO2_FDTD_getNy, libprop3DAcoIsoDenQ_DEO2_FDTD), (Clong), (Ptr{Cvoid},), prop.p)
    nx = ccall((:Prop3DAcoIsoDenQ_DEO2_FDTD_getNx, libprop3DAcoIsoDenQ_DEO2_FDTD), (Clong), (Ptr{Cvoid},), prop.p)
    (nz,ny,nx)
end

for _f in (:V, :B, :PSpace, :PCur, :POld, :DtOmegaInvQ)
    symf = "Prop3DAcoIsoDenQ_DEO2_FDTD_get" * string(_f)
    @eval $(_f)(prop::Prop3DAcoIsoDenQ_DEO2_FDTD) = unsafe_wrap(Array, ccall(($symf, libprop3DAcoIsoDenQ_DEO2_FDTD), Ptr{Float32}, (Ptr{Cvoid},), prop.p), size(prop), own=false)
end

abstract type Prop3DAcoIsoDenQ_DEO2_FDTD_Model end
struct Prop3DAcoIsoDenQ_DEO2_FDTD_Model_V <: Prop3DAcoIsoDenQ_DEO2_FDTD_Model end
struct Prop3DAcoIsoDenQ_DEO2_FDTD_Model_B <: Prop3DAcoIsoDenQ_DEO2_FDTD_Model end
struct Prop3DAcoIsoDenQ_DEO2_FDTD_Model_VB <: Prop3DAcoIsoDenQ_DEO2_FDTD_Model end

propagateforward!(prop::Prop3DAcoIsoDenQ_DEO2_FDTD) = ccall((:Prop3DAcoIsoDenQ_DEO2_FDTD_TimeStep, libprop3DAcoIsoDenQ_DEO2_FDTD), Cvoid, (Ptr{Cvoid},), prop.p)
propagateadjoint!(prop::Prop3DAcoIsoDenQ_DEO2_FDTD) = propagateforward!(prop) # self-adjoint

scale_spatial_derivatives!(prop::Prop3DAcoIsoDenQ_DEO2_FDTD) =
    ccall((:Prop3DAcoIsoDenQ_DEO2_FDTD_ScaleSpatialDerivatives, libprop3DAcoIsoDenQ_DEO2_FDTD), Cvoid, (Ptr{Cvoid},), prop.p)

# v in model-space
function forwardBornInjection!(prop::Prop3DAcoIsoDenQ_DEO2_FDTD, modeltype::Prop3DAcoIsoDenQ_DEO2_FDTD_Model_V, dmodel, wavefields)
    ccall((:Prop3DAcoIsoDenQ_DEO2_FDTD_ForwardBornInjection_V, libprop3DAcoIsoDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat}),
         prop.p,     dmodel["v"], wavefields["pspace"])
end

# v,b in model-space
function forwardBornInjection!(prop::Prop3DAcoIsoDenQ_DEO2_FDTD, modeltype::Prop3DAcoIsoDenQ_DEO2_FDTD_Model_VB, dmodel, wavefields)
    ccall((:Prop3DAcoIsoDenQ_DEO2_FDTD_ForwardBornInjection_VB, libprop3DAcoIsoDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},        Ptr{Cfloat}),
         prop.p,     dmodel["v"], dmodel["b"], wavefields["pold"], wavefields["pspace"])
end

# b in model-space
function forwardBornInjection!(prop::Prop3DAcoIsoDenQ_DEO2_FDTD, modeltype::Prop3DAcoIsoDenQ_DEO2_FDTD_Model_B, dmodel, wavefields)
    ccall((:Prop3DAcoIsoDenQ_DEO2_FDTD_ForwardBornInjection_B, libprop3DAcoIsoDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat},        Ptr{Cfloat}),
         prop.p,     dmodel["b"], wavefields["pold"], wavefields["pspace"])
end

# v in model-space
function adjointBornAccumulation!(prop::Prop3DAcoIsoDenQ_DEO2_FDTD, modeltype::Prop3DAcoIsoDenQ_DEO2_FDTD_Model_V, imagingcondition::ImagingConditionStandard, dmodel, wavefields)
    ccall((:Prop3DAcoIsoDenQ_DEO2_FDTD_AdjointBornAccumulation_V, libprop3DAcoIsoDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat}),
         prop.p,     dmodel["v"], wavefields["pspace"])
end

# v,b in model-space
function adjointBornAccumulation!(prop::Prop3DAcoIsoDenQ_DEO2_FDTD, modeltype::Prop3DAcoIsoDenQ_DEO2_FDTD_Model_VB, imagingcondition::ImagingConditionStandard, dmodel, wavefields)
    ccall((:Prop3DAcoIsoDenQ_DEO2_FDTD_AdjointBornAccumulation_VB, libprop3DAcoIsoDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},        Ptr{Cfloat}),
         prop.p,     dmodel["v"], dmodel["b"], wavefields["pold"], wavefields["pspace"])
end

# b in model-space
function adjointBornAccumulation!(prop::Prop3DAcoIsoDenQ_DEO2_FDTD, modeltype::Prop3DAcoIsoDenQ_DEO2_FDTD_Model_B, imagingcondition::ImagingConditionStandard, dmodel, wavefields)
    ccall((:Prop3DAcoIsoDenQ_DEO2_FDTD_AdjointBornAccumulation_B, libprop3DAcoIsoDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat},        Ptr{Cfloat}),
         prop.p,     dmodel["b"], wavefields["pold"], wavefields["pspace"])
end

# v in model-space with FWI wavefield separation
function adjointBornAccumulation!(prop::Prop3DAcoIsoDenQ_DEO2_FDTD, modeltype::Prop3DAcoIsoDenQ_DEO2_FDTD_Model_V, imagingcondition::ImagingConditionWaveFieldSeparationFWI, dmodel, wavefields)
    ccall((:Prop3DAcoIsoDenQ_DEO2_FDTD_AdjointBornAccumulation_wavefieldsep, libprop3DAcoIsoDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat},          Clong),
         prop.p,     dmodel["v"], wavefields["pspace"], 1)
end

# v in model-space with RTM wavefield separation
function adjointBornAccumulation!(prop::Prop3DAcoIsoDenQ_DEO2_FDTD, modeltype::Prop3DAcoIsoDenQ_DEO2_FDTD_Model_V, imagingcondition::ImagingConditionWaveFieldSeparationRTM, dmodel, wavefields)
    ccall((:Prop3DAcoIsoDenQ_DEO2_FDTD_AdjointBornAccumulation_wavefieldsep, libprop3DAcoIsoDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat},         Clong),
        prop.p,     dmodel["v"], wavefields["pspace"], 0)
end

function show(io::IO, prop::Prop3DAcoIsoDenQ_DEO2_FDTD)
    nx = ccall((:Prop3DAcoIsoDenQ_DEO2_FDTD_getNx, libprop3DAcoIsoDenQ_DEO2_FDTD), (Clong), (Ptr{Cvoid},), prop.p)
    ny = ccall((:Prop3DAcoIsoDenQ_DEO2_FDTD_getNy, libprop3DAcoIsoDenQ_DEO2_FDTD), (Clong), (Ptr{Cvoid},), prop.p)
    nz = ccall((:Prop3DAcoIsoDenQ_DEO2_FDTD_getNz, libprop3DAcoIsoDenQ_DEO2_FDTD), (Clong), (Ptr{Cvoid},), prop.p)
    write(io, "Prop3DAcoIsoDenQ_DEO2_FDTD -- nx,ny,nz; $nx,$ny,$nz")
end

function tilesize(::Type{Prop3DAcoIsoDenQ_DEO2_FDTD}; nz, ny, nx, rngy=4:2:64, rngx=4:2:64, N=10, nthreads=Sys.CPU_THREADS-1)
    speeds = Matrix{Float64}(undef, length(rngy), length(rngx))
    for (iby,by) in enumerate(rngy), (ibx,bx) in enumerate(rngx)
        @info "by=$by, bx=$bx"
        prop = Prop3DAcoIsoDenQ_DEO2_FDTD(nz=nz, ny=ny, nx=nx, nbz=nz, nby=by, nbx=bx, nthreads=nthreads)

        v = WaveFD.V(prop)
        b = WaveFD.B(prop)
        pcur = WaveFD.PCur(prop)
        pold = WaveFD.POld(prop)

        v .= 1500
        b .= 1
        rand!(pcur)
        rand!(pold)

        t = 0.0
        set_zero_subnormals(true)
        for i = 1:N
            t += @elapsed propagateforward!(prop)
        end
        set_zero_subnormals(false)
        speeds[iby,ibx] = N*prod(size(prop))/1000/1000/t
        free(prop)
    end
    rngx,rngy,speeds
end
