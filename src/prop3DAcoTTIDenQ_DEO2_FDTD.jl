mutable struct Prop3DAcoTTIDenQ_DEO2_FDTD
    p::Ptr{Cvoid}
end

function Prop3DAcoTTIDenQ_DEO2_FDTD(;
        nz,
        ny,
        nx,
        nsponge=60,
        nbz=256,
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

    fs = freesurface ? 1 : 0

    p = ccall((:Prop3DAcoTTIDenQ_DEO2_FDTD_alloc, Wave._jl_libprop3DAcoTTIDenQ_DEO2_FDTD),
        Ptr{Cvoid},
        (Clong, Clong,    Clong, Clong, Clong, Clong,   Cfloat, Cfloat, Cfloat, Cfloat, Clong, Clong, Clong),
         fs,    nthreads, nx,    ny,    nz,    nsponge, dx,     dy,     dz,     dt,     nbx,   nby,   nbz)

    ccall((:Prop3DAcoTTIDenQ_DEO2_FDTD_SetupDtOmegaInvQ, Wave._jl_libprop3DAcoTTIDenQ_DEO2_FDTD),
        Cvoid,
        (Ptr{Cvoid}, Cfloat, Cfloat, Cfloat),
         p,          freqQ,  qMin,   qInterior)

    Prop3DAcoTTIDenQ_DEO2_FDTD(p)
end

free(prop::Prop3DAcoTTIDenQ_DEO2_FDTD) = ccall((:Prop3DAcoTTIDenQ_DEO2_FDTD_free, Wave._jl_libprop3DAcoTTIDenQ_DEO2_FDTD), Cvoid, (Ptr{Cvoid},), prop.p)

function size(prop::Prop3DAcoTTIDenQ_DEO2_FDTD)
    nz = ccall((:Prop3DAcoTTIDenQ_DEO2_FDTD_getNz, Wave._jl_libprop3DAcoTTIDenQ_DEO2_FDTD), (Clong), (Ptr{Cvoid},), prop.p)
    ny = ccall((:Prop3DAcoTTIDenQ_DEO2_FDTD_getNy, Wave._jl_libprop3DAcoTTIDenQ_DEO2_FDTD), (Clong), (Ptr{Cvoid},), prop.p)
    nx = ccall((:Prop3DAcoTTIDenQ_DEO2_FDTD_getNx, Wave._jl_libprop3DAcoTTIDenQ_DEO2_FDTD), (Clong), (Ptr{Cvoid},), prop.p)
    (nz,ny,nx)
end

for _f in (:V, :Eps, :Eta, :SinTheta, :CosTheta, :SinPhi, :CosPhi, :B, :F, :PSpace, :MSpace, :PCur, :POld, :MCur, :MOld, :TmpPg1, :TmpPg3, :TmpMg1, :TmpMg3, :DtOmegaInvQ)
    symf = "Prop3DAcoTTIDenQ_DEO2_FDTD_get" * string(_f)
    @eval $(_f)(prop::Prop3DAcoTTIDenQ_DEO2_FDTD) = unsafe_wrap(Array, ccall(($symf, Wave._jl_libprop3DAcoTTIDenQ_DEO2_FDTD), Ptr{Float32}, (Ptr{Cvoid},), prop.p), size(prop), own=false)
end

propagateforward!(prop::Prop3DAcoTTIDenQ_DEO2_FDTD) = ccall((:Prop3DAcoTTIDenQ_DEO2_FDTD_TimeStep, Wave._jl_libprop3DAcoTTIDenQ_DEO2_FDTD), Cvoid, (Ptr{Cvoid},), prop.p)
propagateadjoint!(prop::Prop3DAcoTTIDenQ_DEO2_FDTD) = propagateforward!(prop) # self-adjoint

scale_spatial_derivatives!(prop::Prop3DAcoTTIDenQ_DEO2_FDTD) =
    ccall((:Prop3DAcoTTIDenQ_DEO2_FDTD_ScaleSpatialDerivatives, Wave._jl_libprop3DAcoTTIDenQ_DEO2_FDTD), Cvoid, (Ptr{Cvoid},), prop.p)

abstract type Prop3DAcoTTIDenQ_DEO2_FDTD_Model end

# v,ϵ,η
struct Prop3DAcoTTIDenQ_DEO2_FDTD_Model_VEA <: Prop3DAcoTTIDenQ_DEO2_FDTD_Model end
function forwardBornInjection!(prop::Prop3DAcoTTIDenQ_DEO2_FDTD,::Type{Prop3DAcoTTIDenQ_DEO2_FDTD_Model_VEA},dmodel,wavefield)
    ccall((:Prop3DAcoTTIDenQ_DEO2_FDTD_ForwardBornInjection_VEA, Wave._jl_libprop3DAcoTTIDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},       Ptr{Cfloat},       Ptr{Cfloat},         Ptr{Cfloat}),
         prop.p,     dmodel["v"], dmodel["ϵ"], dmodel["η"], wavefield["pold"], wavefield["mold"], wavefield["pspace"], wavefield["mspace"])
end

function adjointBornAccumulation!(prop::Prop3DAcoTTIDenQ_DEO2_FDTD,::Type{Prop3DAcoTTIDenQ_DEO2_FDTD_Model_VEA},dmodel,wavefield)
    ccall((:Prop3DAcoTTIDenQ_DEO2_FDTD_AdjointBornAccumulation_VEA, Wave._jl_libprop3DAcoTTIDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},       Ptr{Cfloat},       Ptr{Cfloat},         Ptr{Cfloat}),
         prop.p,     dmodel["v"], dmodel["ϵ"], dmodel["η"], wavefield["pold"], wavefield["mold"], wavefield["pspace"], wavefield["mspace"])
end

# v
struct Prop3DAcoTTIDenQ_DEO2_FDTD_Model_V <: Prop3DAcoTTIDenQ_DEO2_FDTD_Model end
function forwardBornInjection!(prop::Prop3DAcoTTIDenQ_DEO2_FDTD,::Type{Prop3DAcoTTIDenQ_DEO2_FDTD_Model_V},dmodel,wavefield)
    ccall((:Prop3DAcoTTIDenQ_DEO2_FDTD_ForwardBornInjection_V, Wave._jl_libprop3DAcoTTIDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat},         Ptr{Cfloat}),
         prop.p,     dmodel["v"], wavefield["pspace"], wavefield["mspace"])
end

function adjointBornAccumulation!(prop::Prop3DAcoTTIDenQ_DEO2_FDTD,::Type{Prop3DAcoTTIDenQ_DEO2_FDTD_Model_V},dmodel,wavefield)
    ccall((:Prop3DAcoTTIDenQ_DEO2_FDTD_AdjointBornAccumulation_V, Wave._jl_libprop3DAcoTTIDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat},         Ptr{Cfloat}),
         prop.p,     dmodel["v"], wavefield["pspace"], wavefield["mspace"])
end

function show(io::IO, prop::Prop3DAcoTTIDenQ_DEO2_FDTD)
    nx = ccall((:Prop3DAcoTTIDenQ_DEO2_FDTD_getNx, Wave._jl_libprop3DAcoTTIDenQ_DEO2_FDTD), (Clong), (Ptr{Cvoid},), prop.p)
    ny = ccall((:Prop3DAcoTTIDenQ_DEO2_FDTD_getNy, Wave._jl_libprop3DAcoTTIDenQ_DEO2_FDTD), (Clong), (Ptr{Cvoid},), prop.p)
    nz = ccall((:Prop3DAcoTTIDenQ_DEO2_FDTD_getNz, Wave._jl_libprop3DAcoTTIDenQ_DEO2_FDTD), (Clong), (Ptr{Cvoid},), prop.p)
    write(io, "Prop3DAcoTTIDenQ_DEO2_FDTD -- nx,ny,nz; $nx,$ny,$nz")
end

function tilesize(::Type{Prop3DAcoTTIDenQ_DEO2_FDTD}; nz, ny, nx, rngy=4:2:64, rngx=4:2:64, N=10, nthreads=Sys.CPU_THREADS-1)
    speeds = Matrix{Float64}(undef, length(rngy), length(rngx))
    for (iby,by) in enumerate(rngy), (ibx,bx) in enumerate(rngx)
        @info "by=$by, bx=$bx"
        prop = Prop3DAcoTTIDenQ_DEO2_FDTD(nz=nz, ny=ny, nx=nx, nbz=nz, nby=by, nbx=bx, nthreads=nthreads)
        t = 0.0
        for i = 1:N
            t += @elapsed propagateforward!(prop)
        end
        speeds[iby,ibx] = N*prod(size(prop))/1000/1000/t
        free(prop)
    end
    rngx,rngy,speeds
end
