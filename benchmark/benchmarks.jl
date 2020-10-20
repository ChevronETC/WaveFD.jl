using BenchmarkTools, Random, Statistics, WaveFD

_nthreads = [2^i for i in 0:floor(Int,log2(Sys.CPU_THREADS))]
if Sys.CPU_THREADS ∉ _nthreads
    push!(_nthreads, Sys.CPU_THREADS)
end

const SUITE = BenchmarkGroup()

z0,y0,x0,dz,dy,dx,nt,dt = 0.0,0.0,0.0,10.0,10.0,10.0,3000,0.001

n_2D = (z=parse(Int,get(ENV,"2D_NZ","501")), x=parse(Int,get(ENV,"2D_NX","1001")))
n_3D = (z=parse(Int,get(ENV,"3D_NZ","101")), y=parse(Int,get(ENV,"3D_NY","201")), x=parse(Int,get(ENV,"3D_NX","301")))

nb_2D = (z=parse(Int,get(ENV,"2D_NBZ","$(n_2D.z)")), x=parse(Int,get(ENV,"2D_NBX","8")))
nb_3D = (z=parse(Int,get(ENV,"3D_NBZ","$(n_3D.z)")), y=parse(Int,get(ENV,"3D_NBY","8")), x=parse(Int,get(ENV,"3D_NBX","8")))

@info "size 2D: $n_2D, use ENV[\"2D_NZ\"], ENV[\"2D_NX\"] to customize"
@info "size 3D: $n_3D, use ENV[\"3D_NZ\"], ENV[\"3D_NY\"], ENV[\"3D_NX\"] to customize"
@info "cache block size 2D: $nb_2D, use ENV[\"2D_NBZ\"], ENV[\"2D_NBX\"] to customize"
@info "cache block size 3D: $nb_3D, use ENV[\"3D_NBZ\"], ENV[\"3D_NBY\"], ENV[\"3D_NBX\"] to customize"

rz,rx = dz*ones(n_2D.x),dx*[0:n_2D.x-1;]

iz,ix,c = WaveFD.hickscoeffs(dz, dx, z0, x0, n_2D.z, n_2D.x, rz, rx)
blocks = WaveFD.source_blocking(n_2D.z, n_2D.x, 4, 4, iz, ix, c)
field = zeros(n_2D.z, n_2D.x)
data = rand(nt, n_2D.x)
it = 1

SUITE["2D Utils"] = BenchmarkGroup()
SUITE["2D Utils"]["source_blocking"] = @benchmarkable WaveFD.source_blocking($(n_2D.z), $(n_2D.x), $(nb_2D.z), $(nb_2D).x, $iz, $ix, $c);
SUITE["2D Utils"]["injectdata!"] = @benchmarkable WaveFD.injectdata!($field, $blocks, $data, $it)

dtmod = 0.0002
ntmod = nt * round(Int,dt / dtmod)
h = WaveFD.interpfilters(dtmod, dt, 0, WaveFD.LangC(), Sys.CPU_THREADS)
data_mod = rand(ntmod, n_2D.x)
SUITE["2D Utils"]["interpadjoint!"] = @benchmarkable WaveFD.interpadjoint!($h, $data_mod, $data)
SUITE["2D Utils"]["interpforward!"] = @benchmarkable WaveFD.interpforward!($h, $data, $data_mod)

compressor = WaveFD.Compressor(Float32, Float32, UInt32, (n_2D.z,n_2D.x), (32,32), 1e-2, 1024, false)
field2 = rand(Float32,n_2D.z,n_2D.x)
SUITE["2D compression"] = BenchmarkGroup()
SUITE["2D compression"]["write"] = @benchmarkable WaveFD.compressedwrite(io, $compressor, 1, $field2) setup=(open($compressor); io=open(tempname(),"w")) teardown=(close($compressor); close(io))
SUITE["2D compression"]["read"] = @benchmarkable WaveFD.compressedread!(io, $compressor, 1, $field2) setup=(open($compressor); tfile=tempname(); _io=open(tfile,"w"); WaveFD.compressedwrite(_io, $compressor, 1, $field2); close(_io); io=open(tfile)) teardown=(close($compressor); close(io))

rng2 = (10:n_2D.z-10,10:n_2D.x)
_field2 = view(field2, rng2...)
_compressor = WaveFD.Compressor(Float32, Float32, UInt32, size(_field2), (32,32), 1e-2, 1024, true)
SUITE["2D compression, interior"] = BenchmarkGroup()
SUITE["2D compression, interior"]["write"] = @benchmarkable WaveFD.compressedwrite(io, $_compressor, 1, $field2, $rng2) setup=(open($_compressor); io=open(tempname(),"w")) teardown=(close($_compressor); close(io))
SUITE["2D compression, interior"]["read"] = @benchmarkable WaveFD.compressedread!(io, $_compressor, 1, $field2, $rng2) setup=(open($_compressor); tfile=tempname(); _io=open(tfile,"w"); WaveFD.compressedwrite(_io, $_compressor, 1, $field2, rng2); close(_io); io=open(tfile)) teardown=(close($_compressor); close(io))

SUITE["2DAcoIsoDenQ_DEO2_FDTD"] = BenchmarkGroup([Dict("ncells"=>n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
function p2diso(nthreads,nz,nx,nbz,nbx)
    p = WaveFD.Prop2DAcoIsoDenQ_DEO2_FDTD(freesurface=false, nz=nz, nx=nx, nbz=nbz, nbx=nbx, dz=10.0, dx=10.0, dt=0.001, nthreads=nthreads)
    v,b,pcur,pold = WaveFD.V(p),WaveFD.B(p),WaveFD.PCur(p),WaveFD.POld(p)
    v .= 1500
    b .= 1
    rand!(pcur)
    rand!(pold)
    p
end
for nthreads in _nthreads
    SUITE["2DAcoIsoDenQ_DEO2_FDTD"]["$nthreads threads"] = @benchmarkable WaveFD.propagateforward!(p) setup=(p=p2diso($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x))) teardown=(free(p))
end

SUITE["2DAcoVTIDenQ_DEO2_FDTD"] = BenchmarkGroup([Dict("ncells"=>n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
function p2dvti(nthreads,nz,nx,nbz,nbx)
    p = WaveFD.Prop2DAcoVTIDenQ_DEO2_FDTD(freesurface=false, nz=nz, nx=nx, nbz=nbz, nbx=nbx, dz=10.0, dx=10.0, dt=0.001, nthreads=nthreads)
    v,b,ϵ,η,f,pcur,pold = WaveFD.V(p),WaveFD.B(p),WaveFD.Eps(p),WaveFD.Eta(p),WaveFD.F(p),WaveFD.PCur(p),WaveFD.POld(p)
    rand!(pcur)
    rand!(pold)
    v .= 1500
    b .= 1
    ϵ .= 0.2
    η .= 0.0
    f .= 0.85
    p
end
for nthreads in _nthreads
    SUITE["2DAcoVTIDenQ_DEO2_FDTD"]["$nthreads threads"] = @benchmarkable WaveFD.propagateforward!(p) setup=(p=p2dvti($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x))) teardown=(free(p))
end

SUITE["2DAcoTTIDenQ_DEO2_FDTD"] = BenchmarkGroup([Dict("ncells"=>n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
function p2dtti(nthreads,nz,nx,nbz,nbx)
    p = WaveFD.Prop2DAcoTTIDenQ_DEO2_FDTD(freesurface=false, nz=nz, nx=nx, nbz=nbz, nbx=nbx, dz=10.0, dx=10.0, dt=0.001, nthreads=nthreads)
    v,b,ϵ,η,f,sinθ,cosθ,pcur,pold = WaveFD.V(p),WaveFD.B(p),WaveFD.Eps(p),WaveFD.Eta(p),WaveFD.F(p),WaveFD.SinTheta(p),WaveFD.CosTheta(p),WaveFD.PCur(p),WaveFD.POld(p)
    v .= 1500
    b .= 1
    ϵ .= 0.2
    η .= 0.0
    f .= 0.85
    cosθ .= cos(pi/4)
    sinθ .= sin(pi/4)
    rand!(pcur)
    rand!(pold)
    p
end
for nthreads in _nthreads
    SUITE["2DAcoTTIDenQ_DEO2_FDTD"]["$nthreads threads"] = @benchmarkable WaveFD.propagateforward!(p) setup=(p=p2dtti($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x))) teardown=(free(p))
end

nz,ny,nx = 501,301,1001

rz = [dz for iy = 1:ny, ix = 1:nx][:]
ry = [(iy-1)*dy for iy = 1:ny, ix = 1:nx][:]
rx = [(ix-1)*dx for iy = 1:ny, ix = 1:nx][:]

iz,iy,ix,c = WaveFD.hickscoeffs(dz, dy, dx, z0, y0, x0, nz, ny, nx, rz, ry, rx)
blocks = WaveFD.source_blocking(nz, ny, nx, 4, 4, 4, iz, iy, ix, c)
field = zeros(nz, ny, nx)
data = rand(nt, nx*ny)
it = 1

SUITE["3D Utils"] = BenchmarkGroup()
SUITE["3D Utils"]["source_blocking"] = @benchmarkable WaveFD.source_blocking($nz, $ny, $nx, 256, 8, 8, $iz, $iy, $ix, $c);
SUITE["3D Utils"]["injectdata!"] = @benchmarkable WaveFD.injectdata!($field, $blocks, $data, $it)

compressor = WaveFD.Compressor(Float32, Float32, UInt32, (n_3D.z,n_3D.y,n_3D.x), (32,32,32), 1e-2, 1024, false)
field3 = rand(Float32,n_3D.z,n_3D.y,n_3D.x)
SUITE["3D compression"] = BenchmarkGroup()
SUITE["3D compression"]["write"] = @benchmarkable WaveFD.compressedwrite(io, $compressor, 1, $field3) setup=(open($compressor); io=open(tempname(), "w")) teardown=(close($compressor); close(io))
SUITE["3D compression"]["read"] = @benchmarkable WaveFD.compressedread!(io, $compressor, 1, $field3) setup=(open($compressor); tfile=tempname(); _io=open(tfile,"w"); WaveFD.compressedwrite(_io, $compressor, 1, $field3); close(_io); io=open(tfile)) teardown=(close($compressor); close(io))

rng3 = (10:n_3D.z-10,10:n_3D.y,10:n_3D.x)
_field3 = view(field3, rng3...)
_compressor = WaveFD.Compressor(Float32, Float32, UInt32, size(_field3), (32,32,32), 1e-2, 1024, true)
SUITE["3D compression, interior"] = BenchmarkGroup()
SUITE["3D compression, interior"]["write"] = @benchmarkable WaveFD.compressedwrite(io, $_compressor, 1, $field3, $rng3) setup=(open($_compressor); io=open(tempname(),"w")) teardown=(close($_compressor); close(io))
SUITE["3D compression, interior"]["read"] = @benchmarkable WaveFD.compressedread!(io, $_compressor, 1, $field3, $rng3) setup=(open($_compressor); tfile=tempname(); _io=open(tfile,"w"); WaveFD.compressedwrite(_io, $_compressor, 1, $field3, rng3); close(_io); io=open(tfile)) teardown=(close($_compressor); close(io))

SUITE["3DAcoIsoDenQ_DEO2_FDTD"] = BenchmarkGroup([Dict("ncells"=>n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
function p3diso(nthreads,nz,ny,nx,nbz,nby,nbx)
    p = WaveFD.Prop3DAcoIsoDenQ_DEO2_FDTD(freesurface=false, nz=nz, ny=ny, nx=nx, nbz=nbz, nby=nby, nbx=nbx, dz=10.0, dy=10.0, dx=10.0, dt=0.001, nthreads=nthreads)
    v,b,pcur,pold = WaveFD.V(p), WaveFD.B(p),WaveFD.PCur(p),WaveFD.POld(p)
    v .= 1500
    b .= 1
    rand!(pcur)
    rand!(pold)
    p
end
for nthreads in _nthreads
    SUITE["3DAcoIsoDenQ_DEO2_FDTD"]["$nthreads threads"] = @benchmarkable WaveFD.propagateforward!(p) setup=(p=p3diso($nthreads,$(n_3D.z),$(n_3D.y),$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x))) teardown=(free(p)) seconds=15
end

SUITE["3DAcoVTIDenQ_DEO2_FDTD"] = BenchmarkGroup([Dict("ncells"=>n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
function p3dvti(nthreads,nz,ny,nx,nbz,nby,nbx)
    p = WaveFD.Prop3DAcoVTIDenQ_DEO2_FDTD(freesurface=false, nz=nz, ny=ny, nx=nx, nbz=nbz, nby=nby, nbx=nbx, dz=10.0, dy=10.0, dx=10.0, dt=0.001, nthreads=nthreads)
    v,b,ϵ,η,f,pcur,pold = WaveFD.V(p),WaveFD.B(p),WaveFD.Eps(p),WaveFD.Eta(p),WaveFD.F(p),WaveFD.PCur(p),WaveFD.POld(p)
    rand!(pcur)
    rand!(pold)
    v .= 1500
    b .= 1
    ϵ .= 0.2
    η .= 0.0
    f .= 0.85
    p
end
for nthreads in _nthreads
    SUITE["3DAcoVTIDenQ_DEO2_FDTD"]["$nthreads threads"] = @benchmarkable WaveFD.propagateforward!(p) setup=(p=p3dvti($nthreads,$(n_3D.z),$(n_3D.y),$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x))) teardown=(free(p)) seconds=15
end

SUITE["3DAcoTTIDenQ_DEO2_FDTD"] = BenchmarkGroup([Dict("ncells"=>n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
function p3dtti(nthreads,nz,ny,nx,nbz,nby,nbx)
    p = WaveFD.Prop3DAcoTTIDenQ_DEO2_FDTD(freesurface=false, nz=nz, ny=ny, nx=nx, nbz=nbz, nby=nby, nbx=nbx, dz=10.0, dx=10.0, dt=0.001, nthreads=nthreads)
    v,b,ϵ,η,f,sinθ,cosθ,sinϕ,cosϕ,pcur,pold = WaveFD.V(p),WaveFD.B(p),WaveFD.Eps(p),WaveFD.Eta(p),WaveFD.F(p),WaveFD.SinTheta(p),WaveFD.CosTheta(p),WaveFD.SinPhi(p),WaveFD.CosPhi(p),WaveFD.PCur(p),WaveFD.POld(p)
    v .= 1500
    b .= 1
    ϵ .= 0.2
    η .= 0.0
    f .= 0.85
    cosθ .= cos(pi/4)
    sinθ .= sin(pi/4)
    cosϕ .= cos(pi/8)
    sinϕ .= sin(pi/8)
    rand!(pcur)
    rand!(pold)
    p
end
for nthreads in _nthreads
    SUITE["3DAcoTTIDenQ_DEO2_FDTD"]["$nthreads threads"] = @benchmarkable WaveFD.propagateforward!(p) setup=(p=p3dtti($nthreads,$(n_3D.z),$(n_3D.y),$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x))) teardown=(free(p)) seconds=15
end

include(joinpath(pkgdir(WaveFD), "benchmark", "mcells_per_second.jl"))

SUITE