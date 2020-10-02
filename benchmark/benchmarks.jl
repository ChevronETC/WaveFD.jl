using BenchmarkTools, Random, Statistics, Wave

@info "Threads.nthreads() = $(Threads.nthreads())"

_nthreads = [2^i for i in 0:floor(Int,log2(Sys.CPU_THREADS))]
if Sys.CPU_THREADS ∉ _nthreads
    push!(_nthreads, Sys.CPU_THREADS)
end

const SUITE = BenchmarkGroup()

z0,x0,z,x,dz,dx,nbz,nbx,nt,dt = 0.0,0.0,5000.0,10000.0,10.0,10.0,4,4,3000,0.001
nz,nx = round(Int,z/dz)+1,round(Int,x/dx)+1

rz,rx = dz*ones(nx),dx*[0:nx-1;]

iz,ix,c = Wave.hickscoeffs(dz, dx, z0, x0, nz, nx, rz, rx)
blocks = Wave.source_blocking(nz, nx, nbz, nbx, iz, ix, c)
field = zeros(nz, nx)
data = rand(nt, nx)
it = 1
nbz,nbx = 256,8

SUITE["2D Utils"] = BenchmarkGroup()
SUITE["2D Utils"]["source_blocking"] = @benchmarkable Wave.source_blocking($nz, $nx, $nbz, $nbx, $iz, $ix, $c);
SUITE["2D Utils"]["injectdata!"] = @benchmarkable Wave.injectdata!($field, $blocks, $data, $it)

dtmod = 0.0002
ntmod = nt * round(Int,dt / dtmod)
h = Wave.interpfilters(dtmod, dt, 0, Wave.LangC(), Sys.CPU_THREADS)
data_mod = rand(ntmod, nx)
SUITE["2D Utils"]["interpadjoint!"] = @benchmarkable Wave.interpadjoint!($h, $data_mod, $data)
SUITE["2D Utils"]["interpforward!"] = @benchmarkable Wave.interpforward!($h, $data, $data_mod)

compressor = Wave.Compressor(Float32, Float32, UInt32, (nz,nx), (32,32), 1e-2, 1024, false)
field2 = rand(Float32,nz,nx)
SUITE["2D compression"] = BenchmarkGroup()
SUITE["2D compression"]["write"] = @benchmarkable Wave.compressedwrite(io, $compressor, 1, $field2) setup=(open($compressor); io=open(tempname(),"w")) teardown=(close($compressor); close(io))
SUITE["2D compression"]["read"] = @benchmarkable Wave.compressedread!(io, $compressor, 1, $field2) setup=(open($compressor); tfile=tempname(); _io=open(tfile,"w"); Wave.compressedwrite(_io, $compressor, 1, $field2); close(_io); io=open(tfile)) teardown=(close($compressor); close(io))

SUITE["2DAcoIsoDenQ_DEO2_FDTD"] = BenchmarkGroup([Dict("ncells"=>nz*nx,"nbz"=>nbz,"nbx"=>nbx,"nthreads"=>_nthreads)])
function p2diso(nthreads,nz,nx,nbz,nbx)
    p = Wave.Prop2DAcoIsoDenQ_DEO2_FDTD(freesurface=false, nz=nz, nx=nx, nbz=nbz, nbx=nbx, dz=10.0, dx=10.0, dt=0.001, nthreads=nthreads)
    v,b,pcur,pold = Wave.V(p),Wave.B(p),Wave.PCur(p),Wave.POld(p)
    v .= 1500
    b .= 1
    rand!(pcur)
    rand!(pold)
    p
end
for nthreads in _nthreads
    SUITE["2DAcoIsoDenQ_DEO2_FDTD"]["$nthreads threads"] = @benchmarkable Wave.propagateforward!(p) setup=(p=p2diso($nthreads,$nz,$nx,$nbz,$nbx)) teardown=(free(p))
end

SUITE["2DAcoVTIDenQ_DEO2_FDTD"] = BenchmarkGroup([Dict("ncells"=>nz*nx,"nbz"=>nbz,"nbx"=>nbx,"nthreads"=>_nthreads)])
function p2dvti(nthreads,nz,nx,nbz,nbx)
    p = Wave.Prop2DAcoVTIDenQ_DEO2_FDTD(freesurface=false, nz=nz, nx=nx, nbz=nbz, nbx=nbx, dz=10.0, dx=10.0, dt=0.001, nthreads=nthreads)
    v,b,ϵ,η,f,pcur,pold = Wave.V(p),Wave.B(p),Wave.Eps(p),Wave.Eta(p),Wave.F(p),Wave.PCur(p),Wave.POld(p)
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
    SUITE["2DAcoVTIDenQ_DEO2_FDTD"]["$nthreads threads"] = @benchmarkable Wave.propagateforward!(p) setup=(p=p2dvti($nthreads,$nz,$nx,$nbz,$nbx)) teardown=(free(p))
end

SUITE["2DAcoTTIDenQ_DEO2_FDTD"] = BenchmarkGroup([Dict("ncells"=>nz*nx,"nbz"=>nbz,"nbx"=>nbx,"nthreads"=>_nthreads)])
function p2dtti(nthreads,nz,nx,nbz,nbx)
    p = Wave.Prop2DAcoTTIDenQ_DEO2_FDTD(freesurface=false, nz=nz, nx=nx, nbz=nbz, nbx=nbx, dz=10.0, dx=10.0, dt=0.001, nthreads=nthreads)
    v,b,ϵ,η,f,sinθ,cosθ,pcur,pold = Wave.V(p),Wave.B(p),Wave.Eps(p),Wave.Eta(p),Wave.F(p),Wave.SinTheta(p),Wave.CosTheta(p),Wave.PCur(p),Wave.POld(p)
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
    SUITE["2DAcoTTIDenQ_DEO2_FDTD"]["$nthreads threads"] = @benchmarkable Wave.propagateforward!(p) setup=(p=p2dtti($nthreads,$nz,$nx,$nbz,$nbx)) teardown=(free(p))
end

z0,y0,x0,z,y,x,dz,dy,dx,nbz,nby,nbx,nt = 0.0,0.0,0.0,5000.0,3000.0,10000.0,10.0,10.0,10.0,4,4,4,3000
nz,ny,nx = round(Int,z/dz)+1,round(Int,y/dy)+1,round(Int,x/dx)+1

rz = [dz for iy = 1:ny, ix = 1:nx][:]
ry = [(iy-1)*dy for iy = 1:ny, ix = 1:nx][:]
rx = [(ix-1)*dx for iy = 1:ny, ix = 1:nx][:]

iz,iy,ix,c = Wave.hickscoeffs(dz, dy, dx, z0, y0, x0, nz, ny, nx, rz, ry, rx)
blocks = Wave.source_blocking(nz, ny, nx, nbz, nby, nbx, iz, iy, ix, c)
field = zeros(nz, ny, nx)
data = rand(nt, nx*ny)
it = 1
nbz,nby,nbx = 256,8,8

SUITE["3D Utils"] = BenchmarkGroup()
SUITE["3D Utils"]["source_blocking"] = @benchmarkable Wave.source_blocking($nz, $ny, $nx, $nbz, $nby, $nbx, $iz, $iy, $ix, $c);
SUITE["3D Utils"]["injectdata!"] = @benchmarkable Wave.injectdata!($field, $blocks, $data, $it)

compressor = Wave.Compressor(Float32, Float32, UInt32, (nz,ny,nx), (32,32,32), 1e-2, 1024, false)
field3 = rand(Float32,nz,ny,nx)
SUITE["3D compression"] = BenchmarkGroup()
SUITE["3D compression"]["write"] = @benchmarkable Wave.compressedwrite(io, $compressor, 1, $field3) setup=(open($compressor); io=open(tempname(), "w")) teardown=(close($compressor); close(io))
SUITE["3D compression"]["read"] = @benchmarkable Wave.compressedread!(io, $compressor, 1, $field3) setup=(open($compressor); tfile=tempname(); _io=open(tfile,"w"); Wave.compressedwrite(_io, $compressor, 1, $field3); close(_io); io=open(tfile)) teardown=(close($compressor); close(io))

z,y,x,dz,dy,dx = 1000.0,2000.0,3000.0,10.0,10.0,10.0
nz,ny,nx = round(Int,z/dz)+1,round(Int,y/dy)+1,round(Int,x/dx)+1

SUITE["3DAcoIsoDenQ_DEO2_FDTD"] = BenchmarkGroup([Dict("ncells"=>nz*ny*nx,"nbz"=>nbz,"nby"=>nby,"nbx"=>nbx,"nthreads"=>_nthreads)])
function p3diso(nthreads,nz,ny,nx,nbz,nby,nbx)
    p = Wave.Prop3DAcoIsoDenQ_DEO2_FDTD(freesurface=false, nz=nz, ny=ny, nx=nx, nbz=nbz, nby=nby, nbx=nbx, dz=10.0, dy=10.0, dx=10.0, dt=0.001, nthreads=nthreads)
    v,b,pcur,pold = Wave.V(p), Wave.B(p),Wave.PCur(p),Wave.POld(p)
    v .= 1500
    b .= 1
    rand!(pcur)
    rand!(pold)
    p
end
for nthreads in _nthreads
    SUITE["3DAcoIsoDenQ_DEO2_FDTD"]["$nthreads threads"] = @benchmarkable Wave.propagateforward!(p) setup=(p=p3diso($nthreads,$nz,$ny,$nx,$nbz,$nby,$nbx)) teardown=(free(p)) seconds=15
end

SUITE["3DAcoVTIDenQ_DEO2_FDTD"] = BenchmarkGroup([Dict("ncells"=>nz*ny*nx,"nbz"=>nbz,"nby"=>nby,"nbx"=>nbx,"nthreads"=>_nthreads)])
function p3dvti(nthreads,nz,ny,nx,nbz,nby,nbx)
    p = Wave.Prop3DAcoVTIDenQ_DEO2_FDTD(freesurface=false, nz=nz, ny=ny, nx=nx, nbz=nbz, nby=nby, nbx=nbx, dz=10.0, dy=10.0, dx=10.0, dt=0.001, nthreads=nthreads)
    v,b,ϵ,η,f,pcur,pold = Wave.V(p),Wave.B(p),Wave.Eps(p),Wave.Eta(p),Wave.F(p),Wave.PCur(p),Wave.POld(p)
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
    SUITE["3DAcoVTIDenQ_DEO2_FDTD"]["$nthreads threads"] = @benchmarkable Wave.propagateforward!(p) setup=(p=p3dvti($nthreads,$nz,$ny,$nx,$nbz,$nby,$nbx)) teardown=(free(p)) seconds=15
end

SUITE["3DAcoTTIDenQ_DEO2_FDTD"] = BenchmarkGroup([Dict("ncells"=>nz*ny*nx,"nbz"=>nbz,"nby"=>nby,"nbx"=>nbx,"nthreads"=>_nthreads)])
function p3dtti(nthreads,nz,ny,nx,nbz,nby,nbx)
    p = Wave.Prop3DAcoTTIDenQ_DEO2_FDTD(freesurface=false, nz=nz, ny=ny, nx=nx, nbz=nbz, nby=nby, nbx=nbx, dz=10.0, dx=10.0, dt=0.001, nthreads=nthreads)
    v,b,ϵ,η,f,sinθ,cosθ,sinϕ,cosϕ,pcur,pold = Wave.V(p),Wave.B(p),Wave.Eps(p),Wave.Eta(p),Wave.F(p),Wave.SinTheta(p),Wave.CosTheta(p),Wave.SinPhi(p),Wave.CosPhi(p),Wave.PCur(p),Wave.POld(p)
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
    SUITE["3DAcoTTIDenQ_DEO2_FDTD"]["$nthreads threads"] = @benchmarkable Wave.propagateforward!(p) setup=(p=p3dtti($nthreads,$nz,$ny,$nx,$nbz,$nby,$nbx)) teardown=(free(p)) seconds=15
end

include(joinpath(pkgdir(Wave), "benchmark", "mcells_per_second.jl"))

SUITE