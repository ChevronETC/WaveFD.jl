using WaveFD, Printf

function testme(nthreads::Int64)
    nt = 100
    nx = 2001
    nz = 2001
    nbx_cache = 8
    nbz_cache = 5000
    dx,dz,dt = 25f0,25f0,0.001f0

    p = WaveFD.Prop2DAcoIsoDenQ_DEO2_FDTD(nz = nz, nx = nx, nbz = nbz_cache, nbx = nbx_cache, nthreads = nthreads, dz = dz, dx = dx, dt = dt);

    WaveFD.V(p) .= 1500;
    WaveFD.B(p) .= 1;
    WaveFD.PCur(p) .= 0;
    WaveFD.POld(p) .= 0;

    # @code_warntype WaveFD.propagateforward!(p);

    t = @elapsed for kt = 1:nt
        WaveFD.propagateforward!(p)
    end
    mc  = nx * nz * nt / (1000 * 1000)
    mcs = mc / t

    @printf("nthreads,time,mcells,mcells/sec; %3d %12.4f %12.4f %12.4f\n", nthreads, t, mc, mcs)

    # @show extrema(p.PCur)
    # @show extrema(p.POld)
    # @show extrema(p.PSpace)
end

for k = 0:4:44
    testme(max(k,1))
end