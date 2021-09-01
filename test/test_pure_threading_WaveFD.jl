using WaveFD, Printf

function testme(nthreads::Int64)
    nt = 500
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

    # warmup
    for kt = 1:2
        WaveFD.propagateforward!(p)
    end
    
    t = @elapsed for kt = 1:nt
        WaveFD.propagateforward!(p)
        
        # source injection 
        ksz, ksx = div(nz,2)+1, div(nx,2)+1
        WaveFD.PCur(p)[ksz,ksx] += dt^2 * WaveFD.V(p)[ksz,ksx]^2 / WaveFD.B(p)[ksz,ksx];
    end
    mc  = nx * nz * nt / (1000 * 1000)
    mcs = mc / t

    @printf("%3d %12.4f\n", nthreads, mcs)
end

for k = 44:-4:0
    testme(max(k,1))
end