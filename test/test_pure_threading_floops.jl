using Polyester, Printf, BenchmarkTools, LoopVectorization, FLoops

mutable struct PureJulia2DAcoIsoDenQ_DEO2_FDTD
    nz::Int64
    nx::Int64
    nbz_cache::Int64
    nbx_cache::Int64
    dz::Float32
    dx::Float32
    dt::Float32
    c8_1::Float32
    c8_2::Float32
    c8_3::Float32
    c8_4::Float32
    V::Array{Float32,2}
    B::Array{Float32,2}
    PSpace::Array{Float32,2}
    PCur::Array{Float32,2}
    POld::Array{Float32,2}
    TmpPx::Array{Float32,2}
    TmpPz::Array{Float32,2}
end

function PureJulia2DAcoIsoDenQ_DEO2_FDTD(; nz, nx, nbz_cache, nbx_cache, dz, dx, dt)
    nz,nx,nbz_cache,nbx_cache = map(x->round(Int64,x), (nz,nx,nbz_cache,nbx_cache))
    dz,dx,dt = map(x->Float32(x), (dz,dx,dt))

    @assert nx > 0
    @assert nz > 0
    @assert nbx_cache > 0
    @assert nbz_cache > 0

    V      = Array{Float32,2}(undef,nz,nx)
    B      = Array{Float32,2}(undef,nz,nx)
    PSpace = Array{Float32,2}(undef,nz,nx)
    PCur   = Array{Float32,2}(undef,nz,nx)
    POld   = Array{Float32,2}(undef,nz,nx)
    TmpPx  = Array{Float32,2}(undef,nz,nx)
    TmpPz  = Array{Float32,2}(undef,nz,nx)

    c8_1 = Float32(+1225.0 / 1024.0)
    c8_2 = Float32(-245.0 / 3072.0)
    c8_3 = Float32(+49.0 / 5120.0)
    c8_4 = Float32(-5.0 / 7168.0)

    PureJulia2DAcoIsoDenQ_DEO2_FDTD(nz, nx, nbz_cache, nbx_cache, dz, dx, dt, c8_1, c8_2, c8_3, c8_4, V, B, PSpace, PCur, POld, TmpPx, TmpPz)
end

function block_first_touch_timestep!(p::PureJulia2DAcoIsoDenQ_DEO2_FDTD, izrng, ixrng, v0, b0)
    for kx in ixrng
        @turbo for kz in izrng
            p.V[kz,kx] = v0
            p.B[kz,kx] = b0
            p.PSpace[kz,kx] = 0
            p.PCur[kz,kx] = 0
            p.POld[kz,kx] = 0
            p.TmpPx[kz,kx] = 0
            p.TmpPz[kz,kx] = 0
        end
    end
    nothing
end

function block_applyFirstDerivatives2D_PlusHalf_Sandwich!(p::PureJulia2DAcoIsoDenQ_DEO2_FDTD, izrng, ixrng)
    invDz = 1f0 / p.dz
    invDx = 1f0 / p.dx

    for kx in ixrng
        @turbo for kz in izrng
            stencilDPx =
                p.c8_1 * (- p.PCur[kz,kx+0] + p.PCur[kz,kx+1]) +
                p.c8_2 * (- p.PCur[kz,kx-1] + p.PCur[kz,kx+2]) +
                p.c8_3 * (- p.PCur[kz,kx-2] + p.PCur[kz,kx+3]) +
                p.c8_4 * (- p.PCur[kz,kx-3] + p.PCur[kz,kx+4]);

            stencilDPz =
                p.c8_1 * (- p.PCur[kz+0,kx] + p.PCur[kz+1,kx]) +
                p.c8_2 * (- p.PCur[kz-1,kx] + p.PCur[kz+2,kx]) +
                p.c8_3 * (- p.PCur[kz-2,kx] + p.PCur[kz+3,kx]) +
                p.c8_4 * (- p.PCur[kz-3,kx] + p.PCur[kz+4,kx]);

            p.TmpPx[kz,kx] = p.B[kz,kx] * invDx * stencilDPx;
            p.TmpPz[kz,kx] = p.B[kz,kx] * invDz * stencilDPz;
        end
    end
    nothing
end

function block_applyFirstDerivatives2D_MinusHalf_TimeUpdate_Nonlinear!(p::PureJulia2DAcoIsoDenQ_DEO2_FDTD, izrng, ixrng)
    invDz = 1f0 / p.dz
    invDx = 1f0 / p.dx

    for kx in ixrng
        @turbo for kz in izrng
            stencilDPx =
                p.c8_1 * (- p.TmpPx[kz,kx-1] + p.TmpPx[kz,kx+0]) +
                p.c8_2 * (- p.TmpPx[kz,kx-2] + p.TmpPx[kz,kx+1]) +
                p.c8_3 * (- p.TmpPx[kz,kx-3] + p.TmpPx[kz,kx+2]) +
                p.c8_4 * (- p.TmpPx[kz,kx-4] + p.TmpPx[kz,kx+3]);

            stencilDPz =
                p.c8_1 * (- p.TmpPz[kz-1,kx] + p.TmpPz[kz+0,kx]) +
                p.c8_2 * (- p.TmpPz[kz-2,kx] + p.TmpPz[kz+1,kx]) +
                p.c8_3 * (- p.TmpPz[kz-3,kx] + p.TmpPz[kz+2,kx]) +
                p.c8_4 * (- p.TmpPz[kz-4,kx] + p.TmpPz[kz+3,kx]);
                
            dP = invDx * stencilDPx + invDz * stencilDPz;

            dt2V2_B = p.dt * p.dt * p.V[kz,kx] * p.V[kz,kx] / p.B[kz,kx];

            p.POld[kz,kx] = dt2V2_B * dP - 0.001 * (p.PCur[kz,kx] - p.POld[kz,kx]) - p.POld[kz,kx] + 2 * p.PCur[kz,kx];
            p.PSpace[kz,kx] = dP;
        end
    end

    nothing
end

function zero_annulus!(p::PureJulia2DAcoIsoDenQ_DEO2_FDTD)
    for k = 1:4
        @turbo for kx = 1:p.nz
            p.TmpPx[k,kx] = 0
            p.TmpPz[k,kx] = 0
            p.PSpace[k,kx] = 0
            
            p.TmpPx[p.nz-k+1,kx] = 0
            p.TmpPz[p.nz-k+1,kx] = 0
            p.PSpace[p.nz-k+1,kx] = 0
        end
        
        @turbo for kz = 1:p.nx
            p.TmpPx[kz,k] = 0
            p.TmpPz[kz,k] = 0
            p.PSpace[kz,k] = 0
            
            p.TmpPx[kz,p.nx-k+1] = 0
            p.TmpPz[kz,p.nx-k+1] = 0
            p.PSpace[kz,p.nx-k+1] = 0
        end
    end
    nothing
end

function first_touch!(p::PureJulia2DAcoIsoDenQ_DEO2_FDTD, v0, b0)
    for iz in 5:p.nbz_cache:p.nz-4
        @floop ThreadedEx() for ix in 5:p.nbx_cache:p.nx-4
            block_first_touch_timestep!(p, iz:min(p.nz-4,iz+p.nbz_cache-1), ix:min(p.nx-4,ix+p.nbx_cache-1), v0, b0)
        end
    end
    nothing
end

function applyFirstDerivatives2D_PlusHalf_Sandwich!(p::PureJulia2DAcoIsoDenQ_DEO2_FDTD)
    for iz in 5:p.nbz_cache:p.nz-4
        @floop ThreadedEx() for ix in 5:p.nbx_cache:p.nx-4
            block_applyFirstDerivatives2D_PlusHalf_Sandwich!(p, 
                iz:min(p.nz-4,iz+p.nbz_cache-1), ix:min(p.nx-4,ix+p.nbx_cache-1))
        end
    end
    nothing
end

function applyFirstDerivatives2D_MinusHalf_TimeUpdate_Nonlinear!(p::PureJulia2DAcoIsoDenQ_DEO2_FDTD)
    for iz in 5:p.nbz_cache:p.nz-4
        @floop ThreadedEx() for ix in 5:p.nbx_cache:p.nx-4
            block_applyFirstDerivatives2D_MinusHalf_TimeUpdate_Nonlinear!(p, 
                iz:min(p.nz-4,iz+p.nbz_cache-1), ix:min(p.nx-4,ix+p.nbx_cache-1))
        end
    end
    nothing
end

function propagateforward!(p::PureJulia2DAcoIsoDenQ_DEO2_FDTD)
    applyFirstDerivatives2D_PlusHalf_Sandwich!(p)
    applyFirstDerivatives2D_MinusHalf_TimeUpdate_Nonlinear!(p)
    
    # source injection 
    ksz, ksx = div(p.nz,2)+1, div(p.nx,2)+1
    p.PCur[ksz,ksx] += p.dt^2 * p.V[ksz,ksx]^2 / p.B[ksz,ksx];
    
    p.PCur, p.POld = p.POld, p.PCur # swap pointers 
end

function testme()
    nt = 5000
    nx = 2001
    nz = 2001
    nbx_cache = 8
    nbz_cache = 5000
    dx,dz,dt = 25f0,25f0,0.001f0

    p = PureJulia2DAcoIsoDenQ_DEO2_FDTD(nz = nz, nx = nx, nbz_cache = nbz_cache, nbx_cache = nbx_cache, dz = dz, dx = dx, dt = dt);
    first_touch!(p,1500.0f0,1.0f0)
    zero_annulus!(p)

    set_zero_subnormals(true)

    # warmup
    for kt = 1:2
        propagateforward!(p)
    end

    # timing
    @show "begin"
    t = @elapsed for kt = 1:nt
        propagateforward!(p)
    end
    @show "end"

    set_zero_subnormals(false)

    mc  = nx * nz * nt / (1000 * 1000)
    mcs = mc / t

    @printf("%3d %12.4f\n", Threads.nthreads(), mcs)
end

testme()