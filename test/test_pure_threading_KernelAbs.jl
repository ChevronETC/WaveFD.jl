using KernelAbstractions, OffsetArrays, Printf, BenchmarkTools

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
    V::OffsetArray{Float32, 2, Array{Float32, 2}}
    B::OffsetArray{Float32, 2, Array{Float32, 2}}
    PSpace::OffsetArray{Float32, 2, Array{Float32, 2}}
    PCur::OffsetArray{Float32, 2, Array{Float32, 2}}
    POld::OffsetArray{Float32, 2, Array{Float32, 2}}
    TmpPx::OffsetArray{Float32, 2, Array{Float32, 2}}
    TmpPz::OffsetArray{Float32, 2, Array{Float32, 2}}
end

function PureJulia2DAcoIsoDenQ_DEO2_FDTD(; nz, nx, nbz_cache, nbx_cache, dz, dx, dt)
    nz,nx,nbz_cache,nbx_cache = map(x->round(Int64,x), (nz,nx,nbz_cache,nbx_cache))
    dz,dx,dt = map(x->Float32(x), (dz,dx,dt))

    @assert nx > 0
    @assert nz > 0
    @assert nbx_cache > 0
    @assert nbz_cache > 0

    V      = B = ones(Float32, nz, nx)
    B      = B = ones(Float32, nz, nx)
    PSpace = B = ones(Float32, nz, nx)
    PCur   = B = ones(Float32, nz, nx)
    POld   = B = ones(Float32, nz, nx)
    TmpPx  = B = ones(Float32, nz, nx)
    TmpPz  = B = ones(Float32, nz, nx)

    c8_1 = Float32(+1225.0 / 1024.0)
    c8_2 = Float32(-245.0 / 3072.0)
    c8_3 = Float32(+49.0 / 5120.0)
    c8_4 = Float32(-5.0 / 7168.0)

    PureJulia2DAcoIsoDenQ_DEO2_FDTD(nz, nx, nbz_cache, nbx_cache, dz, dx, dt, c8_1, c8_2, c8_3, c8_4, 
        V, B, PSpace, PCur, POld, TmpPx, TmpPz)
end

Base.size(p::PureJulia2DAcoIsoDenQ_DEO2_FDTD) = (p.nz,p.nx)

@kernel function block_applyFirstDerivatives2D_PlusHalf_Sandwich!(@Const(PCur), @Const(V), @Const(B), TmpPx, TmpPz, 
        @Const(dz), @Const(dx), @Const(c8_1), @Const(c8_2), @Const(c8_3), @Const(c8_4))

    invDz = 1f0 / dz
    invDx = 1f0 / dx
    
    kz, kx = @index(Global, NTuple)
    
    stencilDPx =
        c8_1 * (- PCur[kz+4,kx+4+0] + PCur[kz+4,kx+4+1]) +
        c8_2 * (- PCur[kz+4,kx+4-1] + PCur[kz+4,kx+4+2]) +
        c8_3 * (- PCur[kz+4,kx+4-2] + PCur[kz+4,kx+4+3]) +
        c8_4 * (- PCur[kz+4,kx+4-3] + PCur[kz+4,kx+4+4]);

    stencilDPz =
        c8_1 * (- PCur[kz+4+0,kx+4] + PCur[kz+4+1,kx+4]) +
        c8_2 * (- PCur[kz+4-1,kx+4] + PCur[kz+4+2,kx+4]) +
        c8_3 * (- PCur[kz+4-2,kx+4] + PCur[kz+4+3,kx+4]) +
        c8_4 * (- PCur[kz+4-3,kx+4] + PCur[kz+4+4,kx+4]);

    TmpPx[kz+4,kx+4] = B[kz+4,kx+4] * invDx * stencilDPx;
    TmpPz[kz+4,kx+4] = B[kz+4,kx+4] * invDz * stencilDPz;

    nothing
end

@kernel function block_applyFirstDerivatives2D_MinusHalf_TimeUpdate_Nonlinear!(@Const(TmpPx), @Const(TmpPz), @Const(V), @Const(B), @Const(PCur),
    POld, PSpace, @Const(dz), @Const(dx), @Const(dt), @Const(c8_1), @Const(c8_2), @Const(c8_3), @Const(c8_4))

    invDz = 1f0 / dz
    invDx = 1f0 / dx

    kz, kx = @index(Global, NTuple)

    stencilDPx =
        c8_1 * (- TmpPx[kz+4,kx+4-1] + TmpPx[kz+4,kx+4+0]) +
        c8_2 * (- TmpPx[kz+4,kx+4-2] + TmpPx[kz+4,kx+4+1]) +
        c8_3 * (- TmpPx[kz+4,kx+4-3] + TmpPx[kz+4,kx+4+2]) +
        c8_4 * (- TmpPx[kz+4,kx+4-4] + TmpPx[kz+4,kx+4+3]);

    stencilDPz =
        c8_1 * (- TmpPz[kz+4-1,kx+4] + TmpPz[kz+4+0,kx+4]) +
        c8_2 * (- TmpPz[kz+4-2,kx+4] + TmpPz[kz+4+1,kx+4]) +
        c8_3 * (- TmpPz[kz+4-3,kx+4] + TmpPz[kz+4+2,kx+4]) +
        c8_4 * (- TmpPz[kz+4-4,kx+4] + TmpPz[kz+4+3,kx+4]);
        
    dP = invDx * stencilDPx + invDz * stencilDPz;

    dt2V2_B = dt * dt * V[kz+4,kx+4] * V[kz+4,kx+4] / B[kz+4,kx+4];

    POld[kz+4,kx+4] = dt2V2_B * dP - 0.001 * (PCur[kz+4,kx+4] - POld[kz+4,kx+4]) - POld[kz+4,kx+4] + 2 * PCur[kz+4,kx+4];
    PSpace[kz+4,kx+4] = dP;

    nothing
end

function propagateforward!(p::PureJulia2DAcoIsoDenQ_DEO2_FDTD, kernel1, kernel2)
    # @tturbo p.TmpPx .= 0;
    # @tturbo p.TmpPz .= 0;
    # @tturbo p.PSpace .= 0;

    wait(kernel1(p.PCur, p.V, p.B, p.TmpPx, p.TmpPz, p.dz, p.dx, p.c8_1, p.c8_2, p.c8_3, p.c8_4))
    wait(kernel2(p.TmpPx, p.TmpPz, p.V, p.B, p.PCur, p.POld, p.PSpace, p.dz, p.dx, p.dt, p.c8_1, p.c8_2, p.c8_3, p.c8_4))

    # source injection 
    ksz, ksx = div(p.nz,2)+1, div(p.nx,2)+1
    p.PCur[ksz,ksx] += p.dt^2 * p.V[ksz,ksx]^2 / p.B[ksz,ksx];
    
    p.PCur, p.POld = p.POld, p.PCur # swap pointers 
end

function testme()
    nt = 10
    nx = 2001
    nz = 2001
    nbx_cache = 8
    nbz_cache = 5000
    dx,dz,dt = 25f0,25f0,0.001f0

    p = PureJulia2DAcoIsoDenQ_DEO2_FDTD(nz = nz, nx = nx, nbz_cache = nbz_cache, nbx_cache = nbx_cache, dz = dz, dx = dx, dt = dt);

    p.V .= 1500;
    p.B .= 1;

    nthrd = 44
    nthrd = 22
    kernel1 = block_applyFirstDerivatives2D_PlusHalf_Sandwich!(CPU(), nthrd, (nz-8,nx-8))
    kernel2 = block_applyFirstDerivatives2D_MinusHalf_TimeUpdate_Nonlinear!(CPU(), nthrd, (nz-8,nx-8))

    # @code_warntype propagateforward!(p);
    propagateforward!(p, kernel1, kernel2)

    set_zero_subnormals(true)
    t = @elapsed for kt = 1:nt
        @time propagateforward!(p, kernel1, kernel2)
    end
    # d = @benchmark propagateforward!($p)
    # display(d)

    set_zero_subnormals(false)
    mc  = nx * nz * nt / (1000 * 1000)
    mcs = mc / t

    @printf("time,mcells,mcells/sec; %12.4f %12.4f %12.4f\n", t, mc, mcs)

    # @show extrema(p.PCur)
    # @show extrema(p.POld)
    # @show extrema(p.PSpace)
end

testme()