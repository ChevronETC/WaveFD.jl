using Polyester, Printf, BenchmarkTools, LoopVectorization

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

    V      = zeros(Float32,nz,nx)
    B      = zeros(Float32,nz,nx)
    PSpace = zeros(Float32,nz,nx)
    PCur   = zeros(Float32,nz,nx)
    POld   = zeros(Float32,nz,nx)
    TmpPx  = zeros(Float32,nz,nx)
    TmpPz  = zeros(Float32,nz,nx)

    c8_1 = Float32(+1225.0 / 1024.0)
    c8_2 = Float32(-245.0 / 3072.0)
    c8_3 = Float32(+49.0 / 5120.0)
    c8_4 = Float32(-5.0 / 7168.0)

    PureJulia2DAcoIsoDenQ_DEO2_FDTD(nz, nx, nbz_cache, nbx_cache, dz, dx, dt, c8_1, c8_2, c8_3, c8_4, V, B, PSpace, PCur, POld, TmpPx, TmpPz)
end

Base.size(p::PureJulia2DAcoIsoDenQ_DEO2_FDTD) = (p.nz,p.nx)

@inline function block_applyFirstDerivatives2D_PlusHalf_Sandwich!(p::PureJulia2DAcoIsoDenQ_DEO2_FDTD, izrng, ixrng)
    invDz = 1f0 / p.dz
    invDx = 1f0 / p.dx

    for kx in ixrng
        @simd ivdep for kz in izrng
            @fastmath @inbounds begin
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
    end
    nothing
end

@inline function block_applyFirstDerivatives2D_MinusHalf_TimeUpdate_Nonlinear!(p::PureJulia2DAcoIsoDenQ_DEO2_FDTD, izrng, ixrng)
    invDz = 1f0 / p.dz
    invDx = 1f0 / p.dx

    for kx in ixrng
        @simd ivdep for kz in izrng
            @fastmath @inbounds begin
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
    end
    nothing
end

@inline function applyFirstDerivatives2D_PlusHalf_Sandwich!(p::PureJulia2DAcoIsoDenQ_DEO2_FDTD)
    for iz in 5:p.nbz_cache:p.nz-4
        Threads.@threads for ix in 5:p.nbx_cache:p.nx-4
            block_applyFirstDerivatives2D_PlusHalf_Sandwich!(p, 
                iz:min(p.nz-4,iz+p.nbz_cache-1), ix:min(p.nx-4,ix+p.nbx_cache-1))
        end
    end
    nothing
end

@inline function applyFirstDerivatives2D_MinusHalf_TimeUpdate_Nonlinear!(p::PureJulia2DAcoIsoDenQ_DEO2_FDTD)
    for iz in 5:p.nbz_cache:p.nz-4
        Threads.@threads for ix in 5:p.nbx_cache:p.nx-4
            block_applyFirstDerivatives2D_MinusHalf_TimeUpdate_Nonlinear!(p, 
                iz:min(p.nz-4,iz+p.nbz_cache-1), ix:min(p.nx-4,ix+p.nbx_cache-1))
        end
    end
    nothing
end

@inline function propagateforward!(p::PureJulia2DAcoIsoDenQ_DEO2_FDTD)
    # @tturbo p.TmpPx .= 0;
    # @tturbo p.TmpPz .= 0;
    # @tturbo p.PSpace .= 0;

    applyFirstDerivatives2D_PlusHalf_Sandwich!(p)
    applyFirstDerivatives2D_MinusHalf_TimeUpdate_Nonlinear!(p)
    
    # source injection 
    ksz, ksx = div(p.nz,2)+1, div(p.nx,2)+1
    p.PCur[ksz,ksx] += p.dt^2 * p.V[ksz,ksx]^2 / p.B[ksz,ksx];
    
    p.PCur, p.POld = p.POld, p.PCur # swap pointers 
end

function testme()
    nt = 100
    nx = 2001
    nz = 2001
    nbx_cache = 8
    nbz_cache = 5000
    dx,dz,dt = 25f0,25f0,0.001f0

    p = PureJulia2DAcoIsoDenQ_DEO2_FDTD(nz = nz, nx = nx, nbz_cache = nbz_cache, nbx_cache = nbx_cache, dz = dz, dx = dx, dt = dt);

    p.V .= 1500;
    p.B .= 1;

    # @code_warntype propagateforward!(p);
    propagateforward!(p)

    set_zero_subnormals(true)
    t = @elapsed for kt = 1:nt
        propagateforward!(p)
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