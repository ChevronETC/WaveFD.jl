using Distributed
addprocs(2)
@everywhere using DistributedArrays, LinearAlgebra, Random, Test, Wave

macro ignore(ex::Expr)
end

@testset "SpaceTime" begin
    @testset "dtmod 2D, T=$(T)" for T in (Float32,Float64)
        for (S,Q,D) in ((:Fornberg,4,1),(:Fornberg,8,1),(:Fornberg,8,2),(:Nihei,8,1))
            dtmod = Wave.default_dtmod(S, Q, D, T(10.0), T(10.0), T(1500.0), T(-1.0), T(0.004), T(0.25))
            @test isapprox(0.004/dtmod, round(0.004/dtmod), rtol=1e-6)
        end
    end

    @testset "dtmod 3D, T=$(T)" for T in (Float32,Float64)
        for (S,Q,D) in ((:Fornberg,8,1),(:Fornberg,8,2))
            dtmod = Wave.default_dtmod(S, Q, D, T(10.0), T(10.0), T(10.0), T(1500.0), T(-1.0), T(0.004), T(0.25))
            @test isapprox(0.004/dtmod, round(0.004/dtmod), rtol=1e-6)
        end
    end

    @testset "ntmod" begin
        ntrec,dtrec,dtmod=128,0.004,0.001
        ntmod = Wave.default_ntmod(dtrec, dtmod, 0.0, ntrec)[2]
        @test (ntrec-1)*dtrec ≈ (ntmod-1)*dtmod
        ntmod = Wave.default_ntmod(dtrec, dtmod, ntrec)
        @test (ntrec-1)*dtrec ≈ (ntmod-1)*dtmod

        t0 = Array{Array}(undef, 2)
        t0[1] = [-0.1]
        t0[2] = [0.0]
        it,ntmod = Wave.default_ntmod(dtrec, dtmod, t0, ntrec)
        @test -(it-1)*dtmod ≈ t0[1][1]
        t0 = distribute(t0)
        it,ntmod = Wave.default_ntmod(dtrec, dtmod, t0, ntrec)
        @test -(it-1)*dtmod ≈ t0[1][1]
    end

    @testset "Time interpolation tests, dot product, T=$(T), mode=$(mode), alg=$(alg), nthreads=$(nthreads), $(length(n)+1)D" for T in (Float32, Float64), mode in (0,1), alg in (Wave.LangJulia(),Wave.LangC()), n in ((),(4,),(4,5)), nthreads=(1,4)
        dtrec=T(.004)
        dtmod=T(.001)
        m = rand(T,256,n...)
        ms = rand(T,256,n...)
        d = rand(T,64,n...)
        ds = rand(T,64,n...)
        Wave.interpforward!(Wave.interpfilters(dtmod,dtrec,mode,alg,nthreads), ds, m)
        Wave.interpadjoint!(Wave.interpfilters(dtmod,dtrec,mode,alg,nthreads), ms, d)
        rhs = dot(ds,d)
        lhs = dot(m,ms)
        err = norm(rhs-lhs)
        write(stdout, "T=$(T), lhs=$(lhs), rhs=$(rhs), mode=$(mode), alg=$(alg), $(length(n)+1)D, err=$(err)\n")
        @test isapprox(err, 0.0, atol=eps(T)*length(m))
    end

    @testset "Time interpolation no-op, mode=$mode, T=$T, n=$n, alg=$alg, nthreads=$nthreads" for mode in (0,1), T in (Float32,Float64), n in ((),(4,),(4,5)), alg in (Wave.LangJulia(), Wave.LangC()), nthreads=(1,4)
        dtrec=T(.004)
        dtmod=T(.004)
        m = rand(T,64,n...)
        ms = rand(T,64,n...)
        d = rand(T,64,n...)
        ds = rand(T,64,n...)
        Wave.interpforward!(Wave.interpfilters(dtmod,dtrec,mode,alg,nthreads), ds, m)
        Wave.interpadjoint!(Wave.interpfilters(dtmod,dtrec,mode,alg,nthreads), ms, d)
        rhs = dot(ds,d)
        lhs = dot(m,ms)
        err = norm(rhs-lhs)
        write(stdout, "T=$(T), lhs=$(lhs), rhs=$(rhs), mode=$(mode), alg=$(alg), $(length(n)+1)D, err=$(err)\n")
        @test isapprox(err, 0.0, atol=eps(T)*length(m))
        @test ds ≈ m
        @test ms ≈ d
    end

    @testset "Time interpolation tests, adjoint, accuracy, T=$(T), alg=$(alg), nthreads=$(nthreads), $(length(n)+1)D" for T in (Float32, Float64), alg=(Wave.LangJulia(),Wave.LangC()), n in ((),(4,),(4,5)), nthreads=(1,4)
        dtrec=T(.004)
        dtmod=T(.001)

        d = Array{T}(undef,64,n...)
        m = Array{T}(undef,256,n...)
        m_check = Array{T}(undef,256,n...)

        td = .004*collect(0:63)
        tm = .001*collect(0:255)

        dd = sin.(10*td) + .2*sin.(30*td)
        mm = sin.(10*tm) + .2*sin.(30*tm)

        d_expected = Array{T}(undef,64,n...)
        m_expected = Array{T}(undef,256,n...)
        for i in CartesianIndices(n)
            d_expected[:,i] .= dd[:]
            m_expected[:,i] .= mm[:]
        end

        I = (8:256-8,)
        I = length(n) == 1 ? (I...,:) : I
        I = length(n) == 2 ? (I...,:,:) : I

        m = similar(m_expected); m[:] .= 0.0
        Wave.interpadjoint!(Wave.interpfilters(dtmod,dtrec,0,alg,nthreads),m,d_expected)
        m *= 4
        err = norm((m-m_expected)[I...])/length(m)
        write(stdout, "T=$(T), interpadjoint!, mode=0, nthreads=$(nthreads), $(length(n)+1)D, err=$(err)\n")
        @test isapprox(err, 0.0, atol=.1)

        m = similar(m_expected); m[:] .= 0.0
        Wave.interpadjoint!(Wave.interpfilters(dtmod,dtrec,1,alg,nthreads),m,d_expected)
        err = norm((m .- m_expected)[I...])/length(m)
        write(stdout, "T=$(T), interpadjoint!, mode=1, nthreads=$(nthreads), $(length(n)+1)D, err=$(err)\n")
        @test isapprox(err, 0.0, atol=.1)
    end

    @testset "Time interpolation tests, forward, accuracy, T=$(T), alg=$(alg), nthreads=$(nthreads), $(length(n)+1)D" for T in (Float32, Float64), alg=(Wave.LangJulia(),Wave.LangC()), n in ((),(4,),(4,5)), nthreads=(1,4)
        dtrec=T(.004)
        dtmod=T(.001)

        d = Array{T}(undef,64,n...)
        m = Array{T}(undef,256,n...)
        m_check = Array{T}(undef,256,n...)

        td = .004*collect(0:63)
        tm = .001*collect(0:255)

        dd = sin.(10 .* td) .+ .2*sin.(30 .* td)
        mm = sin.(10 .* tm) .+ .2*sin.(30 .* tm)

        d_expected = Array{T}(undef,64,n...)
        m_expected = Array{T}(undef,256,n...)
        for i in CartesianIndices(n)
            d_expected[:,i] .= dd[:]
            m_expected[:,i] .= mm[:]
        end

        I = (8:64-8,)
        I = length(n) == 1 ? (I...,:) : I
        I = length(n) == 2 ? (I...,:,:) : I

        d = similar(d_expected); d[:] .= 0.0
        Wave.interpforward!(Wave.interpfilters(dtmod,dtrec,0,alg,nthreads),d,m_expected)
        err = norm((d .- d_expected)[I...])/length(d)
        write(stdout, "T=$(T), interpforward!, mode=0, nthreads=$(nthreads), $(length(n)+1)D, err=$(err)\n")
        @test isapprox(err, 0.0, atol=.1)

        d = similar(d_expected); d[:] .= 0.0
        Wave.interpforward!(Wave.interpfilters(dtmod,dtrec,1,alg,nthreads),d,m_expected)
        d ./= 4
        err = norm((d .- d_expected)[I...])/length(d)
        write(stdout, "T=$(T), interpforward!, mode=1, nthreads=$(nthreads), $(length(n)+1)D, err=$(err)\n")
        @test isapprox(err, 0.0, atol=.1)
    end

    @testset "data injection/extrapolation, 2D off-grid, inner product, T=$(T), F=$(F)" for T in (Float32,Float64), F in (Wave.hickscoeffs, Wave.linearcoeffs), alg=(Wave.LangC(),Wave.LangJulia()), nthreads=(1,4)
        nr = 10
        nz, nx = 100, 100
        dz, dx = 10.0, 10.0
        z0, x0 = 0.0, 0.0
        z1 = 5.0*dz; z2 = dx * (nz - 6)
        x1 = 5.0*dx; x2 = dx * (nx - 6)
        z, x = z1 .+ (z2 - z1) .* rand(nr), x1 .+ (x2 - x1) .* rand(nr)
        iz, ix, c = F(T(dz), T(dx), z0, x0, nz, nx, z, x)

        f1 = rand(T, 1, nr)
        f2 = zeros(T, 1, nr)
        g1 = rand(T, nz, nx)
        g2 = zeros(T, nz, nz)

        Wave.injectdata!(g2, f1, 1, iz, ix, c, 10)
        Wave.extractdata!(f2, g1, 1, iz, ix, c)

        lhs = dot(f1,f2)
        rhs = dot(g1,g2)
        err = abs(lhs - rhs) / abs(lhs + rhs)
        write(stdout, "spacetime, inject/extractdata dot product test (T=$(T), F=$(F)), lhs=$(lhs), rhs=$(rhs), err=$(err)\n")
        @test lhs ≈ rhs
    end

    @testset "data injection/extrapolation, 3D off-grid, inner product, T=$(T), F=$(F)" for T in (Float32,Float64), F in (Wave.hickscoeffs, Wave.linearcoeffs)
        nr = 10
        nz, ny, nx = 100, 100, 100
        dz, dy, dx = 10.0, 10.0, 10.0
        z0, y0, x0 = 0.0, 0.0, 0.0
        z1 = 5.0*dz; z2 = dx * (nz - 6)
        y1 = 5.0*dy; y2 = dy * (ny - 6)
        x1 = 5.0*dx; x2 = dx * (nx - 6)
        z, y, x = z1 .+ (z2 - z1) .* rand(nr), y1 .+ (y2 - y1) .* rand(nr), x1 .+ (x2 - x1) .* rand(nr)
        iz, iy, ix, c = F(T(dz), T(dy), T(dx), z0, y0, x0, nz, ny, nx, z, y, x)

        f1 = rand(T, 1, nr)
        f2 = zeros(T, 1, nr)
        g1 = rand(T, nz, ny, nx)
        g2 = zeros(T, nz, ny, nz)

        Wave.injectdata!(g2, f1, 1, iz, iy, ix, c)
        Wave.extractdata!(f2, g1, 1, iz, iy, ix, c)

        lhs = dot(f1,f2)
        rhs = dot(g1,g2)
        err = abs(lhs - rhs) / abs(lhs + rhs)
        write(stdout, "spacetime, inject/extractdata dot product test (T=$(T), F=$(F)), lhs=$(lhs), rhs=$(rhs), err=$(err)\n")
        @test lhs ≈ rhs
    end

    @testset "data injection/extrapolation, 2D on-grid, dot product tests, T=$(T), F=$(F), lang=$(lang), nthreads=$(nthreads)" for T in (Float32,Float64), F in (Wave.hickscoeffs, Wave.linearcoeffs, Wave.ongridcoeffs), lang in (Wave.LangC(), Wave.LangJulia()), nthreads in (1, 4)
        nz, nx=50, 52
        dz, dx = 10.0, 10.0
        z0, x0 = 0.0, 0.0

        nr = length(5:nx-5)
        rz = 10.0*ones(nr)
        rx = 10.0*collect(5:nx-5)

        m = rand(T,1,nr)    # inject data (source)
        ds = zeros(T,nz,nx) # ... get the field

        d = rand(T,nz,nx)   # given the field
        ms = zeros(T,1,nr)  # ... get the data

        iz, ix, c = F(T(dz), T(dx), z0, x0, nz, nx, rz, rx)

        local optargs
        if F == Wave.ongridcoeffs
            optargs = (lang, nthreads)
        else
            optargs = ()
        end
        Wave.injectdata!(ds, m, 1, iz, ix, c, optargs...)  # m->ds
        Wave.extractdata!(ms, d, 1, iz, ix, c, optargs...) # d->ms
        lhs = dot(m,ms)
        rhs = dot(d,ds)
        err = abs(lhs - rhs) / abs(lhs + rhs)
        write(stdout, "spacetime, inject/extractdata dot product test (T=$(T), F=$(F), lang=$(lang), nthreads=$(nthreads)), lhs=$(lhs), rhs=$(rhs), err=$(err)\n")
        @test lhs ≈ rhs
    end

    @testset "data injection/extrapolation, 3D on-grid, dot product tests, T=$(T), F=$(F)" for T in (Float32,Float64), F in (Wave.hickscoeffs, Wave.linearcoeffs, Wave.ongridcoeffs)
        nz,ny,nx=50,51,52
        dz, dy, dx = 10.0, 10.0, 10.0
        z0, y0, x0 = 0.0, 0.0, 0.0

        rz = 10.0*ones(nx-10,ny-10)
        rx = zeros(nx-10,ny-10)
        ry = zeros(nx-10,ny-10)
        for iy = 1:(ny-10),ix = 1:(nx-10)
            rx[ix,iy] = (ix+5)*dx
            ry[ix,iy] = (iy+5)*dy
        end
        rz = rz[:]
        ry = ry[:]
        rx = rx[:]

        iz, iy, ix, c = F(T(dz), T(dy), T(dx), z0, y0, x0, nz, ny, nx, rz, ry, rx)

        m = rand(T,1,ny*nx)    # inject data (source)
        ds = zeros(T,nz,ny,nx) # ... get the field

        d = rand(T,nz,ny,nx)   # given the field
        ms = zeros(T,1,ny*nx)  # ... get the data

        Wave.injectdata!(ds, m, 1, iz, iy, ix, c)  # m->ds
        Wave.extractdata!(ms, d, 1, iz, iy, ix, c) # d->ms
        lhs = dot(m,ms)
        rhs = dot(d,ds)
        err = abs(lhs - rhs) / abs(lhs + rhs)
        write(stdout, "spacetime, inject/extractdata dot product test (T=$(T), F=$(F)), lhs=$(lhs), rhs=$(rhs), err=$(err)\n")
        @test lhs ≈ rhs

        if F == Wave.ongridcoeffs
            ds_C = copy(ds)
            ms_C = copy(ms)
            ds = zeros(T,nz,ny,nx) # ... get the field
            ms = zeros(T,1,ny*nx)  # ... get the data

            Wave.injectdata!(ds, m, 1, iz, iy, ix, c, Wave.LangJulia())  # m->ds
            Wave.extractdata!(ms, d, 1, iz, iy, ix, c, Wave.LangJulia()) # d->ms

            @test ds ≈ ds_C
            @test ms ≈ ms_C

            lhs = dot(m,ms)
            rhs = dot(d,ds)
            err = abs(lhs - rhs) / abs(lhs + rhs)
            write(stdout, "spacetime, inject/extractdata dot product test (T=$(T), F=$(F)), lhs=$(lhs), rhs=$(rhs), err=$(err), Julia code\n")
            @test lhs ≈ rhs
        end
    end

    @testset "data injection, modeling 2D off-grid accuracy tests, T=$(T),F=$(F),freesurface=$(fs)" for 
            T in (Float32,), F in (Wave.hickscoeffs,Wave.linearcoeffs), fs in (true, false)

        function modeling(T,F,ongrid::Bool)
            nthreads = Sys.CPU_THREADS
            z,x,tmax,dt,nbz,nbx,nsponge = 2.0,2.0,0.5,0.001,10,10,10
            dz = ongrid ? 0.01 : 0.02
            dx = ongrid ? 0.01 : 0.02
            fpeak = ongrid ? 1.5/(5*2*dx)/3 : 1.5/(5*dx)/3
            sz,sx = z/2+0.01,x/2+0.01
            nz,nx,nt = round(Int,z/dz)+1, round(Int,x/dx)+1, round(Int,tmax/dt)+1

            prop = Prop2DAcoIsoDenQ_DEO2_FDTD(
                nz=nz,nx=nx,nsponge=nsponge,dz=dx,dx=dx,dt=dt,freesurface=fs)

            Wave.V(prop) .= 1.5
            Wave.B(prop) .= 1.0
            pcur = Wave.PCur(prop)
            pold = Wave.POld(prop)

            wavelet = convert(Array{T},reshape(get(Wave.WaveletCausalRicker(f=fpeak), dt*collect(0:nt-1)), nt, 1))

            iz, ix, c =  F(T(dz), T(dx), 0.0, 0.0, nz, nx, [sz], [sx])
            blocks = Wave.source_blocking(nz, nx, nbz, nbx, iz, ix, c)

            set_zero_subnormals(true)
            for it = 1:nt
                rem(it,10) == 0 && write(stdout, "..$(it/nt*100) percent..\r")
                Wave.propagateforward!(prop)
                pcur,pold = pold,pcur
                Wave.injectdata!(pcur, blocks, wavelet, it)
            end
            set_zero_subnormals(false)
            pcur./(dz*dx)
        end

        write(stdout, "Computing wavefields, be patient, 2D, T=$(T), F=$(F), freesurface=$(fs) ...\n")
        poff = modeling(T,F,false)
        pon = modeling(T,F,true)[1:2:end,1:2:end]
        r = maximum(vec((poff .- pon) ./ maximum(abs, pon)))
        @test r < .02
    end

    @testset "data injection, modeling 3D off-grid accuracy tests, T=$(T),F=$(F),freesurface=$(fs)" for 
            T in (Float32,), F in (Wave.hickscoeffs,Wave.linearcoeffs), fs in (true, false)

        function modeling(T,F,ongrid::Bool)
            nthreads = Sys.CPU_THREADS
            z,y,x,tmax,dt,nbz,nby,nbx,nsponge = 2.0,2.0,2.0,0.5,0.001,10,10,10,10
            dz = ongrid ? 0.01 : 0.02
            dy = ongrid ? 0.01 : 0.02
            dx = ongrid ? 0.01 : 0.02
            fpeak = ongrid ? 1.5/(5*2*dx)/3 : 1.5/(5*dx)/3
            nz,ny,nx,nt = round(Int,z/dz)+1, round(Int,y/dy)+1, round(Int,x/dx)+1, round(Int,tmax/dt)+1
            sz,sy,sx = z/2+0.01,y/2+0.01,x/2+0.01
            nz,nx,nt = round(Int,z/dz)+1, round(Int,x/dx)+1, round(Int,tmax/dt)+1

            prop = Wave.Prop3DAcoIsoDenQ_DEO2_FDTD(
                nz=nz,ny=ny,nx=nx,nsponge=nsponge,dz=dz,dy=dy,dx=dx,dt=dt,freesurface=fs)

            Wave.V(prop) .= 1.5
            Wave.B(prop) .= 1.0
            pcur = Wave.PCur(prop)
            pold = Wave.POld(prop)

            wavelet = convert(Array{T},reshape(get(Wave.WaveletCausalRicker(f=fpeak), dt*collect(0:nt-1)), nt, 1))

            iz, iy, ix, c = F(T(dz), T(dy), T(dx), 0.0, 0.0, 0.0, nz, ny, nx, [sz], [sy], [sx])
            blocks = Wave.source_blocking(nz, ny, nx, nbz, nby, nbx, iz, iy, ix, c)

            set_zero_subnormals(true)
            for it = 1:nt
                rem(it,10) == 0 && write(stdout, "..$(it/nt*100) percent..\r")
                Wave.propagateforward!(prop)
                pcur,pold = pold,pcur
                Wave.injectdata!(pcur, blocks, wavelet, it)
            end
            set_zero_subnormals(false)
            pcur./(dz*dy*dx)
        end

        write(stdout, "Computing wavefields, be patient, 3D, T=$(T), F=$(F), freesurface=$(fs) ...\n")
        poff = modeling(T,F,false)
        pon = modeling(T,F,true)[1:2:end,1:2:end,1:2:end]
        r = maximum(vec((poff .- pon) ./ maximum(abs, pon)))
        @test r < .1
    end

    @testset "data injection/extrapolation, 2D hicks, on-grid tests" for T in (Float32,Float64)
        nz,nx,dz,dx,z0,x0 = 50,51,10.0,10.0,0.0,0.0
        rng = 6:(nx-6)
        rz = 7 .* dz .* ones(length(rng))
        rx = dx .* collect(rng)
        iz, ix, c = Wave.hickscoeffs(T(dz), T(dx), z0, x0, nz, nx, rz, rx)

        for i = 1:length(rz)
            @test size(c[i]) == (1,1)
            @test c[i] ≈ ones(T,1,1)
            @test length(ix[i]) == 1
            @test length(iz[i]) == 1
            @test ix[i][1,1] == round(Int,(rx[i]-x0)/dx) + 1
            @test iz[i][1,1] == round(Int,(rz[i]-z0)/dz) + 1
        end
    end

    @testset "data injection/extrapolation, 3D hicks, on-grid tests" for T in (Float32,Float64)
        nz,ny,nx,dz,dy,dx,z0,y0,x0 = 50,51,52,10.0,10.0,10.0,0.0,0.0,0.0
        rngx = 6:(nx-6)
        rngy = 6:(ny-6)
        nr = length(rngx)*length(rngy)
        rz = 7 .* dz .* ones(nr)
        ry = zeros(nr)
        rx = zeros(nr)
        ir = 1
        for x in rngx, y in rngy
            rx[ir], ry[ir] = dx*x, dy*y
            ir += 1
        end
        iz, iy, ix, c = Wave.hickscoeffs(T(dz), T(dy), T(dx), z0, y0, x0, nz, ny, nx, rz, ry, rx)

        c_expected = ones(T, 1, 1)
        for i = 1:length(rz)
            @test c[i] ≈ c_expected
        end
    end

    @testset "data injection/extrapolation, bilinear, on-grid tests" for T in (Float32,Float64)
        nz,nx,dz,dx,z0,x0 = 50,51,10.0,10.0,0.0,0.0
        rng = 2:(nx-1)
        rz = 2 .* dz .* ones(length(rng))
        rx = dx .* collect(rng .- 1)
        iz, iy, c = Wave.linearcoeffs(T(dz), T(dx), z0, x0, nz, nx, rz, rx)

        c_expected = zeros(T, 2, 2)
        c_expected[1,1] = 1.0
        for i = 1:length(rz)
            @test c[i] ≈ c_expected
        end
    end

    @testset "data injection/extrapolation, trilinear, on-grid tests" for T in (Float32,Float64)
        nz,ny,nx,dz,dy,dx,z0,y0,x0 = 50,51,52,10.0,10.0,10.0,0.0,0.0,0.0
        rngx = 2:(nx-1)
        rngy = 2:(ny-1)
        nr = length(rngx)*length(rngy)
        rz = 2 .* dz .* ones(nr)
        ry = zeros(nr)
        rx = zeros(nr)
        ir = 1
        for x in rngx, y in rngy
            rx[ir], ry[ir] = dx*(x-1), dy*(y-1)
            ir += 1
        end
        iz, iy, ix, c = Wave.linearcoeffs(T(dz), T(dy), T(dx), z0, y0, x0, nz, ny, nx, rz, ry, rx)

        c_expected = zeros(T, 2, 2, 2)
        c_expected[1,1,1] = 1.0
        for i = 1:length(rz)
            @test c[i] ≈ c_expected
        end
    end

    @testset "ongrid detection tests, 2D" for T in (Float32,Float64)
        z0 = 10.0
        x0 = 20.0
        dz = 5.0
        dx = 3.0
        sz = z0 .+ dz.*(randperm(128) .- 1)
        sx = x0 .+ dx.*(randperm(128) .- 1)
        @test Wave.allongrid(dz, dx, z0, x0, sz, sx) == true
        sz[10] += .1
        @test Wave.allongrid(dz, dx, z0, x0, sz, sx) == false
        sz[10] -= .1
        sx[10] += .1
        @test Wave.allongrid(dz, dx, z0, x0, sz, sx) == false
    end

    @testset "ongrid detection tests, 3D" for T in (Float32,Float64)
        z0 = 10.0
        y0 = 10.0
        x0 = 20.0
        dz = 5.0
        dy = 4.0
        dx = 3.0
        sz = z0 .+ dz.*(randperm(128) .- 1)
        sy = y0 .+ dy.*(randperm(128) .- 1)
        sx = x0 .+ dx.*(randperm(128) .- 1)
        @test Wave.allongrid(dz, dy, dx, z0, y0, x0, sz, sy, sx) == true
        sz[10] += .1
        @test Wave.allongrid(dz, dy, dx, z0, y0, x0, sz, sy, sx) == false
        sz[10] -= .1
        sx[10] += .1
        @test Wave.allongrid(dz, dy, dx, z0, y0, x0, sz, sy, sx) == false
        sx[10] -= .1
        sy[10] += .1
        @test Wave.allongrid(dz, dy, dx, z0, y0, x0, sz, sy, sx) == false
    end

    @testset "ongrid injection coefficients, 2D" for T in (Float32, Float64)
        z0 = 10.0
        x0 = 20.0
        dz = 5.0
        dx = 3.0
        nz = 128
        nx = 128
        z = z0 .+ dz.*(randperm(nx) .- 1)
        x = x0 .+ dx.*(randperm(nx) .- 1)

        iz, ix, c = Wave.ongridcoeffs(dz, dx, z0, x0, nz, nx, z, x)
        @test iz ≈ round.((z .- z0)./dz).+1
        @test ix ≈ round.((x .- x0)./dx).+1
        @test c ≈ ones(size(iz))
    end

    @testset "ongrid injection coefficients, 3D" for T in (Float32, Float64)
        z0 = 10.0
        y0 = 10.0
        x0 = 20.0
        dz = 5.0
        dy = 5.0
        dx = 3.0
        nz = 128
        ny = 128
        nx = 128
        z = z0 .+ dz.*(randperm(nx) .- 1)
        y = y0 .+ dy.*(randperm(nx) .- 1)
        x = x0 .+ dx.*(randperm(nx) .- 1)

        iz, iy, ix, c = Wave.ongridcoeffs(dz, dy, dx, z0, y0, x0, nz, ny, nx, z, y, x)
        @test iz ≈ round.((z .- z0)./dz).+1
        @test iy ≈ round.((y .- y0)./dy).+1
        @test ix ≈ round.((x .- x0)./dx).+1
        @test c ≈ ones(size(iz))
    end
end

nothing