using Distributed
addprocs(2)
@everywhere using DistributedArrays, LinearAlgebra, Random, Test, WaveFD

macro ignore(ex::Expr)
end

@testset "SpaceTime" begin
    @testset "dtmod 2D, T=$(T)" for T in (Float32,Float64)
        for (S,Q,D) in ((:Fornberg,4,1),(:Fornberg,8,1),(:Fornberg,8,2),(:Nihei,8,1))
            dtmod = WaveFD.default_dtmod(S, Q, D, T(10.0), T(10.0), T(1500.0), T(-1.0), T(0.004), T(0.25))
            @test isapprox(0.004/dtmod, round(0.004/dtmod), rtol=1e-6)
        end
    end

    @testset "dtmod 3D, T=$(T)" for T in (Float32,Float64)
        for (S,Q,D) in ((:Fornberg,8,1),(:Fornberg,8,2))
            dtmod = WaveFD.default_dtmod(S, Q, D, T(10.0), T(10.0), T(10.0), T(1500.0), T(-1.0), T(0.004), T(0.25))
            @test isapprox(0.004/dtmod, round(0.004/dtmod), rtol=1e-6)
        end
    end

    @testset "ntmod" begin
        ntrec,dtrec,dtmod=128,0.004,0.001
        ntmod = WaveFD.default_ntmod(dtrec, dtmod, 0.0, ntrec)[2]
        @test (ntrec-1)*dtrec ≈ (ntmod-1)*dtmod
        ntmod = WaveFD.default_ntmod(dtrec, dtmod, ntrec)
        @test (ntrec-1)*dtrec ≈ (ntmod-1)*dtmod

        t0 = Array{Array}(undef, 2)
        t0[1] = [-0.1]
        t0[2] = [0.0]
        it,ntmod = WaveFD.default_ntmod(dtrec, dtmod, t0, ntrec)
        @test -(it-1)*dtmod ≈ t0[1][1]
        t0 = distribute(t0)
        it,ntmod = WaveFD.default_ntmod(dtrec, dtmod, t0, ntrec)
        @test -(it-1)*dtmod ≈ t0[1][1]
    end

    @testset "Time shift test, dot product, T=$(T), $(length(n)+1)D, shift=$(shift)" for T in (Float32, Float64), n in ((),(4,),(4,5)), shift in (0, 9, -4, 7.3, -5.7)
        dtrec=T(.004)
        dtmod=T(.001)
        m = rand(T,256,n...)
        ms = rand(T,256,n...)
        d = rand(T,256,n...)
        ds = rand(T,256,n...)
        WaveFD.shiftforward!(WaveFD.shiftfilter(shift), ds, m)
        WaveFD.shiftadjoint!(WaveFD.shiftfilter(shift), ms, d)
        rhs = dot(ds,d)
        lhs = dot(m,ms)
        err = norm(rhs-lhs)
        write(stdout, "T=$(T), lhs=$(lhs), rhs=$(rhs), $(length(n)+1)D, shift=$(shift) samples, err=$(err)\n")
        @test isapprox(err, 0.0, atol=eps(T)*length(m))
    end

    @testset "Time interpolation tests, dot product, T=$(T), mode=$(mode), alg=$(alg), nthreads=$(nthreads), $(length(n)+1)D" for T in (Float32, Float64), mode in (0,1), alg in (WaveFD.LangJulia(),WaveFD.LangC()), n in ((),(4,),(4,5)), nthreads=(1,4)
        dtrec=T(.004)
        dtmod=T(.001)
        m = rand(T,256,n...)
        ms = rand(T,256,n...)
        d = rand(T,64,n...)
        ds = rand(T,64,n...)
        WaveFD.interpforward!(WaveFD.interpfilters(dtmod,dtrec,mode,alg,nthreads), ds, m)
        WaveFD.interpadjoint!(WaveFD.interpfilters(dtmod,dtrec,mode,alg,nthreads), ms, d)
        rhs = dot(ds,d)
        lhs = dot(m,ms)
        err = norm(rhs-lhs)
        write(stdout, "T=$(T), lhs=$(lhs), rhs=$(rhs), mode=$(mode), alg=$(alg), $(length(n)+1)D, err=$(err)\n")
        @test isapprox(err, 0.0, atol=eps(T)*length(m))
    end

    @testset "Time interpolation no-op, mode=$mode, T=$T, n=$n, alg=$alg, nthreads=$nthreads" for mode in (0,1), T in (Float32,Float64), n in ((),(4,),(4,5)), alg in (WaveFD.LangJulia(), WaveFD.LangC()), nthreads=(1,4)
        dtrec=T(.004)
        dtmod=T(.004)
        m = rand(T,64,n...)
        ms = rand(T,64,n...)
        d = rand(T,64,n...)
        ds = rand(T,64,n...)
        WaveFD.interpforward!(WaveFD.interpfilters(dtmod,dtrec,mode,alg,nthreads), ds, m)
        WaveFD.interpadjoint!(WaveFD.interpfilters(dtmod,dtrec,mode,alg,nthreads), ms, d)
        rhs = dot(ds,d)
        lhs = dot(m,ms)
        err = norm(rhs-lhs)
        write(stdout, "T=$(T), lhs=$(lhs), rhs=$(rhs), mode=$(mode), alg=$(alg), $(length(n)+1)D, err=$(err)\n")
        @test isapprox(err, 0.0, atol=eps(T)*length(m))
        @test ds ≈ m
        @test ms ≈ d
    end

    @testset "Time interpolation tests, adjoint, accuracy, T=$(T), alg=$(alg), nthreads=$(nthreads), $(length(n)+1)D" for T in (Float32, Float64), alg=(WaveFD.LangJulia(),WaveFD.LangC()), n in ((),(4,),(4,5)), nthreads=(1,4)
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
        WaveFD.interpadjoint!(WaveFD.interpfilters(dtmod,dtrec,0,alg,nthreads),m,d_expected)
        m *= 4
        err = norm((m-m_expected)[I...])/length(m)
        write(stdout, "T=$(T), interpadjoint!, mode=0, nthreads=$(nthreads), $(length(n)+1)D, err=$(err)\n")
        @test isapprox(err, 0.0, atol=.1)

        m = similar(m_expected); m[:] .= 0.0
        WaveFD.interpadjoint!(WaveFD.interpfilters(dtmod,dtrec,1,alg,nthreads),m,d_expected)
        err = norm((m .- m_expected)[I...])/length(m)
        write(stdout, "T=$(T), interpadjoint!, mode=1, nthreads=$(nthreads), $(length(n)+1)D, err=$(err)\n")
        @test isapprox(err, 0.0, atol=.1)
    end

    @testset "Time interpolation tests, forward, accuracy, T=$(T), alg=$(alg), nthreads=$(nthreads), $(length(n)+1)D" for T in (Float32, Float64), alg=(WaveFD.LangJulia(),WaveFD.LangC()), n in ((),(4,),(4,5)), nthreads=(1,4)
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
        WaveFD.interpforward!(WaveFD.interpfilters(dtmod,dtrec,0,alg,nthreads),d,m_expected)
        err = norm((d .- d_expected)[I...])/length(d)
        write(stdout, "T=$(T), interpforward!, mode=0, nthreads=$(nthreads), $(length(n)+1)D, err=$(err)\n")
        @test isapprox(err, 0.0, atol=.1)

        d = similar(d_expected); d[:] .= 0.0
        WaveFD.interpforward!(WaveFD.interpfilters(dtmod,dtrec,1,alg,nthreads),d,m_expected)
        d ./= 4
        err = norm((d .- d_expected)[I...])/length(d)
        write(stdout, "T=$(T), interpforward!, mode=1, nthreads=$(nthreads), $(length(n)+1)D, err=$(err)\n")
        @test isapprox(err, 0.0, atol=.1)
    end

    @testset "injection partitioning, 2D" begin
        dz,dx,z0,x0,nz,nx,nthreads = 10.0,10.0,0.0,0.0,100,101,48

        z = rand(100)*dz*nz
        x = rand(100)*dx*nx

        points = WaveFD.hickscoeffs(dz,dx,z0,x0,nz,nx,z,x)
        blks = WaveFD.source_blocking(points, nthreads)
        @test length(blks) <= nthreads
        @test length(blks) <= length(points)

        # ensure that the same point is not in multiple blocks
        for i in eachindex(blks)
            pointsi = [p.iu for p in blks[i]]
            for j in eachindex(blks)
                if i != j
                    for point in blks[j]
                        @test point.iu ∉ pointsi
                    end
                end
            end
        end
    end

    @testset "injection partitioning, 3D" begin
        dz,dy,dx,z0,y0,x0,nz,ny,nx,nthreads = 10.0,10.0,10.0,0.0,0.0,0.0,100,101,102,48

        z = rand(100)*dz*nz
        y = rand(100)*dy*ny
        x = rand(100)*dx*nx

        points = WaveFD.hickscoeffs(dz,dy,dx,z0,y0,x0,nz,ny,nx,z,y,x)
        blks = WaveFD.source_blocking(points, nthreads)
        @test length(blks) <= nthreads
        @test length(blks) <= length(points)

        # ensure that the same point is not in multiple blocks
        for i in eachindex(blks)
            pointsi = [p.iu for p in blks[i]]
            for j in eachindex(blks)
                if i != j
                    for point in blks[j]
                        @test point.iu ∉ pointsi
                    end
                end
            end
        end
    end

    @testset "injection partitioning with empty partitions, 2D" begin
        dz,dx,z0,x0,nz,nx,nthreads = 10.0,10.0,0.0,0.0,100,101,48

        z = 0.25*dz*nz .+ 0.5*rand(3)*dz*nz
        x = 0.25*dz*nz .+ 0.5*rand(3)*dx*nx

        x = [ [x[1] for i=1:10]; [x[2] for i=1:10]; [x[3] for i=1:10] ]
        z = [ [z[1] for i=1:10]; [z[2] for i=1:10]; [z[3] for i=1:10] ]

        points = WaveFD.linearcoeffs(dz,dx,z0,x0,nz,nx,z,x)
        blks = WaveFD.source_blocking(points, nthreads)

        # ensure that there are exactly 4*3 partitions since there are only 3 uniqe ingection locations
        @test length(blks) == 4*3

        # ensure that the same point is not in multiple blocks
        for i in eachindex(blks)
            pointsi = [p.iu for p in blks[i]]
            for j in eachindex(blks)
                if i != j
                    for point in blks[j]
                        @test point.iu ∉ pointsi
                    end
                end
            end
        end
    end

    @testset "injection partitioning with empty partitions, 3D" begin
        dz,dy,dx,z0,y0,x0,nz,ny,nx,nthreads = 10.0,10.0,10.0,0.0,0.0,0.0,100,101,102,48

        z = 0.25*dz*nz .+ 0.5*rand(3)*dz*nz
        y = 0.25*dy*ny .+ 0.5*rand(3)*dy*ny
        x = 0.25*dz*nz .+ 0.5*rand(3)*dx*nx

        z = [ [z[1] for i=1:10]; [z[2] for i=1:10]; [z[3] for i=1:10] ]
        y = [ [y[1] for i=1:10]; [y[2] for i=1:10]; [y[3] for i=1:10] ]
        x = [ [x[1] for i=1:10]; [x[2] for i=1:10]; [x[3] for i=1:10] ]

        points = WaveFD.linearcoeffs(dz,dx,z0,x0,nz,nx,z,x)
        blks = WaveFD.source_blocking(points, nthreads)

        # ensure that there are exactly 4*3 partitions since there are only 3 uniqe ingection locations
        @test length(blks) == 4*3

        # ensure that the same point is not in multiple blocks
        for i in eachindex(blks)
            pointsi = [p.iu for p in blks[i]]
            for j in eachindex(blks)
                if i != j
                    for point in blks[j]
                        @test point.iu ∉ pointsi
                    end
                end
            end
        end
    end

    @testset "injection partitions with on-grid points, 2D, F=$F" for F in (WaveFD.hickscoeffs, WaveFD.linearcoeffs)
        dz,dx,z0,x0,nz,nx,nthreads = 10.0,10.0,0.0,0.0,100,101,48

        x = rand(5:(nx-5), 25)*dx
        z = rand(5:(nz-5), 25)*dz

        points = F(dz,dx,z0,x0,nz,nx,z,x)
        blks = WaveFD.source_blocking(points, nthreads)

        @test mapreduce(length, +, blks) == 25
        for blk in blks
            for point in blk
                @test point.c ≈ 1
            end
        end
    end

    @testset "injection partitions with on-grid points, 3D, F=$F" for F in (WaveFD.hickscoeffs, WaveFD.linearcoeffs)
        dz,dy,dx,z0,y0,x0,nz,ny,nx,nthreads = 10.0,10.0,10.0,0.0,0.0,0.0,100,101,102,48

        x = rand(5:(nx-5), 25)*dx
        y = rand(5:(ny-5), 25)*dy
        z = rand(5:(nz-5), 25)*dz

        points = F(dz,dy,dx,z0,y0,x0,nz,ny,nx,z,y,x)
        blks = WaveFD.source_blocking(points, nthreads)

        @test mapreduce(length, +, blks) == 25
        for blk in blks
            for point in blk
                @test point.c ≈ 1
            end
        end
    end

    @testset "extraction partitioning, 2D, F=$F" for F in (WaveFD.hickscoeffs, WaveFD.linearcoeffs)
        dz,dx,z0,x0,nz,nx,nthreads = 10.0,10.0,0.0,0.0,100,101,48

        z = rand(100)*dz*nz
        x = rand(100)*dx*nx

        points = WaveFD.hickscoeffs(dz,dx,z0,x0,nz,nx,z,x)
        blks = WaveFD.receiver_blocking(points, nthreads)
        @test length(blks) <= nthreads
        @test length(blks) <= length(points)

        # ensure that the same point is not in multiple blocks
        for i in eachindex(blks)
            pointsi = [p.ir for p in blks[i]]
            for j in eachindex(blks)
                if i != j
                    for point in blks[j]
                        @test point.ir ∉ pointsi
                    end
                end
            end
        end
    end

    @testset "extraction partitioning, 3D" begin
        dz,dy,dx,z0,y0,x0,nz,ny,nx,nthreads = 10.0,10.0,10.0,0.0,0.0,0.0,100,101,102,48

        z = rand(100)*dz*nz
        y = rand(100)*dy*ny
        x = rand(100)*dx*nx

        points = WaveFD.hickscoeffs(dz,dy,dx,z0,y0,x0,nz,ny,nx,z,y,x)
        blks = WaveFD.receiver_blocking(points, nthreads)
        @test length(blks) <= nthreads
        @test length(blks) <= length(points)

        # ensure that the same point is not in multiple blocks
        for i in eachindex(blks)
            pointsi = [p.ir for p in blks[i]]
            for j in eachindex(blks)
                if i != j
                    for point in blks[j]
                        @test point.ir ∉ pointsi
                    end
                end
            end
        end
    end

    @testset "data injection/extrapolation, 2D off-grid, inner product, T=$(T), F=$(F), nthreads=$nthreads" for T in (Float32,Float64), F in (WaveFD.hickscoeffs, WaveFD.linearcoeffs), alg=(WaveFD.LangC(),WaveFD.LangJulia()), nthreads=(1,4)
        Random.seed!(0)
        nr = 10
        nz, nx = 100, 100
        dz, dx = 10.0, 10.0
        z0, x0 = 0.0, 0.0
        z1 = 5.0*dz; z2 = dx * (nz - 6)
        x1 = 5.0*dx; x2 = dx * (nx - 6)
        z, x = z1 .+ (z2 - z1) .* rand(nr), x1 .+ (x2 - x1) .* rand(nr)
        points = F(T(dz), T(dx), z0, x0, nz, nx, z, x)

        f1 = rand(T, 1, nr)
        f2 = zeros(T, 1, nr)
        g1 = rand(T, nz, nx)
        g2 = zeros(T, nz, nz)

        WaveFD.injectdata!(g2, f1, 1, points, nthreads)
        WaveFD.extractdata!(f2, g1, 1, points, nthreads)

        lhs = dot(f1,f2)
        rhs = dot(g1,g2)
        err = abs(lhs - rhs) / abs(lhs + rhs)
        write(stdout, "spacetime, inject/extractdata dot product test (T=$(T), F=$(F)), lhs=$(lhs), rhs=$(rhs), err=$(err)\n")
        @test lhs ≈ rhs
    end

    @testset "data injection, modeling 2D off-grid accuracy tests, T=$(T),F=$(F),freesurface=$(fs)" for 
            T in (Float32,), F in (WaveFD.hickscoeffs,WaveFD.linearcoeffs), fs in (true, false)

        function modeling(T,F,ongrid::Bool)
            nthreads = Sys.CPU_THREADS
            z,x,tmax,dt,nsponge = 2.0,2.0,0.5,0.001,10
            dz = ongrid ? 0.01 : 0.02
            dx = ongrid ? 0.01 : 0.02
            fpeak = ongrid ? 1.5/(5*2*dx)/3 : 1.5/(5*dx)/3
            sz,sx = z/2+0.01,x/2+0.01
            nz,nx,nt = round(Int,z/dz)+1, round(Int,x/dx)+1, round(Int,tmax/dt)+1

            prop = Prop2DAcoIsoDenQ_DEO2_FDTD(
                nz=nz,nx=nx,nsponge=nsponge,dz=dx,dx=dx,dt=dt,freesurface=fs)

            WaveFD.V(prop) .= 1.5
            WaveFD.B(prop) .= 1.0
            pcur = WaveFD.PCur(prop)
            pold = WaveFD.POld(prop)

            wavelet = convert(Array{T},reshape(get(WaveFD.WaveletCausalRicker(f=fpeak), dt*collect(0:nt-1)), nt, 1))

            points = F(T(dz), T(dx), 0.0, 0.0, nz, nx, [sz], [sx])
            blocks = WaveFD.source_blocking(points, nthreads)

            set_zero_subnormals(true)
            for it = 1:nt
                rem(it,10) == 0 && write(stdout, "..$(it/nt*100) percent..\r")
                WaveFD.propagateforward!(prop)
                pcur,pold = pold,pcur
                WaveFD.injectdata!(pcur, blocks, wavelet, it, nthreads)
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
            T in (Float32,), F in (WaveFD.hickscoeffs,WaveFD.linearcoeffs), fs in (true, false)

        function modeling(T,F,ongrid::Bool)
            nthreads = Sys.CPU_THREADS
            z,y,x,tmax,dt,nsponge = 2.0,2.0,2.0,0.5,0.001,10
            dz = ongrid ? 0.01 : 0.02
            dy = ongrid ? 0.01 : 0.02
            dx = ongrid ? 0.01 : 0.02
            fpeak = ongrid ? 1.5/(5*2*dx)/3 : 1.5/(5*dx)/3
            nz,ny,nx,nt = round(Int,z/dz)+1, round(Int,y/dy)+1, round(Int,x/dx)+1, round(Int,tmax/dt)+1
            sz,sy,sx = z/2+0.01,y/2+0.01,x/2+0.01
            nz,nx,nt = round(Int,z/dz)+1, round(Int,x/dx)+1, round(Int,tmax/dt)+1

            prop = WaveFD.Prop3DAcoIsoDenQ_DEO2_FDTD(
                nz=nz,ny=ny,nx=nx,nsponge=nsponge,dz=dz,dy=dy,dx=dx,dt=dt,freesurface=fs)

            WaveFD.V(prop) .= 1.5
            WaveFD.B(prop) .= 1.0
            pcur = WaveFD.PCur(prop)
            pold = WaveFD.POld(prop)

            wavelet = convert(Array{T},reshape(get(WaveFD.WaveletCausalRicker(f=fpeak), dt*collect(0:nt-1)), nt, 1))

            points = F(T(dz), T(dy), T(dx), 0.0, 0.0, 0.0, nz, ny, nx, [sz], [sy], [sx])
            blocks = WaveFD.source_blocking(points, nthreads)

            set_zero_subnormals(true)
            for it = 1:nt
                rem(it,10) == 0 && write(stdout, "..$(it/nt*100) percent..\r")
                WaveFD.propagateforward!(prop)
                pcur,pold = pold,pcur
                WaveFD.injectdata!(pcur, blocks, wavelet, it, nthreads)
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

end

nothing