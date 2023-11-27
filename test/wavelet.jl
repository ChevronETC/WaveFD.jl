using FFTW, LinearAlgebra, Test, WaveFD

@testset "Wavelets" begin
    @testset "Minimum phase Ricker wavelet tests" begin
        dt = .001
        nt = 512
        wav = WaveletMinPhaseRicker()
        for t0 in [-51 -50 -49 0 49 50 51]
            w = get(wav, dt*t0 .+ dt * collect(0:nt-1))
            @test length(w) == nt
        end

        # check amplitude spectrum of min-phase ricker
        nt=1024
        dt=.004
        w1 = get(WaveletRicker(f=15.0), dt*collect(-div(nt,2):div(nt,2)-1))
        w2 = get(WaveletMinPhaseRicker(f=15.0), dt*collect(0:nt-1))
        W1 = abs.(rfft(w1))
        W2 = abs.(rfft(w2))
        err = norm(abs.(W2) - abs.(W1))/length(W1)
        @test isapprox(err, 0.0, atol=.01*maximum(W1))
    end

    @testset "Causal ormsby wavelet tests" begin
        # check amplitude spectrum of Causal Ormsby
        nt=1024
        dt=.004
        f1,f2,f3,f4 = 2.0,4.0,6.0,8.0

        w1 = get(WaveletOrmsby(a=1.0, f1=f1, f2=f2, f3=f3, f4=f4), dt*collect(-div(nt,2):div(nt,2)-1))
        w2 = get(WaveletCausalOrmsby(a=1.0, f1=f1, f2=f2, f3=f3, f4=f4,tol=0.004), dt*collect(-div(nt,2):div(nt,2)-1))
        W1 = abs.(rfft(w1))
        W2 = abs.(rfft(w2))
        err = norm(abs.(W2) - abs.(W1))/length(W1)
        @test isapprox(err, 0.0, atol=.01*maximum(W1))
    end

    @testset "Types, wtype=$(wtype)" for wtype in (WaveletSine, WaveletRicker, WaveletMinPhaseRicker, WaveletDerivRicker, WaveletOrmsby, WaveletMinPhaseOrmsby, WaveletCausalOrmsby)
        @test eltype(get(wtype(), Float64(.004)*collect(0:511))) == Float64
        @test eltype(get(wtype(), Float32(.004)*collect(0:511))) == Float32
    end

    @testset "copy methods" begin
        for W in (WaveletSine, WaveletRicker, WaveletMinPhaseRicker, WaveletDerivRicker, WaveletCausalRicker, WaveletOrmsby, WaveletMinPhaseOrmsby, WaveletCausalOrmsby)
            global w = W()
            global wcopy = copy(w)
            for f in fieldnames(typeof(w))
                v = eval(Meta.parse("w.$(f)"))
                vcopy = eval(Meta.parse("wcopy.$(f)"))
                @test v ≈ vcopy
            end
        end
    end
end



nothing