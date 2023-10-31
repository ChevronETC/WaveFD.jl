abstract type Wavelet end

function minphase(wav::Array{T}) where T<:AbstractFloat
    n = length(wav)

    # zero pad
    wav = [wav; zeros(T,n)]

    # fft
    WAV = fft(wav)

    # log(|WAV|)
    logabswav = log.(abs.(WAV) .+ Base.eps(T))

    # Hilbert transform of log(|WAV|)
    hlogabswav = imag.(hilbert(logabswav)) # DSP.hilbert gives the analytic signal

    # min-phase:
    PHI = -hlogabswav

    # min-phase wavelet:
    WAV = abs.(WAV) .* exp.(im * PHI)
    wav = real.(ifft(WAV))
    wav[1:n]
end

function phaseshift(wav::Array{T}, phi::T) where T<:AbstractFloat
    if isapprox(phi, 0.0)
        return wav
    end
    n = length(wav)

    # zero pad
    wav = [wav; zeros(n)]

    # fft
    WAV = rfft(wav)

    # phase shift
    for i = 1:length(WAV)
        WAV[i] *= exp.(-im*phi)
    end
    irfft(WAV, length(wav))[1:n]
end

function costap(w::Wavelet, t::Array{T}) where T<:AbstractFloat
    if w.tmax < 0
        return ones(T,length(t))
    end
    itstrt = clamp(round(Int, .9*w.tmax/t[end]),1,length(t))
    itends = clamp(round(Int, (w.tmax/t[end])*length(t)),1,length(t))
    tap = ones(T,length(t))
    for i = 1:length(t)
        if itstrt <= i <= itends
            tap[i] = .5*(1.0 + cos(pi*(i-itstrt)/(itends-itstrt)))
        elseif i > itends
            tap[i] = 0.0
        end
    end
    tap
end

mutable struct WaveletSine <: Wavelet
    a::Float64
    f::Float64
    integrate::Bool
end
WaveletSine(;a = 1.0, f = 10.0, integrate = false) = WaveletSine(a, f, integrate)
copy(w::WaveletSine) = WaveletSine(w.a,w.f,w.integrate)

mutable struct WaveletRicker <: Wavelet
    a::Float64
    f::Float64
    tmax::Float64
    phase::Float64
    integrate::Bool
end
WaveletRicker(;a = 1.0, f = 15.0, tmax = -1.0, phase = 0.0, integrate = false) = WaveletRicker(a, f, tmax, phase, integrate)
copy(w::WaveletRicker) = WaveletRicker(w.a,w.f,w.tmax,w.phase,w.integrate)

mutable struct WaveletMinPhaseRicker <: Wavelet
    a::Float64
    f::Float64
    tmax::Float64
    integrate::Bool
end
WaveletMinPhaseRicker(;a = 1.0, f = 15.0, tmax = -1.0, integrate = false) = WaveletMinPhaseRicker(a, f, tmax, integrate)
copy(w::WaveletMinPhaseRicker) = WaveletMinPhaseRicker(w.a,w.f,w.tmax,w.integrate)

mutable struct WaveletDerivRicker <: Wavelet
    a::Float64
    f::Float64
    tmax::Float64
end
WaveletDerivRicker(;a = 1.0, f = 15.0, tmax = -1.0) = WaveletDerivRicker(a, f, tmax)
copy(w::WaveletDerivRicker) = WaveletDerivRicker(w.a,w.f,w.tmax)

mutable struct WaveletCausalRicker <: Wavelet
    a::Float64
    f::Float64
    tol::Float64
    tmax::Float64
    integrate::Bool
end
WaveletCausalRicker(;a = 1.0, f = 15.0, tol = 1.e-4, tmax = -1.0, integrate = false) = WaveletCausalRicker(a, f, tol, tmax, integrate)
copy(w::WaveletCausalRicker) = WaveletCausalRicker(w.a,w.f,w.tol,w.tmax,w.integrate)

mutable struct WaveletOrmsby <: Wavelet
    a::Float64
    f1::Float64
    f2::Float64
    f3::Float64
    f4::Float64
    tmax::Float64
    integrate::Bool
end
WaveletOrmsby(;a = 1.0, f1 = 5.0, f2 = 10.0, f3 = 40.0, f4 = 45.0, tmax = -1.0, integrate = false) = WaveletOrmsby(a, f1, f2, f3, f4, tmax, integrate)
copy(w::WaveletOrmsby) = WaveletOrmsby(w.a,w.f1,w.f2,w.f3,w.f4,w.tmax,w.integrate)

mutable struct WaveletMinPhaseOrmsby <: Wavelet
    a::Float64
    f1::Float64
    f2::Float64
    f3::Float64
    f4::Float64
    tmax::Float64
    integrate::Bool
end
WaveletMinPhaseOrmsby(;a = 1.0, f1 = 5.0, f2 = 10.0, f3 = 40.0, f4 = 45.0, tmax = -1.0, integrate = false) = WaveletMinPhaseOrmsby(a, f1, f2, f3, f4, tmax, integrate)
copy(w::WaveletMinPhaseOrmsby) = WaveletMinPhaseOrmsby(w.a,w.f1,w.f2,w.f3,w.f4,w.tmax,w.integrate)

mutable struct WaveletCausalOrmsby <: Wavelet
    a::Float64
    f1::Float64
    f2::Float64
    f3::Float64
    f4::Float64
    tmax::Float64
    integrate::Bool
    tol::Float64
end
WaveletCausalOrmsby(;a = 1.0, f1 = 5.0, f2 = 10.0, f3 = 40.0, f4 = 45.0, tmax = -1.0, integrate = false,tol=1e-4) = WaveletCausalOrmsby(a, f1, f2, f3, f4, tmax, integrate,tol)
copy(w::WaveletCausalOrmsby) = WaveletCausalOrmsby(w.a,w.f1,w.f2,w.f3,w.f4,w.tmax,w.integrate,w.tol)

function get(w::WaveletSine, t::Array{T}) where T<:AbstractFloat
    wav = T(w.a) .* sin.(T(2 * pi * w.f) .* t)
    w.integrate ? cumsum(wav, dims=1) : wav
end

function get(w::WaveletRicker, t::Array{T}) where T<:AbstractFloat
    wav = convert(Array{T},w.a * (1.0 .- 2.0 * (pi * w.f * t).^2) .* exp.(-(pi * w.f * t).^2))
    wav = phaseshift(wav, T(w.phase))
    wav .*= costap(w,t)
    w.integrate ? cumsum(wav, dims=1) : wav
end

function get(w::WaveletMinPhaseRicker, t::Array{T}) where T<:AbstractFloat
    wav = minphase(get(WaveletRicker(w.a,w.f,-1.0,0.0,false),t .- mean(t)))
    wav .*= costap(w,t)
    w.integrate ? cumsum(wav, dims=1) : wav
end

function get(w::WaveletDerivRicker, t::Array{T}) where T<:AbstractFloat
    wav1 = get(WaveletRicker(w.a,w.f,-1.0,0.0,false),t)
    wav = similar(wav1)
    wav[1] = wav1[1]
    for i = 2:length(wav1)
        wav[i] = wav1[i] - wav1[i-1]
    end
    wav .* costap(w,t)
end

function get(w::WaveletCausalRicker, t::Array{T}) where T<:AbstractFloat
    wav = get(WaveletRicker(w.a,w.f,-1.0,0.0,false),t .- mean(t))
    idx = findall(ww->abs(ww)>w.tol*w.a,wav)[1]
    rng = idx:length(wav)
    wav[1:length(rng)] = wav[rng]
    wav[length(rng)+1:end] .= 0.0
    wav .*= costap(w,t)
    w.integrate ? cumsum(wav, dims=1) : wav
end

function get(w::WaveletOrmsby, t::Array{T}) where T<:AbstractFloat
    numerator(f,t) = sinc.(f*t).^2 .* (pi*f)^2
    pf43 = T(pi*(w.f4-w.f3))
    pf21 = T(pi*(w.f2-w.f1))
    wav = numerator(T(w.f4),t)/pf43 - numerator(T(w.f3),t)/pf43 - numerator(T(w.f2),t)/pf21 + numerator(T(w.f1),t)/pf21
    wav ./= maximum(abs, wav)
    wav .*= w.a
    wav .-= mean(wav)
    wav = wav .* costap(w,t)
    w.integrate ? cumsum(wav, dims=1) : wav
end

function get(w::WaveletMinPhaseOrmsby, t::Array{T}) where T<:AbstractFloat
    wav = minphase(get(WaveletOrmsby(w.a,w.f1,w.f2,w.f3,w.f4,w.tmax,false),t .- mean(t)))
    wav ./= maximum(abs, wav)
    wav .*= w.a
    wav .-= mean(wav)
    wav .*= costap(w,t)
    w.integrate ? cumsum(wav, dims=1) : wav
end

function get(w::WaveletCausalOrmsby, t::Array{T}) where T<:AbstractFloat
    wav = get(WaveletOrmsby(w.a,w.f1,w.f2,w.f3,w.f4,-1.0,false),t .- mean(t))
    idx = findall(ww->abs(ww)>w.tol*w.a,wav)[1]
    rng = idx:length(wav)
    wav[1:length(rng)] = wav[rng]
    wav[length(rng)+1:end] .= 0.0
    wav .*= WaveFD.costap(w,t)
    w.integrate ? cumsum(wav, dims=1) : wav
end