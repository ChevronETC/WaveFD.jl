#
# time utility functions
#
function default_dtmod_helper(stenciltype::Symbol, Q::Int, D::Int, mindel::T, velmax::T, dtmod::T, dtrec::T, alpha::T, ndim::Int) where T<:Real
    a = stencilcoeffs(:Fornberg, Q, D, ndim, dtmod, velmax, mindel, T)
    c = courantnumber(a, ndim)
    default_dtmod_helper_helper(mindel, velmax, c, dtmod, dtrec, alpha)
end
function default_dtmod_helper_helper(mindel::T, velmax::T, c::T, dtmod::T, dtrec::T, alpha::T) where T<:Real
    dtmod_auto = alpha*c*mindel/velmax
    if dtmod < 0
        dtmod = dtmod_auto
    end
    if 0.99*dtmod > dtmod_auto
        throw(ArgumentError("dtmod=$(dtmod), dtmod_auto=$(dtmod_auto), mindel=$(mindel), velmax=$(velmax)"))
    end
    T(dtrec/ceil(dtrec/min(dtmod,dtrec) - .1)) # tricky
end

"""
# 2D
    dtmod = Wave.default_dtmod(stenciltype, Q, D, dz, dx, velmax, dtmod, dtrec, alpha)
    dtmod = Wave.default_dtmod(alpha, dz, dx, velmax, dtrec, dtmod) # assumes courant=1 .. should go away once we become less lazy about doing analysis
"""
default_dtmod(stenciltype::Symbol, Q::Int, D::Int, dz::T, dx::T, velmax::T, dtmod::T, dtrec::T, alpha::T) where {T<:AbstractFloat} = default_dtmod_helper(stenciltype, Q, D, min(dz,dx), velmax, dtmod, dtrec, alpha, 2)
default_dtmod(alpha::T, dz::T, dx::T, velmax::T, dtrec::T, dtmod::T) where {T<:AbstractFloat} = default_dtmod_helper_helper(min(dz,dx), velmax, one(T), dtmod, dtrec, alpha)

"""
# 3D
    dtmod = Wave.default_dtmod(stenciltype, Q, D, dz, dy, dx, velmax, dtmod, dtrec, alpha)
    dtmod = Wave.default_dtmod(alpha, dz, dy, dx, velmax, dtrec, dtmod) # assumes courant=1, lazy option, rather than doing proper analysis

where `dtmod::AbstractFloat` is a stable finite-difference time step determined by the C-F-L condition (https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition).
A modified Courant number is used for the C-F-L condition, and is determined by the choice of finite-difference stencil and a scaling factor `alpha`.  Making `alpha` smaller should reduce
numerical dispersion, but at the expense of a smaller `dtmod`.  We have found that setting `alpha=0.25` produces results with small dispersion.  For simplicity, we compute `dtmod`
such that `dtrec` is a scalar multiple of `dtmod`.

## paramters
* `proptype` choose betweeen `:O28_C`,`:O28_Julia`,`:O24_C`, or `:O24_Julia`.  The Courant number depends on the finite difference stencil.  In this case, either 4th or 8th order
* `stenciltype` choose between `:Fornberg` and `:Nihei`.  `:Fornberg` are standard coefficients derived from Taylor. `:Nihei` is an attempt at producing optimal coefficients but only work for constant earth models.
* `Q` is the accuracy order of the stencil, choose between `4` and `8`
* `D` is the derivative order of the stencil, choose between `1` and `2`. Note that `1` assumes staggered, and `2` assumes centered.
* `dz`,`dy`,`dx` are the grid spacing
* `velmax` expected maximum velocity in the earth model
* `dtmod` desired modeling time interval.  If `dtmod<0`, then `dtmod` will be computed according to C-F-L and the recording dt.  Otherwise, we will attempt to use the user specified `dtmod` and throw
an error if it violates C-F-L condition.  However, we will, if needed, reduce `dtmod` so that `dtrec` is a scalar multiple of `dtmod`.
* `dtrec` recording time interval
* `alpha` Courant number multiplier

## Example
```julia
dtmod=-1.0
dz,dx,velmax,dtrec,alpha=0.01,0.01,5.0,0.004,0.25
Q,D=8,1
dtmod = Wave.default_dtmod(:Fornberg, Q, D, dz, dx, velmax, dtmod, dtrec, alpha)
```

## Notes
This method is not exported, because it is not the expectation that the user will need to call this method unless they want/need to by-pass the logic in the
Jot Wavefield propagation operators.
"""
default_dtmod(stenciltype::Symbol, Q::Int, D::Int, dz::T, dy::T, dx::T, velmax::T, dtmod::T, dtrec::T, alpha::T) where {T<:AbstractFloat} = default_dtmod_helper(stenciltype, Q, D, min(dz,dy,dx), velmax, dtmod, dtrec, alpha, 3)
default_dtmod(alpha::T, dz::T, dy::T, dx::T, velmax::T, dtrec::T, dtmod::T) where {T<:AbstractFloat} = default_dtmod_helper_helper(min(dz,dy,dx), velmax, one(T), dtmod, dtrec, alpha)

"""
    it0, ntmod = Wave.default_ntmod(dtrec,dtmod,t0,ntrec)

`it0` the time index corresponding to zero-time, and `ntmod` the total number of time points, including negative time.
`dtrec`,`dtmod` and `ntrec` are `Real`, while `t0` can either be `Real`, `Array{Array{Real}}` or `DArray{Array{Real}}`.  If `t0` is
of type `Array{Real}`, then `it0` is determined using `minimum(t0)`.
"""
function default_ntmod(dtrec::Real, dtmod::Real, t0::Real, ntrec::Int)
    ntmod = round(Int,dtrec/dtmod)*(ntrec - 1) + 1
    it0 = max(round(Int64, -t0/dtmod) + 1, 1)
    ntmod += it0-1
    it0, ntmod
end
default_ntmod(dtrec::Real, dtmod::Real, t0::Array, ntrec::Int) = default_ntmod(dtrec, dtmod, mapreduce(i->minimum(t0[i]), min, 1:length(t0)), ntrec)

function default_ntmod(dtrec::Real, dtmod::Real, t0::DArray, ntrec::Int)
    pids = procs(t0)
    npids = length(pids)
    r = Array{Future}(undef, npids)
    @sync for (i,pid) in enumerate(pids)
        r[i] = @spawnat pid begin
            t0_local = localpart(t0)
            t0min_local = minimum(t0_local[1])
            for j = 2:length(t0_local)
                t0min_local = minimum(t0_local[j]) < t0min_local ? minimum(t0_local[j]) : t0min_local
            end
            t0min_local
        end
    end
    t0min = fetch(r[1])
    for i = 2:npids
        t0test = fetch(r[i])
        t0min = t0test < t0min ? t0test : t0min
    end
    default_ntmod(dtrec, dtmod, t0min, ntrec)
end

"""
    ntmod = default_ntmod(dtrec,dtmod,ntrec)

`ntmod` is the total number of time points.  It is assuming that the first time index corresponds to zero-time.
"""
default_ntmod(dtrec::Real, dtmod::Real, ntrec::Int) = default_ntmod(dtrec, dtmod, 0.0, ntrec)[2]

#
# time interpolation codes
#

mutable struct TimeInterp{T<:AbstractFloat,C<:Language}
    h::Array{Array{T,1},1}
    nthreads::Int64
end

"""
    h = Wave.interpfilters(dtmod, dtrec [, mode=0, impl=Wave.LangC, nthreads=Sys.CPU_THREADS])

Build an 8 point sinc filter, mapping between `dtmod` and `dtrec` sampling.  The optional parameters are:

* `mode::Int` determines if amplitude is preserved in the forward (mode=0) or adjoint operation (mode=1)
* `impl::Wave.Language` can be set to either `Wave.LangJulia()` or `Wave.LangC()` to determine which code-path to follow
* `nthreads`  If `impl=Wave.LangC()`, then OMP is used for threading the 2D and 3D arrays where time is assumed to be along the fast dimension

# Notes
It is assumed that `dtmod<dtrec`:

* forward operator (see `Wave.interpforward!`) - interpolates from `dtmod` to `dtrec`, preserves amplitude if `mode=0`.  This is the default behaviour.
* adjoint operator (see `Wave.interpadjoint!`) - interpolates from `dtrec` to `dtmod`, preserves amplitude if `mode=1`
"""
function interpfilters(dtmod::T, dtrec::T, mode::Int=0, impl::Language=LangC(), nthreads=Sys.CPU_THREADS) where T<:Real
    @assert dtmod <= dtrec
    n = round(Int, dtrec/dtmod)
    if abs(n*dtmod - dtrec)/dtrec > 1e-6
        throw(ArgumentError("abs(n*dtmod-dtrec)/dtrec > 1e-6, n=$(n), dtrec=$(dtrec), dtmod=$(dtmod), abs(n*dtmod-dtrec)/dtrec=$(abs(n*dtmod-dtrec)/n)"))
    end

    d = dtmod/dtrec*[0:n-1;]
    fmax = T(0.066 + 0.265*log(8.0))
    j = [1:8;]

    a = @. sinc((j-1)*fmax)
    A = [a[1] a[2] a[3] a[4] a[5] a[6] a[7] a[8] ;
         a[2] a[1] a[2] a[3] a[4] a[5] a[6] a[7] ;
         a[3] a[2] a[1] a[2] a[3] a[4] a[5] a[6] ;
         a[4] a[3] a[2] a[1] a[2] a[3] a[4] a[5] ;
         a[5] a[4] a[3] a[2] a[1] a[2] a[3] a[4] ;
         a[6] a[5] a[4] a[3] a[2] a[1] a[2] a[3] ;
         a[7] a[6] a[5] a[4] a[3] a[2] a[1] a[2] ;
         a[8] a[7] a[6] a[5] a[4] a[3] a[2] a[1]]

    h = Array{T}[]
    for i = 1:n
        c = @. sinc((4-j+d[i])*fmax)
        push!(h, A\c)
        if mode == 0
            h[i] ./= n # denom is to make sure that the amplitude in the forward is correct
        end
    end

    TimeInterp{T,typeof(impl)}(h,nthreads)
end

"""
    Wave.interpadjoint!(h, m, d)

Interpolate from `d::Array{T,N}` (coarse) to `m::Array{T,N}` (fine) using the sinc filter coefficients in `h::Array{Array{T,1},1}`.  `h` is built using
`Wave.interpfilters`.  For example:

    Wave.interpadjoint!(Wave.interpfilters(.001,.004), m, d)

Note that we support, `N=1`, `N=2` or `N=3`. If `N=2` or `N=3`, then interpolation is done along the fast dimension.  By default, `interpadjoint!` does not preserve
amplitude (see `Wave.interpfilters`).
"""
function interpadjoint!(H::TimeInterp{T,LangJulia}, m::StridedArray{T,1}, d::StridedArray{T,1}) where T
    if length(m) == length(d)
        m .= d
        return nothing
    end
    h = H.h
    nh = length(h)
    nd = length(d)
    nm = length(m)
    fill!(m,0.0)

    interpadjoint_helper!(m,d,h,nm,nh,1,4,8) # i=1
    interpadjoint_helper!(m,d,h,nm,nh,2,3,8) # i=2
    interpadjoint_helper!(m,d,h,nm,nh,3,2,8) # i=3
    for i = 4:nd-4
        interpadjoint_helper!(m,d,h,nm,nh,i,1,8)
    end
    interpadjoint_helper!(m,d,h,nm,nh,nd-3,1,7) # i=nd-3
    interpadjoint_helper!(m,d,h,nm,nh,nd-2,1,6) # i=nd-2
    interpadjoint_helper!(m,d,h,nm,nh,nd-1,1,5) # i=nd-1
    nothing
end

@inline function interpadjoint_helper!(m,d,h,nm,nh,i,lo,hi)
    j = (i-1)*nh
    im4 = i-4
    for k=1:nh
        jk = j+k
        jk > nm && continue
        for l = lo:hi
            m[jk] += h[k][l] * d[im4+l]
        end
    end
end

"""
    Wave.interpforward!(h, d, m)

Interpolate from `m::Array{T,N}` (fine) to `d::Array{T,N}` (coarse) using the sinc filter coefficients in `h::Array{Array{T,1},1}`.  `h` is built using
`Wave.interpfilters`.  For example:

    Wave.interpadjoint!(Wave.interpfilters(.001,.004), m, d)

Note that we support, `N=1`, `N=2` or `N=3`. If `N=2` or `N=3`, then interpolation is done along the fast dimension.  By default, `interpforward!` preserves
amplitude (see `Wave.interpfilters!`).
"""
function interpforward!(H::TimeInterp{T,LangJulia}, d::StridedArray{T,1}, m::StridedArray{T,1}) where T
    if length(d) == length(m)
        d .= m
        return nothing
    end
    h = H.h
    nh = length(h)
    nd = length(d)
    nm = length(m)
    fill!(d,0.0)
    interpforward_helper!(d,m,h,nh,1,4,8) # i=1
    interpforward_helper!(d,m,h,nh,2,3,8) # i=2
    interpforward_helper!(d,m,h,nh,3,2,8) # i=3
    for i = 4:nd-4
        interpforward_helper!(d,m,h,nh,i,1,8)
    end
    interpforward_helper!(d,m,h,nh,nd-3,1,7) # i=nd-3
    interpforward_helper!(d,m,h,nh,nd-2,1,6) # i=nd-2
    interpforward_helper!(d,m,h,nh,nd-1,1,5) # i=nd-1
    nothing
end

@inline function interpforward_helper!(d,m,h,nh,i,lo,hi)
    j = (i-1)*nh
    im4 = i-4
    for k = 1:nh
        jk = j + k
        jk > length(m) && continue
        for l = lo:hi
            d[im4+l] += h[k][l]*m[jk]
        end
    end
end

function interpforward!(H::TimeInterp{T,LangJulia}, d::StridedArray{T,2}, m::StridedArray{T,2}) where T
    for i2 = 1:size(m,2)
        d_trace = @view d[:,i2]
        m_trace = @view m[:,i2]
        interpforward!(H, d_trace, m_trace)
    end
    nothing
end

function interpforward!(H::TimeInterp{T,LangJulia}, d::StridedArray{T,3}, m::StridedArray{T,3}) where T
    for i3 = 1:size(m,3), i2 = 1:size(m,2)
        d_trace = @view d[:,i2,i3]
        m_trace = @view m[:,i2,i3]
        interpforward!(H, d_trace, m_trace)
    end
    nothing
end

function interpadjoint!(H::TimeInterp{T,LangJulia}, m::StridedArray{T,2}, d::StridedArray{T,2}) where T
    for i2 = 1:size(m,2)
        d_trace = @view d[:,i2]
        m_trace = @view m[:,i2]
        interpadjoint!(H, m_trace, d_trace)
    end
    nothing
end

function interpadjoint!(H::TimeInterp{T,LangJulia}, m::StridedArray{T,3}, d::StridedArray{T,3}) where T
    for i3 = 1:size(m,3), i2=1:size(m,2)
        d_trace = @view d[:,i2,i3]
        m_trace = @view m[:,i2,i3]
        interpadjoint!(H, m_trace, d_trace)
    end
    nothing
end

#
# fast-path (OpenMP) codes for time interpolation (should be able to delete these once Julia gets shared memory parallel support)
#
function interpadjoint!(H::TimeInterp{Float32,LangC}, m::StridedArray{Float32,1}, d::StridedArray{Float32,1})
    if length(m) == length(d)
        m .= d
        return nothing
    end
    ccall(
        (:interpadjoint_1d_float, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cfloat}}, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t,     Csize_t,   Csize_t),
         H.h,              m,           d,           length(H.h), length(m), length(d)
    )
end

function interpadjoint!(H::TimeInterp{Float64,LangC}, m::StridedArray{Float64,1}, d::StridedArray{Float64,1})
    if length(m) == length(d)
        m .= d
        return nothing
    end
    ccall(
        (:interpadjoint_1d_double, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cdouble}}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,     Csize_t,   Csize_t),
         H.h,               m,            d,            length(H.h), length(m), length(d)
    )
end

function interpadjoint!(H::TimeInterp{Float32,LangC}, m::StridedArray{Float32,2}, d::StridedArray{Float32,2})
    if length(m) == length(d)
        m .= d
        return nothing
    end
    ccall(
        (:interpadjoint_nd_float, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cfloat}}, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t,     Csize_t,   Csize_t,   Csize_t,   Csize_t),
         H.h,              m,           d,           length(H.h), size(m,1), size(d,1), size(d,2), H.nthreads
    )
    nothing
end

function interpadjoint!(H::TimeInterp{Float64,LangC}, m::StridedArray{Float64,2}, d::StridedArray{Float64,2})
    if length(m) == length(d)
        m .= d
        return nothing
    end
    ccall(
        (:interpadjoint_nd_double, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cdouble}}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,     Csize_t,   Csize_t,   Csize_t,   Csize_t),
         H.h,               m,            d,            length(H.h), size(m,1), size(d,1), size(d,2), H.nthreads
    )
    nothing
end

function interpadjoint!(H::TimeInterp{Float32,LangC}, m::StridedArray{Float32,3}, d::StridedArray{Float32,3})
    if length(m) == length(d)
        m .= d
        return nothing
    end
    ccall(
        (:interpadjoint_nd_float, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cfloat}}, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t,     Csize_t,   Csize_t,   Csize_t,             Csize_t),
         H.h,              m,           d,           length(H.h), size(m,1), size(d,1), size(d,2)*size(d,3), H.nthreads
    )
    nothing
end

function interpadjoint!(H::TimeInterp{Float64,LangC}, m::StridedArray{Float64,3}, d::StridedArray{Float64,3})
    if length(m) == length(d)
        m .= d
        return nothing
    end
    ccall(
        (:interpadjoint_nd_double, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cdouble}}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,     Csize_t,   Csize_t,   Csize_t,             Csize_t),
         H.h,               m,            d,            length(H.h), size(m,1), size(d,1), size(d,2)*size(d,3), H.nthreads
    )
    nothing
end

function interpforward!(H::TimeInterp{Float32,LangC}, d::StridedArray{Float32,1}, m::StridedArray{Float32,1})
    if length(d) == length(m)
        d .= m
        return nothing
    end
    ccall(
        (:interpforward_1d_float, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cfloat}}, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t,     Csize_t,   Csize_t),
         H.h,              d,           m,           length(H.h), length(d), length(m)
    )
    nothing
end

function interpforward!(H::TimeInterp{Float64,LangC}, d::StridedArray{Float64,1}, m::StridedArray{Float64,1})
    if length(d) == length(m)
        d .= m
        return nothing
    end
    ccall(
        (:interpforward_1d_double, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cdouble}}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,     Csize_t,   Csize_t),
         H.h,               d,            m,            length(H.h), length(d), length(m)
    )
    nothing
end

function interpforward!(H::TimeInterp{Float32,LangC}, d::StridedArray{Float32,2}, m::StridedArray{Float32,2})
    if length(d) == length(m)
        d .= m
        return nothing
    end
    ccall(
        (:interpforward_nd_float, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cfloat}}, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t,      Csize_t,   Csize_t,   Csize_t,   Csize_t),
         H.h,              d,           m,           length(H.h),  size(d,1), size(m,1), size(m,2), H.nthreads
    )
    nothing
end

function interpforward!(H::TimeInterp{Float64,LangC}, d::StridedArray{Float64,2}, m::StridedArray{Float64,2})
    if length(d) == length(m)
        d .= m
        return nothing
    end
    ccall(
        (:interpforward_nd_double, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cdouble}}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,     Csize_t,   Csize_t,   Csize_t,   Csize_t),
         H.h,               d,            m,            length(H.h), size(d,1), size(m,1), size(m,2), H.nthreads
    )
    nothing
end

function interpforward!(H::TimeInterp{Float32,LangC}, d::StridedArray{Float32,3}, m::StridedArray{Float32,3})
    if length(d) == length(m)
        d .= m
        return nothing
    end
    ccall(
        (:interpforward_nd_float, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cfloat}}, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t,      Csize_t,   Csize_t,   Csize_t,             Csize_t),
         H.h,               d,           m,           length(H.h), size(d,1), size(m,1), size(m,2)*size(m,3), H.nthreads
    )
    nothing
end

function interpforward!(H::TimeInterp{Float64,LangC}, d::StridedArray{Float64,3}, m::StridedArray{Float64,3})
    if length(d) == length(m)
        d .= m
        return nothing
    end
    ccall(
        (:interpforward_nd_double, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cdouble}}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,     Csize_t,   Csize_t,   Csize_t,             Csize_t),
         H.h,               d,            m,            length(H.h), size(d,1), size(m,1), size(m,2)*size(m,3), H.nthreads
    )
    nothing
end

#
# space interpolation codes
#
#
# Graham Hicks' Kaiser windowed sinc function method, Geophysics 67(1), P. 156-166
# paper suggests r=4, with b=6.31 for kmax=pi/2 and with b=4.14 for kmax=2pi/3
#
# For our problem we are using 5 grid points per wave-length --
# k_max = w_max / c_min = 2 pi fmax / c_min
# k_nyq = pi / del = pi / (c_min / (5fmax)) = 5 fmax * pi / c_min
#
# k_max   2 pi fmax / c_min   2   4        1   5
# ----- = ----------------- = - = --    <  - = --
# k_nyq   5 pi fmax / c_min   5   10       2   10
#
# Hicks suggests b=6.42 for kmax=1/2q
kaiser(b::T, r::Int64, x::T) where {T<:Union{Float32,Float64}} = abs(x) > r ? zero(T) : T(besseli(0,b*sqrt(1-(x/r)^2))/besseli(0,b))

function hicks!(d::Matrix, b::Real, r::Integer, alphaz::Real, alphax::Real, fz::Vector, fx::Vector)
    l = 2*r
    @inbounds for i = 1:l
        a = i - r
        z = a - alphaz
        x = a - alphax
        fz[i] = kaiser(b,r,z)*sinc(z)
        fx[i] = kaiser(b,r,x)*sinc(x)
    end
    @inbounds for j = 1:l, i = 1:l
        d[i,j] = fz[i]*fx[j]
    end
end

function hicks(b::Real, r::Integer, alphaz::Real, alphax::Real)
    l = 2*r
    d = Array{eltype(b)}(undef, l, l)
    fz = Array{eltype(b)}(undef, l)
    fx = Array{eltype(b)}(undef, l)
    hicks!(d, b, r, alphaz, alphax, fz, fx)
    d
end

function hicks!(d::Array{T,3}, b::Real, r::Integer, alphaz::Real, alphay::Real, alphax::Real, fz::Vector, fy::Vector, fx::Vector) where T<:Real
    l = 2*r
    @inbounds for i = 1:l
        a = i - r
        z = a - alphaz
        y = a - alphay
        x = a - alphax
        fz[i] = kaiser(b,r,z)*sinc(z)
        fy[i] = kaiser(b,r,y)*sinc(y)
        fx[i] = kaiser(b,r,x)*sinc(x)
    end
    @inbounds for k = 1:l, j = 1:l, i = 1:l
        d[i,j,k] = fz[i]*fy[j]*fx[k]
    end
end

function hicks(b::Real, r::Integer, alphaz::Real, alphay::Real, alphax::Real)
    l = 2*r
    d = Array{eltype(b)}(undef, l, l, l)
    fz = Array{eltype(b)}(undef, l)
    fy = Array{eltype(b)}(undef, l)
    fx = Array{eltype(b)}(undef, l)
    hicks!(d, b, r, alphaz, alphay, alphax, fz, fy, fx)
    d
end

function hickscoeffs(dz::T, dx::T, z0::Float64, x0::Float64, nz::Int64, nx::Int64, z::Array{Float64,1}, x::Array{Float64,1}, fs_index=1, zstagger=0.0, xstagger=0.0) where T
    hicks_b = T(6.42)
    hicks_r = 4
    hicks_l = 2*hicks_r
    hicks_fz = Array{T}(undef, hicks_l)
    hicks_fx = Array{T}(undef, hicks_l)
    iz = Array{Array{Int64,1}}(undef, length(z))
    ix = Array{Array{Int64,1}}(undef, length(z))
    c = Array{Array{T,2}}(undef, length(z))
    zerotol = Base.eps(T)*1e3
    for i = 1:length(z)
        iz_mid = floor(Int, (z[i] - z0)/dz) + 1
        ix_mid = floor(Int, (x[i] - x0)/dx) + 1
        alphaz = T((z[i] - (z0 + (iz_mid - 1 + zstagger)*dz))/dz)
        alphax = T((x[i] - (x0 + (ix_mid - 1 + xstagger)*dx))/dx)

        ongrid = abs(alphaz) < zerotol && abs(alphax) < zerotol
        if ongrid == true
            iz[i] = [iz_mid]
            ix[i] = [ix_mid]
            c[i] = ones(T,1,1)
        else
            iz[i] = collect(iz_mid-hicks_r+1:iz_mid+hicks_r)
            ix[i] = collect(ix_mid-hicks_r+1:ix_mid+hicks_r)
            c[i] = Array{T}(undef, hicks_l, hicks_l)
            hicks!(c[i], hicks_b, hicks_r, alphaz, alphax, hicks_fz, hicks_fx)

            # reflect source terms above the free-surface
            # we assume that if there is no free-surface, then
            # the model would have being padded accordingly, making
            # the below if statement always evaluate to false
            if iz[i][1] == fs_index - 3
                c[i][5,:] -= c[i][3,:]
                c[i][6,:] -= c[i][2,:]
                c[i][7,:] -= c[i][1,:]
            end
            if iz[i][1] == fs_index - 2
                c[i][4,:] -= c[i][2,:]
                c[i][5,:] -= c[i][1,:]
            end
            if iz[i][1] == fs_index - 1
                c[i][3,:] -= c[i][1,:]
            end

            # cut the coefficients beyond the edge of the domain:
            iz_in = findall(z->(fs_index+1)<=z<=nz,iz[i]) # exclude the free-surface since its coefficients should always be zero
            ix_in = findall(x->1<=x<=nx,ix[i])
            c[i] = c[i][iz_in,ix_in]
            iz[i] = iz[i][iz_in]
            ix[i] = ix[i][ix_in]
        end
    end
    iz, ix, c
end

function hickscoeffs(dz::T, dy::T, dx::T, z0::Float64, y0::Float64, x0::Float64, nz::Int64, ny::Int64, nx::Int64, z::Array{Float64,1}, y::Array{Float64,1}, x::Array{Float64,1}, fs_index=1, zstagger=0.0, ystagger=0.0, xstagger=0.0) where T
    hicks_b = T(6.42)
    hicks_r = 4
    hicks_l = 2*hicks_r
    hicks_fz = Array{T}(undef, hicks_l)
    hicks_fy = Array{T}(undef, hicks_l)
    hicks_fx = Array{T}(undef, hicks_l)
    iz = Array{Array{Int64,1}}(undef, length(z))
    iy = Array{Array{Int64,1}}(undef, length(z))
    ix = Array{Array{Int64,1}}(undef, length(z))
    c = Array{Array{T,3}}(undef, length(z))
    zerotol = Base.eps(T)*1e3
    for i = 1:length(z)
        iz_mid = floor(Int, (z[i] - z0)/dz) + 1
        iy_mid = floor(Int, (y[i] - y0)/dy) + 1
        ix_mid = floor(Int, (x[i] - x0)/dx) + 1
        alphaz = T((z[i] - (z0 + (iz_mid - 1 + zstagger)*dz))/dz)
        alphay = T((y[i] - (y0 + (iy_mid - 1 + ystagger)*dy))/dy)
        alphax = T((x[i] - (x0 + (ix_mid - 1 + xstagger)*dx))/dx)

        ongrid = abs(alphaz) < zerotol && abs(alphay) < zerotol && abs(alphax) < zerotol
        if ongrid == true
            iz[i] = [iz_mid]
            iy[i] = [iy_mid]
            ix[i] = [ix_mid]
            c[i] = ones(T,1,1,1)
        else
            iz[i] = collect(iz_mid-hicks_r+1:iz_mid+hicks_r)
            iy[i] = collect(iy_mid-hicks_r+1:iy_mid+hicks_r)
            ix[i] = collect(ix_mid-hicks_r+1:ix_mid+hicks_r)
            c[i] = Array{T}(undef, hicks_l, hicks_l, hicks_l)
            hicks!(c[i], hicks_b, hicks_r, alphaz, alphay, alphax, hicks_fz, hicks_fy, hicks_fx)

            # reflect source terms above the free-surface
            # we assume that if there is no free-surface, then
            # the model would have being padded accordingly, making
            # the below if statement always evaluate to false
            if iz[i][1] == fs_index-3
                c[i][5,:,:] -= c[i][3,:,:]
                c[i][6,:,:] -= c[i][2,:,:]
                c[i][7,:,:] -= c[i][1,:,:]
            end
            if iz[i][1] == fs_index-2
                c[i][4,:,:] -= c[i][2,:,:]
                c[i][5,:,:] -= c[i][1,:,:]
            end
            if iz[i][1] == fs_index-1
                c[i][3,:,:] -= c[i][1,:,:]
            end

            # cut the coefficients beyond the edge of the domain:
            iz_in = findall(z->(fs_index+1)<=z<=nz,iz[i]) # exclude the free-surface since its coefficients should always be zero
            iy_in = findall(y->1<=y<=ny,iy[i])
            ix_in = findall(x->1<=x<=nx,ix[i])
            c[i] = c[i][iz_in,iy_in,ix_in]
            iz[i] = iz[i][iz_in]
            iy[i] = iy[i][iy_in]
            ix[i] = ix[i][ix_in]
        end
    end
    iz, iy, ix, c
end

function linearcoeffs(dz::T, dx::T, z0::Float64, x0::Float64, nz::Int64, nx::Int64, z::Array{Float64,1}, x::Array{Float64,1}, fs_index=1) where T
    iz = Array{Array{Int64,1}}(undef, length(z))
    ix = Array{Array{Int64,1}}(undef, length(z))
    c = Array{Array{T,2}}(undef, length(z))
    for i = 1:length(z)
        kx = floor(Int64, (x[i] - x0) / dx) + 1
        kz = floor(Int64, (z[i] - z0) / dz) + 1

        rx = (x[i] - (x0 + dx*(kx-1))) / dx
        rz = (z[i] - (z0 + dz*(kz-1))) / dz

        c[i] = Array{T}(undef, 2,2)
        c[i][1,1] = (1.0 - rx) * (1.0 - rz)
        c[i][2,1] = (1.0 - rx) *        rz
        c[i][1,2] =        rx  * (1.0 - rz)
        c[i][2,2] =        rx  *        rz

        iz[i] = collect(kz:kz+1)
        ix[i] = collect(kx:kx+1)

        # cut the coefficients beyond the edge of the domain:
        iz_in = findall(z->(fs_index+1)<=z<=nz,iz[i]) # exclude the free-surface since its coefficients should always be zero
        ix_in = findall(x->1<=x<=nx,ix[i])
        c[i] = c[i][iz_in,ix_in]
        iz[i] = iz[i][iz_in]
        ix[i] = ix[i][ix_in]
    end
    iz, ix, c
end

function linearcoeffs(dz::T, dy::T, dx::T, z0::Float64, y0::Float64, x0::Float64, nz::Int64, ny::Int64, nx::Int64, z::Array{Float64,1}, y::Array{Float64,1}, x::Array{Float64,1}, fs_index=1) where T
    iz = Array{Array{Int64,1}}(undef, length(z))
    iy = Array{Array{Int64,1}}(undef, length(z))
    ix = Array{Array{Int64,1}}(undef, length(z))
    c = Array{Array{T,3}}(undef, length(z))
    for i = 1:length(z)
        kx = floor(Int64, (x[i] - x0) / dx) + 1
        ky = floor(Int64, (y[i] - y0) / dy) + 1
        kz = floor(Int64, (z[i] - z0) / dz) + 1

        rx = (x[i] - (x0 + dx*(kx-1))) / dx
        ry = (y[i] - (y0 + dy*(ky-1))) / dy
        rz = (z[i] - (z0 + dz*(kz-1))) / dz

        c[i] = Array{T}(undef, 2,2,2)
        c[i][1,1,1] = (1.0 - rx) * (1.0 - ry) * (1.0 - rz)
        c[i][2,1,1] = (1.0 - rx) * (1.0 - ry) *        rz
        c[i][1,2,1] = (1.0 - rx) *        ry  * (1.0 - rz)
        c[i][2,2,1] = (1.0 - rx) *        ry  *        rz
        c[i][1,1,2] =        rx  * (1.0 - ry) * (1.0 - rz)
        c[i][2,1,2] =        rx  * (1.0 - ry) *        rz
        c[i][1,2,2] =        rx  *        ry  * (1.0 - rz)
        c[i][2,2,2] =        rx  *        ry  *        rz

        iz[i] = collect(kz:kz+1)
        iy[i] = collect(ky:ky+1)
        ix[i] = collect(kx:kx+1)

        # cut the coefficients beyond the edge of the domain:
        iz_in = findall(z->(fs_index+1)<=z<=nz,iz[i]) # exclude the free-surface since its coefficients should always be zero
        iy_in = findall(y->1<=y<=ny,iy[i])
        ix_in = findall(x->1<=x<=nx,ix[i])
        c[i] = c[i][iz_in,iy_in,ix_in]
        iz[i] = iz[i][iz_in]
        iy[i] = iy[i][iy_in]
        ix[i] = ix[i][ix_in]
    end
    iz, iy, ix, c
end

function ongridcoeffs(dz::T, dx::T, z0::Float64, x0::Float64, nz::Int64, nx::Int64, z::Array{Float64,1}, x::Array{Float64,1}) where T
    iz = Array{Int64}(undef, length(z))
    ix = Array{Int64}(undef, length(z))
    c  = Array{T}(undef, length(z))
    for i = 1:length(z)
        iz[i] = round(Int64, (z[i] - z0)/dz) + 1
        ix[i] = round(Int64, (x[i] - x0)/dx) + 1
        c[i]  = one(T)
    end
    iz, ix, c
end

function ongridcoeffs(dz::T, dy::T, dx::T, z0::Float64, y0::Float64, x0::Float64, nz::Int64, ny::Int64, nx::Int64, z::Array{Float64,1}, y::Array{Float64,1}, x::Array{Float64,1}) where T
    iz = Array{Int64}(undef, length(z))
    iy = Array{Int64}(undef, length(z))
    ix = Array{Int64}(undef, length(z))
    c  = Array{T}(undef, length(z))
    for i = 1:length(z)
        iz[i] = round(Int64, (z[i] - z0)/dz) + 1
        iy[i] = round(Int64, (y[i] - y0)/dy) + 1
        ix[i] = round(Int64, (x[i] - x0)/dx) + 1
        c[i]  = one(T)
    end
    iz, iy, ix, c
end

function sourceblock_range(nb, irng)
    b,rm = divrem(length(irng),nb)
    blocks = Vector{UnitRange{Int}}(undef, nb)
    i1 = irng[1]
    for iblock = 1:nb
        i2 = i1 + b - 1
        if iblock <= rm
            i2 += 1
        end
        blocks[iblock] = i1:i2
        i1 = i2+1
    end
    blocks
end

function get_blocks_for_source_point(i, blocks)
    iblock_min = findfirst(block->in(i[1],block), blocks)
    iblock_max = findfirst(block->in(i[end],block), blocks)
    iblock_min:iblock_max
end

struct SourcePoint{T,N}
    c::Array{T,N}
    r::NTuple{N,Vector{Int}}
    index::Int
end

struct SourceBlock{T,N}
    block_range::NTuple{N,UnitRange{Int}}
    source_points::Vector{SourcePoint{T,N}}
end

Base.range(block::SourceBlock, i) = block.block_range[i]

function source_blocking_range(i)
    a = mapreduce(minimum, min, i)::Int
    b = mapreduce(maximum, max, i)::Int
    a:b
end

"""
    sourceblocks = source_blocking(nz, nx, nbz, nbx, iz, ix, c)

Blocks the axes with `nbz*nbx` blocks, and partitions `iz`, `ix` and `c` into
sourceblocks::Vector{SourceBlock}.

# Inputs
* nz,nx is the finite-difference model size
* iz,ix are the injection points
* c are the injection filters (e.g. Hick's coefficients)
* nbz,nbx are the number of blocks in each dimension

see also Wave.injectdata!
"""
function source_blocking(nz, nx, nbz, nbx, iz, ix, c::Vector{<:AbstractMatrix{T}}) where {T}
    izrng = source_blocking_range(iz)
    ixrng = source_blocking_range(ix)
    zblocks = sourceblock_range(nbz, izrng) # TODO change name
    xblocks = sourceblock_range(nbx, ixrng)

    nblocks = nbz*nbx

    source_blocks = [SourceBlock((zblock,xblock), SourcePoint{T,2}[]) for zblock in zblocks, xblock in xblocks]
    for kshot = 1:length(iz)
        source_point = SourcePoint(c[kshot], (iz[kshot],ix[kshot]), kshot)

        jz_blocks = get_blocks_for_source_point(iz[kshot], zblocks)
        jx_blocks = get_blocks_for_source_point(ix[kshot], xblocks)

        for jx_block in jx_blocks, jz_block in jz_blocks
            push!(source_blocks[jz_block,jx_block].source_points, source_point)
        end
    end
    source_blocks
end

function source_blocking(nz, ny, nx, nbz, nby, nbx, iz, iy, ix, c::Vector{<:AbstractArray{T,3}}) where {T}
    izrng = source_blocking_range(iz)
    iyrng = source_blocking_range(iy)
    ixrng = source_blocking_range(ix)
    zblocks = sourceblock_range(nbz, izrng)
    yblocks = sourceblock_range(nby, iyrng)
    xblocks = sourceblock_range(nbx, ixrng)

    nblocks = nbz*nbx

    source_blocks = [SourceBlock((zblock,yblock,xblock), SourcePoint{T,3}[]) for zblock in zblocks, yblock in yblocks, xblock in xblocks]
    for kshot = 1:length(iz)
        source_point = SourcePoint(c[kshot], (iz[kshot],iy[kshot],ix[kshot]), kshot)

        jz_blocks = get_blocks_for_source_point(iz[kshot], zblocks)
        jy_blocks = get_blocks_for_source_point(iy[kshot], yblocks)
        jx_blocks = get_blocks_for_source_point(ix[kshot], xblocks)

        for jx_block in jx_blocks, jy_block in jy_blocks, jz_block in jz_blocks
            push!(source_blocks[jz_block,jy_block,jx_block].source_points, source_point)
        end
    end
    source_blocks
end

"""
  injectdata!(field, data, it, irblk, izblk, ixblk, cblk, izblk_rngs, ixblk_rngs)

Inject data from data[it] into field. `irblk, izblk, ixblk, cblk, izblk_rngs, ixblk_rngs`
are computed using `Wave.injectdata_range`.

The general work-flow is:
```
nz,nx=size(field)
nthreads=20
irblk, izblk, ixblk, cblk, izblk_rngs, ixblk_rngs = Wave.injectdata_range(nz,nx,nbz,nbx,iz,ix,c)
for it = 1:ntrec # time loop
    injectdata!(field, data, it, irblk, izblk, ixblk, cblk, izblk_rngs, ixblk_rngs)
    ...
end
```
"""
function injectdata!(field::AbstractArray, blocks::AbstractArray{<:SourceBlock}, data::AbstractArray, it::Int)
    @sync for block in blocks
        Threads.@spawn injectdata_block!(field, block, data, it)
    end
end

function injectdata_block!(field::AbstractArray{T,2}, block::SourceBlock, data::AbstractArray{T}, it::Int) where {T}
    @inbounds for source_point in block.source_points
        iz,ix,i,c = source_point.r[1],source_point.r[2],source_point.index,source_point.c
        for jz = 1:length(iz)
            kz = iz[jz]
            kz ∈ range(block, 1) || continue
            for jx = 1:length(ix)
                kx = ix[jx]
                kx ∈ range(block, 2) || continue
                
                field[kz,kx] += c[jz,jx]*data[it,i]
            end
        end
    end
end

function injectdata_block!(field::AbstractArray{T,3}, block::SourceBlock, data::AbstractArray{T}, it::Int) where {T}
    @inbounds for source_point in block.source_points
        iz,iy,ix,i,c = source_point.r[1],source_point.r[2],source_point.r[3],source_point.index,source_point.c
        for jz = 1:length(iz)
            kz = iz[jz]
            kz ∈ range(block, 1) || continue
            for jy = 1:length(iy)
                ky = iy[jy]
                ky ∈ range(block, 2) || continue
                for jx = 1:length(ix)
                    kx = ix[jx]
                    kx ∈ range(block, 3) || continue
                
                    field[kz,ky,kx] += c[jz,jy,jx]*data[it,i]
                end
            end
        end
    end
end

"""
  injectdata!(field, data, it, iz, ix, c[, bz=10])
"""
function injectdata!(field::AbstractArray{T,2}, data::AbstractArray{T,2}, it::Integer, iz::AbstractVector{Vector{Int64}}, ix::AbstractVector{Vector{Int64}}, c::AbstractVector{Array{C,2}}, nbz::Integer=10, nbx::Integer=10) where {T,C}
    blocks = Wave.source_blocking(size(field)..., nbz, nbx, iz, ix, c)
    injectdata!(field, blocks, data, it)
end

"""
  injectdata!(field, data, it, iz, iy, ix, c[,bz=10])
"""
function injectdata!(field::AbstractArray{T,3}, data::AbstractArray{T,2}, it::Integer, iz::AbstractVector{Vector{Int64}}, iy::AbstractVector{Vector{Int64}}, ix::AbstractVector{Vector{Int64}}, c::AbstractVector{Array{C,3}}, nbz::Integer=1, nby::Integer=10, nbx::Integer=10) where {T,C}
    blocks = Wave.source_blocking(size(field)..., nbz, nby, nbx, iz, iy, ix, c)
    injectdata!(field, blocks, data, it)
end

function extractdata!(data::AbstractArray{C,2}, field::AbstractArray{C,2}, it::Integer, iz::AbstractArray{Array{Int64,1},1}, ix::AbstractArray{Array{Int64,1},1}, c::AbstractArray{Array{T,2},1}) where {T,C}
    for i = 1:length(iz)
        nc_z, nc_x = size(c[i])
        for ihx=1:nc_x
            @simd ivdep for ihz=1:nc_z
                @inbounds begin
                    data[it,i] += c[i][ihz,ihx]*field[iz[i][ihz],ix[i][ihx]]
                end
            end
        end
    end
end

function extractdata!(data::AbstractArray{C,2}, field::AbstractArray{C,3}, it::Integer, iz::AbstractArray{Array{Int64,1},1}, iy::AbstractArray{Array{Int64,1},1}, ix::AbstractArray{Array{Int64,1},1}, c::AbstractArray{Array{T,3},1}) where {T,C}
    Threads.@threads for i = 1:length(iz)
        nc_z, nc_y, nc_x = size(c[i])
        for ihx=1:nc_x, ihy=1:nc_y, ihz=1:nc_z
            @inbounds begin
                data[it,i] += c[i][ihz,ihy,ihx]*field[iz[i][ihz],iy[i][ihy],ix[i][ihx]]
            end
        end
    end
end

#
# injection/extraction methods specialized to on-grid source/receivers:
#
function allongrid(dz::T, dx::T, z0::Float64, x0::Float64, z::Array{Float64,1}, x::Array{Float64,1}; rtol=1e-6) where T
    for i = 1:length(z)
        if isapprox(z0 + round(Int, (z[i] - z0)/dz)*dz, z[i], rtol=rtol) == false
            return false
        end
        if isapprox(x0 + round(Int, (x[i] - x0)/dx)*dx, x[i], rtol=rtol) == false
            return false
        end
    end
    true
end

function allongrid(dz::T, dy::T, dx::T, z0::Float64, y0::Float64, x0::Float64, z::Array{Float64,1}, y::Array{Float64,1}, x::Array{Float64,1}; rtol=1e-6) where T
    for i = 1:length(z)
        if isapprox(z0 + round(Int, (z[i] - z0)/dz)*dz, z[i], rtol=rtol) == false
            return false
        end
        if isapprox(y0 + round(Int, (y[i] - y0)/dy)*dy, y[i], rtol=rtol) == false
            return false
        end
        if isapprox(x0 + round(Int, (x[i] - x0)/dx)*dx, x[i], rtol=rtol) == false
            return false
        end
    end
    true
end

function injectdata!(field::Array{T,2}, data::Array{T,2}, it::Integer, iz::Array{Int64,1}, ix::Array{Int64,1}, c::Array{T,1}, language::Language=LangC(), nthreads::Int=Sys.CPU_THREADS) where T
    if isa(language, LangC)
        if T == Float32
            ccall(
                (:injectdata_2d_ongrid_float, libspacetime),
                Cvoid,
                (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Clong}, Ptr{Clong}, Csize_t, Csize_t,      Csize_t,   Csize_t,       Csize_t,            Csize_t),
                 field,       data,        c,           iz,         ix,         it,      size(data,1), length(c), size(field,1), size(field,2),      nthreads
            )
        else
            ccall(
                (:injectdata_2d_ongrid_double, libspacetime),
                Cvoid,
                (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Clong}, Ptr{Clong}, Csize_t, Csize_t,      Csize_t,   Csize_t,       Csize_t,       Csize_t),
                 field,       data,          c,            iz,         ix,         it,      size(data,1), length(c), size(field,1), size(field,2), nthreads
            )
        end
        return
    end

    @inbounds for i = 1:length(iz)
        field[iz[i],ix[i]] += c[i]*data[it,i]
    end
end

function injectdata!(field::Array{T,3}, data::Array{T,2}, it::Integer, iz::Array{Int64,1}, iy::Array{Int64,1}, ix::Array{Int64,1}, c::Array{T,1}, language::Language=LangC(), nthreads::Int=Sys.CPU_THREADS) where T
    if isa(language, LangC)
        if T == Float32
            ccall(
                (:injectdata_3d_ongrid_float, libspacetime),
                Cvoid,
                (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Clong}, Ptr{Clong}, Ptr{Clong}, Csize_t, Csize_t,      Csize_t,   Csize_t,       Csize_t,       Csize_t,       Csize_t),
                 field,       data,        c,           iz,         iy,         ix,         it,      size(data,1), length(c), size(field,1), size(field,2), size(field,3), nthreads
            )
        else
            ccall(
                (:injectdata_3d_ongrid_double, libspacetime),
                Cvoid,
                (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Clong}, Ptr{Clong}, Ptr{Clong}, Csize_t, Csize_t,      Csize_t,   Csize_t,       Csize_t,       Csize_t,       Csize_t),
                 field,       data,          c,            iz,         iy,         ix,         it,      size(data,1), length(c), size(field,1), size(field,2), size(field,3), nthreads
            )
        end
        return
    end

    @inbounds for i = 1:length(iz)
        field[iz[i],iy[i],ix[i]] += c[i]*data[it,i]
    end
end

function extractdata!(data::Array{T,2}, field::Array{T,2}, it::Integer, iz::Array{Int64,1}, ix::Array{Int64,1}, c::Array{T,1}, language::Language=LangC(), nthreads::Int=Sys.CPU_THREADS) where T
    if isa(language, LangC)
        if T == Float32
            ccall(
                (:extractdata_2d_ongrid_float, libspacetime),
                Cvoid,
                (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Clong}, Ptr{Clong}, Clong, Clong,        Clong,     Clong,         Clong,         Clong),
                 data,        field,       c,           iz,         ix,         it,    size(data,1), length(c), size(field,1), size(field,2), nthreads
            )
        else
            ccall(
                (:extractdata_2d_ongrid_double, libspacetime),
                Cvoid,
                (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Clong}, Ptr{Clong}, Clong, Clong,        Clong,     Clong,         Clong,         Clong),
                 data,         field,        c,            iz,         ix,         it,    size(data,1), length(c), size(field,1), size(field,2), nthreads
            )
        end
        return
    end
    @inbounds for i = 1:length(iz)
        data[it,i] += c[i]*field[iz[i],ix[i]]
    end
end

function extractdata!(data::Array{T,2}, field::Array{T,3}, it::Integer, iz::Array{Int64,1}, iy::Array{Int64,1}, ix::Array{Int64,1}, c::Array{T,1}, language::Language=LangC(), nthreads::Int=Sys.CPU_THREADS) where T
    if isa(language, LangC)
        if T == Float32
            ccall(
                (:extractdata_3d_ongrid_float, libspacetime),
                Cvoid,
                (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Clong}, Ptr{Clong}, Ptr{Clong}, Clong, Clong,        Clong,     Clong,         Clong,         Clong,         Clong),
                 data,        field,       c,           iz,         iy,         ix,         it,    size(data,1), length(c), size(field,1), size(field,2), size(field,3), nthreads
            )
        else
            ccall(
                (:extractdata_3d_ongrid_double, libspacetime),
                Cvoid,
                (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Clong}, Ptr{Clong}, Ptr{Clong}, Clong, Clong,        Clong,     Clong,         Clong,         Clong,         Clong),
                 data,         field,        c,            iz,         iy,         ix,         it,    size(data,1), length(c), size(field,1), size(field,2), size(field,3), nthreads
            )
        end
        return
    end
    @inbounds for i = 1:length(iz)
        data[it,i] += c[i]*field[iz[i],iy[i],ix[i]]
    end
end

#
# taper methods
#
function taper_region(nz, nx, iz, ix, iradius, iflat; interior=false)
    region = falses(nz, nx)
    addz = interior ? iflat : iflat+iradius
    addx = interior ? iflat : iflat+iradius
    for is = 1:length(ix)
        ilbz,iubz = clamp(minimum(iz[is])-addz,1,nz),clamp(maximum(iz[is])+addz,1,nz)
        ilbx,iubx = clamp(minimum(ix[is])-addx,1,nx),clamp(maximum(ix[is])+addx,1,nx)
        region[ilbz:iubz,ilbx:iubx] = true
    end

    region
end
function taper_region(nz, ny, nx, iz, iy, ix, iradius, iflat; interior=false)
    region = falses(nz, ny, nx)
    addz = interior ? iflat : iflat+iradius
    addy = interior ? iflat : iflat+iradius
    addx = interior ? iflat : iflat+iradius
    @inbounds for is = 1:length(ix)
        ilbz,iubz = clamp(minimum(iz[is])-addz,1,nz),clamp(maximum(iz[is])+addz,1,nz)
        ilby,iuby = clamp(minimum(iy[is])-addy,1,ny),clamp(maximum(iy[is])+addy,1,ny)
        ilbx,iubx = clamp(minimum(ix[is])-addx,1,nx),clamp(maximum(ix[is])+addx,1,nx)
        region[ilbz:iubz,ilby:iuby,ilbx:iubx] = true
    end

    region
end

function taper_region(taper_int::AbstractArray{Bool,2}, taper_ext)
    region_tpr = SVector{2,Float32}[]
    nz, nx = size(taper_int)
    @inbounds for jx = 1:nx, jz = 1:nz
        if !taper_int[jz,jx] && taper_ext[jz,jx]
            push!(region_tpr, SVector{2,Float32}(Float32(jz),Float32(jx)))
        end
    end
    region_tpr
end
function taper_region(taper_int::AbstractArray{Bool,3}, taper_ext)
    region_tpr = SVector{3,Float32}[]
    nz, ny, nx = size(taper_int)
    @inbounds for jx = 1:nx, jy = 1:ny, jz = 1:nz
        if !taper_int[jz,jy,jx] && taper_ext[jz,jy,jx]
            push!(region_tpr, SVector{3,Float32}(Float32(jz),Float32(jy),Float32(jx)))
        end
    end
    region_tpr
end

function taper_edge(region::AbstractArray{Bool,2})
    nz,nx = size(region)
    edge = SVector{2,Float32}[]
    @inbounds for ix = 2:nx-1, iz = 2:nz-1
        if region[iz,ix]
            if !region[iz+1,ix]
                push!(edge, SVector{2,Float32}(iz, ix))
                continue
            end
            if !region[iz-1,ix]
                push!(edge, SVector{2,Float32}(iz, ix))
                continue
            end
            if !region[iz,ix+1]
                push!(edge, SVector{2,Float32}(iz, ix))
                continue
            end
            if !region[iz,ix-1]
                push!(edge, SVector{2,Float32}(iz, ix))
            end
        end
    end
    KDTree(edge)
end
function taper_edge(region::AbstractArray{Bool,3})
    nz,ny,nx = size(region)
    edge = SVector{3,Float32}[]
    @inbounds for ix = 2:nx-1, iy = 2:ny-1, iz = 2:nz-1
        if region[iz,iy,ix]
            if !region[iz+1,iy,ix]
                push!(edge, SVector{3,Float32}(iz, iy, ix))
                continue
            end
            if !region[iz-1,iy,ix]
                push!(edge, SVector{3,Float32}(iz, iy, ix))
                continue
            end
            if !region[iz,iy+1,ix]
                push!(edge, SVector{3,Float32}(iz, iy, ix))
                continue
            end
            if !region[iz,iy-1,ix]
                push!(edge, SVector{3,Float32}(iz, iy, ix))
                continue
            end
            if !region[iz,iy,ix+1]
                push!(edge, SVector{3,Float32}(iz, iy, ix))
                continue
            end
            if !region[iz,iy,ix-1]
                push!(edge, SVector{3,Float32}(iz, iy, ix))
            end
        end
    end
    KDTree(edge)
end

function taper_build(t::Type{T}, nz, nx, region_tpr, region_int_edge, region_ext_edge) where T
    tpr = zeros(T, nz, nx)
    dint = knn(region_int_edge, region_tpr, 1)[2]
    dext = knn(region_ext_edge, region_tpr, 1)[2]
    for (i,tpr_pt) in enumerate(region_tpr)
        jz = Int(tpr_pt[1])
        jx = Int(tpr_pt[2])
        tpr[jz,jx] = dint[i][1]/(dint[i][1]+dext[i][1])
    end
    tpr
end
function taper_build(t::Type{T}, nz, ny, nx, region_tpr, region_int_edge, region_ext_edge) where T
    tpr = zeros(T, nz, ny, nx)
    dint = knn(region_int_edge, region_tpr, 1)[2]
    dext = knn(region_ext_edge, region_tpr, 1)[2]
    for (i,tpr_pt) in enumerate(region_tpr)
        jz = Int(tpr_pt[1])
        jy = Int(tpr_pt[2])
        jx = Int(tpr_pt[3])
        tpr[jz,jy,jx] = dint[i][1]/(dint[i][1]+dext[i][1])
    end
    tpr
end

function taper_etatozero(η, iz, ix, iradius, iflat)
    T = eltype(η)
    nz,nx = size(η)

    region_int = taper_region(nz, nx, iz, ix, iradius, iflat, interior=true)
    region_ext = taper_region(nz, nx, iz, ix, iradius, iflat, interior=false)

    region_int_edge = taper_edge(region_int)
    region_ext_edge = taper_edge(region_ext)
    region_tpr = taper_region(region_int, region_ext)

    tpr = taper_build(T, nz, nx, region_tpr, region_int_edge, region_ext_edge)

    for jx = 1:nx, jz = 1:nz
        if region_int[jz,jx] == 1
            η[jz,jx] = 0.0
            continue
        end
        if region_ext[jz,jx] == 1
            η[jz,jx] *= tpr[jz,jx]
        end
    end
    nothing
end
function taper_etatozero(η, iz, iy, ix, iradius, iflat)
    T = eltype(η)
    nz,ny,nx = size(η)

    region_int = taper_region(nz, ny, nx, iz, iy, ix, iradius, iflat, interior=true)
    region_ext = taper_region(nz, ny, nx, iz, iy, ix, iradius, iflat, interior=false)

    region_int_edge = taper_edge(region_int)
    region_ext_edge = taper_edge(region_ext)
    region_tpr = taper_region(region_int, region_ext)

    tpr = taper_build(T, nz, ny, nx, region_tpr, region_int_edge, region_ext_edge)

    for jx = 1:nx, jy = 1:ny, jz = 1:nz
        if region_int[jz,jy,jx] == 1
            η[jz,jy,jx] = 0.0
            continue
        end
        if region_ext[jz,jy,jx] == 1
            η[jz,jy,jx] *= tpr[jz,jy,jx]
        end
    end
    nothing
end

function taper_epsilon2delta(δ, ϵ, iz, ix, iradius, iflat)
    T = eltype(δ)
    nz,nx = size(δ)

    region_int = taper_region(nz, nx, iz, ix, iradius, iflat, interior=true)
    region_ext = taper_region(nz, nx, iz, ix, iradius, iflat, interior=false)

    region_int_edge = taper_edge(region_int)
    region_ext_edge = taper_edge(region_ext)
    region_tpr = taper_region(region_int, region_ext)

    tpr = taper_build(T, nz, nx, region_tpr, region_int_edge, region_ext_edge)

    for jx = 1:nx, jz = 1:nz
        if region_int[jz,jx]
            δ[jz,jx] = ϵ[jz,jx]
            continue
        end
        if region_ext[jz,jx]
            δ[jz,jx] = (1-tpr[jz,jx])*ϵ[jz,jx] + tpr[jz,jx]*δ[jz,jx]
        end
    end
    nothing
end
function taper_epsilon2delta(δ, ϵ, iz, iy, ix, iradius, iflat)
    T = eltype(δ)
    nz,ny,nx = size(δ)

    region_int = taper_region(nz, ny, nx, iz, iy, ix, iradius, iflat, interior=true)
    region_ext = taper_region(nz, ny, nx, iz, iy, ix, iradius, iflat, interior=false)

    region_int_edge = taper_edge(region_int)
    region_ext_edge = taper_edge(region_ext)
    region_tpr = taper_region(region_int, region_ext)

    tpr = taper_build(T, nz, ny, nx, region_tpr, region_int_edge, region_ext_edge)

    for jx = 1:nx, jy = 1:ny, jz = 1:nz
        if region_int[jz,jy,jx]
            δ[jz,jy,jx] = ϵ[jz,jy,jx]
            continue
        end
        if region_ext[jz,jy,jx]
            δ[jz,jy,jx] = (1-tpr[jz,jy,jx])*ϵ[jz,jy,jx] + tpr[jz,jy,jx]*δ[jz,jy,jx]
        end
    end
    nothing
end
