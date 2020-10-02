abstract type StencilOrder end
struct StencilOrder4 <: StencilOrder end
struct StencilOrder8 <: StencilOrder end
show(io::IO, s::StencilOrder4) = write(io, "O4")
show(io::IO, s::StencilOrder8) = write(io, "O8")

# order 4 deriv :  a1*(f3-f2) + a2*(f4-f1)
@inline dfdr4(a1,a2,f1,f2,f3,f4) = a1*(f3 - f2) + a2*(f4 - f1)

# order 8 deriv : a1*(f5-f4) + a2*(f6-f3) + a3*(f7-f2) + a4*(f8-f1)
@inline dfdr8(a1,a2,a3,a4,f1,f2,f3,f4,f5,f6,f7,f8) = a1*(f5-f4) + a2*(f6-f3) + a3*(f7-f2) + a4*(f8-f1)

function stencilcoeffs(stenciltype::Symbol, Q::Int, D::Int, ndim::Int, dtmod::Real, velmax::Real, mindel::Real, T::Type)
    if stenciltype == :Fornberg
        return stencilcoeffs_fornberg(Q,D,T)
    end
    if stenciltype == :Nihei
        return stencilcoeffs_nihei(Q,ndim,velmax,mindel,dtmod,T)
    end
    throw(ArgumentError("Unknonw stenciltype: $(stenciltype)"))
end

courantnumber(a::Array{T,1}, ndim) where {T<:Real} = T(1.0/(sqrt(ndim)*sum(abs.(a))))

function stencilcoeffs_fornberg(Q,D,T)
    a = D == 1 ? Array{T}(undef, div(Q,2)) : Array{T}(undef, div(Q,2)+1)
    if Q == 4 && D == 1 # assumes staggered
        a[1] = 9.0/8.0
        a[2] = -1.0/24.0
    elseif Q == 8 && D == 1 # assumes staggered
        a[1] = +1225.0/1024.0
        a[2] = -245.0/3072.0
        a[3] = +49.0/5120.0
        a[4] = -5.0/7168.0
    elseif Q == 8 && D == 2 # assumes centered
        a[1] = -205.0/72.0 # centered coeff
        a[2] = 8.0/5.0
        a[3] = -1.0/5.0
        a[4] = 8.0/315.0
        a[5] = -1.0/560.0
    else
        throw(ArgumentError("Unknown combo of accruacy order=$(Q) and derivative order=$(D)"))
    end
    a
end

function stencilcoeffs_nihei(Q,ndim,velmax,mindel,dtmod,T;nkdx=500,pts_per_wavelength=2.8,rtol=1e-9)
    a = stencilcoeffs_fornberg(Q, T)
    csafe = 0.95*courantnumber(a, ndim) # Approximate and safe courant number of O(8)
    c = velmax*dtmod/mindel/.99
    @assert dtmod > 0 && c < csafe
    kdx_max = 2pi/pts_per_wavelength
    dkdx = kdx_max/nkdx

    # Newton's method
    itmax=10
    aold = copy(a)
    for it = 1:itmax
        g = stencilcoeffs_nihei_gradient(a,Q,c,nkdx,dkdx)
        H = stencilcoeffs_nihei_hessian(a,Q,c,nkdx,dkdx)
        a -= H\g
        phi = stencilcoeffs_nihei_cost(a,Q,c,nkdx,dkdx)
        if norm(a-aold) < rtol
            break
        end
        copyto!(aold,a)
    end
    a
end

function stencilcoeffs_nihei_gradient(a, Q, c, nkdx, dkdx)
    Q2 = round(Int,Q/2)
    grad = zeros(Q2)
    for ikdx = 1:nkdx
        kdx = ikdx*dkdx

        A = 2.0/(kdx*c)
        E = 0.0
        for r = 1:Q2
            E += c*(a[r]*sin(kdx*(r - 0.5)))
        end
        vnorm = A*asin(E)  # normalized velocity
        diff_vnorm =  vnorm - 1.0  # normalized velocity misfit
        for p = 1:Q2
            B = c*sin(kdx*(p - 0.5))
            grad[p] += diff_vnorm*A*B/sqrt(1.0 - E^2)  # Gradient for a[p]
        end
    end
    grad
end

function stencilcoeffs_nihei_hessian(a, Q, c, nkdx, dkdx)
    Q2 = round(Int,Q/2)
    hess = zeros(Q2, Q2)
    for ikdx = 1:nkdx
        kdx = convert(Float64, ikdx)*dkdx

            A = 2.0/(kdx*c)
        E = 0.0
        for r = 1:Q2
            E += c*(a[r]*sin(kdx*(r - 0.5)))
        end
        vnorm = A*asin(E)  # normalized velocity
        diff_vnorm =  vnorm - 1.0  # normalized velocity misfit

        for p = 1:Q2
            B = c*sin(kdx*(p - 0.5))
            for q = 1:Q2
                D = c*sin(kdx*(q - 0.5))
                hess[p,q] += diff_vnorm*A*B*D*E/(1.0 - E^2)^1.5 + (A^2)*B*D/(1.0 - E^2)  # Hpq
            end
        end
    end
    hess
end

function stencilcoeffs_nihei_cost(a, Q, c, nkdx, dkdx)
    Q2 = round(Int,Q/2)
    obj = 0.0
    for ikdx = 1:nkdx
        kdx = convert(Float64, ikdx)*dkdx

        A = 2.0/(kdx*c)
        E = 0.0
        for r = 1:Q2
            E += c*(a[r]*sin(kdx*(r - 0.5)))
        end
        vnorm = A*asin(E)  # normalized velocity
        obj += 0.5*(vnorm - 1.0)^2  # L2 norm objective function of normalized velocity misfit
    end
    return obj
end
