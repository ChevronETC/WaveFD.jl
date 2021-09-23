global NDAMP = 50

function absorb_maxwell_john!(abc::AbstractArray{T,3}, freesurface, nsponge, dt, freqQ, qMin, qInterior) where {T}
    qprof = Vector{T}(undef, nsponge)

    qmin = Float32(qMin)
    qmax = Float32(qInterior)

    lqmin = log(qmin)
    lqmax = log(qmax)

    for ksponge = 1:nsponge
        dk = (ksponge-one(T)) / (nsponge - 1)
        lq = lqmin + dk * (lqmax - lqmin)
        qprof[ksponge] = exp(lq)
    end

    nz,ny,nx = size(abc)
    c = dt * 2 * pi * freqQ
    Threads.@threads for kx = 1:nx
        for ky = 1:ny
            @simd for kz = 1:nz
                @inbounds begin
                    ksx = min(kx, nx-kx+1)
                    ksy = min(ky, ny-ky+1)
                    ksz = freesurface ? nz-kz+1 : min(kz, nz-kz+1)

                    ksponge = min(ksx, ksy, ksz)

                    if ksponge <= nsponge
                        abc[kz,ky,kx] = c / qprof[ksponge]
                    else
                        abc[kz,ky,kx] = c / qInterior
                    end
                end
            end
        end
    end
end

# Computes the absorbing boundary condition spatial decay function, abc=w*dt/(2Q), for a Maxwell rheologic Q model.
function absorb_maxwell!(abc::Matrix, freesurface, freqQ, Qmin, dt, nzabc, nxabc)
    @assert(nxabc>0 && nzabc>0)

    # set maximum attenuation value
    w = 2*pi*freqQ
    abc_max = w*dt/(2*Qmin)

    # initialize Maxwell attenuation coefficient: abc=w*dt/(2Q) 
    abc .= 0  # initialize to zero attenuation
    nz,nx = size(abc)

    # define 1D decay functions (Hamming window):
    decay_x = 0.505 .+ 0.495 .* cos.(pi .* range(0., stop=nxabc-1, length=nxabc) / float(nxabc-1))
    decay_z = 0.505 .+ 0.495 .* cos.(pi .* range(0., stop=nzabc-1, length=nzabc) / float(nzabc-1))

    abc_x = ones(Float32, nx)
    abc_z = ones(Float32, nz)

    abc_x[1:nxabc] = decay_x[nxabc:-1:1]
    if !freesurface
        abc_z[1:nzabc] = decay_z[nzabc:-1:1]
    end

    abc_x[nx-nxabc+1:nx] = decay_x[1:nxabc]
    abc_z[nz-nzabc+1:nz] = decay_z[1:nzabc]

    # generate 2D abc by taking products of 1D decay functions
    for k=1:nz, i=1:nx
        abc[k,i] = abc_x[i]*abc_z[k]
    end

    # take negative image & apply abc_max scaling
    abc .= abc_max*(1 .- abc)

    abc
end

function absorb(vp::AbstractArray{T,2}, nz::Int64, nx::Int64, dz::T, dx::T, dt::T; ndamp::Int64=100, freesurface::Bool=false) where T<:Union{Float32,Float64}

# Generates an array of size of the model that contains the absorbing
# boundary coefficient for a Maxwell viscoelastic damper. The spatial
# decay of Maxwell damping factor is taken from PML (Collino & Tsogka,
# 2001, Geophys., 66, 294-307).

    # number of points in the absorbing boundary
    ndamp_top = freesurface == true ? 1 : ndamp
    ndamp_bot = ndamp
    ndamp_sides = ndamp

    # determine maximum values of the damping coefficient on top, bottom and sides of model
    vp_top_ave = mean(vp[ndamp_top,:])
    vp_bot_ave = mean(vp[(nz - ndamp_top + 1),:])

    damp_top_max = 18.0 * vp_top_ave / (float(ndamp_top) * dz)
    damp_bot_max = 18.0 * vp_bot_ave / (float(ndamp_bot) * dz)

    # ..for sides, use a linear gradient from top to bottom
    damp_sides_top = 18.0 * vp_top_ave / (float(ndamp_sides) * dx)
    damp_sides_bot = 18.0 * vp_bot_ave / (float(ndamp_sides) * dx)
    damp_sides_grad = (damp_sides_bot - damp_sides_top) / float(nz - 1)
    damp_sides_max = zeros(T, nz)
    for k = 1:nz
        damp_sides_max[k] = damp_sides_top + damp_sides_grad * float(k - 1)
    end

    # initialize Maxwell damping coefficient to zero (i.e., no damping)
    damp = zeros(T, nz, nx)

    #
    # Sides:
    #

    # define Maxwell damping profile for sides of model
    damp_sides = zeros(T, nz, ndamp_sides)
    for i = 1:ndamp_sides, k = 1:nz
        damp_sides[k,i] = damp_sides_max[k]*(float(ndamp_sides - i) / ndamp_sides)^2
    end

    # assign damping coefficient for right absorbing boundary
    icnt = 0
    for i = (nx - ndamp_sides + 1):nx
        icnt += 1
        for k = 1:nz
            damp[k, i] = damp_sides[k, ndamp_sides - icnt + 1]
        end
    end

    # assign damping coefficient for left absorbing boundary
    icnt = 0
    for i = 1:ndamp_sides
        icnt += 1
        for k = 1:nz
            damp[k, i] = damp_sides[k, icnt]
        end
    end

    #
    # Top, and top corners
    #

    if freesurface == false
        # define Maxwell damping profile for top of model
        damp_top = zeros(T, ndamp_top)
        for k = 1:ndamp_top
            damp_top[k] = damp_top_max*(float(ndamp_top - k) / ndamp_top)^2
        end

        # assign damping coefficient for top absorbing boundary
        for i = 1:nx, k = 1:ndamp_top
            damp[k, i] = damp_top[k]
        end

        # define Maxwell damping profile for top corners of model
        damp_topcorners = zeros(T, ndamp_top, ndamp_sides)
        for i = 1:ndamp_sides, k = 1:ndamp_top
            damp_topcorners[k,i] = sqrt((damp_sides_max[1]^2)*(float(ndamp_sides - i) / ndamp_sides)^4 + (damp_top_max^2)*(float(ndamp_top - k) / ndamp_top)^4)
        end

        # assign damping coefficient for top-left corner absorbing boundary
        kcnt = 0
        for k = 1:ndamp_top
            icnt = 0
            kcnt += 1
            for i = 1:ndamp_sides
                icnt += 1
                damp[k, i] = damp_topcorners[kcnt, icnt]
            end
        end

        # assign damping coefficient for top-right corner absorbing boundary
        kcnt = 0
        for k = 1:ndamp_top
            icnt = 0
            kcnt += 1
            for i = (nx - ndamp_sides + 1):nx
                icnt += 1
                damp[k, i] = damp_topcorners[kcnt, ndamp_sides - icnt + 1]
            end
        end
    end

    #
    # Bottom, and bottom corners:
    #

    # define Maxwell damping profile for bottom of model
    damp_bot = zeros(T, ndamp_bot)
    for k = 1:ndamp_bot
        damp_bot[k] = damp_bot_max*(float(ndamp_bot - k) / ndamp_bot)^2
    end

    # assign damping coefficient for bottom absorbing boundary
    kcnt = 0
    for k = (nz - ndamp_bot + 1):nz
        kcnt += 1
        for i = 1:nx
            damp[k, i] = damp_bot[ndamp_bot - kcnt + 1]
        end
    end

    # define Maxwell damping profile for bottom corners of model
    damp_botcorners = zeros(T, ndamp_bot, ndamp_sides)
    for i = 1:ndamp_sides, k = 1:ndamp_bot
        damp_botcorners[k,i] = sqrt((damp_sides_max[nz]^2)*(float(ndamp_sides - i) / ndamp_sides)^4 + (damp_bot_max^2)*(float(ndamp_bot - k) / ndamp_bot)^4)
    end

    # assign damping coefficient for bottom-right corner absorbing boundary
    kcnt = 0
    for k = (nz - ndamp_bot + 1):nz
        icnt = 0
        kcnt += 1
        for i = (nx - ndamp_sides + 1):nx
            icnt += 1
            damp[k, i] = damp_botcorners[ndamp_bot - kcnt + 1, ndamp_sides - icnt + 1]
        end
    end

    # assign damping coefficient for bottom-left corner absorbing boundary
    kcnt = 0
    for k = (nz - ndamp_bot + 1):nz
        icnt = 0
        kcnt += 1
        for i = 1:ndamp_sides
            icnt += 1
            damp[k, i] = damp_botcorners[ndamp_bot - kcnt + 1, icnt]
        end
    end


    # Convert the Maxwell damping coefficient into a scaling factor [0,1]
    # that is to be used in the pressure and particle velocity updates.
    # Note that this scaling factor is the result of using the Lax-Wendroff
    # approximation the Maxwell viscoelastic damping terms that appear on
    # the right hand sides of the equations of motion and stress-strain
    # law.

    dabc = ones(T, nz, nx)

    for i = 1:nx, k = 1:nz
        dabc[k,i] = (1.0 - 0.5*damp[k,i]*dt) / (1.0 + 0.5*damp[k,i]*dt)
    end

    return dabc

end

function absorb(c33::Array{T,3}, den::Array{T,3}, nz::Int64, ny::Int64, nx::Int64, dz::T, dy::T, dx::T, dt::T, nzabc_top::Int64, nzabc_bot::Int64, nyabc::Int64, nxabc::Int64; freesurface=false) where T<:Union{Float32,Float64}

    # reference frequency for Maxwell viscoelasticity
    fq = 10.    # reference freq is used purely for QC'ing invQ
    w = 2. * pi * fq

    # empirically-derived tuning parameter for optimal ABC damping
    tuneabc = 18.0  # Qmin=0.5; note: tuneabc=xx is unstable

    # compute average Vpvert along top &  bottom row of model, and use this to define the ABC minimum Q
    vp_ave_top = mean(sqrt.(c33[1,:,:] ./ den[1,:,:]))
    vp_ave_bot = mean(sqrt.(c33[nz,:,:] ./ den[nz,:,:]))

    # for top and bot invQmax, use average values
    invQ_max_ztop = tuneabc * vp_ave_top / (w * nzabc_top * dz)
    invQ_max_zbot = tuneabc * vp_ave_bot / (w * nzabc_bot * dz)

    # for sides invQmax, use linear gradient from top to bot
    invQ_max_xtop = tuneabc * vp_ave_top / (w * nxabc * dx)
    invQ_max_xbot = tuneabc * vp_ave_bot / (w * nxabc * dx)
    invQ_max_ytop = tuneabc * vp_ave_top / (w * nyabc * dy)
    invQ_max_ybot = tuneabc * vp_ave_bot / (w * nyabc * dy)

    invQ_max_xgrad = (invQ_max_xbot - invQ_max_xtop) / (nz - 1)
    invQ_max_ygrad = (invQ_max_ybot - invQ_max_ytop) / (nz - 1)
    invQ_max_x = zeros(Float32, nz)
    invQ_max_y = zeros(Float32, nz)
    for iz = 1:nz
        invQ_max_x[iz] = invQ_max_xtop + invQ_max_xgrad * (iz - 1)
        invQ_max_y[iz] = invQ_max_ytop + invQ_max_ygrad * (iz - 1)
    end

    # initialize Maxwell "transmission coefficient"
    abc = ones(T, nz, ny, nx)  # initialize to complete transmission

    # define a compact function for computing Maxwell "transmission coefficient"
    trans(invQ, w, dt) = (1. - 0.5 * w * dt * invQ) / (1. + 0.5 * w * dt * invQ)

    # 1D abc for 6 plates (note: won't worry about beginning and ends of non-1D indices for plates since these will be overwritten subsequently by beams and cubes)
    # ..top (-k) plate
    if freesurface == false
        for ix = 1:nx, iy = 1:ny, iz = 1:nzabc_top
            invQ = invQ_max_ztop * ((nzabc_top - iz) / nzabc_top)^2 # decay factor
            abc[iz,iy,ix] = trans(invQ, w, dt)
        end
    end

    # ..bottom (+k) plate
    for ix = 1:nx, iy = 1:ny, iz = 1:nzabc_bot
        invQ = invQ_max_zbot * ((iz - 1) / nzabc_bot)^2
        iiz = nz - nzabc_bot + iz
        abc[iiz,iy,ix] = trans(invQ, w, dt)
    end

    for ix = 1:nxabc, iy = 1:ny, iz = 1:nz
        # ..west (-i) plate
        invQ = invQ_max_x[iz] * ((nxabc - ix) / nxabc)^2
        abc[iz,iy,ix] = trans(invQ, w, dt)

        # ..east (+i) plate
        invQ = invQ_max_x[iz] * ((ix - 1) / nxabc)^2
        iix = nx - nxabc + ix
        abc[iz,iy,iix] = trans(invQ, w, dt)
    end

    for ix = 1:nx, iy = 1:nyabc, iz = 1:nz
        # ..south (-j) plate
        invQ = invQ_max_y[iz] * ((nyabc - iy) / nyabc)^2
        abc[iz,iy,ix] = trans(invQ, w, dt)

        # ..north (+j) plate
        invQ = invQ_max_y[iz] * ((iy - 1) / nyabc)^2
        iiy = ny - nyabc + iy
        abc[iz,iiy,ix] = trans(invQ, w, dt)
    end

    # 2D decay for 12 beams (note: won't worry about beginning and ends of non-2D indices for beams since these will be overwritten subsequently by cubes)
    # ..top beams
    if freesurface == false
        for iz = 1:nzabc_top
            for ix=1:nxabc, iy=1:ny
                # ....top-west beam
                invQ = sqrt(invQ_max_x[iz]^2 * ((nxabc - ix) / nxabc)^4 + invQ_max_ztop^2 * ((nzabc_top - iz) / nzabc_top)^4)
                abc[iz,iy,ix] = trans(invQ, w, dt)

                # ....top-east beam
                invQ = sqrt(invQ_max_x[iz]^2 * ((ix - 1) / nxabc)^4 + invQ_max_ztop^2 * ((nzabc_top - iz) / nzabc_top)^4)
                iix = nx - nxabc + ix
                abc[iz,iy,iix] = trans(invQ, w, dt)
            end

            for ix = 1:nx, iy = 1:nyabc
                # ....top-south beam
                invQ = sqrt(invQ_max_y[iz]^2 * ((nyabc - iy) / nyabc)^4 + invQ_max_ztop^2 * ((nzabc_top - iz) / nzabc_top)^4)
                abc[iz,iy,ix] = trans(invQ, w, dt)

                # ....top-north beam
                invQ = sqrt(invQ_max_y[iz]^2 * ((iy - 1) / nyabc)^4 + invQ_max_ztop^2 * ((nzabc_top - iz) / nzabc_top)^4)
                iiy = ny - nyabc + iy
                abc[iz,iiy,ix] = trans(invQ, w, dt)
            end
        end
    end

    # ..bottom beams
    for iz = 1:nzabc_bot
        iiz = nz - nzabc_bot + iz
        for ix = 1:nxabc, iy = 1:ny
            # ....bot-west beam
            invQ = sqrt(invQ_max_x[iiz]^2 * ((nxabc - ix) / nxabc)^4 + invQ_max_zbot^2 * ((iz - 1) / nzabc_bot)^4)
            abc[iiz,iy,ix] = trans(invQ, w, dt)

            # ....bot-east beam
            invQ = sqrt(invQ_max_x[iiz]^2 * ((ix - 1) / nxabc)^4 + invQ_max_zbot^2 * ((iz - 1) / nzabc_bot)^4)
            iix = nx - nxabc + ix
            abc[iiz,iy,iix] = trans(invQ, w, dt)
        end

        for ix = 1:nx, iy = 1:nyabc
            # ....bot-south beam
            invQ = sqrt(invQ_max_y[iiz]^2 * ((nyabc - iy) / nyabc)^4 + invQ_max_zbot^2 * ((iz - 1) / nzabc_bot)^4)
            abc[iiz,iy,ix] = trans(invQ, w, dt)

            # ....bot-north beam
            invQ = sqrt(invQ_max_y[iiz]^2 * ((iy - 1) / nyabc)^4 + invQ_max_zbot^2 * ((iz - 1) / nzabc_bot)^4)
            iiy = ny - nyabc + iy
            abc[iiz,iiy,ix] = trans(invQ, w, dt)
        end
    end

    # ..side beams
    for ix = 1:nxabc, iy = 1:nyabc, iz = 1:nz
        # ....south-west beam
        invQ = sqrt(invQ_max_x[iz]^2 * ((nxabc - ix) / nxabc)^4 + invQ_max_y[iz]^2 * ((nyabc - iy) / nyabc)^4)
        abc[iz,iy,ix] = trans(invQ, w, dt)

        # ....south-east beam
        invQ = sqrt(invQ_max_x[iz]^2 * ((ix - 1) / nxabc)^4 + invQ_max_y[iz]^2 * ((nyabc - iy) / nyabc)^4)
        iix = nx - nxabc + ix
        abc[iz,iy,iix] = trans(invQ, w, dt)

        # ....north-west beam
        invQ = sqrt(invQ_max_x[iz]^2 * ((nxabc - ix) / nxabc)^4 + invQ_max_y[iz]^2 * ((iy - 1) / nyabc)^4)
        iiy = ny - nyabc + iy
        abc[iz,iiy,ix] = trans(invQ, w, dt)

        # ....north-east beam
        invQ = sqrt(invQ_max_x[iz]^2 * ((ix - 1) / nxabc)^4 + invQ_max_y[iz]^2 * ((iy - 1) / nyabc)^4)
        iix = nx - nxabc + ix
        iiy = ny - nyabc + iy
        abc[iz,iiy,iix] = trans(invQ, w, dt)
    end

    # 3D decay for 8 corner cubes
    # ..top cubes
    if freesurface == false
        for ix = 1:nxabc, iy = 1:nyabc, iz = 1:nzabc_top
            # ....top-south-west cube
            invQ = sqrt(invQ_max_x[iz]^2 * ((nxabc - ix) / nxabc)^4 + invQ_max_y[iz]^2 * ((nyabc - iy) / nyabc)^4 + invQ_max_ztop^2 * ((nzabc_top - iz) / nzabc_top)^4)
            abc[iz,iy,ix] = trans(invQ, w, dt)

            # ....top-southeast cube
            invQ = sqrt(invQ_max_x[iz]^2 * ((ix - 1) / nxabc)^4 + invQ_max_y[iz]^2 * ((nyabc - iy) / nyabc)^4 + invQ_max_ztop^2 * ((nzabc_top - iz) / nzabc_top)^4)
            iix = nx - nxabc + ix
            abc[iz,iy,iix] = trans(invQ, w, dt)

            # ....top-northwest cube
            invQ = sqrt(invQ_max_x[iz]^2 * ((nxabc - ix) / nxabc)^4 + invQ_max_y[iz]^2 * ((iy - 1) / nyabc)^4 + invQ_max_ztop^2 * ((nzabc_top - iz) / nzabc_top)^4)
            iiy = ny - nyabc + iy
            abc[iz,iiy,ix] = trans(invQ, w, dt)

            # ....top-northeast cube
            invQ = sqrt(invQ_max_x[iz]^2 * ((ix - 1) / nxabc)^4 + invQ_max_y[iz]^2 * ((iy - 1) / nyabc)^4 + invQ_max_ztop^2 * ((nzabc_top - iz) / nzabc_top)^4)
            iix = nx - nxabc + ix
            iiy = ny - nyabc + iy
            abc[iz,iiy,iix] = trans(invQ, w, dt)
        end
    end

    # ..bottom cubes
    for ix = 1:nxabc, iy = 1:nyabc, iz = 1:nzabc_bot
        iiz = nz - nzabc_bot + iz

        # ....bot-southwest cube
        invQ = sqrt(invQ_max_x[iiz]^2 * ((nxabc - ix) / nxabc)^4 + invQ_max_y[iiz]^2 * ((nyabc - iy) / nyabc)^4 + invQ_max_zbot^2 * ((iz - 1) / nzabc_bot)^4)
        abc[iiz,iy,ix] = trans(invQ, w, dt)

        # ....bot-southeast cube
        invQ = sqrt(invQ_max_x[iiz]^2 * ((ix - 1) / nxabc)^4 + invQ_max_y[iiz]^2 * ((nyabc - iy) / nyabc)^4 + invQ_max_zbot^2 * ((iz - 1) / nzabc_bot)^4)
        iix = nx - nxabc + ix
        abc[iiz,iy,iix] = trans(invQ, w, dt)

        # ....bot-northwest cube
        invQ = sqrt(invQ_max_x[iiz]^2 * ((nxabc - ix) / nxabc)^4 + invQ_max_y[iiz]^2 * ((iy - 1) / nyabc)^4 + invQ_max_zbot^2 * ((iz - 1) / nzabc_bot)^4)
        iiy = ny - nyabc + iy
        abc[iiz,iiy,ix] = trans(invQ, w, dt)

        # ....bot-northeast cube
        invQ = sqrt(invQ_max_x[iiz]^2 * ((ix - 1) / nxabc)^4 + invQ_max_y[iiz]^2 * ((iy - 1) / nyabc)^4 + invQ_max_zbot^2 * ((iz - 1) / nzabc_bot)^4)
        iix = nx - nxabc + ix
        iiy = ny - nyabc + iy
        abc[iiz,iiy,iix] = trans(invQ, w, dt)
    end

    return abc
end

# TODO: investigate LoopVectorization.jl for performance?

function setup_q_profile_2D_serial!(w_inv_q::Array{T,2}, freesurface::Bool, nsponge::Int64, 
        dt::T, q_freq::T, q_min::T, q_interior::T) where T<:Float32
    nz,nx = size(w_inv_q)
    lqmin = log(q_min)
    lqmax = log(q_interior)
    qprof = zeros(Float32, nsponge)

    for ksponge = 1:nsponge
        dk = (ksponge - 1) / (nsponge - 1)
        lq = lqmin + dk * (lqmax - lqmin)
        qprof[ksponge] = T(exp(lq))
    end

    for kz = 1:nz
        ksz = (freesurface) ? (nz - 1 - (kz - 1)) : min((kz - 1), (nz - 1 - (kz - 1)))
        for kx = 1:nx
            ksx = min((kx - 1),   (nx - 1 - (kx - 1)))
            ksponge = min(ksx, ksz) + 1 # add one to get back to julia indexing
            @inbounds w_inv_q[kz,kx] = (dt * 2.0 * π * q_freq / q_interior)
            if ksponge ∈ 1:nsponge
                @inbounds w_inv_q[kz,kx] = dt * 2.0 * π * q_freq / qprof[ksponge]
            end
        end
    end
end

function make_qprof(nsponge::Int64, q_min::T, q_interior::T) where T<:Float32
    lqmin = log(q_min)
    lqmax = log(q_interior)
    qprof = zeros(Float32, nsponge)

    @threads for ksponge = 1:nsponge
        dk = (ksponge - 1) / (nsponge - 1)
        lq = lqmin + dk * (lqmax - lqmin)
        @inbounds qprof[ksponge] = exp(lq)
    end

    return qprof
end

function setup_q_profile_2D_threaded!(w_inv_q::Array{T,2}, freesurface, nsponge::Int64, 
        dt::T, q_freq::T, q_min::T, q_interior::T) where T<:Float32
    
    function compute_w_inv_q_2D!(w_inv_q::Array{T,2}, qprof::Array{T,1}, freesurface::Bool, nsponge::Int64, q_freq::T, q_interior::T) where T<:Float32
        nz,nx = size(w_inv_q)

        @threads for kz = 1:nz
            ksz = (freesurface) ? (nz - 1 - (kz - 1)) : min((kz - 1), (nz - 1 - (kz - 1)))
            for kx = 1:nx
                ksx = min((kx - 1),   (nx - 1 - (kx - 1)))
                ksponge = min(ksx, ksz) + 1 # add one to get back to julia indexing
                @inbounds w_inv_q[kz,kx] = dt * 2.0 * π * q_freq / q_interior
                if ksponge ∈ 1:nsponge
                    @inbounds w_inv_q[kz,kx] = dt * 2.0 * π * q_freq / qprof[ksponge]
                end
            end
        end
    end

    qprof = make_qprof(nsponge, q_min, q_interior)
    compute_w_inv_q_2D!(w_inv_q, qprof, freesurface, nsponge, q_freq, q_interior)
end

function setup_q_profile_3D_serial!(w_inv_q::Array{T,3}, freesurface::Bool, nsponge::Int64, 
    dt::T, q_freq::T, q_min::T, q_interior::T) where T<:Float32
    nz,ny,nx = size(w_inv_q)
    lqmin = log(q_min)
    lqmax = log(q_interior)
    qprof = zeros(Float32, nsponge)

    for ksponge = 1:nsponge
        dk = (ksponge - 1) / (nsponge - 1)
        lq = lqmin + dk * (lqmax - lqmin)
        qprof[ksponge] = T(exp(lq))
    end

    for kz = 1:nz
        ksz = (freesurface) ? (nz - 1 - (kz - 1)) : min((kz - 1), (nz - 1 - (kz - 1)))
        
        for ky = 1:ny
            ksy = min((ky - 1),   (ny - 1 - (ky - 1)))
            for kx = 1:nx
                ksx = min((kx - 1),   (nx - 1 - (kx - 1)))
                ksponge = min(ksx,min(ksy, ksz)) + 1 # add one to get back to julia indexing
                @inbounds w_inv_q[kz,ky,kx] = (dt * 2.0 * π * q_freq / q_interior)
                if ksponge ∈ 1:nsponge
                    @inbounds w_inv_q[kz,ky,kx] = dt * 2.0 * π * q_freq / qprof[ksponge]
                end
            end
        end
    end
end

function setup_q_profile_3D_threaded!(w_inv_q::Array{T,3}, freesurface, nsponge::Int64, 
    dt::T, q_freq::T, q_min::T, q_interior::T) where T<:Float32

    function compute_w_inv_q_3D!(w_inv_q::Array{T,3}, qprof::Array{T,1}, freesurface::Bool, nsponge::Int64, q_freq::T, q_interior::T) where T<:Float32
        nz,ny,nx = size(w_inv_q)

        @threads for kz = 1:nz
            ksz = (freesurface) ? (nz - 1 - (kz - 1)) : min((kz - 1), (nz - 1 - (kz - 1)))
            
            for ky = 1:ny
                ksy = min((ky - 1),   (ny - 1 - (ky - 1)))
                for kx = 1:nx
                    ksx = min((kx - 1),   (nx - 1 - (kx - 1)))
                    ksponge = min(ksx,min(ksy, ksz)) + 1 # add one to get back to julia indexing
                    @inbounds w_inv_q[kz,ky,kx] = dt * 2.0 * π * q_freq / q_interior
                    if ksponge ∈ 1:nsponge
                        @inbounds w_inv_q[kz,ky,kx] = dt * 2.0 * π * q_freq / qprof[ksponge]
                    end
                end
            end
        end
    end

    qprof = make_qprof(nsponge, q_min, q_interior)
    compute_w_inv_q_3D!(w_inv_q, qprof, freesurface, nsponge, q_freq, q_interior)
end
