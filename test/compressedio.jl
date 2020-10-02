using LinearAlgebra, Test, Wave

@testset "compressedio, 3D Array, T=$(T), C=$(C), style=$(style)" for (T,C) in ((Float32, UInt32), (Float64, UInt32), (Float32, Float32), (Float64, Float64), (Float64, Float32)), style in (:file, :buffer)
    nz = 512
    ny = 256
    nx = 128

    bz = 32
    by = 32
    bx = 32

    x = -1 .+ 2 .* rand(T,nz,ny,nx)
    y = -1 .+ 2 .* rand(T,nz,ny,nx)

    C,P = Wave.comptype(C, T)
    comp = Wave.Compressor(T, P, C, (nz,ny,nx), (bz,by,bx), 1e-2, 2, false)
    open(comp)
    io = style == :file ? open("test.bin","w") : IOBuffer()
    Wave.compressedwrite(io, comp, 1, x)
    Wave.compressedwrite(io, comp, 2, y)
    style == :file && close(io)

    xx = similar(x)
    yy = similar(y)

    io = style == :file ? open("test.bin") : io
    Wave.compressedread!(io, comp, 1, xx)
    Wave.compressedread!(io, comp, 2, yy)
    close(io)

    snr_x = 10*log10(norm(x)^2 / norm(x .- xx)^2)
    snr_y = 10*log10(norm(y)^2 / norm(y .- yy)^2)
    err_x = norm(x .- xx) / length(x)
    err_y = norm(y .- yy) / length(y)
    if C == UInt32
        @test snr_x > 40
        @test snr_y > 40
    else
        @test err_x < 1e-6
        @test err_y < 1e-6
    end
    close(comp)
end

@testset "compressedio, 3D SubArray, T=$(T), C=$(C), style=$(style)" for (T,C) in ((Float32, UInt32), (Float64, UInt32), (Float32, Float32), (Float64, Float64), (Float64, Float32)), style in (:file, :buffer)
    nz = 512
    ny = 256
    nx = 128

    bz = 32
    by = 32
    bx = 32

    x = -1 .+ 2 .* rand(T,nz,ny,nx)
    y = -1 .+ 2 .* rand(T,nz,ny,nx)

    x_sub = @view x[10:end-10,10:end-10,10:end-10]
    y_sub = @view y[10:end-10,10:end-10,10:end-10]

    nz, ny, nx = size(x_sub)

    C,P = Wave.comptype(C, T)
    comp = Wave.Compressor(T, P, C, (nz,ny,nx), (bz,by,bx), 1e-2, 2, true)
    open(comp)

    io = style == :file ? open("test.bin", "w") : IOBuffer()
    Wave.compressedwrite(io, comp, 1, x_sub)
    Wave.compressedwrite(io, comp, 2, y_sub)
    style == :file && close(io)

    xx = copy(x)
    yy = copy(y)

    xx_sub = @view xx[10:end-10,10:end-10,10:end-10]
    yy_sub = @view yy[10:end-10,10:end-10,10:end-10]

    xx_sub[:] .= 0.0
    yy_sub[:] .= 0.0

    io = style == :file ? open("test.bin") : io
    Wave.compressedread!(io, comp, 1, xx_sub)
    Wave.compressedread!(io, comp, 2, yy_sub)
    close(io)

    snr_x = 10*log10(norm(x)^2 / norm(x .- xx)^2)
    snr_y = 10*log10(norm(y)^2 / norm(y .- yy)^2)
    err_x = norm(x .- xx) / length(x)
    err_y = norm(y .- yy) / length(y)
    if C == UInt32
        @test snr_x > 40
        @test snr_y > 40
    else
        @test err_x < 1e-6
        @test err_y < 1e-6
    end
    close(comp)
end

@testset "compressedio, 2D Array, T=$(T), C=$(C), style=$(style)" for (T,C) in ((Float32, UInt32), (Float64, UInt32), (Float32, Float32), (Float64, Float64), (Float64, Float32)), style in (:file, :buffer)
    nz = 512
    nx = 128

    bz = 32
    bx = 32

    x = -1 .+ 2 .* rand(T,nz,nx)
    y = -1 .+ 2 .* rand(T,nz,nx)

    C,P = Wave.comptype(C, T)
    comp = Wave.Compressor(T, P, C, (nz,nx), (bz,bx), 1e-2, 2, false)
    open(comp)

    io = style == :file ? open("test.bin", "w") : IOBuffer()
    Wave.compressedwrite(io, comp, 1, x)
    Wave.compressedwrite(io, comp, 2, y)
    style == :file && close(io)

    xx = similar(x)
    yy = similar(y)

    io = style == :file ? open("test.bin") : io
    Wave.compressedread!(io, comp, 1, xx)
    Wave.compressedread!(io, comp, 2, yy)
    close(io)

    snr_x = 10*log10(norm(x)^2 / norm(x .- xx)^2)
    snr_y = 10*log10(norm(y)^2 / norm(y .- yy)^2)
    err_x = norm(x .- xx) / length(x)
    err_y = norm(y .- yy) / length(y)
    if C == UInt32
        @test snr_x > 40
        @test snr_y > 40
    else
        @test err_x < 1e-6
        @test err_y < 1e-6
    end
    close(comp)
end

@testset "compressedio 2D SubArray, T=$(T), C=$(C), style=$(style)" for (T,C) in ((Float32, UInt32), (Float64, UInt32), (Float32, Float32), (Float64, Float64), (Float64, Float32)), style in (:file, :buffer)
    nz = 512
    nx = 128

    bz = 32
    bx = 32

    x = -1 .+ 2 .* rand(T,nz,nx)
    y = -1 .+ 2 .* rand(T,nz,nx)

    x_sub = @view x[10:end-10,10:end-10]
    y_sub = @view y[10:end-10,10:end-10]

    nz, nx = size(x_sub)

    C,P = Wave.comptype(C, T)
    comp = Wave.Compressor(T, P, C, (nz,nx), (bz,bx), 1e-2, 2, true)
    open(comp)

    io = style == :file ? open("test.bin", "w") : IOBuffer()
    Wave.compressedwrite(io, comp, 1, x_sub)
    Wave.compressedwrite(io, comp, 2, y_sub)
    style == :file && close(io)

    xx = copy(x)
    yy = copy(y)

    xx_sub = @view xx[10:end-10,10:end-10]
    yy_sub = @view y[10:end-10,10:end-10]

    xx_sub[:] .= 0.0
    yy_sub[:] .= 0.0

    io = style == :file ? open("test.bin") : io
    Wave.compressedread!(io, comp, 1, xx_sub)
    Wave.compressedread!(io, comp, 2, yy_sub)
    close(io)

    snr_x = 10*log10(norm(x)^2 / norm(x .- xx)^2)
    snr_y = 10*log10(norm(y)^2 / norm(y .- yy)^2)
    err_x = norm(x .- xx) / length(x)
    err_y = norm(y .- yy) / length(y)
    if C == UInt32
        @test snr_x > 40
        @test snr_y > 40
    else
        @test err_x < 1e-6
        @test err_y < 1e-6
    end
    close(comp)
end

@testset "compressedio, exception" for file in ("test.bin",)
    io = open(file,"w")
    @test_throws ErrorException Wave.compressedwrite_exception(io,0,1000)
    rm(file)
end

nothing