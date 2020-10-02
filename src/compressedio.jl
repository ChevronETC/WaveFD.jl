mutable struct Compressor{T,P,C,N}
    offsets::Array{Int64,1}
    lengths::Array{Int64,1}
    n::NTuple{N,Int}
    issubarray::Bool
    volume::Array{P,N}
    compressed_volume::Array{UInt32,1}
    compressor::CvxCompressor{N}
end

function Compressor(::Type{T}, ::Type{P}, ::Type{C}, n::NTuple{N,Int}, b::NTuple{N,Int}, scale, nfields, issubarray) where {T,C,P,N}
    volume = zeros(P,ntuple(i->0,N))
    compressed_volume = zeros(UInt32,0)
    compressor = CvxCompressor(b, scale)
    Compressor{T,P,C,N}(zeros(Int64, nfields), zeros(Int64, nfields), n, issubarray, volume, compressed_volume, compressor)
end

function Base.open(compressor::Compressor{T,P,C,N}) where {T,P,C,N}
    compressor.volume = ((T == Float64 && C != Float64) || compressor.issubarray) ? zeros(P,compressor.n) : zeros(P,ntuple(i->0,N))
    compressor.compressed_volume = C == UInt32 ? zeros(UInt32,2*prod(compressor.n)) : zeros(UInt32,0)
    nothing
end

function Base.close(compressor::Compressor{T,P,C,N}) where {T,P,C,N}
    compressor.volume = zeros(T,ntuple(i->0,N))
    compressor.compressed_volume = zeros(UInt32,0)
    nothing
end

copy(compressor::Compressor{T,P,C,N}) where {T,P,C,N} = Compressor{T,P,C,N}(copy(compressor.offsets), copy(compressor.lengths), compressor.n,
    compressor.issubarray, copy(compressor.volume), copy(compressor.compressed_volume), copy(compressor.compressor))

function compressedwrite_exception(io, nbytes_written, nbytes)
    dir = join(split(split(io.name, " ")[2],"/")[1:end-1], "/")
    dir = dir == "" ? "." : dir
    throw(ErrorException("wrote $(nbytes_written) of $(nbytes) on $(gethostname()) for $(io.name), bytes_available=\n$(run(`df -h $(dir)`))\n"))
end

compressedwrite(io::IO, compressor::Compressor{Float32,Float32,UInt32,N}, ifield::Integer, field::Array{Float32,N}) where {N} = compressedwrite_helper(io, compressor, ifield, field)

function compressedwrite(io::IO, compressor::Compressor{Float32,Float32,UInt32,N}, ifield::Integer, field::SubArray{Float32,N}) where N
    copyto!(compressor.volume, field) # converts from SubArray to Array
    compressedwrite_helper(io, compressor, ifield, compressor.volume)
end

function compressedwrite(io::IO, compressor::Compressor{Float64,Float32,UInt32,N}, ifield::Integer, field::AbstractArray{Float64,N}) where N
    copyto!(compressor.volume, field) # converts from Float64 to Float32, and possibly from SubArray to Array
    compressedwrite_helper(io, compressor, ifield, compressor.volume)
end

function compressedwrite_helper(io::IO, compressor::Compressor{T,P,UInt32,N}, ifield::Integer, field::Array{Float32,N}) where {T,P,N}
    compressor.lengths[ifield] = compress!(compressor.compressed_volume, compressor.compressor, field)
    nbytes = unsafe_write(io, convert(Ptr{UInt8},pointer(compressor.compressed_volume)), compressor.lengths[ifield])
    nbytes == compressor.lengths[ifield] || compressedwrite_exception(io, nbytes, compressor.lengths[ifield])
    if ifield < length(compressor.offsets)
        @inbounds compressor.offsets[ifield+1] = compressor.offsets[ifield] + compressor.lengths[ifield]
    end
end

function compressedwrite(io::IO, compressor::Compressor{T,T,T,N}, ifield::Integer, field::Array{T,N}) where {T<:Union{Float32,Float64},N}
    nbytes = write(io, field)
    nbytes == length(field)*sizeof(T) || compressedwrite_exception(io, nbytes, compressor.lengths[ifield])
end

function compressedwrite(io::IO, compressor::Compressor{T,T,T,N}, ifield::Integer, field::SubArray{T,N}) where {T<:Union{Float32,Float64},N}
    copyto!(compressor.volume, field)
    nbytes = write(io, compressor.volume)
    nbytes == length(field)*sizeof(T) || compressedwrite_exception(io, nbytes, compressor.lengths[ifield])
end

function compressedwrite(io::IO, compressor::Compressor{Float64,Float32,Float32,N}, ifield::Integer, field::AbstractArray{Float64,N}) where N
    copyto!(compressor.volume, field)
    nbytes = write(io, compressor.volume)
    nbytes == length(field)*sizeof(Float32) || compressedwrite_exception(io, nbytes, compressor.lengths[ifield])
end

compressedread!(io::IO, compressor::Compressor{Float32,Float32,UInt32,N}, ifield::Integer, field::Array{Float32,N}) where {N} = compressedread_helper!(io, compressor, ifield, field)

function compressedread!(io::IO, compressor::Compressor{Float32,Float32,UInt32,N}, ifield::Integer, field::SubArray{Float32,N}) where N
    compressedread_helper!(io, compressor, ifield, compressor.volume)
    copyto!(field, compressor.volume) # converts from Array to SubArray
end

function compressedread!(io::IO, compressor::Compressor{Float64,Float32,UInt32,N}, ifield::Integer, field::AbstractArray{Float64,N}) where N
    compressedread_helper!(io, compressor, ifield, compressor.volume)
    copyto!(field, compressor.volume)  # converts from Float32 to Float64, and from Array to SubArray
end

function compressedread_helper!(io::IO, compressor::Compressor{T,Float32,UInt32,N}, ifield::Integer, field::Array{Float32,N}) where {T,N}
    seek(io, compressor.offsets[ifield])
    compressed_volume_ptr = convert(Ptr{UInt8}, pointer(compressor.compressed_volume))
    unsafe_read(io, compressed_volume_ptr, compressor.lengths[ifield])
    decompress!(field, compressor.compressor, compressor.compressed_volume, compressor.lengths[ifield])
end

function compressedread!(io::IO, compressor::Compressor{T,T,T,N}, ifield::Integer, field::Array{T,N}) where {T<:Union{Float32,Float64},N}
    seek(io, (ifield-1)*length(field)*sizeof(T))
    read!(io, field)
end

function compressedread!(io::IO, compressor::Compressor{T,T,T,N}, ifield::Integer, field::SubArray{T,N}) where {T<:Union{Float32,Float64},N}
    seek(io, (ifield-1)*length(field)*sizeof(T))
    read!(io, compressor.volume)
    copyto!(field, compressor.volume)
end

function compressedread!(io::IO, compressor::Compressor{Float64,Float32,Float32,N}, ifield::Integer, field::AbstractArray{Float64,N}) where N
    seek(io, (ifield-1)*length(field)*sizeof(Float32))
    read!(io, compressor.volume)
    copyto!(field, compressor.volume)
end

function comptype(comptype, T)
    comptype = comptype == nothing ? T : comptype
    if T == Float64
        @assert comptype == Float64 || comptype == Float32 || comptype == UInt32
    else
        @assert comptype == Float32 || comptype == UInt32
    end
    P = (T == Float64 && comptype == Float64) ? Float64 : Float32
    comptype, P
end
