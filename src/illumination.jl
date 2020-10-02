srcillum_helper(field_ginsu::Array{T}, nthreads) where {T<:Float32} =
    ccall((:srcillum_helper_float,Wave._jl_libillumination), Cvoid,
        (Ptr{Float32},Csize_t,Cint), field_ginsu, length(field_ginsu), nthreads)

srcillum_helper(field_ginsu::Array{T}, nthreads) where {T<:Float64} =
    ccall((:srcillum_helper_double,Wave._jl_libillumination), Cvoid,
        (Ptr{Float64},Csize_t,Cint), field_ginsu, length(field_ginsu), nthreads)

illum_accumulate(field_ginsu_accum::Array{T}, field_ginsu::Array{T}, nthreads) where {T<:Float32} =
    ccall((:illum_accumulate_float,Wave._jl_libillumination), Cvoid,
        (Ptr{Float32},Ptr{Float32},Csize_t,Cint), field_ginsu_accum, field_ginsu, length(field_ginsu), nthreads)

illum_accumulate(field_ginsu_accum::Array{T}, field_ginsu::Array{T}, nthreads) where {T<:Float64} =
    ccall((:illum_accumulate_double,Wave._jl_libillumination), Cvoid,
        (Ptr{Float64},Ptr{Float64},Csize_t,Cint), field_ginsu_accum, field_ginsu, length(field_ginsu), nthreads)
