@everywhere module mat_op
    ##===================================================================================
    ##  using directives
    ##===================================================================================
    using cnv
    using f

    ##===================================================================================
    ## normalize
    ##===================================================================================
    export normalize, normalize_sta, normalize_sta_parallel, normalize_sta_parallel_shared

    ##-----------------------------------------------------------------------------------
    function normalize_sta{T<:Number}(m::Array{T, 2})                                   # sets variance to 1 and mean to 0
        d = size(m, 1)
        for w = 1:size(m, 2)
            m[1:d, w] = (m[1:d, w] - median(m[1:d, w])) / std(m[1:d, w])
        end
        return m
    end

    ##-----------------------------------------------------------------------------------
    function normalize_sta_parallel{T<:Number}(m::Array{T, 2})
        d = size(m, 1); m = convert(SharedArray, m)
        @sync @parallel for w = 1:size(m, 2)
            m[1:d, w] = (m[1:d, w] - median(m[1:d, w])) / std(m[1:d, w])
        end
        return convert(Array, m)
    end

    ##-----------------------------------------------------------------------------------
    function normalize_sta_parallel_shared{T<:Number}(m::Array{T, 2})
        d = size(m, 1);
        @sync @parallel for w = 1:size(m, 2)
            m[1:d, w] = (m[1:d, w] - median(m[1:d, w])) / std(m[1:d, w])
        end
        return m
    end


    ##===================================================================================
    ## rotation_matrix
    ##===================================================================================
    export rotation_matrix

    ##-----------------------------------------------------------------------------------
    function rotation_matrix{T<:Number}(axis::Array{T, 1}, angle)
        axis = axis'
        m = [ 0 -axis[3] axis[2]; axis[3] 0 -axis[1]; -axis[2] axis[1] 0 ]
        return eye(3) + m * sind(alpha) + (1 - cosd(alpha)) * m^2
    end


    ##===================================================================================
    ## median column/row
    ##===================================================================================
    export mmed, vmed

    ##-----------------------------------------------------------------------------------
    function mmed{T<:Float64}(arr::Array{T, 2}, column::Bool = true)
        n = size(X, (column ? 1 : 2))
        return BLAS.gemv((column ? 'N' : 'T'), 1/n, arr, ones(n))
    end

    ##-----------------------------------------------------------------------------------
    function mmed{T<:Float64}(arr::Array{T, 2}, weights::Array{T, 1}, column::Bool = true)
        n = size(X, (column ? 1 : 2))
        return BLAS.gemv((column ? 'N' : 'T'), 1/n, weights.*arr, ones(n))
    end

    ##-----------------------------------------------------------------------------------
    vmed{T<:Float64}(v::Array{T, 1}) = bdot(v, ones(l))/l


    ##===================================================================================
    ## covariance
    ##===================================================================================
    export cov

    ##-----------------------------------------------------------------------------------
    cov(x, mx, y, my, p) = p*(x-mx)*(y-my)'                                             # m* median of * | p = probability


    ##===================================================================================
    ## covariance matrices from observations
    ##===================================================================================
    export covp, covs

    ##-----------------------------------------------------------------------------------
    function covp{T<:Real}(samples::Array{T, 2})                                        # cov population
        n = size(samples, 1)
        m = BLAS.gemv('T', samples, ones(n))
        return BLAS.gemm('T', 'N', 1/n, samples, samples) - (BLAS.nrm2(n) / n)^2
    end

    ##-----------------------------------------------------------------------------------
    function covs{T<:Real}(samples::Array{T, 2})                                        # cov sample
        n = size(samples, 1)
        m = BLAS.gemv('T', samples, ones(n))
        return BLAS.gemm('T', 'N', 1/(n-1), samples, samples) - (bdot(n, n) / (n*(n-1)))
    end


    ##===================================================================================
    ## cross covariance
    ##===================================================================================
    export ccov

    ##-----------------------------------------------------------------------------------
    function ccov{T<:Real, N<:Real}(x::Array{T, 1}, y::Array{N, 1})
        xs = length(x); ys = length(y)
        xm = vmed(x); ym = vmed(y)
        m = zeros(xs, ys); sc = 1/(xs*ys)
        for xi = 1:xs, yi = 1:ys
            m = cov(x[xi], xm, y[yi], ym, sc)
        end
        return m
    end


    ##===================================================================================
    ## cross covariance sumed (with delay)
    ##===================================================================================
    export ccovs

    ##-----------------------------------------------------------------------------------
    function ccovs{T<:Real, N<:Real}(v::Array{T, 1}, u::Array{N, 1}, tau::Int64 = 1)    # ccov sumed
        return bdot(l, (v-vmed(v)), (circshift(u, tau)-vmed(u)))/l
    end


    ##===================================================================================
    ## cross correlation (with delay)
    ##===================================================================================
    export ccor

    ##-----------------------------------------------------------------------------------
    ccor{T<:Real, N<:Real}(v::Array{T, 1}, u::Array{N, 1}, tau::Int64 = 1) = ccov(v, u, tau)/(std(v)*std(u))


    ##===================================================================================
    ## supp (support)
    ##===================================================================================
    export supp

    ##-----------------------------------------------------------------------------------
    function supp{T<:Number}(v::Array{T, 1})
        u = Array{T, 1}
        for x in v if x != 0 push!(u, v) end end
        return u
    end

    ##-----------------------------------------------------------------------------------
    function supp{T<:Number}(vl::Array{Array{T, 1}, 1})                                 # supp for vector lists
        ul = Array{Array{T, 1}, 1}
        for v in vl push!(ul, supp(v)) end
        return ul
    end


    ##===================================================================================
    ## random stochastic matrix
    ##===================================================================================
    export rand_sto_mat

    ##-----------------------------------------------------------------------------------
    function rand_sto_mat(sx::Int, sy::Int)
        m = APL.rand_sto_vec(sy)'
        for i = 2:sx m = vcat(m, APL.rand_sto_vec(sy)') end
        return m
    end

    ##-----------------------------------------------------------------------------------
    function rand_sto_mat(s::Int)
        return rand_sto_mat(s, s)
    end


    ##===================================================================================
    ## random vl
    ##===================================================================================
    export randvl

    ##-----------------------------------------------------------------------------------
    function randvl{T<:Int, N<:Int}(l::T, w::N)
        vl = Array{Any, 1}
        for i = 1:l push!(vl, rand(w)) end
        return vl
    end


    ##===================================================================================
    ## samples
    ##===================================================================================
    export samples

    ##-----------------------------------------------------------------------------------
    function samples{T<:Any}(data::Array{T, 1}, size::Int)
        L = length(data)
        @assert size < L ["The number of samples musn't be bigger than the data!"]
        return shuffle(getindex(data, sort(sample(1:L, Size, replace = false))))
    end


    ##===================================================================================
    ## checks
    ##===================================================================================
    export iszero, levi_civita_tensor, index_permutations_count

    ##-----------------------------------------------------------------------------------
    iszero(v) = sumabs(v) == 0

    ##-----------------------------------------------------------------------------------
    function levi_civita_tensor{T<:Number}(v::Array{T, 1})
        return ifelse(0 == index_permutations_count(v) % 2, 1, -1)
    end

    ##-----------------------------------------------------------------------------------
    function index_permutations_count{T<:Any}(v::Array{T, 1})                           # [3,4,5,2,1] -> [1,2,3,4,5]
        c = 0; s = length(v)                                                            # 3 inversions needed
        t = linspace(1, s, s)
        while v != t
            for i = 1:length(v)
                if v[i] != i
                    s = find(v .== i)
                    if s != i
                        v[s] = v[i]
                        v[i] = i
                        c += 1
                    end
                end
            end
        end
        return c
    end


    ##===================================================================================
    ## random vectors (colour/stochastic/orthonormal)
    ##===================================================================================
    export rand_colour_vec, rand_sto_vec, rand_orthonormal_vec

    ##-----------------------------------------------------------------------------------
    function rand_colour_vec(rgba = false)
        return rand(0:255, rgba ? 4 : 3)
    end

    ##-----------------------------------------------------------------------------------
    function rand_sto_vec(size::Int = 3)
        v = rand(size)
        v = (v' \ [1.0]) .* v
        v[find(v .== maximum(v))] += 1.0 - sum(v)
        return v
    end

    ##-----------------------------------------------------------------------------------
    function rand_orthonormal_vec{T<:Number}(v::Array{T, 1})
        u = [rand(), rand(), 0]
        u[3] = (v[1] * u[1] + v[2] * u[2]) / (-1 * (v[3] == 0 ? 1 : v[3]))
        return normalize(u)
    end


    ##===================================================================================
    ## rm_column
    ##===================================================================================
    export rm_column, rm_column_many, rm_column_many_sorted, rm_column_range

    ##-----------------------------------------------------------------------------------
    function rm_column{T<:Any}(m::Array{T, 2}, c)
        return hcat(m[:, 1:(c-1)], m[:, (c+1):end])
    end

    ##-----------------------------------------------------------------------------------
    function rm_column_many_sorted{T<:Any}(m::Array{T, 2}, c::Array{Any, 1})
        for x in c
            m = hcat(m[:, 1:(x-1)], m[:, (x+1):end])
            c .-= 1
        end
        return m
    end

    ##-----------------------------------------------------------------------------------
    function rm_column_many{T<:Any}(m::Array{T, 2}, c::Array{Any, 1})
        for x in sort(c)
            m = hcat(m[:, 1:(x-1)], m[:, (x+1):end])
            c .-= 1
        end
        return m
    end

    ##-----------------------------------------------------------------------------------
    function rm_column_range{T<:Any}(m::Array{T, 2}, upper_bound = 1, lower_bound = 0)
        return hcat(m[:, 1:(lower_bound-1)], m[:, (upper_bound+1):end])
    end


    ##===================================================================================
    ## rm
    ##===================================================================================
    export rm, rm_sorted

    ##-----------------------------------------------------------------------------------
    function rm{T<:Any, N<:Int}(v::Array{T, 1}, i::Array{N, 1})
        i = sort(i)
        for j=1:length(i)
            v = cat(1, v[1:(i[j]-1)], v[(i[j]+1):end])
            i .-= 1
        end
        return v
    end

    ##-----------------------------------------------------------------------------------
    function rm_sorted{T<:Any, N<:Int}(v::Array{T, 1}, i::Array{N, 1})
        for j=1:length(i)
            v = cat(1, v[1:(i[j]-1)], v[(i[j]+1):end])
            i .-= 1
        end
        return v
    end


    ##===================================================================================
    ## union overload
    ##===================================================================================
    import Base.union
    export union

    ##-----------------------------------------------------------------------------------
    function union{T<:Any}(vl::Array{Array{T, 1}, 1})
        v = vl[1]
        for i=2:length(vl)
            v = union(v, vl[i])
        end
        return v
    end


    ##===================================================================================
    ## intersect overload
    ##===================================================================================
    import Base.intersect
    export intersect

    ##-----------------------------------------------------------------------------------
    function intersect{T<:Any}(vl::Array{Array{T, 1}, 1})
        v = vl[1]
        for i=2:length(vl)
            v = intersect(v, vl[i])
        end
        return v
    end


    ##===================================================================================
    ## prepend
    ##===================================================================================
    export prepend, prepend!

    ##-----------------------------------------------------------------------------------
    function prepend{T<:Any}(v::Array{T, 1}, w)
        return cat(1, [w], v)
    end

    ##-----------------------------------------------------------------------------------
    function prepend{T<:Any}(v::Array{T, 1}, w::Array{T, 1})
        return cat(1, w, v)
    end

    ##-----------------------------------------------------------------------------------
    function prepend!{T<:Any}(v::Array{T, 1}, w)
        return v = cat(1, [w], v)
    end

    ##-----------------------------------------------------------------------------------
    function prepend!{T<:Any}(v::Array{T, 1}, w::Array{T, 1})
        return v = cat(1, w, v)
    end


    ##===================================================================================
    ## fill (square matrix, diagonal matrix, triangular)
    ##===================================================================================
    export sq_zeros, sq_ones, sq_fill, dia_fill, dia_rand, dia_randn, tri_fill, tri_ones, tri_rand,
        tri_randn, vl_zeros, vl_rand, vl_randn

    ##-----------------------------------------------------------------------------------
    sq_zeros{T<:Int}(s::T) = zeros(s, s)

    ##-----------------------------------------------------------------------------------
    sq_ones{T<:Int}(s::T) = ones(s, s)

    ##-----------------------------------------------------------------------------------
    sq_fill{T<:Int}(s::T, x) = fill(x, s, s)

    ##-----------------------------------------------------------------------------------
    function dia_fill{T<:Int}(value, s::T)
        m = sq_zeros(s)
        for i = 1:s m[i, i] = value end
        return m
    end

    ##-----------------------------------------------------------------------------------
    function dia_rand{T<:Int}(s::T)
        m = sq_zeros(s); r = rand(s)
        for i = 1:s m[i, i] = r[i] end
        return m
    end

    ##-----------------------------------------------------------------------------------
    function dia_randn{T<:Int}(s::T)
        m = sq_zeros(s); r = randn(s)
        for i = 1:s m[i, i] = r[i] end
        return m
    end

    ##-----------------------------------------------------------------------------------
    function tri_fill{T<:Int}(value, s::T, upper = true)
        m = apply_tri_upper((x) -> value, sq_zeros(s))
        return upper ? m : m'
    end

    ##-----------------------------------------------------------------------------------
    function tri_ones{T<:Int}(s::T, upper = true)
        return tri_fill(1, s, upper)
    end

    ##-----------------------------------------------------------------------------------
    function tri_rand{T<:Int}(s::T, upper = true)
        m = apply_tri_upper((x) -> rand(), sq_zeros(s))
        return upper ? m : m'
    end

    ##-----------------------------------------------------------------------------------
    function tri_randn{T<:Int}(s::T, upper = true)
        m = apply_tri_upper((x) -> randn(), sq_zeros(s))
        return upper ? m : m'
    end

    ##-----------------------------------------------------------------------------------
    function vl_zeros{T<:Int}(l::T, d::T)                                               # l = length of vl | d = dimension of the vectors
        vl = []; v = zeros(d)
        for i = 1:l push!(vl, v) end
        return vl
    end

    ##-----------------------------------------------------------------------------------
    function vl_rand{T<:Int}(l::T, d::T)                                               # l = length of vl | d = dimension of the vectors
        vl = [];
        for i = 1:l push!(vl, rand(d)) end
        return vl
    end

    ##-----------------------------------------------------------------------------------
    function vl_randn{T<:Int}(l::T, d::T)                                               # l = length of vl | d = dimension of the vectors
        vl = [];
        for i = 1:l push!(vl, randn(d)) end
        return vl
    end


    ##===================================================================================
    ## fills an d^l hypercube with zeros
    ##===================================================================================
    export hs_zero

    ##-----------------------------------------------------------------------------------
    function hs_zero{T<:Int}(lg::T, dim::T)
        hs = zeros(lg)
        for d = 2:(dim)
            el = hs
            for i = 2:lg
                hs = cat(d, el, hs)
            end
        end
        return hs
    end


    ##===================================================================================
    ## split
    ##===================================================================================
    export msplit, msplit_half

    ##-----------------------------------------------------------------------------------
    function msplit(m, i, lrows)
        if !lrows m = m' end;
        @assert i < 0 || i >= size(m, 1)
        return (m[1:i, :], m[(i+1):end, :])
    end

    ##-----------------------------------------------------------------------------------
    function msplit_half(m, lrows = true)
        if !lrows m = m' end;
        l = convert(Int, round(size(m, 1)/2))
        return (m[1:l, :], m[(l+1):end, :])
    end


    ##===================================================================================
    ## map (overload)
    ##===================================================================================
    import Base.map
    export map

    ##===================================================================================
    function map{T<:Any}(f::Function, vl::Array{Array{T, 1}, 1})
        ul = Array{Array{T, 1}, 1}
        @simd for i = 1:length(ul)
            push!(ul, f(ul[i]))
        end
        return ul
    end


    ##===================================================================================
    ## apply
    ##===================================================================================
    export apply, apply_parallel, apply_parallel_shared, apply_tri_upper, apply_tri_lower

    ##-----------------------------------------------------------------------------------
    function apply(f::Function, m)
        for i in eachindex(m)
            m[i] = f(m[i])
        end
        return m
    end

    ##-----------------------------------------------------------------------------------
    function apply_parallel(f::Function, m)
        m = convert(SharedArray, m)
        @sync @parallel for i in eachindex(m)
            m[i] = f(m[i])
        end
        return convert(Array, m)
    end

    ##-----------------------------------------------------------------------------------
    function apply_parallel_shared(f::Function, m)
        @sync @parallel for i in eachindex(m)
            m[i] = f(m[i])
        end
        return m
    end

    ##------------------------------------------------------------------------------------
    function apply_tri_upper(f::Function, m)
        for j = 2:size(m, 2), i = 1:j-1
            m[i, j] = f(m[i, j])
        end
        return m
    end

    ##------------------------------------------------------------------------------------
    function apply_tri_lower(f::Function, m)
        for i = 2:size(m, 2), j = 1:i-1
            m[i, j] = f(m[i, j])
        end
        return m
    end
end
