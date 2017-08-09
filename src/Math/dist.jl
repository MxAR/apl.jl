@everywhere module dist
    ##===================================================================================
    ## distance function wrapper (returns upper triangle matrix)
    ##===================================================================================
    export distance, distance_parallel

    ##-----------------------------------------------------------------------------------
    function distance{T<:Number}(m::Array{T}, df::Function = (x, y) -> norm(x-y))
        s = size(m, 1); d = sqm_zero(s)
        for y = 2:s, x = 1:(y-1)
            d[x, y] = df(m[x,:], m[y,:])
        end
        return d
    end

    ##-----------------------------------------------------------------------------------
    function distance_parallel{T<:Number}(m::Array{T}, df::Function = (x, y) -> norm(x-y))
        s = size(m, 1); d = convert(SharedArray, sqm_zero(s))
        @sync @parallel for y = 2:s
            for x = 1:(y-1)
                d[x, y] = df(m[x,:], m[y,:])
            end
        end
        return convert(Array, d)
    end


    ##===================================================================================
    ## hamming distance
    ##===================================================================================
    export hamming_distance

    ##-----------------------------------------------------------------------------------
    function hamming_distance{T<:Union{Bool, Char, Int8, Int16, Int32, Int64, Float16, Float32, Float64, UInt8, UInt16, UInt32, UInt64}}(x::T, y::T)
        xb = bits(x); yb = bits(y); c = 0
        @assert length(xb) == length(yb)
        for i = 1:length(xb)
            c = xb[i] == yb[i] ? c : c+1
        end
        return c
    end

    function hamming_distance(x::String, y::String)
        @assert length(x) == length(y)
        return exitsum(map((x, y) -> hamming_distance(x, y), split(x, ""), split(y, "")))
    end


    ##===================================================================================
    ## orthodromic distance
    ##===================================================================================
    export orthodromic_distance

    ##-----------------------------------------------------------------------------------
    function orthodromic_distance(pla, plo, sla, slo, radius, precision = 1)
        sigma = 0.0
        if precision == 1 sigma = central_angle(pla, plo, sla, slo)
        elseif precision == 2 sigma = haversine_central_angle(pla, plo, sla, slo)
        else sigma = vincenty_central_angle(pla, plo, sla, slo) end
        return radius*sigma
    end

    ##-----------------------------------------------------------------------------------
    function orthodromic_distance{T<:Number, N<:Int}(u::Array{T, 1}, v::Array{T, 1}, radius, center::Array{T, 1} = zeros(N), precision = 1)
        sigma = 0.0
        u = normalize(u, center)
        v = normalize(v, center)
        if precision == 1 sigma = acos_central_angle(u, v)
        elseif precision == 2 sigma = asin_central_angle(u, v)
        else sigma = atan_central_angle(u, v) end
        return radius*sigma
    end


    ##===================================================================================
    ## mahalanobis distance
    ##===================================================================================
    export mahal_dist, mahal_dist_d_vec, mahal_dist_d_cov, mahal_distsq, mahal_distsq_derivate_vec,
        mahal_distsq_derivate_cov

    ##-----------------------------------------------------------------------------------
    function mahal_dist{T<:Real}(X::Array{T, 1}, Y::Array{T, 1}, CovMatrix::Array{T, 2})
        @fastmath d = X - Y                                                             # the mahalanobis_distance can be used either as
        @fastmath return sqrt(abs(bdot(d, (CovMatrix \ d))))                            # point-point distance (Y == second point) or as a
    end                                                                                 # point-distribution distance (Y == median of the distribution)

    ##-----------------------------------------------------------------------------------
    function mahal_dist_d_vec{T<:Real}(X::Array{T, 1}, Y::Array{T, 1}, CovMatrix::Array{T, 2})
        @fastmath d = X - Y                                                             # derivation after X or Y
        p = CovMatrix \ d
        @fastmath return bdot(fill(4.0, length(X)), p) / sqrt(abs(bdot(d, p)))
    end

    ##-----------------------------------------------------------------------------------
    function mahal_dist_d_cov{T<:Real}(X::Array{T, 1}, Y::Array{T, 1}, CovMatrix::Array{T, 2})
        @fastmath d = X - Y
        @fastmath return (-2 * bdot(d, (CovMatrix^2 \ d))) / sqrt(abs(bdot(d, (CovMatrix \ d))))
    end

    ##-----------------------------------------------------------------------------------
    function mahal_distsq{T<:Real}(X::Array{T, 1}, Y::Array{T, 1}, CovMatrix::Array{T, 2})
        @fastmath d = X - Y
        @fastmath return bdot(d, (CovMatrix \ d))
    end

    ##-----------------------------------------------------------------------------------
    function mahal_distsq_derivate_vec{T<:Real}(X::Array{T, 1}, Y::Array{T, 1}, CovMatrix::Array{T, 2})
        @fastmath return bdot(fill(2.0, length(X)), (CovMatrix \ (X - Y)))              # derivation after X or Y
    end

    ##-----------------------------------------------------------------------------------
    function mahal_distsq_derivate_cov{T<:Real}(X::Array{T, 1}, Y::Array{T, 1}, CovMatrix::Array{T, 2})
        @fastmath d = X - Y
        @fastmath return (-2 * bdot(d, (CovMatrix^2 \ d)))
    end
end
