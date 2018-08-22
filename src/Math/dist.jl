@everywhere module dist
	##===================================================================================
	##	using directives
	##===================================================================================
	using Distributed


    ##===================================================================================
	## itakura-saito distance
	##	(all values have to be bigger than 0)
	##===================================================================================
	export isdist

	##-----------------------------------------------------------------------------------
	function isdist(m::Array{R}, n::Array{R}) where R<:AbstractFloat
		s = min(length(m), length(n))
		r = R(-s)
		a = R(0)
		i = 1

		@inbounds while i <= s
			@fastmath a = m[i] / n[i]
			@fastmath r = r + a - log(a)
			i = i + 1
		end

		return r
	end
	
	
	##===================================================================================
    ## distance function wrapper (returns upper triangle matrix)
    ##===================================================================================
    export gdist, gdist_p

    ##-----------------------------------------------------------------------------------
    function gdist(m::Array{T}, df::Function = (x, y) -> norm(x-y)) where T<:Number
        s = size(m, 1)
        d = fill(0, s, s)

        @inbounds for y = 2:s, x = 1:(y-1)
            d[x, y] = df(m[x,:], m[y,:])
        end

        return d
    end

    ##-----------------------------------------------------------------------------------
    function gdist_p(m::Array{T}, df::Function = (x, y) -> norm(x-y)) where T<:Number
        s = size(m, 1);
        d = convert(SharedArray, fill(0, s, s))

        @inbounds @sync @distributed for y = 2:s
            for x = 1:(y-1)
                d[x, y] = df(m[x,:], m[y,:])
            end
        end

        return convert(Array, d)
    end


    ##===================================================================================
    ## hamming distance
    ##===================================================================================
    export hdist

    ##-----------------------------------------------------------------------------------
    function hdist(x::Z, y::Z) where Z<:Integer
        xb = bits(x)
        yb = bits(y)
        c = 0

        @assert(size(xb, 1) == size(yb, 1))
        @inbounds for i = 1:size(xb, 1)
            if xb[i] != yb[i]
                c += 1
            end
        end

        return c
    end

    ##-----------------------------------------------------------------------------------
    function hdist(x::String, y::String)
        @assert(size(x, 1) == size(y, 1))
        return sum(map((x, y) -> hamming_distance(x, y), split(x, ""), split(y, "")))
    end


    ##===================================================================================
    ## orthodromic distance
    ##===================================================================================
    export odist

    ##-----------------------------------------------------------------------------------
    function odist(pla::R, plo::R, sla::R, slo::R, radius::R, precision::R = R(1)) where R<:AbstractFloat
        sigma = R(0)

        if precision == 1
            sigma = central_angle(pla, plo, sla, slo)
        elseif precision == 2
            sigma = haversine_central_angle(pla, plo, sla, slo)
        else
            sigma = vincenty_central_angle(pla, plo, sla, slo)
        end

        return radius*sigma
    end

    ##-----------------------------------------------------------------------------------
    function odist(u::Array{R, 1}, v::Array{R, 1}, radius::R, center::Array{R, 1}, precision::R = (1)) where R<:AbstractFloat
        u = normalize(u, center)
        v = normalize(v, center)
        sigma = R(0)

        if precision == 1
            sigma = acos_central_angle(u, v)
        elseif precision == 2
            sigma = asin_central_angle(u, v)
        else
            sigma = atan_central_angle(u, v)
        end

        return radius*sigma
    end


    ##===================================================================================
    ## mahalanobis distance
    ##===================================================================================
    export mdist, mdist_dvec, mdist_dcov, mdist_sq, mdist_sq_dvec, mdist_sq_dcov

    ##-----------------------------------------------------------------------------------
    function mdist(X::Array{R, 1}, Y::Array{R, 1}, CovMatrix::Array{R, 2}) where R<:AbstractFloat
        @fastmath d = X - Y                                                             # the mahalanobis_distance can be used either as
        @fastmath return sqrt(abs(bdot(d, (CovMatrix \ d))))                            # point-point distance (Y == second point) or as a
    end                                                                                 # point-distribution distance (Y == median of the distribution)

    ##-----------------------------------------------------------------------------------
    function mdist_dvec(X::Array{R, 1}, Y::Array{R, 1}, CovMatrix::Array{R, 2}) where R<:Real
        @fastmath d = X - Y                                                             # derivation after X or Y
        p = CovMatrix \ d
        @fastmath return bdot(fill(4.0, length(X)), p) / sqrt(abs(bdot(d, p)))
    end

    ##-----------------------------------------------------------------------------------
    function mdist_dcov(X::Array{R, 1}, Y::Array{R, 1}, CovMatrix::Array{R, 2}) where R<:Real
        @fastmath d = X - Y
        @fastmath return (-2 * bdot(d, (CovMatrix^2 \ d))) / sqrt(abs(bdot(d, (CovMatrix \ d))))
    end

    ##-----------------------------------------------------------------------------------
    function mdist_sq(X::Array{R, 1}, Y::Array{R, 1}, CovMatrix::Array{R, 2}) where R<:Real
        @fastmath d = X - Y
        @fastmath return bdot(d, (CovMatrix \ d))
    end

    ##-----------------------------------------------------------------------------------
    function mdist_sq_dvec(X::Array{R, 1}, Y::Array{R, 1}, CovMatrix::Array{R, 2}) where R<:Real
        @fastmath return bdot(fill(2.0, length(X)), (CovMatrix \ (X - Y)))              # derivation after X or Y
    end

    ##-----------------------------------------------------------------------------------
    function mdist_sq_dcov(X::Array{R, 1}, Y::Array{R, 1}, CovMatrix::Array{R, 2}) where R<:Real
        @fastmath d = X - Y
        @fastmath return (-2 * bdot(d, (CovMatrix^2 \ d)))
    end
end
