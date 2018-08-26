@everywhere module dist
	##===================================================================================
	##	using directives
	##===================================================================================
	using Distributed


	##===================================================================================
	## beta divergence/distance
	##===================================================================================
	export beta_dist

	##-----------------------------------------------------------------------------------
	function beta_dist(beta::R, l::Z, x::Array{T}, inc_x, y::Array{T}, inc_y::Z) where T<:Number where Z<:Integer where R<:Real
		@assert(l > 0, "l needs to be a positive number")
		@assert(inc_x > 0, "index increment for vector x needs to be positive (inc_x)")
		@assert(inc_y > 0, "index increment for vector y needs to be positive (inc_y)")
		@assert(inc_x*l <= length(x), "the number of requested computations musn't be larger than the dimension of x")
		@assert(inc_y*l <= length(y), "the number of requested computations musn't be larger than the dimension of y")
		@assert(0 <= beta && beta <= 1, "beta must lie in [0, 1]") 

		s = T(1)
		r = T(0)
		f = NaN

		x_i = 1
		y_i = 1
		i = 1

		if beta == 1
			f = (a, b) -> a * log(a / b) - a + b
		elseif beta == 0
			f = (a, b) -> (a / b) - log(a / b)
			r = T(-l)
		else
			f = (a, b) -> a^beta + ((beta - 1) * b^beta) - (beta * a * b^(beta - 1))
			s = (beta * (beta - 1))
		end

		@inbounds while i <= l
			@fastmath r = r + f(x[x_i], y[y_i])

			x_i = x_i + inc_x
			y_i = y_i + inc_y
			i += 1
		end

		return r / (s * l)
	end


	##===================================================================================
	## log specteal distance
	##===================================================================================
	export log_spectral_dist

	##-----------------------------------------------------------------------------------
	function log_spectral_dist(l::Z, x::Array{T}, inc_x, y::Array{T}, inc_y::Z) where T<:Number where Z<:Integer
		@assert(l > 0, "l needs to be a positive number")
		@assert(inc_x > 0, "index increment for vector x needs to be positive (inc_x)")
		@assert(inc_y > 0, "index increment for vector y needs to be positive (inc_y)")
		@assert(inc_x*l <= length(x), "the number of requested computations musn't be larger than the dimension of x")
		@assert(inc_y*l <= length(y), "the number of requested computations musn't be larger than the dimension of y")
		
		r = T(0)
		x_i = 1
		y_i = 1
		i = 1

		@inbounds while i <= l
			@fastmath r = r + log10(x[x_i] / y[y_i])^2

			x_i = x_i + inc_x
			y_i = y_i + inc_y
			i += 1
		end

		r = @fastmath sqrt(r / (.01 * l))
		return r
	end

    
	##===================================================================================
	## itakura-saito distance/divergence
	##	(all values have to be bigger than 0)
	##===================================================================================
	export itakura_sait_dist

	##-----------------------------------------------------------------------------------
	function itakura_saito_dist(l::Z, x::Array{T}, inc_x::Z, y::Array{T}, inc_y::Z) where T<:Number where Z<:Integer
		@assert(l > 0, "l needs to be a positive number")
		@assert(inc_x > 0, "index increment for vector x needs to be positive (inc_x)")
		@assert(inc_y > 0, "index increment for vector y needs to be positive (inc_y)")
		@assert(inc_x*l <= length(x), "the number of requested computations musn't be larger than the dimension of x")
		@assert(inc_y*l <= length(y), "the number of requested computations musn't be larger than the dimension of y")

		r = T(-l)
		q = T(0)

		x_i = 1
		y_i = 1
		i = 1

		@inbounds while i <= l
			@fastmath q = x[x_i] / y[y_i]
			@fastmath r = r + q - log(q)

			x_i = x_i + inc_x
			y_i = y_i + inc_y
			i = i + 1
		end

		return r / l
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
