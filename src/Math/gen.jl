@everywhere module gen
	##===================================================================================
	## fill (square matrix, diagonal matrix, triangular)
	##===================================================================================
	export zeros_sq, ones_sq, rand_sq, randn_sq, fill_sq, fill_dia, ones_dia, rand_dia,
	randn_dia, fill_tri, ones_tri, rand_tri, randn_tri, zeros_vl, ones_vl, rand_vl, randn_vl

	##-----------------------------------------------------------------------------------
	zeros_sq{T<:Integer}(s::T) = fill(0., s, s)

	##-----------------------------------------------------------------------------------
	ones_sq{T<:Integer}(s::T) = fill(1., s, s)

	##-----------------------------------------------------------------------------------
	fill_sq{T<:Integer}(s::T, x) = fill(x, s, s)

	##-----------------------------------------------------------------------------------
	rand_sq{T<:Integer}(s::T) = rand(s, s)

	##-----------------------------------------------------------------------------------
	randn_sq{T<:Integer}(s::T) = randn(s, s)

	##-----------------------------------------------------------------------------------
	ones_dia{T<:Integer}(s::T) = [i == j ? 1. : 0. for i = 1:s, j = 1:s]

	##-----------------------------------------------------------------------------------
	fill_dia{T<:Number, N<:Integer}(x::T, s::N) = [i == j ? x : T(0) for i = 1:s, j = 1:s]

	##-----------------------------------------------------------------------------------
	rand_dia{T<:Integer}(s::T) = [i == j ? rand() : 0 for i = 1:s, j = 1:s]

	##-----------------------------------------------------------------------------------
	randn_dia{T<:Integer}(s::T) = [i == j ? randn() : 0 for i = 1:s, j = 1:s]

	##-----------------------------------------------------------------------------------
	function fill_tri{T<:Any, N<:Integer}(v::T, s::N, upper::Bool = true)
		m = apply_tri_upper((x) -> v, fill(0., s, s))
		return upper ? m : m'
	end

	##-----------------------------------------------------------------------------------
	function ones_tri{T<:Integer}(s::T, upper::Bool = true)
		return tri_fill(1., s, upper)
	end

	##-----------------------------------------------------------------------------------
	function rand_tri{T<:Integer}(s::T, upper::Bool = true)
		m = apply_tri_upper((x) -> rand(), fill(0., s, s))
		return upper ? m : m'
	end

	##-----------------------------------------------------------------------------------
	function randn_tri{T<:Integer}(s::T, upper::Bool = true)
		m = apply_tri_upper((x) -> randn(), fill(0., s, s))
		return upper ? m : m'
	end

	##-----------------------------------------------------------------------------------
	function zeros_vl{T<:Integer}(l::T, d::T)                                               # l = length of vl | d = dimension of the vectors
		v = zeros(d)
		vl = []

		for i = 1:l
			push!(vl, v)
		end

		return vl
	end

	##-----------------------------------------------------------------------------------
	function ones_vl{T<:Integer}(l::T, d::T)                                               # l = length of vl | d = dimension of the vectors
		v = ones(d)
		vl = []

		for i = 1:l
			push!(vl, v)
		end

		return vl
	end

	##-----------------------------------------------------------------------------------
	function rand_vl{T<:Integer}(l::T, d::T)                                               # l = length of vl | d = dimension of the vectors
		vl = []

		for i = 1:l
			push!(vl, rand(d))
		end

		return vl
	end

	##-----------------------------------------------------------------------------------
	function randn_vl{T<:Integer}(l::T, d::T)                                               # l = length of vl | d = dimension of the vectors
		vl = []

		for i = 1:l
			push!(vl, randn(d))
		end

		return vl
	end


	##===================================================================================
	## fills an d^l hypercube with zeros
	##===================================================================================
	export zeros_hs

	##-----------------------------------------------------------------------------------
	function zeros_hs{T<:Integer}(lg::T, dim::T)
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
    ## random vectors (colour/stochastic/orthonormal)
    ##===================================================================================
    export rand_colour_vec, rand_sto_vec, rand_orthonormal_vec

    ##-----------------------------------------------------------------------------------
    rand_colour_vec(rgba = false) = rand(0:1:255, rgba ? 4 : 3)

    ##-----------------------------------------------------------------------------------
    function rand_sto_vec(s::Integer)
        v = rand(s)
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
    ## random circulant matrix
    ##===================================================================================
	export rand_circ_mat

	##-----------------------------------------------------------------------------------
	function rand_circ_mat(s::Integer)
		m = zeros(s, s)
		m[1, :] = rand(s)

		@inbounds for i = 2:s
			m[i, :] = circshift(m[i-1, :], 1)
		end

		return m
	end


    ##===================================================================================
    ## random stochastic matrix
    ##===================================================================================
    export rand_sto_mat

    ##-----------------------------------------------------------------------------------
    function rand_sto_mat{T<:Integer}(sx::T, sy::T)
        m = APL.rand_sto_vec(sy)'

        for i = 2:sx
			m = vcat(m, APL.rand_sto_vec(sy)')
		end

        return m
    end

    ##-----------------------------------------------------------------------------------
    function rand_sto_mat(s::Integer)
        return rand_sto_mat(s, s)
    end


    ##===================================================================================
    ## random vl
    ##===================================================================================
    export vl_rand

    ##-----------------------------------------------------------------------------------
    function vl_rand{T<:Integer}(l::T, w::T)
        vl = Array{Any, 1}

		for i = 1:l
			push!(vl, rand(w))
		end

		return vl
    end

    ##-----------------------------------------------------------------------------------
	function vl_rand(ncbd::tncbd, l::Integer)
		vl = Array{Array{Float64, 1}, 1}(l)												# create an empty vl of length l
		set_zero_subnormals(true)														# to save computing time
		@inbounds for i = 1:l															# fill the list
			vl[i] = ncbd.alpha+(rand(ncbd.n).*ncbd.delta)								# (filling)
		end
		return vl																		# return of the vl
	end


	##===================================================================================
    ## pascal matrix
    ##===================================================================================
	export pasc

	##-----------------------------------------------------------------------------------
	function pasc(s::Integer)
		m = zeros(s, s)

		@inbounds for x = 1:s, y = 1:x
			m[x, y] = binomial((x+y-2), x-1)
			if (x != y)
				m[y, x] = m[x, y]
			end
		end

		return m
	end


	##===================================================================================
    ## exchange matrix
    ##===================================================================================
	export exm

	##-----------------------------------------------------------------------------------
	exm(s::Integer) = [((s+1)-x == y) ? 1 : 0 for x = 1:s, y = 1:s]


	##===================================================================================
    ## hilbert matrix
    ##===================================================================================
	export hbm

	##-----------------------------------------------------------------------------------
	function hbm(s::Integer)
		m = zeros(s, s); c = s
		m[1, :] = ones(s) ./ [x for x = 1:s]

		@inbounds for i = 2:s
			c += 1
			m[i, :] = circshift(m[i-1, :], -1)
			m[i, s] = 1/c
		end

		return m
	end


	##===================================================================================
    ## lehmer matrix
    ##===================================================================================
	export lehm

	##-----------------------------------------------------------------------------------
	function lehm(s::Integer)
		m = eye(s)

		@inbounds for y = 2:s, x = 1:(y-1)
			m[x, y] = x/y
			m[y, x] = m[x, y]
		end

		return m
	end


	##===================================================================================
    ## pauli matrices
    ##===================================================================================
	export pauli

	##-----------------------------------------------------------------------------------
	pauli() = ([.0 1; 1 0], [0 -im; im 0], [1 0.; 0 -1])


	##===================================================================================
    ## redheffer matrix
    ##===================================================================================
	export redh

	##-----------------------------------------------------------------------------------
	function redh(s::Integer)
		m = eye(s)

		@inbounds for i = 2:s
			m[i, 1] = 1
			m[1, i] = 1
		end

		@inbounds for i = 2:s, j = (i << 1):i:s
			m[i, j] = 1
		end

		return m
	end


	##===================================================================================
	## shift matrix
	##===================================================================================
	export shift

	##-----------------------------------------------------------------------------------
	function shift(s::Integer, sup::Bool = true)
		b = sup ? (s+1) : 2
		m = zeros(s, s)

		@inbounds for i = b:(s+1):(s^2)
			m[i] = 1.
		end

		return m
	end


	##===================================================================================
	##  vandermonde
	##===================================================================================
	export vandermonde

	##-----------------------------------------------------------------------------------
	function vandermonde{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, d::N)
		m = ones(v)

		@inbounds for i = 1:d
			m = hcat(m, v.^i)
		end

		return m
	end
end
