@everywhere module gen
	##===================================================================================
	##  types
	##===================================================================================
	mutable struct tncbd{T<:AbstractFloat, N<:Integer}	# n dimensional cuboid
		alpha::Array{T, 1}								# infimum (point)
		delta::Array{T, 1}								# diference between supremum and infimum	(s-i)
		n::N											# n
	end


	##===================================================================================
	##	allocates a random vector of unique integers
	##===================================================================================
	export urand

	##-----------------------------------------------------------------------------------
	function urand(q::UnitRange{Z}, s::Z) where Z<:Integer
		j = abs(q[end] - q[1] + 1)
		r = Array{Z, 1}(s)
		w = Array{Z, 1}(q)
		i = 1
		
		@inbounds while i <= s
			k = rand(1:j)
			r[i] = w[k]
			w = vcat(w[1:(k - 1)], w[(k + 1):j])
			j = j - 1
			i = i + 1
		end

		return r
	end

	##===================================================================================
	##	fill (square matrix, diagonal matrix, triangular)
	##		l: length (when l is a parameter)
	##		d: dimension (when d is a paramter)
	##		z: zeros
	##		o: ones
	##		f: fill
	##		r: random
	##		rn: normal distribtion
	##===================================================================================
	export zsqm, osqm, rsqm, rnsqm, fsqm, fdiam, odiam, rdiam,
	rndiam, ftrim, otrim, rtrim, rntrim, zvl, ovl, rvl, rnvl

	##-----------------------------------------------------------------------------------
	zsqm(s::Z) where Z<:Integer = fill(0., s, s)

	##-----------------------------------------------------------------------------------
	osqm(s::Z) where Z<:Integer = fill(1., s, s)

	##-----------------------------------------------------------------------------------
	fsqm(s::Z, x::T) where T<:Any where Z<:Integer = fill(x, s, s)

	##-----------------------------------------------------------------------------------
	rsqm(s::Z) where Z<:Integer = rand(s, s)

	##-----------------------------------------------------------------------------------
	rnsqm(s::Z) where Z<:Integer = randn(s, s)

	##-----------------------------------------------------------------------------------
	fdiam(x::T, s::Z) where T<:Number where Z<:Integer = [i == j ? x : T(0) for i = 1:s, j = 1:s]

	##-----------------------------------------------------------------------------------
	rdiam(s::Z) where Z<:Integer = [i == j ? rand() : 0 for i = 1:s, j = 1:s]

	##-----------------------------------------------------------------------------------
	rndiam(s::Z) where Z<:Integer = [i == j ? randn() : 0 for i = 1:s, j = 1:s]

	##-----------------------------------------------------------------------------------
	function ftrim(v::T, s::Z, upper::Bool = true) where T<:Any where Z<:Integer
		m = apply_tri_upper((x) -> v, fill(0., s, s))
		return upper ? m : m'
	end

	##-----------------------------------------------------------------------------------
	function otrim(s::Z, upper::Bool = true) where Z<:Integer
		return tri_fill(1., s, upper)
	end

	##-----------------------------------------------------------------------------------
	function rtrim(s::Z, upper::Bool = true) where Z<:Integer
		m = apply_tri_upper((x) -> rand(), fill(0., s, s))
		return upper ? m : m'
	end

	##-----------------------------------------------------------------------------------
	function rntrim(s::Z, upper::Bool = true) where Z<:Integer
		m = apply_tri_upper((x) -> randn(), fill(0., s, s))
		return upper ? m : m'
	end

	##-----------------------------------------------------------------------------------
	function zvl(l::Z, d::Z) where Z<:Integer
		v = zeros(d)
		vl = []

		@inbounds for i = 1:l
			push!(vl, v)
		end

		return vl
	end

	##-----------------------------------------------------------------------------------
	function ovl(l::Z, d::Z) where Z<:Integer
		v = ones(d)
		vl = []

		@inbounds for i = 1:l
			push!(vl, v)
		end

		return vl
	end

	##-----------------------------------------------------------------------------------
	function rvl(l::Z, d::Z) where Z<:Integer
		vl = []

		@inbounds for i = 1:l
			push!(vl, rand(d))
		end

		return vl
	end

	##-----------------------------------------------------------------------------------
	function rnvl(l::Z, d::Z) where Z<:Integer
		vl = []

		@inbounds for i = 1:l
			push!(vl, randn(d))
		end

		return vl
	end


	##===================================================================================
	## fills an d^l hypercube with zeros
	##===================================================================================
	export zhs

	##-----------------------------------------------------------------------------------
	function zhs(lg::Z, dim::Z) where Z<:Integer
		hs = zeros(lg)

		@inbounds for d = 2:(dim)
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
    export rrgbv, rstov, rorthv

    ##-----------------------------------------------------------------------------------
    rrgbv(rgba = false) = rand(0:1:255, rgba ? 4 : 3)

    ##-----------------------------------------------------------------------------------
	function rstov(s::Z) where Z<:Integer
        v = rand(s)
        v .*= (v'\[1.])
		
		me = -Inf
		mi = N(1)
		sm = 1.
		@inbounds for i = 1:size(v, 1)
			sm -= v[i]
			if v[i] > me
				me = v[i]
				mi = i
			end
		end 

        v[i] .+= sm
        return v
    end

    ##-----------------------------------------------------------------------------------
    function rorthv(v::Array{T, 1}) where T<:Number
        u = [rand(), rand(), 0]
        u[3] = (v[1] * u[1] + v[2] * u[2]) / (-1 * (v[3] == 0 ? 1 : v[3]))
        return normalize(u)
    end


	##===================================================================================
	## hadamard matrix
	##	- remember s has to be a multiple of two and bigger than zero
	##===================================================================================
	import ..bin.bit_dot
	export hadamard

	##-----------------------------------------------------------------------------------
	function hadamard(s::Z) where Z<:Integer
		r = fill(-1, s, s)
		i = UInt(1)
		k = UInt(0)

		@inbounds while i <= s
			j = UInt(1)
			while j <= s
				if bit_dot(k, j - 1) % 2 == 0
					r[i, j] = 1
				end
				j = j + 1
			end
			k = k + 1
			i = i + 1
		end

		return r
	end


	##===================================================================================
    ## random circulant matrix
    ##===================================================================================
	export rcircm, rncircm

	##-----------------------------------------------------------------------------------
	function rcircm(s::Z) where Z<:Integer
		m = zeros(s, s)
		m[1, :] = rand(s)

		@inbounds for i = 2:s
			m[i, :] = circshift(m[i-1, :], 1)
		end

		return m
	end

	##-----------------------------------------------------------------------------------
	function rncircm(s::Z) where Z<:Integer
		m = zeros(s)
		m[1, :] = randn(s)

		@inbounds for i = 2:s
			m[i, :] = circshift(m[i-1, :], 1)
		end

		return m
	end 


    ##===================================================================================
    ## random stochastic matrix
    ##===================================================================================
    export rstom

    ##-----------------------------------------------------------------------------------
    function rstom(sx::Z, sy::Z) where Z<:Integer
        m = APL.rstov(sy)'

        @inbounds for i = 2:sx
			m = vcat(m, APL.rand_sto_vec(sy)')
		end

        return m
    end

    ##-----------------------------------------------------------------------------------
    function rstom(s::Z) where Z<:Integer
        return rstom(s, s)
    end

	##===================================================================================
    ## pascal matrix
    ##===================================================================================
	export pasc

	##-----------------------------------------------------------------------------------
	function pasc(s::Z) where Z<:Integer
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
	exm(s::Z) where Z<:Integer = @inbounds [((s+1)-x == y) ? 1 : 0 for x = 1:s, y = 1:s]


	##===================================================================================
    ## hilbert matrix
    ##===================================================================================
	export hbm

	##-----------------------------------------------------------------------------------
	function hbm(s::Z) where Z<:Integer
		m = zeros(s, s)
		m[1, :] = ones(s) ./ [x for x = 1:s]
		c = s

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
	function lehm(s::Z) where Z<:Integer
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
	function redh(s::Z) where Z<:Integer
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
	export shiftm

	##-----------------------------------------------------------------------------------
	function shiftm(s::Z, sup::Bool = true) where Z<:Integer
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
	function vandermonde(v::Array{T, 1}, d::Z) where T<:Number where Z<:Integer
		m = ones(v, T)

		@inbounds for i = 1:d
			m = hcat(m, v.^i)
		end

		return m
	end


	##===================================================================================
    ## rotation matrix
    ##===================================================================================
    export rotm3d, rotm2d

	##-----------------------------------------------------------------------------------
	rotm2d(angle::R = R(90)) where R<:AbstractFloat = [cos(angle) -sin(angle); sin(angle) cos(angle)]

    ##-----------------------------------------------------------------------------------
    function rotm3d(axis::Array{R, 1}, angle::R = R(90)) where R<:AbstractFloat
        axis = axis'
        m = [ 0 -axis[3] axis[2]; axis[3] 0 -axis[1]; -axis[2] axis[1] 0 ]
        return eye(T, 3) + m * sind(alpha) + (1 - cosd(alpha)) * m^2
    end


	##===================================================================================
	##  general evaluation matrix
	##      l = [(x)->1, (x)->x, (x)->x^2] for polynomial of degree two
	##===================================================================================
	export gevam

	##-----------------------------------------------------------------------------------
	function gevam(l::Array{Function, 1}, v::Array{R, 1}) where R<:AbstractFloat
		m = map(l[1], v)

		@inbounds for i = 2:size(l, 1)
			m = hcat(m, map(l[i], v))
		end

		return m
	end


	##===================================================================================
	##	random affine matrix
	##===================================================================================
	export raffine, rnaffine

	##-----------------------------------------------------------------------------------
	function raffine(s::Z) where Z<:Integer
		r = rand(s, s)
		d = det(r)
		
		@inbounds while d == 0
			r = rand(s, s)
			d = det(r)
		end 

		d = 1 / d
		i = 1

		@inbounds while i <= s
			r[i, 1] = r[i, 1] * d
			i = i + 1
		end

		return r
	end

	##-----------------------------------------------------------------------------------
	function rnaffine(s::Z) where Z<:Integer
		r = randn(s, s)
		d = det(r)

		@inbounds while d == 0
			r = randn(s, s)
			d = det(r)
		end

		d = 1 / d
		i = 1

		@inbounds while i <= s
			r[i, 1] = r[i, 1] * d
			i = i + 1
		end

		return r
	end


	##===================================================================================
	##	random hermitian matrix
	##		u: uniform distributed
	##		_: normal distributed
	##===================================================================================
	export uhermitian, hermitian

	##-----------------------------------------------------------------------------------
	function uhermitian(n::Z, d::Bool = true) where Z<:Integer
		t0 = d ? Complex128 : Complex64
		t1 = d ? Float64 : Float32
		r = zeros(t0, n, n)
		i = N(1)

		while i<=n
			r[i, i] = rand(t1)
			i += 1
		end

		j = N(0)
		i = 1

		while i<=n
			j = i+1
			while j<=n
				r[i, j] = rand(t0)
				r[j, i] = r[i, j]'
				j += 1
			end
			i += 1
		end

		return r
	end

	##-----------------------------------------------------------------------------------
	function hermitian(n::Z, d::Bool = true) where Z<:Integer
		r = zeros(d ? Complex128 : Complex64, n, n)
		t = d ? Float64 : Float32
		i = N(1)

		while i<=n
			r[i, i] = randn(t)
			i += 1
		end

		j = N(0)
		i = 1

		while i<=n
			j = i+1
			while j<=n
				r[i, j] = randn(t)+randn(t)im
				r[j, i] = r[i, j]'
				j += 1
			end
			i += 1
		end

		return r
	end 
end
