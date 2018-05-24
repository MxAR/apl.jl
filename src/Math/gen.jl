@everywhere module gen
	##===================================================================================
	##  types
	##===================================================================================
	type tncbd{T<:AbstractFloat, N<:Integer}	# n dimensional cuboid
		alpha::Array{T, 1}						# infimum (point)
		delta::Array{T, 1}						# diference between supremum and infimum	(s-i)
		n::N									# n
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
	zsqm{N<:Integer}(s::N) = fill(0., s, s)

	##-----------------------------------------------------------------------------------
	osqm{N<:Integer}(s::N) = fill(1., s, s)

	##-----------------------------------------------------------------------------------
	fsqm{T<:Any, N<:Integer}(s::N, x::T) = fill(x, s, s)

	##-----------------------------------------------------------------------------------
	rsqm{N<:Integer}(s::N) = rand(s, s)

	##-----------------------------------------------------------------------------------
	rnsqm{N<:Integer}(s::N) = randn(s, s)

	##-----------------------------------------------------------------------------------
	fdiam{R<:Number, N<:Integer}(x::R, s::N) = [i == j ? x : T(0) for i = 1:s, j = 1:s]

	##-----------------------------------------------------------------------------------
	rdiam{N<:Integer}(s::N) = [i == j ? rand() : 0 for i = 1:s, j = 1:s]

	##-----------------------------------------------------------------------------------
	rndiam{N<:Integer}(s::N) = [i == j ? randn() : 0 for i = 1:s, j = 1:s]

	##-----------------------------------------------------------------------------------
	function ftrim{T<:Any, N<:Integer}(v::T, s::N, upper::Bool = true)
		m = apply_tri_upper((x) -> v, fill(0., s, s))
		return upper ? m : m'
	end

	##-----------------------------------------------------------------------------------
	function otrim{N<:Integer}(s::N, upper::Bool = true)
		return tri_fill(1., s, upper)
	end

	##-----------------------------------------------------------------------------------
	function rtrim{N<:Integer}(s::N, upper::Bool = true)
		m = apply_tri_upper((x) -> rand(), fill(0., s, s))
		return upper ? m : m'
	end

	##-----------------------------------------------------------------------------------
	function rntrim{N<:Integer}(s::N, upper::Bool = true)
		m = apply_tri_upper((x) -> randn(), fill(0., s, s))
		return upper ? m : m'
	end

	##-----------------------------------------------------------------------------------
	function zvl{N<:Integer}(l::N, d::N)
		v = zeros(d)
		vl = []

		@inbounds for i = 1:l
			push!(vl, v)
		end

		return vl
	end

	##-----------------------------------------------------------------------------------
	function ovl{N<:Integer}(l::N, d::N)
		v = ones(d)
		vl = []

		@inbounds for i = 1:l
			push!(vl, v)
		end

		return vl
	end

	##-----------------------------------------------------------------------------------
	function rvl{N<:Integer}(l::N, d::N)
		vl = []

		@inbounds for i = 1:l
			push!(vl, rand(d))
		end

		return vl
	end

	##-----------------------------------------------------------------------------------
	function rnvl{N<:Integer}(l::N, d::N)
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
	function zhs{N<:Integer}(lg::N, dim::N)
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
	function rstov{N<:Integer}(s::N)
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
    function rorthv{R<:Number}(v::Array{R, 1})
        u = [rand(), rand(), 0]
        u[3] = (v[1] * u[1] + v[2] * u[2]) / (-1 * (v[3] == 0 ? 1 : v[3]))
        return normalize(u)
    end


	##===================================================================================
	## random hadamard matrix
	##	- remember s has to be a multiple of two bigger than zero
	##===================================================================================
	import bin.bit_dot
	export hadamard

	##-----------------------------------------------------------------------------------
	function hadamard{Z<:Integer}(s::Z)
		r = Array{Z, 2}(s, s)
		i = UInt(1)
		k = UInt(0)

		while i <= s
			j = UInt(1)
			while j <= s
				@inbounds r[i, j] = bit_dot(k, j - 1) % 2 == 0 ? 1 : -1
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
	function rcircm{N<:Integer}(s::N)
		m = zeros(s, s)
		m[1, :] = rand(s)

		@inbounds for i = 2:s
			m[i, :] = circshift(m[i-1, :], 1)
		end

		return m
	end

	##-----------------------------------------------------------------------------------
	function rncircm{N<:Integer}(s::N)
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
    function rstom{N<:Integer}(sx::N, sy::N)
        m = APL.rstov(sy)'

        @inbounds for i = 2:sx
			m = vcat(m, APL.rand_sto_vec(sy)')
		end

        return m
    end

    ##-----------------------------------------------------------------------------------
    function rstom(s::Integer)
        return rstom(s, s)
    end

	##===================================================================================
    ## pascal matrix
    ##===================================================================================
	export pasc

	##-----------------------------------------------------------------------------------
	function pasc{N<:Integer}(s::N)
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
	exm{N<:Integer}(s::N) = @inbounds [((s+1)-x == y) ? 1 : 0 for x = 1:s, y = 1:s]


	##===================================================================================
    ## hilbert matrix
    ##===================================================================================
	export hbm

	##-----------------------------------------------------------------------------------
	function hbm{N<:Integer}(s::N)
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
	function lehm{N<:Integer}(s::N)
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
	function redh{N<:Integer}(s::N)
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
	function shiftm{N<:Integer}(s::N, sup::Bool = true)
		b = sup?(s+1):2
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
	function vandermonde{R<:Number, N<:Integer}(v::Array{R, 1}, d::N)
		m = ones(v)

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
	rotm2d{R<:AbstractFloat}(angle::R = R(90)) = [cos(angle) -sin(angle); sin(angle) cos(angle)]

    ##-----------------------------------------------------------------------------------
    function rotm3d{R<:AbstractFloat}(axis::Array{R, 1}, angle::R = R(90))
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
	function gevama{R<:AbstractFloat}(l::Array{Function, 1}, v::Array{R, 1})
		m = map(l[1], v)

		@inbounds for i = 2:size(l, 1)
			m = hcat(m, map(l[i], v))
		end

		return m
	end


	##===================================================================================
	##	random affine matrix
	##		u: uniform distribution
	##		n: normal distribution
	##===================================================================================
	export uaffine, naffine

	##-----------------------------------------------------------------------------------
	function uaffine{N<:Integer}(s::N)
		@assert(s > 0, "out of bounds error")
		r = rand(s, s)
		d = det(r)

		while d == 0
			r = rand(s, s)
			d = det(r)
		end 

		return r./(abs(d)^(1/s))
	end

	##-----------------------------------------------------------------------------------
	function naffine{N<:Integer}(s::N)
		@assert(s > 0, "out of bounds error")
		r = randn(s, s)
		d = det(r)

		while d == 0
			r = randn(s, s)
			d = det(r)
		end

		if d < 0 
			r[1, :] .*= -1
		end 

		return r./(abs(d)^(1/s))
	end


	##===================================================================================
	##	random hermitian matrix
	##		u: uniform distributed
	##		_: normal distributed
	##===================================================================================
	export uhermitian, hermitian

	##-----------------------------------------------------------------------------------
	function uhermitian{N<:Integer}(n::N, d::Bool = true)
		t0 = d?Complex128:Complex64
		t1 = d?Float64:Float32
		r = zeros(t0,n,n)
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
	function hermitian{N<:Integer}(n::N, d::Bool = true)
		r = zeros(d?Complex128:Complex64,n,n)
		t = d?Float64:Float32
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
