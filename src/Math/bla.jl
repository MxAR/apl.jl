@everywhere module bla    
	##===================================================================================
	## using directives
	##===================================================================================
	using LinearAlgebra.LAPACK
	using LinearAlgebra.BLAS
	using SparseArrays
	using SharedArrays
	using Distributed


	##===================================================================================
	## hypotenuse (numerically stable)
	##===================================================================================
	export hypot, hypot_sq

	##-----------------------------------------------------------------------------------
	function hypot(x::R, y::R) where R<:AbstractFloat
		a = undef
		b = undef
		
		if x == R(0)
			return x == y ? R(0) : abs(y)
		end

		if y == R(0)
			return abs(x)
		end

		if abs(x) > abs(y)
			a = y / x
			b = abs(x)
		else
			a = x / y
			b = abs(y)
		end

		return b * Base.Math.sqrt_llvm(R(1) + a^2)
	end

	##-----------------------------------------------------------------------------------
	function hypot(x::R, y::R) where R<:AbstractFloat
		a = undef
		b = undef
		
		if x == R(0)
			return x == y ? R(0) : y^2
		end

		if y == R(0)
			return x^2
		end

		if abs(x) > abs(y)
			a = y / x
			b = x^2
		else
			a = x / y
			b = y^2
		end

		return b * (R(1) + a^2)
	end	


	##===================================================================================
	## basis pursuit denosing via [singular] in-crowd algorithm (L = 1)
	##===================================================================================
	export bpdn_sic

	##-----------------------------------------------------------------------------------
	function bpdn_sic(A::Array{R, 2}, y::Array{R, 1}, lambda::R) where R<:AbstractFloat
		s = size(A)

		x = zeros(R, s[2])
		r = copy(y)
	
		a_suparg = Int(0)
		a_sup = R(0)
		a = R(0)

		while true
			a_suparg = Int(0)
			a_sup = R(-Inf)
			
			@inbounds for i = 1:s[2]
				if x[i] != 0
					continue
				end

				a = R(0)
				for j = 1:s[1]
					a = a + r[j] * A[j, i]
				end

				a = abs(a)	
				if a > lambda && a > a_sup
					a_suparg = i
					a_sup = a
				end
			end
	
			if a_suparg == 0
				return x
			end

			a = R(0)
			@inbounds for i = 1:s[1]
				a = a + A[i, a_suparg]^2
			end

			x[a_suparg] = a_sup / a
			a = x[a_suparg]

			@inbounds for i = 1:s[1]
				r[i] = r[i] - a * A[i, a_suparg] 
			end
		end
	end


	##===================================================================================
	## BLAS wrapper
	##	l = length of the vector
	##===================================================================================
	export bdot, bdotu, bdotc, bnrm

	##-----------------------------------------------------------------------------------
	function bdot(l::Z, v::Array{R, 1}, u::Array{R, 1}) where R<:AbstractFloat where Z<:Integer
		return BLAS.dot(l, v, 1, u, 1)
	end

	##-----------------------------------------------------------------------------------
	function bdot(v::Array{R, 1}, u::Array{R, 1}) where R<:AbstractFloat
		return BLAS.dot(size(v, 1), v, 1, u, 1)
	end

	##-----------------------------------------------------------------------------------
	function bdotu(l::Z, v::Array{C, 1}, u::Array{C, 1}) where C<:Complex where Z<:Integer
		return BLAS.dotu(l, v, 1, u, 1)
	end

	##-----------------------------------------------------------------------------------
	function bdotu(v::Array{C, 1}, u::Array{C, 1}) where C<:Complex 
		return BLAS.dotu(size(v, 1), v, 1, u, 1)
	end

	##-----------------------------------------------------------------------------------
	function bdotc(l::Z, v::Array{C, 1}, u::Array{C, 1}) where C<:Complex where Z<:Integer 
		return BLAS.dotc(l, v, 1, u, 1)
	end

	##-----------------------------------------------------------------------------------
	function bdotc(v::Array{C, 1}, u::Array{C, 1}) where C<:Complex 
		return BLAS.dotc(size(v, 1), v, 1, u, 1)
	end

	##-----------------------------------------------------------------------------------
	function bnrm(l::Z, v::Array{R, 1}) where R<:AbstractFloat where Z<:Integer 
		BLAS.nrm2(l, v, 1)
	end

	##-----------------------------------------------------------------------------------
	function bnrm(v::Array{R, 1}) where R<:AbstractFloat
		BLAS.nrm2(size(v, 1), v, 1)
	end


	##===================================================================================
	## compute gramian matrix (G = A'*A)
	##===================================================================================
	export gram

	##-----------------------------------------------------------------------------------
	function gram(m::Array{R, 2}) where R<:AbstractFloat
		return BLAS.gemm('T', 'N', m, m)
	end


	##===================================================================================
	##  soq (sum of squares)
	##===================================================================================
	export soq, soqu, soqc

	##-----------------------------------------------------------------------------------
	function soq(l::Z, v::Array{R, 1}, n::Z = 1) where R<:AbstractFloat where Z<:Integer 
		return BLAS.dot(l, v, n, v, n)
	end

	##-----------------------------------------------------------------------------------
	function soq(v::Array{R, 1}) where R<:AbstractFloat
		return bdot(v, v)
	end

	##-----------------------------------------------------------------------------------
	function soqu(l::Z, v::Array{C, 1}, n::Z = 1) where C<:Complex where Z<:Integer
		return BLAS.dot(l, v, n, v, n)
	end

	##-----------------------------------------------------------------------------------
	function soqu(v::Array{C, 1}) where C<:Complex
		return bdotu(v, v)
	end

	##-----------------------------------------------------------------------------------
	function soqc(l::Z, v::Array{C, 1}, n::Z = 1) where C<:Complex where Z<:Integer
		return bdotc(l, v, n, v, n)
	end

	##-----------------------------------------------------------------------------------
	function soc(v::Array{C, 1}) where C<:Complex 
		return bdotc(v, v)
	end

	
	##===================================================================================
	## kullback-leibler divergence
	##	gkld = generalized kld
	##	(all elements in m and n must be positive)
	##===================================================================================
	export kld, gkld

	##-----------------------------------------------------------------------------------
	function kld(m::Array{R}, n::Array{R}) where R<:AbstractFloat
		s = min(length(m), length(n))
		r = R(0)
		i = 1

		@inbounds while i <= s
			@fastmath r = r + (m[i] * log(m[i] / n[i]))
			i = i + 1
		end

		return r
	end

	##-----------------------------------------------------------------------------------
	function gkld(m::Array{R}, n::Array{R}) where R<:AbstractFloat
		s = min(length(m), length(n))
		r = R(0)
		i = 1

		@inbounds while i <= s
			@fastmath r = r + n[i] - m[i] + (m[i] * log(m[i] / n[i]))
			i = i + 1
		end

		return r
	end
	

	##===================================================================================
	## k statistics
	##===================================================================================
	export kstat, kstat_1, kstat_2, kstat_3, kstat_4
	import ..mean.gamean

	##-----------------------------------------------------------------------------------
	function kstat(v::Array{R, 1}, k::Z) where R<:Real where Z<:Integer
		if k == 1
			return kstat_1(v)
		elseif k == 2
			return kstat_2(v)	
		elseif k == 3
			return kstat_3(v)
		elseif k == 4
			return kstat_4(v)
		end
	end
	
	##-----------------------------------------------------------------------------------
	function kstat_1(v::Array{R, 1}) where R<:Real
		return gamean(v)
	end

	##-----------------------------------------------------------------------------------
	function kstat_2(v::Array{R, 1}) where R<:Real
		m = AbstractFloat(v[1])
		s = size(v, 1)

		i = 2
		@inbounds while i <= s
			m = m + v[i]
			i = i + 1
		end

		m = m / s
		@fastmath r = AbstractFloat((v[1] - m)^2)

		i = 2
		@inbounds while i <= s
			@fastmath r = r + (v[i] - m)^2 
			i = i + 1
		end

		r = r / (s - 1)
		return r
	end

	##-----------------------------------------------------------------------------------
	function kstat_3(v::Array{R, 1}) where R<:Real
		m = AbstractFloat(v[1])
		s = size(v, 1)

		i = 2
		@inbounds while i <= s
			m = m + v[i]
			i = i + 1
		end

		m = m / s
		@fastmath r = AbstractFloat((v[1] - m)^3)

		i = 2
		@inbounds while i <= s
			@fastmath r = r + (v[i] - m)^3
			i = i + 1
		end

		r = (r * s) / ((s - 1) * (s - 2))
		return r
	end

	##-----------------------------------------------------------------------------------
	function kstat_4(v::Array{R, 1}) where R<:Real
		m = AbstractFloat(v[1])
		s = size(v, 1)

		i = 2
		@inbounds while i <= s
			m = m + v[i]
			i = i + 1
		end

		m = m / s
		@fastmath r0 = AbstractFloat((v[1] - m)^2)
		@fastmath r1 = AbstractFloat(r0^2)
		x = AbstractFloat(0)

		i = 2
		@inbounds while i <= s
			@fastmath x = (v[i] - m)^2
			r0 = r0 + x
			@fastmath r1 = r1 + x^2
			i = i + 1
		end

		r = (s * (((s + 1) * r1) - ((s - 1) * (r0 / s)))) / ((s - 1) * (s - 2) * (s - 3))
		return r
	end


	##===================================================================================
	## fast johnson-lindenstrauss transfor n
	##===================================================================================
	export fjlt

	##-----------------------------------------------------------------------------------
	function fjlt(n::Z, d::Z, eps::R, p::Z) where Z<:Integer where R<:Real
		@fastmath begin
			q = min(1., ((eps^(p - 2)) * (log(n)^p)) / d)
			k = Z(ceil(log(n)*eps^-2)) 
			a = d^-.5
		end

		m_p = SharedArray{R, 2}(k, d)
		m_h = SharedArray{R, 2}(d, d)
		m_d = SharedArray{R, 2}(d, d)

		@sync @distributed for i = 1:(k * d)
			@fastmath m_p[i] = rand() >= q ? randn() / q : 0.
		end

		@sync @distributed for i = 1:d 
			for j = 1:d
				m_d[i, j] = i == j ? (rand(Bool) ? 1. : -1.) : 0.

				if j < i 
					continue
				end

				if Bool((((((i - 1) & (j - 1)) * 0x0101010101010101) & 0x8040201008040201) % 0x1FF) & 1)
					m_h[i, j] = a
					m_h[j, i] = a
				else
					m_h[i, j] = -a
					m_h[j, i] = -a
				end
			end
		end

		return ((dropzeros(sparse(m_p); trim=true) *
				 dropzeros(sparse(m_d); trim=true)) *
				sparse(m_h))
	end


	##===================================================================================
	## kth moment
	##===================================================================================
	export moment, central_moment

	##-----------------------------------------------------------------------------------
	function moment(v::Array{R, 1}, k::Z) where R<:Real where Z<:Integer
		@fastmath r = AbstractFloat(v[1]^k)
		s = size(v, 1)

		i = 2
		@inbounds while i <= s
			@fastmath r = r + v[i]^k
			i = i + 1
		end

		r = r / s
		return r
	end

	##-----------------------------------------------------------------------------------
	function moment(v::Array{C, 1}, k::Z) where C<:Complex where Z<:Integer
		s = size(v, 1)
		r = C(v[1]^k)

		i = 2
		@inbounds while i <= s
			@fastmath r = r + v[i]^k
			i = i + 1
		end

		return r / s
	end

	##-----------------------------------------------------------------------------------
	function central_moment(v::Array{R, 1}, c::R, k::Z) where R<:Real where Z<:Integer
		@fastmath r = AbstractFloat((v[1] - c)^k)
		s = size(v, 1)

		i = 2
		@inbounds while i <= s
			@fastmath r = r + (v[i] - c)^k
			i = i + 1
		end

		return r / s
	end

	##-----------------------------------------------------------------------------------
	function central_moment(v::Array{C, 1}, c::C, k::Z) where C<:Complex where Z<:Integer
		r = C((v[i] - c)^k)
		s = size(v, 1)

		i = 2
		@inbounds while i <= s
			@fastmath r = r + (v[i] - c)^k
			i = i + 1
		end

		return r / s
	end


	##===================================================================================
	## spectral norm
	##===================================================================================
	export snorm

	##-----------------------------------------------------------------------------------
	function snorm(m::Array{T, 2}) where T<:Number
		C = Complex{AbstractFloat}
		a = Base.LinAlg.Eigen{C,C,Array{C,2},Array{C,1}}(eigfact(m; scale=false, permute=false))
		return sqrt(a[:values][1])
	end

	##-----------------------------------------------------------------------------------
	function snorm_posdef(m::Array{T, 2}) where T<:Number
		R = AbstractFloat 
		a = Base.LinAlg.Eigen{R,R,Array{R,2},Array{R,1}}(eigfact(m; scale=false, permute=false))
		return sqrt(a[:values][1])
	end


	##===================================================================================
	##  QR (faster than vanilla)
	##===================================================================================
	export qrd_sq, qrd

	##-----------------------------------------------------------------------------------
	function qrd_sq(m::Array{R, 2}) where R<:AbstractFloat
		s = size(m, 1)
		t = Array{R, 2}(s, s)
		v = Array{R, 1}(s)
		r = copy(m)
		w = R(0)

		@inbounds for i = 1:(s-1)
			w = 0.
			for j = i:s
				v[j] = r[j, i]
				w = w + v[j]*v[j]
			end

			v[i] = v[i] + (r[i, i] >= 0 ? 1. : -1.)*sqrt(w)
			w = 0.

			for j = i:s 
				w = w + v[j]*v[j] 
			end
			
			w = 2.0 / w

			for j = 1:s
				for k = 1:s
					t[j, k] = k == j ? 1. : 0.
					if j>=i && k>=i
						t[j, k] = t[j, k] - w*v[j]*v[k]
					end
				end
			end

			for j = 1:s
				for k = 1:s
					v[k] = r[k, j]
				end

				for l = 1:s
					w = 0.
					for h = 1:s
						w = w + v[h]*t[l, h]
					end
					r[l, j] = w
				end
			end
		end

		for j = 1:(s-1)
			for k = (j+1):s
				r[k, j] = 0.
			end
		end

		return (m * inv(r), r)
	end

	##-----------------------------------------------------------------------------------
	function qrd(m::Array{R, 2}) where R<:AbstractFloat
		s = size(m, 1)
		t = Array{R, 2}(s, s)
		v = Array{R, 1}(s)
		r = copy(m)
		w = R(0)
		i = 1

		@inbounds while i <= (s-1)
			w = 0.
			j = i
			
			while j <= s
				v[j] = r[j, i]
				w = w + v[j]*v[j]
				j = j + 1
			end

			v[i] = v[i] + (r[i, i] >= 0 ? 1. : -1.) * sqrt(w)
			w = 0.
			j = i

			while j <= s 
				w = w + v[j]*v[j]
				j = j + 1
			end

			w = 2.0 / w
			j = 1

			while j <= s
				k = 1
				while k <= s
					t[j, k] = k == j ? 1. : 0.
					
					if j>=i && k>=i
						t[j, k] = t[j, k] - w * v[j] * v[k]
					end
					
					k = k + 1
				end
				j = j + 1
			end

			if i == 1
				r = t * r
			else
				j = 1
				while j <= s
					k = 1
					while k <= s
						v[k] = r[k, j]
						k = k + 1
					end

					k = 1
					while k <= s
						w = 0.
						l = 1

						while l <= s
							w = w + v[l] * t[k, l]
							l = l + 1
						end

						r[k, j] = w
						k = k + 1
					end

					j = j + 1
				end
			end

			i = i + 1
		end

		i = 1
		while i <= (s-1)
			k = i + 1
			while k <= s
				r[k, i] = 0.
				k = k + 1
			end
			i = i + 1
		end

		return (m * inv(r), r)
	end


	##===================================================================================
	## diagonal expansion of matrices
	##	- ul: upper left
	##===================================================================================
	export ul_x_expand

	##-----------------------------------------------------------------------------------
	function ul_x_expand(m::Array{T, 2}, s::Tuple{Z, Z}, x::T = 1.0) where T<:Number where Z<:Integer
		d = (s[1] - size(m, 1), s[2] - size(m,2))
		r = Array{T, 2}(s[1], s[2])
		i = 1

		while i <= s[1]
			j = 1
			while j <= s[2]
				if i > d[1] && j > d[2]
					@inbounds r[i, j] = m[i-d[1], j-d[2]]
				elseif i == j
					@inbounds r[i, j] = x
				else
					@inbounds r[i, j] = T(0)
				end 
				j = j + 1
			end
			i = i + 1 
		end

		return r
	end


	##===================================================================================
	##  minor of matrix (lower submatrix)
	##===================================================================================
	export minor

	##-----------------------------------------------------------------------------------
	function minor(m::Array{R, 2}, p::Tuple{Z, Z} = (1, 1)) where R<:AbstractFloat where Z<:Integer
		s = size(m)
		r = Array{R, 2}(s[1] - p[1], s[2] - p[1])
		i = 1 + p[1]
		a = 1 + p[2]
		k = 1

		while i <= s[1]
			j = a
			l = 1
			while j <= s[2]
				@inbounds r[k, l] = m[i, j]
				j = j + 1
				l = l + 1
			end
			i = i + 1
			k = k + 1
		end

		return r
	end


	##===================================================================================
	##  outer product implementation (faster than vanila)
	##===================================================================================
	export outer_product

	##-----------------------------------------------------------------------------------
	function outer_product(v::Array{R, 1}, w::Array{R, 1}) where R<:AbstractFloat
		s = (size(v, 1), size(w, 1))
		m = Array{T, 2}(s)
		i = 1

		while i <= s[1]
			j = 1
			while j <= s[2]
				@inbounds m[i, j] = v[i] * w[j]
				j = j + 1
			end
			i = i + 1
		end

		return m
	end


	##===================================================================================
	##  gram schmidt proces
	##===================================================================================
    export grsc, grscn

	##-----------------------------------------------------------------------------------
	function grsc(m::Array{R, 2}) where R<:AbstractFloat
    	s = size(m)
		r = Array{T, 2}(s[1], s[2])
		d = Array{T, 1}(s[2])

		i = 1
		@inbounds while i <= s[2]
			j = 1
			while j <= s[1]
				r[j, i] = m[j, i]
				j = j + 1
			end

			j = 1
			while j <= (i-1)
				n = R(0)
				k = 1

				while k <= s[1]
					n = n + r[k, j] * m[k, i]
					k = k + 1
				end

				n = n / d[j]
				k = 1

				while k <= s[1]
					r[k, i] = r[k, i] - n * r[k, j] 
					k = k + 1
				end

				j = j + 1
			end

			d[i] = 0
			j = 1

			while j <= s[1]
				d[i] = d[i] + r[j, i] * r[j, i]
				j = j + 1
			end

			i = i + 1
		end

    	return r
	end

    ##-----------------------------------------------------------------------------------
   	function grscn(m::Array{R, 2}) where R<:AbstractFloat
    	s = size(m)
		r = Array{T, 2}(s[1], s[2])
		d = R(0)

		i = 1
		@inbounds while i <= s[2]
			j = 1
			while j <= s[1]
				r[j, i] = m[j, i]
				j = j + 1
			end

			j = 1
			while j <= (i - 1)
				n = T(0)
				
				k = 1
				while k <= s[1]
					n = n + r[k, j] * m[k, i]
					k = k + 1
				end

				k = 1
				while k <= s[1]
					r[k, i] = r[k, i] - n * r[k, j]
					k = k + 1
				end

				j = j + 1
			end 

			d = T(0)
			
			j = 1
			while j <= s[1]
				d = d + r[j, i] * r[j, i]
				j = j + 1
			end
			
			d = sqrt(d)

			j = 1
			while j <= s[1]
				r[j, i] = r[j, i] / d[i]
				j = j + 1
			end

			i = i +1
		end

    	return r
    end


    ##===================================================================================
    ##  orthogonal projection
    ##===================================================================================
    export proj, projn

    ##-----------------------------------------------------------------------------------
    function proj(v::Array{T, 1}, m::Array{T, 2}) where T<:Number
		s = size(m)
		r = Array{T, 1}(s[1])
		n = T(0)
		d = T(0)

		i = 1
		@inbounds while i <= s[1]
			r[i] = T(0) 
			i = i + 1
		end

		i = 1
		@inbounds while i <= s[2]
			j = 1
			while j <= s[1]
				d = d + m[j, i] * m[j, i]
				n = n + v[j] * m[j, i]
				j = j + 1
			end

			n = n / d

			j = 1
			while j <= s[1]
				r[j] = r[j] + n * m[j, i]
				j = j + 1
			end 

    		i = i + 1
			n = T(0)
			d = T(0)
		end
    	
		return r
    end

    ##-----------------------------------------------------------------------------------
    function projn(v::Array{R, 1}, m::Array{R, 2}) where R<:AbstractFloat
		s = size(m)
		r = Array{R, 1}(s[1])
		n = R(0)

		i = 1
		@inbounds while i <= s[1]
			r[i] = R(0)
			i = i + 1
		end

		i = 1
		@inbounds while i <= s[2]
			j = 1
			while j <= s[2]
				n = n + v[j] * m[j, i]
				j = j + 1
			end

			j = 1
			while j <= s[1]
				r[j] = r[j] + n * m[j, i]
				j = j + 1
			end

			i = i + 1
			n = R(0)
		end

		return r
	end


	##===================================================================================
    ##	Cofactor Matrix of a Matrix
	##		TODO needs better time complexity
    ##===================================================================================
    export cof

    ##-----------------------------------------------------------------------------------
    function cof(m::Array{T, 2}) where T<:Number
		s = size(m)
		n = Array{T, 2}(s[1]-1, s[2]-1)
		r = Array{T, 2}(s[1], s[2])

		i = 1
		@inbounds while i <= s[1]-1
			j = i
			while j <= s[2]-1
				n[i, j] = T(0)
				n[j, i] = T(0)
				j = j + 1
			end
			i = i + 1
		end 

		i = 1
		@inbounds while i <= s[1]
			j = 1
			while j <= s[2]
				x = 1
				while x <= s[1]
					y = 1
					while y <= s[2]
						if x != i && y != j
							n[(x > i ? x-1 : x), (y > j ? y-1 : y)] = m[x, y]
						end
						y = y + 1
					end
					x = x + 1
				end
				r[i, j] = det(n) * (i + j % 2 == 0 ? T(1) : T(-1))
				j = j + 1
			end
			i = i + 1
		end

		return r
    end


	##===================================================================================
	##	Adjugate of a Matrix
	##===================================================================================
	export adj

	##-----------------------------------------------------------------------------------
	adj(m::Array{T, 2}) where T<:Number = return cof(m)'


	##===================================================================================
	##	Nullspace of a matrix i.e its kernel
	##===================================================================================
	export ker

	##-----------------------------------------------------------------------------------
	function ker(m::Array{R, 2}) where R<:AbstractFloat
		s = size(m)
		d = s[2] - s[1]

		if d <= 0
			r = Array{R, 2}(1, s[2])
			
			i = 1
			@inbounds while i <= s[2]
				r[i] = T(0)
				i = i + 1
			end
		
			return r
		else
			x = Array{R, 2}(lufact(m)[:U])
			r = Array{R, 2}(s[1] + d, d)
			xi = inv(x[1:s[1], 1:s[1]])

			i = s[1] + 1
			j = 1

			set_zero_subnormals(true)
			@inbounds while i <= s[2]
				k = 1
				while k <= s[1]
					r[k, j] = BLAS.dot(s[1], xi[k, :], 1, x[:, i], 1)
					k = k + 1
				end

				i = i + 1
				j = j + 1
			end
			set_zero_subnormals(false)

			i = s[1] + 1
			k = 1

			@inbounds while i <= s[2]
				j = 1
				while j <= (k - 1)
					r[i, j] = T(0)
					j = j + 1
				end

				j = k + 1
				while j <= d
					r[i, j] = T(0)
					j = j + 1
				end

				r[i, k] = T(-1)

				i = i + 1
				k = k + 1
			end

			return r 
		end
	end


	##===================================================================================
    ## split
    ##===================================================================================
    export split, splith

    ##-----------------------------------------------------------------------------------
    function split(m::Array{T, 2}, i::Z, by_rows::Bool = true) where T<:Number where Z<:Integer
		return by_rows ? (m[1:i, :], m[(i+1):end, :]) : (m[:, 1:i], m[:, (i+1):end]) 
    end

    ##-----------------------------------------------------------------------------------
    function splith(m::Array{T, 2}, by_rows::Bool = true) where T<:Number
		return msplit(m, convert(Int, round(size(m, by_rows ? 1 : 2)/2)), by_rows)
    end

	
	##===================================================================================
	## projection matrix X*inv(X'*X)*X'
	##===================================================================================
	export projm

	##-----------------------------------------------------------------------------------
	function projm(m::Array{R, 2}) where R<:AbstractFloat
		A = gram(m)

		(A, ipiv, _) = LAPACK.getrf!(A)
		A = LAPACK.getri!(A, ipiv)
		
		A = BLAS.gemm('N', 'T', A, m)
		A = BLAS.gemm('N', 'N', m, A)

		return A
	end

	##===================================================================================
	## ordinary least squares (qr)
	##===================================================================================
	export ols

	##-----------------------------------------------------------------------------------
	function ols(m::Array{R, 2}, v::Array{R, 1}) where R<:AbstractFloat
		y = BLAS.gemv('T', m, v)
		A = gram(m)

		return LAPACK.gelsy!(A, y)[1]
	end


	##===================================================================================
	##  householder reflection
	##      reflects v about a hyperplane given by u
	##		hhr: u = normal vector
	##===================================================================================
	export hhr, hhm

	##-----------------------------------------------------------------------------------
	function hhr(v::Array{R, 1}, u::Array{R, 1}) where R<:AbstractFloat
		s = size(u, 1)
		r = Array{R, 1}(s)
		a = R(2) * BLAS.dot(s, v, 1, u, 1)
		
		i = 1
		@inbounds while i <= s
			r[i] = u[i] - a * v[i]
			i = i + 1
		end

		return r
	end

	##-----------------------------------------------------------------------------------
	function hhm(v::Array{R, 1}) where R<:AbstractFloat
		s = size(v, 1)
		m = Array{R, 2}(s, s)

		i = 1
		@inbounds while i <= s
			j = 1
			while j <= s
				m[i, j] = R(-2) * v[i] * v[j]
				j = j + 1
			end
			i = i + 1
		end

		i = 1
		@inbounds while i <= s
			m[i, i] = m[i, i] + R(1)
			i = i + 1
		end

		return m
	end


	##===================================================================================
    ##	kronecker product
    ##===================================================================================
    export kep

    ##-----------------------------------------------------------------------------------
    function kep(m::Array{R, 2}, n::Array{R, 2}) where R<:AbstractFloat
		sm = size(m)
		sn = size(n)
		r = Array{R, 2}(sm[1]^2, sn[1]^2)
		
		i = 1
		x = 0
		while i <= sm[1]
			j = 1
			y = 0
			while j <= sm[2]
				k = 1
				while k <= sn[1]
					l = 1
					while l <= sn[2]
						r[(x + k), (y + l)] = m[i, j] * n[k, l]
						l = l + 1
					end
					k = k + 1
				end
				y = y + sm[2]
				j = j + 1
			end
			x = x + sm[1]
			i = i + 1
		end

		return r
    end


	##===================================================================================
	##	normalize matrix columns
	##===================================================================================
	export normalize

	##-----------------------------------------------------------------------------------
	function normalize(m::Array{R, 2}) where R<:Real
		T = AbstractFloat
		s = size(m)
		r = Array{T, 2}(s[1], s[2])

		i = 1
		@inbounds while i <= s[2]
			a = T(m[1, i])
			j = s[1]
			while j >= 2
				a = a + m[j, i]^2
				j = j - 1
			end 

			@fastmath a = a^-.5
			while j <= s[1]
				r[j, i] = m[j, i] * a
				j = j + 1 
			end

			i = i + 1
		end

		return r
	end


	##===================================================================================
	##	phi (transform n dimensional points into their polar form)
	##		- first column represents the radius
	##		- iphi is the inverse of phi
	##===================================================================================
	export phi, iphi

	##-----------------------------------------------------------------------------------
	function phi(m::Array{R, 2}) where R<:AbstractFloat
		s = size(m)
		r = Array{R, 2}(s[1], s[2])

		i = 1
		@inbounds while i <= s[1]
			r[i, 1] = BLAS.nrm2(s[2], m[i, :], 1)
			i = i + 1
		end

		i = 1
		@inbounds while i <= s[1]
			a = R(1)
			j = 2
			while j <= s[2]
				r[i, j] = acosd(m[i, j - 1] / (a * r[i, 1]))
				r[i, j] = isnan(r[i, j]) ? 0 : r[i, j]
				a = a * sind(r[i, j])
				j = j + 1
			end
			i = i + 1
		end

		return r
	end

	##-----------------------------------------------------------------------------------
	function iphi(m::Array{R, 2}) where R<:Real
		s = size(m)
		r = Array{R, 2}(s[1], s[2])

		i = 1
		@inbounds while i <= s[1]
			a = R(1)
			j = 1
			@inbounds while j <= (s[2] - 1)
				r[i, j] = m[i, 1] * a * cosd(m[i, j + 1])
				a = a * sind(m[i, j + 1])
				j = j +1
			end
			r[i, s[2]] = m[i, 1] * a
			i = i + 1
		end

		return r
	end 


	##===================================================================================	
	##	scalar projection 
	##		get the length of the projection of a onto b
	##===================================================================================
	export scal

	##-----------------------------------------------------------------------------------
	function scal(v::Array{R, 1}, u::Array{R, 1}) where R<:AbstractFloat
		s = size(v, 1)
		c = Array{R, 1}(s)
		a = BLAS.nrm2(s, v, 1)

		i = 1
		@inbounds while i <= s
			c[i] = v[i] / a
			i = i + 1
		end
 
		return BLAS.dot(s, c, 1, u, 1)
	end


	##===================================================================================
	##	vector projection (vector -> vector)
	##===================================================================================
	export proj

	##-----------------------------------------------------------------------------------
	function proj(v::Array{R, 1}, u::Array{R, 1}) where R<:AbstractFloat
		s = size(v, 1)
		r = Array{R, 1}(s)

		a = BLAS.nrm2(s, v, 1)
		i = 1
		
		@inbounds while i <= s 
			r[i] = v[i] / a
			i = i + 1
		end

		a = BLAS.dot(s, r, 1, u, 1)
		i = 1

		@inbounds while i <= s
			r[i] = r[i] * a
			i = i + 1
		end

		return r
	end


    ##===================================================================================
    ## multi linear qr decomposition regression
    ##===================================================================================
    export rg_qr

    ##-----------------------------------------------------------------------------------
    function rg_qr(X::Array{R, 2}) where R<:AbstractFloat
        QR = qr(hcat(ones(size(X, 1)), X[:, 1:end-1]))
        return QR[2] \ QR[1]' * X[:, end]
    end


	##===================================================================================	
	## pca (principal component analysis)												
	##		mat: matrix of data points where each row presents a point
	## 		t: wether or not the matrix is transposed
	##===================================================================================
	export pca

	##-----------------------------------------------------------------------------------
	function pca(m::Array{R, 2}) where R<:AbstractFloat
		s = size(m, 1)
		d = svd(m)
		
		i = 1
		@inbounds while i <= s
			j = 1
			while j <= s
				d[1][j, i] = d[1][j, i] * d[2][i]
				j = j + 1
			end
			i = i + 1
		end
		
		return d[1]
	end


	##===================================================================================
	##	normalize matrix columns
	##===================================================================================
	export normalize

	##-----------------------------------------------------------------------------------
	function normalize(m::Array{T, 2}) where T<:Number
		s = size(m)
		r = zeros(s[1], s[2])

		@inbounds for i = 1:s[2]
			a = 0.
			@inbounds for j = 1:s[1]
				a = a + m[j, i]^2
			end 

			a = a^.5
			@inbounds for j = 1:s[1]
				r[j, i] = m[j, i] / a
			end
		end

		return r
	end 


	##===================================================================================	
	##	scalar projection 
	##		get the length of the projection of a onto b
	##===================================================================================
	export scal

	##-----------------------------------------------------------------------------------
	function scal(a::Array{T, 1}, b::Array{T, 1}) where T<:Number
		l = size(a, 1)
		c = b ./ BLAS.nrm2(l, b, 1) 
		return BLAS.dot(l, a, 1, c, 1)
	end


	##===================================================================================
	##	vector projection (vector -> vector)
	##===================================================================================
	export proj

	##-----------------------------------------------------------------------------------
	function proj(a::Array{T, 1}, b::Array{T, 1}) where T<:Number
		l = size(a, 1)
		c = b ./ BLAS.nrm2(l, b, 1)
		return c.*BLAS.dot(l, a, 1, c, 1)
	end
end
