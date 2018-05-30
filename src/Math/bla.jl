@everywhere module bla    
	##===================================================================================
	##	BLAS wrapper
	##		l = length of the vector
	##===================================================================================
	export bdot, bdotu, bdotc, bnrm

	##-----------------------------------------------------------------------------------
	function bdot{T<:AbstractFloat, N<:Integer}(l::N, v::Array{T, 1}, u::Array{T, 1}) 
		return BLAS.dot(l, v, 1, u, 1)
	end

	##-----------------------------------------------------------------------------------
	function bdot{T<:AbstractFloat}(v::Array{T, 1}, u::Array{T, 1}) 
		return BLAS.dot(size(v, 1), v, 1, u, 1)
	end

	##-----------------------------------------------------------------------------------
	function bdotu{C<:Complex, N<:Integer}(l::N, v::Array{C, 1}, u::Array{C, 1}) 
		return BLAS.dotu(l, v, 1, u, 1)
	end

	##-----------------------------------------------------------------------------------
	function bdotu{C<:Complex}(v::Array{C, 1}, u::Array{C, 1}) 
		return BLAS.dotu(size(v, 1), v, 1, u, 1)
	end

	##-----------------------------------------------------------------------------------
	function bdotc{C<:Complex, N<:Integer}(l::N, v::Array{C, 1}, u::Array{C, 1}) 
		return BLAS.dotc(l, v, 1, u, 1)
	end

	##-----------------------------------------------------------------------------------
	function bdotc{C<:Complex}(v::Array{C, 1}, u::Array{C, 1}) 
		return BLAS.dotc(size(v, 1), v, 1, u, 1)
	end

	##-----------------------------------------------------------------------------------
	function bnrm{T<:AbstractFloat, N<:Integer}(l::N, v::Array{T, 1}) 
		BLAS.nrm2(l, v, 1)
	end

	##-----------------------------------------------------------------------------------
	function bnrm{T<:AbstractFloat}(v::Array{T, 1})
		BLAS.nrm2(size(v, 1), v, 1)
	end


	##===================================================================================
	##  soq (sum of squares)
	##===================================================================================
	export soq, soqu, soqc

	##-----------------------------------------------------------------------------------
	function soq{T<:AbstractFloat, N<:Integer}(l::N, v::Array{T, 1}, n::N = 1) 
		return BLAS.dot(l, v, n, v, n)
	end

	##-----------------------------------------------------------------------------------
	function soq{T<:AbstractFloat}(v::Array{T, 1}) 
		return bdot(v, v)
	end

	##-----------------------------------------------------------------------------------
	function soqu{C<:Complex, N<:Integer}(l::N, v::Array{C, 1}, n::N = N(1)) 
		return BLAS.dot(l, v, n, v, n)
	end

	##-----------------------------------------------------------------------------------
	function soqu{C<:Complex}(v::Array{C, 1})
		return bdotu(v, v)
	end

	##-----------------------------------------------------------------------------------
	function soqc{C<:Complex, N<:Integer}(l::N, v::Array{C, 1}, n::N = 1) 
		return bdotc(l, v, n, v, n)
	end

	##-----------------------------------------------------------------------------------
	function soqc{C<:Complex}(v::Array{C, 1}) 
		return bdotc(v, v)
	end

	##===================================================================================
	## k statistics
	##===================================================================================
	export kstat, kstat_1, kstat_2, kstat_3, kstat_4
	import mean.gamean

	##-----------------------------------------------------------------------------------
	function kstat{R<:Real, N<:Integer}(v::Array{R, 1}, k::N)
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
	function kstat_1{R<:Real}(v::Array{R, 1})
		return gamean(v)
	end

	##-----------------------------------------------------------------------------------
	function kstat_2{R<:Real}(v::Array{R, 1})
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
	function kstat_3{R<:Real}(v::Array{R, 1})
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
	function kstat_4{R<:Real}(v::Array{R, 1})
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
	## kth moment
	##===================================================================================
	export moment, central_moment

	##-----------------------------------------------------------------------------------
	function moment{R<:Real, N<:Integer}(v::Array{R, 1}, k::N)
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
	function moment{C<:Complex, N<:Integer}(v::Array{C, 1}, k::N)
		s = size(v, 1)
		r = C(v[1]^k)

		i = 2
		@inbounds while i <= s
			@fastmath r = r + v[i]^k
			i = i + 1
		end

		r = r / s
		return r
	end

	##-----------------------------------------------------------------------------------
	function central_moment{R<:Real, N<:Integer}(v::Array{R, 1}, c::R, k::N)
		@fastmath r = AbstractFloat((v[1] - c)^k)
		s = size(v, 1)

		i = 2
		@inbounds while i <= s
			@fastmath r = r + (v[i] - c)^k
			i = i + 1
		end

		r = r / s
		return r
	end

	##-----------------------------------------------------------------------------------
	function central_moment{C<:Complex, N<:Integer}(v::Array{C, 1}, c::C, k::N)
		r = C((v[i] - c)^k)
		s = size(v, 1)

		i = 2
		@inbounds while i <= s
			@fastmath r = r + (v[i] - c)^k
			i = i + 1
		end

		r = r / s
		return r
	end


	##===================================================================================
	## spectral norm
	##===================================================================================
	export snorm

	##-----------------------------------------------------------------------------------
	function snorm{N<:Number}(m::Array{N, 2})
		T = Complex{Float64}
		a = Base.LinAlg.Eigen{T,T,Array{T,2},Array{T,1}}(eigfact(m; scale=false, permute=false))
		return sqrt(a[:values][1])
	end

	##-----------------------------------------------------------------------------------
	function snorm_posdef{N<:Number}(m::Array{N, 2})
		T = Float64
		a = Base.LinAlg.Eigen{T,T,Array{T,2},Array{T,1}}(eigfact(m; scale=false, permute=false))
		return sqrt(a[:values][1])
	end


	##===================================================================================
	##  QR (faster than vanilla)
	##===================================================================================
	export qrd_sq, qrd

	##-----------------------------------------------------------------------------------
	function qrd_sq{T<:AbstractFloat}(m::Array{T, 2})
		s = size(m, 1)
		t = Array{T, 2}(s, s)
		v = Array{T, 1}(s)
		r = copy(m)
		w = T(0)

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

		return (m*inv(r), r)
	end

	##-----------------------------------------------------------------------------------
	function qrd{T<:AbstractFloat}(m::Array{T, 2})
		s = size(m, 1)
		t = Array{T, 2}(s, s)
		v = Array{T, 1}(s)
		r = copy(m)
		w = T(0)
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

		return (m*inv(r), r)
	end


	##===================================================================================
	## diagonal expansion of matrices
	##	- ul: upper left
	##===================================================================================
	export ul_x_expand

	##-----------------------------------------------------------------------------------
	function ul_x_expand{T<:Number, N<:Integer}(m::Array{T, 2}, s::Tuple{N, N}, x::T = 1.0)
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
	function minor{T<:AbstractFloat, N<:Integer}(m::Array{T, 2}, p::Tuple{N, N} = (1, 1))
		s = size(m)
		r = Array{T, 2}(s[1]-p[1], s[2]-p[1])
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
	export otr

	##-----------------------------------------------------------------------------------
	function otr{T<:AbstractFloat}(v::Array{T, 1}, w::Array{T, 1})
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
	function grsc{T<:AbstractFloat}(m::Array{T, 2})
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
				n = T(0)
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
   	function grscn{T<:AbstractFloat}(m::Array{T, 2})
    	s = size(m)
		r = Array{T, 2}(s[1], s[2])
		d = T(0)

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
    function proj{T<:Number}(v::Array{T, 1}, m::Array{T, 2})
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
    function projn{T<:AbstractFloat}(v::Array{T, 1}, m::Array{T, 2})
		s = size(m)
		r = Array{T, 1}(s[1])
		n = T(0)

		i = 1
		@inbounds while i <= s[1]
			r[i] = T(0)
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
			n = T(0)
		end

		return r
	end


	##===================================================================================
    ##	Cofactor Matrix of a Matrix
	##		TODO needs better time complexity
    ##===================================================================================
    export cof

    ##-----------------------------------------------------------------------------------
    function cof{T<:Number}(m::Array{T, 2})
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
	adj{T<:Number}(m::Array{T, 2}) = return cof(m)'


	##===================================================================================
	##	Nullspace of a matrix
	##===================================================================================
	export ker

	##-----------------------------------------------------------------------------------
	function ker{T<:AbstractFloat}(m::Array{T, 2})
		s = size(m)
		d = s[2] - s[1]

		if d <= 0
			r = Array{T, 2}(1, s[2])
			
			i = 1
			@inbounds while i <= s[2]
				r[i] = T(0)
				i = i + 1
			end
		
			return r
		else
			x = Array{T, 2}(lufact(m)[:U])
			r = Array{T, 2}(s[1] + d, d)
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
    export msplit, msplit_half

    ##-----------------------------------------------------------------------------------
    function msplit{T<:Number, N<:Integer}(m::Array{T, 2}, i::N, lrows::Bool = true)
        r = lrows ? m : m'
        @assert(i < 0 || i >= size(r, 1))
        return (r[1:i, :], r[(i+1):end, :])
    end

    ##-----------------------------------------------------------------------------------
    function msplit_half{T<:Number}(m::Array{T, 2}, lrows::Bool = true)
        r = lrows ? m : m'
        l = convert(Int, round(size(r, 1)/2))
        return (r[1:l, :], r[(l+1):end, :])
    end


	##===================================================================================
	##  ordinary least squares
	##===================================================================================
	export ols

	##-----------------------------------------------------------------------------------
	ols{T<:AbstractFloat}(m::Array{T, 2}, y::Array{T, 1}) = (m'*m)\(m'*y)


	##===================================================================================
	##  householder reflection
	##      reflects v about a hyperplane given by u
	##		hh_rfl: u = normal vector
	##===================================================================================
	export hh_rfl, hh_mat

	##-----------------------------------------------------------------------------------
	hh_rfl{T<:AbstractFloat}(v::Array{T, 1}, u::Array{T, 1}) = u-(v*(2.0*bdot(u, v)))

	##-----------------------------------------------------------------------------------
	function hh_mat{T<:AbstractFloat}(v::Array{T, 1})
		s = size(v, 1)
		m = Array{T, 2}(s, s)

		@inbounds for i=1:s, j=1:s
			if i == j
				m[i, j]= 1-(2*v[i]*v[j])
			else
				m[i, j]= -2*v[i]*v[j]
			end
		end

		return m
	end


	##===================================================================================
    ##	kronecker product
    ##===================================================================================
    export kep

    ##-----------------------------------------------------------------------------------
    function kep{T<:AbstractFloat}(m::Array{T, 2}, n::Array{T, 2})
		sm = size(m); sn = size(n)
		r = zeros(T, sm[1]^2, sn[1]^2)
		px = py = T(0)

		@inbounds for i = 1:sm[1], j = 1:sm[2]
			px = (i-1)*sm[1]
			py = (j-1)*sm[2]

			for k = 1:sn[1], l = 1:sn[2]
				r[(px+k), (py+l)] = m[i,j]*n[k,l]
			end
		end

		return r
    end


	##===================================================================================
	##	normalize matrix columns
	##===================================================================================
	export normalize

	##-----------------------------------------------------------------------------------
	function normalize{T<:Number}(m::Array{T, 2})
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
	##	phi (transform n dimensional points into their polar form)
	##		- first column represents the radius
	##		- iphi is the inverse of phi
	##===================================================================================
	export phi, iphi

	##-----------------------------------------------------------------------------------
	function phi{T<:Real}(m::Array{T, 2})
		s = size(m)
		r = zeros(s[1], s[2])

		@inbounds for i = 1:s[1]
			r[i, 1] = BLAS.nrm2(s[2], m[i, :], 1)
		end

		@inbounds for i = 1:s[1]
			a = 1.
			@inbounds for j = 2:s[2]
				r[i, j] = acosd(m[i, j-1]/(a*r[i, 1]))
				a = a * sind(r[i, j])
			end
		end

		return r
	end

	##-----------------------------------------------------------------------------------
	function iphi{T<:Real}(m::Array{T, 2})
		s = size(m)
		r = zeros(s[1], s[2])

		@inbounds for i = 1:s[1]
			a = 1.
			@inbounds for j = 1:(s[2]-1)
				r[i, j] = m[i, 1]*a*cosd(m[i, j+1])
				a  = a * sind(m[i, j+1])
			end
			r[i, s[2]] = m[i, 1]*a
		end

		return r
	end 


	##===================================================================================	
	##	scalar projection 
	##		get the length of the projection of a onto b
	##===================================================================================
	export scal

	##-----------------------------------------------------------------------------------
	function scal{T<:Number}(a::Array{T, 1}, b::Array{T, 1})
		l = size(a, 1)
		c = b ./ BLAS.nrm2(l, b, 1) 
		return BLAS.dot(l, a, 1, c, 1)
	end


	##===================================================================================
	##	vector projection (vector -> vector)
	##===================================================================================
	export proj

	##-----------------------------------------------------------------------------------
	function proj{T<:Number}(a::Array{T, 1}, b::Array{T, 1})
		l = size(a, 1)
		c = b ./ BLAS.nrm2(l, b, 1)
		return c.*BLAS.dot(l, a, 1, c, 1)
	end


	##===================================================================================
	##	get the diagonal of matrix
	##===================================================================================
	export diagonal

	##-----------------------------------------------------------------------------------
	function diagonal{T<:Number}(m::Array{T, 2})
		s = op.min(size(m))
		r = zeros(s)

		@inbounds for i = 1:s
			r[i] = m[i, i]
		end

		return r
	end 


    ##===================================================================================   	
    ##	logistic regression
    ##		last column is y the rest are x's
	##===================================================================================
    export rg_log

    ##-----------------------------------------------------------------------------------
    function rg_log{T<:AbstractFloat}(X::Array{T, 2})
        reve = zeros(T, 2, 1)
        coeff = zeros(T, 2, 2)
        rows = size(X, 1)
        L = ceil(maximum(X[1:rows,2]))

        coeff[1, 1] = size(X, 1)
        coeff[1, 2] = sum(X[1:rows,1])
        coeff[2, 2] = sumabs2(X[1:rows,1])
        coeff[2, 1] = coeff[1, 2]

        X[1:rows,2] = map((x) -> log((L - x) / x), X[1:rows,2])

        reve[2, 1] = bdot(X[1:rows,1], X[1:rows,2])
        reve[1, 1] = sum(X[1:rows,2])

        S = coeff \ reve
        return (x) -> L / (1 + exp(S[1] + (x*S[2])))
    end


    ##===================================================================================
    ## linear statistic regression
    ##===================================================================================
    export rg_sta

    ##-----------------------------------------------------------------------------------
    function rg_sta{T<:AbstractFloat}(X::Array{T, 2})
        m = mamean(X)
        a = cov(X[:, 1], X[:, 2])/var(X[:, 1])
        return [(m[2] - (a * m[1])), a]
    end
    

    ##===================================================================================
    ## multi linear qr decomposition regression
    ##===================================================================================
    export rg_qr

    ##-----------------------------------------------------------------------------------
    function rg_qr{T<:AbstractFloat}(X::Array{T, 2})
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
	function pca{T<:AbstractFloat}(m::Array{T, 2}, t::Bool = false)
		if t; m = m' end 
		s = size(m, 1)
		d = svd(m)

		@inbounds for i = 1:s, j = 1:s
			d[1][j, i] *= d[2][i] 	
		end
		
		return d[1]
	end
end
