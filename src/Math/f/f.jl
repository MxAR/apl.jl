@everywhere module f
	##===================================================================================
	##  using directives
	##===================================================================================
	using StatsBase
	using cnv
	using op


	##===================================================================================
	##  import directives
	##===================================================================================
	import Base.map, Base.std


	##===================================================================================
	##  types
	##===================================================================================
	type tncbd{T<:AbstractFloat, N<:Integer}	# n dimensional cuboid
		alpha::Array{T, 1}						# infimum (point)
		delta::Array{T, 1}						# diference between supremum and infimum	(s-i)
		n::N									# n
	end


	##===================================================================================
	## hyperbolic tangent
	##===================================================================================
	export tanh, tanhd

	##-----------------------------------------------------------------------------------
	tanh(x, eta) = @. Base.tanh(x-eta)

	##-----------------------------------------------------------------------------------
	tanhd(x, eta = 0) = @. 1-(tanh(x, eta)^2)


	##===================================================================================
	## std (standart deviation (overload))
	##		m = median
	##===================================================================================
	export std

	##-----------------------------------------------------------------------------------
	function std{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, l::N, n::N = 1)
		s = n * l
		m = T(0)

		@assert(s <= size(v, 1), "out of bounds error")
		@inbounds for i = 1:n:(n*l)
			m = m + v[i]
		end

		return sqrt(BLAS.dot(l, v, n, v, n)-(m/l)^2)
	end

	##-----------------------------------------------------------------------------------
	std{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, m::T, l::N, n::N = 1) = sqrt(BLAS.dot(l, v, n, v, n)-m^2)


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

	##===================================================================================
	##  chi distribution
	##===================================================================================
	export chi

	##-----------------------------------------------------------------------------------
	chi(k::Integer) = norm(randn(k))


	##===================================================================================
	##  general evaluation matrix
	##      l = [(x)->1, (x)->x, (x)->x^2] for polynomial of degree two
	##===================================================================================
	export gevamat

	##-----------------------------------------------------------------------------------
	function gevamat{T<:AbstractFloat}(l::Array{Function, 1}, v::Array{T, 1})
		m = map(l[1], v)

		@inbounds for i = 2:size(l, 1)
			m = hcat(m, map(l[i], v))
		end

		return m
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
	##===================================================================================
	export hh_rfl, hh_mat

	##-----------------------------------------------------------------------------------
	hh_rfl{T<:AbstractFloat}(v::Array{T, 1}, u::Array{T, 1}) = u-(v*(2.0*bdot(u, v))) 	# hh reflection (u = normal vector)

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
	##  dice sampling
	##===================================================================================
	export w6, nw6

	##-----------------------------------------------------------------------------------
	w6() = Int(ceil(rand()*6))

	##-----------------------------------------------------------------------------------
	nw6(n::Integer) = sum([w6() for i = 1:n])


	##===================================================================================
	##  QR (faster than vanilla)
	##===================================================================================
	export qrd_sq, qrd

	##-----------------------------------------------------------------------------------
	function qrd_sq{T<:AbstractFloat}(m::Array{T, 2})
		s = size(m, 1)
		t = zeros(T, s, s)
		v = zeros(T, s)
		r = copy(m)
		w = T(0)

		@inbounds for i=1:(s-1)
			w = 0.
			for j=i:s
				v[j] = r[j, i]
				w += v[j]*v[j]
			end

			v[i] += (r[i, i] >= 0 ? 1. : -1.)*sqrt(w)
			w = 0.

			for j=i:s w += v[j]*v[j] end
			w = 2.0/w

			for j=1:s, k=1:s
				t[j, k] = k == j ? 1. : 0.
				if j>=i && k>=i
					t[j, k] -= w*v[j]*v[k]
				end
			end

			for j=1:s
				for k=1:s
					v[k] = r[k, j]
				end

				for l=1:s
					w = 0.
					for h=1:s
						w += v[h]*t[l, h]
					end
					r[l, j] = w
				end
			end
		end

		for j=1:(s-1), k=(j+1):s
			r[k, j] = 0.
		end

		return (m*inv(r), r)
	end

	##-----------------------------------------------------------------------------------
	function qrd{T<:AbstractFloat}(m::Array{T, 2})
		s = size(m, 1)
		t = zeros(T, s, s)
		v = zeros(T, s)
		r = copy(m)
		w = T(0)

		for i=1:(s-1)
			w = 0.
			for j=i:s
				v[j] = r[j, i]
				w += v[j]*v[j]
			end

			v[i] += (r[i, i] >= 0 ? 1. : -1.)*sqrt(w)
			w = 0.

			for j=i:s w += v[j]*v[j] end
			w = 2.0/w

			for j=1:s, k=1:s
				t[j, k] = k == j ? 1. : 0.
				if j>=i && k>=i
				    t[j, k] -= w*v[j]*v[k]
				end
			end

			if i == 1
				r .= t*r
			else
				for j=1:s
					for k=1:s
						v[k] = r[k, j]
					end

					for l=1:s
						w = 0.
						for h=1:s
							w += v[h]*t[l, h]
						end
						r[l, j] = w
					end
				end
			end
		end

		for j=1:(s-1), k=(j+1):s
			r[k, j] = 0.
		end

		return (m/r, r)
	end


	##===================================================================================
	## diagonal expansion of matrices
	##===================================================================================
	export ul_x_expand

	##-----------------------------------------------------------------------------------
	function ul_x_expand{T<:AbstractFloat, N<:Integer}(m::Array{T, 2}, s::Tuple{N, N}, x::T = 1.0)# ul = upper left
		d = (s[1]-size(m, 1), s[2]-size(m,2))
		r = zeros(T, s)

		@inbounds for i = 1:s[1], j = 1:s[2]
			if i>d[1] && j>d[2]
				r[i, j] = m[i-d[1], j-d[2]]
			elseif i == j
				r[i, j] = x
			end
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

		@inbounds for i=(1+p[1]):s[1], j=(1+p[2]):s[2]
			r[i-p[1], j-p[2]] = m[i, j]
		end

		return r
	end


	##===================================================================================
	##	generalized mean
	##===================================================================================
	export gamean, ghmean, ggmean, gpmean, gfmean, grmean

	##-----------------------------------------------------------------------------------
	gamean{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, l::N, n::N) = gfmean(v, (x) -> x, (x) -> x, l, n)					# arithmetic mean

	##-----------------------------------------------------------------------------------
	ghmean{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, l::N, n::N) = gfmean(v, (x) -> 1/x, (x) -> 1/x, l, n)				# harmonic mean

	##-----------------------------------------------------------------------------------
	ggmean{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, l::N, n::N) = gfmean(v, (x) -> log(x), (x) -> exp(x), l, n)		# geometric mean

	##-----------------------------------------------------------------------------------
	gpmean{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, l::N, n::N, p::T) = gfmean(v, (x) -> x^p, (x) -> x^(1/p), l, n)	# power mean

	##-----------------------------------------------------------------------------------
	grmean{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, l::N, n::N, p::T) = gfmean(v, (x) -> x^2, (x) -> sqrt(x), l, n)  		# root squared mean

	##-----------------------------------------------------------------------------------
	function gfmean{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, g::Function, g_inv::Function, l::N, n::N) 				# generalized f mean
		@assert(size(v, 1) >= (n*l), "out of bounds error")
		u = Float64(0)

		@inbounds for i = 1:n:(n*l)
			u += g(v[i])
		end

		return g_inv(u/l)
	end


	##===================================================================================
	##  outer product implementation (faster than vanila)
	##===================================================================================
	export otr

	##-----------------------------------------------------------------------------------
	function otr{T<:AbstractFloat}(v::Array{T, 1}, w::Array{T, 1})
		s = (size(v, 1), size(w, 1))
		m = Array{T, 2}(s)

		@inbounds for i=1:s[1], j=1:s[2]
			m[i, j] = v[i] * w[j]
		end

		return m
	end


	##===================================================================================
	##  gram schmidt proces
	##===================================================================================
    export grsc, grscn

	##-----------------------------------------------------------------------------------
	function grsc{T<:AbstractFloat}(m::Array{T, 2})
    	s = size(m, 2)
    	d = zeros(T, s)
    	ob = []

    	@inbounds for i = 1:s
        		push!(ob, m[:, i])
        		for j = 1:(i-1)
            		ob[i] -= (dot(ob[j], ob[i])/d[j])*ob[j]
        		end
     		d[i] = dot(ob[i], ob[i])
   		end

    	return ob
	end

    ##-----------------------------------------------------------------------------------
   	function grscn{T<:AbstractFloat}(m::Array{T, 2})
    	ob = []

    	@inbounds for i = 1:size(m, 2)
        		push!(ob, m[:, i])
        		for j = 1:(i-1)
            		ob[i] -= dot(ob[j], ob[i])*ob[j]
        		end
        		normalize!(ob[i])
    	end

    	return ob
    end


    ##===================================================================================
    ##  orthogonal projection
    ##===================================================================================
    export proj, projn

    ##-----------------------------------------------------------------------------------
    function proj{T<:AbstractFloat}(v::Array{T, 1}, m::Array{T, 2})
    	r = zeros(size(v))
    	@inbounds for i = 1:size(m, 2)
        	r += m[:, i]*(bdot(v, m[:, i])/bdot(m[:, i], m[:, i]))
    	end
    	return r
    end

	##-----------------------------------------------------------------------------------
    function proj{T<:Complex}(v::Array{T, 1}, m::Array{T, 2})
    	r = zeros(size(v))
    	@inbounds for i = 1:size(m, 2)
        	r += m[:, i]*(bdotc(v, m[:, i])/bdotc(m[:, i], m[:, i]))
    	end
    	return r
    end

    ##-----------------------------------------------------------------------------------
    projn{T<:AbstractFloat}(v::Array{T, 1}, m::Array{T, 2}) = m*m'*v

	##-----------------------------------------------------------------------------------
    projn{T<:Complex}(v::Array{T, 1}, m::Array{T, 2}) = m*m.'*v


    ##===================================================================================
    ##  boolean operators (aggregation)
    ##===================================================================================
    export AND, OR

    ##-----------------------------------------------------------------------------------
    function AND(v::BitArray{1})
        @inbounds for i = 1:size(v, 1)
            if v[i] == false
                return false
            end
        end

        return true
    end

    ##-----------------------------------------------------------------------------------
    function OR(v::BitArray{1})
        @inbounds for i = 1:size(v, 1)
            if v[i] == true
                return true
            end
        end

        return false
    end


    ##===================================================================================
    ##  / (overload)
    ##===================================================================================
	import Base./
	export /

    ##-----------------------------------------------------------------------------------
    function /{T<:AbstractFloat}(x::T, v::Array{T, 1})
        @inbounds for i = 1:size(v, 1)
            v[i] = x/v[i]
        end

        return v
    end


    ##===================================================================================
    ## variance (overload)
    ##===================================================================================
    export var

    ##-----------------------------------------------------------------------------------
    function var{T<:AbstractFloat}(v::Array{T, 1}, m::T)                                # faster implementation
        l = size(v, 1)
        return (soq(l, v)/l) - (m^2)
    end

    ##-----------------------------------------------------------------------------------
    function var{T<:AbstractFloat}(v::Array{T, 1})                                      # faster implementation
        l = size(v, 1)
        return (soq(l, v)/l) - (gamean(l, v)^2)
    end

    ##-----------------------------------------------------------------------------------
    var{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, m::T, l::N) = (soq(l, v)/l) - (m^2)

    ##-----------------------------------------------------------------------------------
    var{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, l::N) = (soq(l, v)/l) - (gamean(l, v)^2)


    ##===================================================================================
    ## BLAS wrapper
    ##===================================================================================
    export bdot, bdotu, bdotc, bnrm

    ##-----------------------------------------------------------------------------------
    bdot{T<:AbstractFloat, N<:Integer}(l::N, v::Array{T, 1}, u::Array{T, 1}) = BLAS.dot(l, v, 1, u, 1)	# l = length of the vectors

    ##-----------------------------------------------------------------------------------
    bdot{T<:AbstractFloat}(v::Array{T, 1}, u::Array{T, 1}) = BLAS.dot(size(v, 1), v, 1, u, 1)

    ##-----------------------------------------------------------------------------------
    bdotu{C<:Complex, N<:Integer}(l::N, v::Array{C, 1}, u::Array{C, 1}) = BLAS.dotu(l, v, 1, u, 1)		# l = length of the vectors

    ##-----------------------------------------------------------------------------------
    bdotu{C<:Complex}(v::Array{C, 1}, u::Array{C, 1}) = BLAS.dotu(size(v, 1), v, 1, u, 1)

    ##-----------------------------------------------------------------------------------
    bdotc{C<:Complex, N<:Integer}(l::N, v::Array{C, 1}, u::Array{C, 1}) = BLAS.dotc(l, v, 1, u, 1)		# l = length of the vectors

    ##-----------------------------------------------------------------------------------
    bdotc{C<:Complex}(v::Array{C, 1}, u::Array{C, 1}) = BLAS.dotc(size(v, 1), v, 1, u, 1)

    ##-----------------------------------------------------------------------------------
    bnrm{T<:AbstractFloat, N<:Integer}(l::N, v::Array{T, 1}) = BLAS.nrm2(l, v, 1)                       # l = length of the vector

    ##-----------------------------------------------------------------------------------
    bnrm{T<:AbstractFloat}(v::Array{T, 1}) = BLAS.nrm2(size(v, 1), v, 1)


    ##===================================================================================
    ##  soq (sum of squares)
    ##===================================================================================
    export soq, soqu, soqc

    ##-----------------------------------------------------------------------------------
    soq{T<:AbstractFloat, N<:Integer}(l::N, v::Array{T, 1}, n::N = 1) = BLAS.dot(l, v, n, v, n)

    ##-----------------------------------------------------------------------------------
    soq{T<:AbstractFloat}(v::Array{T, 1}) = bdot(v, v)

    ##-----------------------------------------------------------------------------------
    soqu{C<:Complex, N<:Integer}(l::N, v::Array{C, 1}, n::N = N(1)) = BLAS.dot(l, v, n, v, n)

    ##-----------------------------------------------------------------------------------
    soqu{C<:Complex}(v::Array{C, 1}) = bdotu(v, v)

    ##-----------------------------------------------------------------------------------
    soqc{C<:Complex, N<:Integer}(l::N, v::Array{C, 1}, n::N = 1) = bdotc(l, v, n, v, n)

    ##-----------------------------------------------------------------------------------
    soqc{C<:Complex}(v::Array{C, 1}) = bdotc(v, v)


    ##===================================================================================
    ##  collatz conjecture
    ##===================================================================================
    export collatz

    ##-----------------------------------------------------------------------------------
    function collatz{T<:Integer}(x::T)
		@assert(x > 0, "out of bounds error")
		c = UInt(0)

        while x != 4
            if x & 1 == 1
                x = 3*x +1
            end

            x = x >> 1
            c = c + 1
        end

        return c
    end


    ##===================================================================================
    ## veconomy core function
    ##===================================================================================
    function veconomy_core{T<:AbstractFloat}(v::Array{T, 1}, cc::T = 0.4)
        lumV = norm(v) / MAX_LUM
        o = prison(rotmat_3d(rand_orthonormal_vec(v), 90)*(v-127.5), -127.5, 127.5)
        while abs(lumv-(norm(o)/MAX_LUM)) < cc; o = map((x) -> x*(lumV>0.5?.5:1.5), o) end
        return map((x) -> prison(round(x), 0, 255), o)
    end


    ##===================================================================================
    ##  dirac delta/impulse
    ##===================================================================================
    export dcd

    ##-----------------------------------------------------------------------------------
    function dcd{T<:AbstractFloat}(x::T)
        set_zero_subnormals(true)
        return x == 0 ? Inf : 0
    end

    ##===================================================================================
    ## kronecker delta
    ##===================================================================================
    export ked

    ##-----------------------------------------------------------------------------------
    ked{T<:Number}(x::T, y::T) = x == y ? 1 : 0


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
    ## auto regressive model
    ##===================================================================================
	export ar

	##-----------------------------------------------------------------------------------
	function ar{T<:AbstractFloat, N<:Integer}(q::N, v::Array{T, 1})
		l = size(v, 1) - q
		m = zeros(T, l, q+1)
		y = zeros(T, l)

		@inbounds for i = 1:l
			m[i, 1] = 1.
			m[i, 2:end] = v[i:(q+i-1)]
			y[i] = v[q+i]
		end

		return map((x) -> isnan(x) ? 0. : x, (m'*m)\(m'*y))
	end


	##===================================================================================
    ## [augumented] dickey fuller test for stationarity
    ##===================================================================================
	export difut, adifut

	##-----------------------------------------------------------------------------------
	difut{T<:AbstractFloat}(v::Array{T, 1}, p::T = .01) = ar(1, v)[2] <= p

	##-----------------------------------------------------------------------------------
	difut{T<:AbstractFloat}(v::Array{T, 1}, p::T, d::T) = ar(1, v-[d*x for x = 1:size(v, 1)])[2] <= p

	##-----------------------------------------------------------------------------------
	difut{T<:AbstractFloat}(v::Array{T, 1}, p::T, d::T, t::T) = ar(1, v-[(d*x)+(t*sum(x)) for x = 1:size(v, 1)])[2] <= p

	##-----------------------------------------------------------------------------------
	function adifut{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, q::N, p::T = .01)
		@assert(l > p, "sample size to small")
		d = (circshift(v, -1) - v)[1:end-1]
		l = size(v, 1) - q - 1
		m = zeros(T, l, q + 2)
		y = d[(q+1):end]

		@inbounds for i = 1:l
			m[i, 1] = 1.
			m[i, 2] = v[q+i-1]
			m[i, 3:end] = d[(q+i-1):-1:i]
		end

		return map((x) -> isnan(x) ? 0. : x, (m'*m)\(m'*y))[2] <= p
	end

	##-----------------------------------------------------------------------------------
	adifut{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, q::N, p::T, d::N) = adifut((v-[d*x for x = 1:size(v, 1)]), q, p)

	##-----------------------------------------------------------------------------------
	adifut{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, q::N, p::T, d::T, t::T) = adifut((v-[(d*x)+(t*sum(x)) for x = 1:size(v, 1)]), q, p)


	##===================================================================================
    ## angle granger test for cointegration
    ##===================================================================================
	export angrat, aangrat

	##-----------------------------------------------------------------------------------
	angrat{T<:AbstractFloat}(x::Array{T, 1}, y::Array{T, 1}, p::T = .01) = difut(y - (((soq(x))\dot(x, y))*x), p)

	##-----------------------------------------------------------------------------------
	angrat{T<:AbstractFloat}(x::Array{T, 1}, y::Array{T, 1}, p::T, d::T) = difut(y - (((soq(x))\dot(x, y))*x), p, d)

	##-----------------------------------------------------------------------------------
	aangrat{T<:AbstractFloat}(x::Array{T, 1}, y::Array{T, 1}, p::T, d::T, t::T) = difut(y - (((soq(x))\dot(x, y))*x), p, d, t)

	##-----------------------------------------------------------------------------------
	aangrat{T<:AbstractFloat, N<:Integer}(x::Array{T, 1}, y::Array{T, 1}, q::N, p::T = .01) = difut(y - (((soq(x))\dot(x, y))*x), q, p)

	##-----------------------------------------------------------------------------------
	aangrat{T<:AbstractFloat, N<:Integer}(x::Array{T, 1}, y::Array{T, 1}, q::N, p::T, d::T) = difut(y - (((soq(x))\dot(x, y))*x), q, p, d)

	##-----------------------------------------------------------------------------------
	aangrat{T<:AbstractFloat, N<:Integer}(x::Array{T, 1}, y::Array{T, 1}, q::N, p::T, d::T, t::T) = difut(y - (((soq(x))\dot(x, y))*x), q, p, d, t)


    ##===================================================================================
    ##	Cofactor Matrix of a Matrix
    ##===================================================================================
    export cof

    ##-----------------------------------------------------------------------------------
    function cof{T<:AbstractFloat}(m::Array{T, 2})										# TODO needs better performance
		s = size(m)
		n = zeros(T, s[1]-1, s[2]-1)
		r = zeros(T, s)

		@inbounds for i = 1:s[1], j = 1:s[2]
			for x = 1:s[1], y = 1:s[2]
				if x != i && y != j
					n[(x > i ? x-1 : x), (y > j ? y-1 : y)] = m[x, y]
				end
			end

			r[i, j] = det(n)*(i+j % 2 == 0 ? 1. : -1)
		end

		return r
    end


	##===================================================================================
	##	Adjugate of a Matrix
	##===================================================================================
	export adj

	##-----------------------------------------------------------------------------------
	adj{T<:AbstractFloat}(m::Array{T, 2}) = return cof(m)'


	##===================================================================================
	##	Nullspace of a matrix
	##===================================================================================
	export nul

	##-----------------------------------------------------------------------------------
	function nul{T<:AbstractFloat}(m::Array{T, 2})
		s = size(m)

		if s[1] >= s[2]
			return zeros(T, s[2])
		end

		r = zeros(T, s[2], s[2]-s[1])

		@inbounds for i = 1:s[1]
			for j = 1:s[1]
				if i != j
					m[j,:] -= (m[j,i]/m[i,i]) * m[i,:]
				end
			end
			m[i,:] /= m[i,i]
		end

		r[1:s[1], :] = -m[:,(s[1]+1):s[2]]
		r[(s[1]+1):end, :] = eye(s[2]-s[1])

		return r
	end


	##===================================================================================
	##	kingsman equation
	##===================================================================================
	export king

	##-----------------------------------------------------------------------------------
	king{T<:AbstractFloat, N<:Integer}(p::T, ca::T, cs::T, c::N) = ((p^(sqrt(2*(c+1))))/(1-p))x*(0.5)*((ca^2)+(cs^2))


    ##===================================================================================
    ## sigmoid
    ##===================================================================================
    export sigmoid, sigmoidd

    ##-----------------------------------------------------------------------------------
    sigmoid(x, eta = 0) = @. 1/(1+exp(-(x-eta)))

    ##-----------------------------------------------------------------------------------
    sigmoidd(x, eta = 0) =  @. sigmoid(x, eta) * (1-sigmoid(x, eta))


    ##===================================================================================
    ## norm derivate
    ##===================================================================================
    export normd

    ##-----------------------------------------------------------------------------------
    nromd(v::Array{Float64, 1}, p::Int64 = 2) = @. sign(v)*(abs(v)/ifelse(iszero(v), 1, norm(v, p)))^(p-1)


    ##===================================================================================
    ## radial basis functions
    ##===================================================================================
    export rbf_gaussian, rbf_gaussian_d_lambda, rbf_gaussian_d_delta, rbf_triang, rbf_cos_decay,
        rbf_psq, rbf_inv_psq, rbf_inv_sq, rbf_exp, rbf_thin_plate_spline

    ##-----------------------------------------------------------------------------------
    rbf_gaussian(delta, lambda::Float64 = 1) = @. exp(-(delta/(2*lambda))^2)

    ##-----------------------------------------------------------------------------------
    function rbf_gaussian_d_lambda(delta, lambda::Float64 = 1)
        @. delta ^= 2
        lam = lambda^2
        return @. (delta/(lam*lambda))*exp(-delta/lam)
    end

    ##-----------------------------------------------------------------------------------
    function rbf_gaussian_d_delta(delta, lambda::Float64 = 1)
        @. lambda ^= 2
        return @. (delta./lambda) .* exp(-(delta.^2)./(2*lambda))
    end

    ##-----------------------------------------------------------------------------------
    rbf_triang(delta::Float64, lambda::Float64 = 1) = delta > lambda ? 0. : (1 - (delta/lambda))

    ##-----------------------------------------------------------------------------------
    function rbf_triang(delta::Array{Float64, 1}, lambda::Float64 = 1)
        if AND(delta .> lambda)
            return zeros(delta)
        else
            return (1.-(delta./lambda))
        end
    end

    ##-----------------------------------------------------------------------------------
    rbf_cos_decay(delta::Float64, lambda::Float64 = 1) = delta > lambda ? 0. : ((cos((pi*delta)/(2*lambda)))+1)/2

    ##-----------------------------------------------------------------------------------
    function rbf_cos_decay(delta::Array{Float64, 1}, lambda::Float64 = 1)
        if AND(delta .> lambda)
            return zeros(delta)
        else
            return @. ((cos((pi*delta)/(2*lambda))).+1)/2
        end
    end

    ##-----------------------------------------------------------------------------------
    rbf_psq(delta, lambda::Float64 = 1) = @. sqrt(1+(lambda*delta)^2)

    ##-----------------------------------------------------------------------------------
    rbf_inv_psq(delta, lambda::Float64 = 1) = @. 1/sqrt(1+(lambda*delta)^2)

    ##-----------------------------------------------------------------------------------
    rbf_inv_sq(delta, lambda::Float64 = 1) = @. 1/(1+(lambda*delta)^2)

    ##-----------------------------------------------------------------------------------
    rbf_exp(delta, expt::Float64 = 2) = delta.^expt

    ##-----------------------------------------------------------------------------------
    rbf_thin_plate_spline(delta, expt::Float64 = 2) = @. (delta^expt)*log(delta)


    ##===================================================================================
    ## ramp
    ##===================================================================================
    export ramp, rampd

    ##-----------------------------------------------------------------------------------
    ramp(x, eta) = @. max(0, x-eta)

    ##-----------------------------------------------------------------------------------
    rampd{T<:AbstractFloat}(x::T, eta::T) = x-eta > 0 ? T(1) : T(0)

    ##-----------------------------------------------------------------------------------
    rampd{T<:AbstractFloat}(x::Array{T, 1}, eta::T) = AND(x.-eta .> 0) ? ones(T, x) : zeros(T, x)

    ##-----------------------------------------------------------------------------------
    function rampd{T<:AbstractFloat}(x::Array{T, 1}, eta::Array{T, 1})
        r = zeros(T, x)

        @inbounds for i = 1:size(x, 1)
            if x[i]-eta[i] > 0
                r[i] = 1.
            end
        end

        return r
    end

    ##===================================================================================
    ## semi linear
    ##===================================================================================
    export semilin, semilind

    ##-----------------------------------------------------------------------------------
    semilin(x, eta, sigma = 0.5) = prison(x, (x) -> x-eta+sigma, eta-sigma, eta+sigma)

    ##-----------------------------------------------------------------------------------
    semilind(x, eta, sigma = 0.5) = (x > eta+sigma || x < eta-sigma) ? 0. : 1.


    ##===================================================================================
    ## sine saturation
    ##===================================================================================
    export sine_saturation, sine_saturation_d

    ##-----------------------------------------------------------------------------------
    sine_saturation(x, eta, sigma = pi/2) = prison(x, (x) -> (sin(x-eta)+1)/2, eta-sigma, eta+sigma)

    ##-----------------------------------------------------------------------------------
    sine_saturation_d(x, eta, sigma = pi/2) = (x > eta+sigma || x < eta-sigma) ? 0 : cos(x-eta)/2


    ##===================================================================================
    ## softplus
    ##===================================================================================
    export softplus, softplusd

    ##-----------------------------------------------------------------------------------
    softplus(x, eta) = @. log(1+exp(x-eta))

    ##-----------------------------------------------------------------------------------
    softplusd(x, eta) = @. 1/(exp(x-eta)+1)


    ##===================================================================================
    ## mutual incoherence
    ##===================================================================================
    export mut_incoherent

    ##-----------------------------------------------------------------------------------
    function mut_incoherent{T<:Number, N<:Integer}(m::Array{T, 2}, rows = true, p::N = 2)         # the lower the better the mutual incoherence property
        m = rows ? m : m'
		inf = 0

		for x = 2:size(m, 1), y = 1:(x-1)
            inf = max(norm(bdot(m[x, :], m[y, :]), p), inf)
        end

		return inf
    end

    ##-----------------------------------------------------------------------------------
    function mut_incoherent{T<:Number, N<:Integer}(vl::Array{Array{T, 1}}, p::N = 2)
        inf = 0

        @inbounds for x = 2:lenght(vl), y = 1:(x-1)
            inf = max(norm(bdot(m[x], m[y]), p), inf)
        end

		return inf
    end


    ##===================================================================================
    ## step
    ##===================================================================================
    export step, stepd

    ##-----------------------------------------------------------------------------------
    step(x, eta = 0.5) = @. ifelse(x >= eta, 1, 0)

    ##-----------------------------------------------------------------------------------
    stepd{T<:Number}(x::T, eta) = @. ifelse(x == eta, typemax(T), 0)


    ##===================================================================================
    ## trigonometric
    ##===================================================================================
    export sin2, cos2, versin, aversin, vercos, avercos, coversin, acoversin, covercos, acovercos,
        havsin, ahavsin, havcos, ahavcos, hacoversin, hacovercos

    ##-----------------------------------------------------------------------------------
    sin2(alpha) = @. sin(alpha)^2

    ##-----------------------------------------------------------------------------------
    cos2(alpha) = @. cos(alpha)^2

    ##-----------------------------------------------------------------------------------
    versin(alpha) = @. 1-cos(alpha)

    ##-----------------------------------------------------------------------------------
    aversin(alpha) = @. acos(1-alpha)

    ##-----------------------------------------------------------------------------------
    vercos(alpha) = @. 1+cos(alpha)

    ##-----------------------------------------------------------------------------------
    avercos(alpha) = @. acos(1+alpha)

    ##-----------------------------------------------------------------------------------
    coversin(alpha) = @. 1-sin(alpha)

    ##-----------------------------------------------------------------------------------
    acoversin(alpha) = @. asin(1-alpha)

    ##-----------------------------------------------------------------------------------
    covercos(alpha) = @. 1+sin(alpha)

    ##-----------------------------------------------------------------------------------
    acovercos(alpha) = @. asin(1+alpha)

    ##-----------------------------------------------------------------------------------
    havsin(alpha) = @. versin(alpha)/2

    ##-----------------------------------------------------------------------------------
    ahavsin(alpha) = @. 2*asin(sqrt(alpha))

    ##-----------------------------------------------------------------------------------
    havcos(alpha) = @. vercos(alpha)/2

    ##-----------------------------------------------------------------------------------
    ahavcos(alpha) = @. 2*acos(sqrt(alpha))

    ##-----------------------------------------------------------------------------------
    hacoversin(alpha) = @. coversin(alpha)/2

    ##-----------------------------------------------------------------------------------
    hacovercos(alpha) = @. covercos(alpha)/2


    ##===================================================================================
    ## angle
    ##===================================================================================
    export angle, ccentral_angle, scentral_angle, tcentral_angle, central_angle,
        hcentral_angle, vincenty_central_angle

    ##-----------------------------------------------------------------------------------
    angle{T<:AbstractFloat}(u::Array{T, 1}, v::Array{T, 1}, bias::T = T(0)) = @. acosd((abs(bdot(u, v))/(bnrm(v)*bnrm(u)))+bias)

    ##-----------------------------------------------------------------------------------
    ccentral_angle{T<:AbstractFloat}(u::Array{T, 1}, v::Array{T, 1}) = @. acos(bdot(u, v))           			# returns radians | u&v = normal vectors on the circle

    ##-----------------------------------------------------------------------------------
    scentral_angle{T<:AbstractFloat}(u::Array{T, 1}, v::Array{T, 1}) = @. asin(bnrm(cross(u, v)))     			# returns radians | u&v = normal vectors on the circle

    ##-----------------------------------------------------------------------------------
    tcentral_angle{T<:AbstractFloat}(u::Array{T, 1}, v::Array{T, 1}) = @. atan(bnrm(cross(u, v))/bdot(u, v))  	# returns radians | u&v = normal vectors on the circle

    ##-----------------------------------------------------------------------------------
    central_angle{T<:AbstractFloat}(pla::T, plo::T, sla::T, slo::T) = acos((sin(pla)*sin(sla))+(cos(pla)*cos(sla)*cos(abs(plo-slo)))) 						# returns radians | pla/sla = primary/secondary latitude / plo/slo = primary/secondary longitude

    ##-----------------------------------------------------------------------------------
    hcentral_angle{T<:AbstractFloat}(pla::T, plo::T, sla::T, slo::T) = 2*asin(sqrt(havsin(abs(pla-sla))+cos(pla)*cos(sla)*havsin(abs(plo-slo)))) 	# returns radians | pla/sla = primary/secondary latitude / plo/slo = primary/secondary longitude

    ##-----------------------------------------------------------------------------------
    function vcentral_angle{T<:AbstractFloat}(pla::T, plo::T, sla::T, slo::T)
        longitude_delta = abs(plo-slo)                                                                                      								# returns radians | pla/sla = primary/secondary latitude / plo/slo = primary/secondary longitude
        return atan2(sqrt((cos(sla)*sin(longitude_delta))^2+((cos(pla)*sin(sla))-(sin(pla)*cos(sla)*cos(longitude_delta)))^2), (sin(pla)*sin(sla)+cos(pla)*cos(sla)*cos(longitude_delta)))
    end


    ##===================================================================================
    ## normalize (statiscally)
    ##===================================================================================
    export normalize_s, normalize_sp, normalize_sps

    ##-----------------------------------------------------------------------------------
    function normalize_s{T<:Number}(m::Array{T, 2}, column::Bool = true)                # sets variance to 1 and mean to 0
		r = column ? m : m'
		d = size(r, 1)

        @inbounds for w = 1:size(r, 2)
            r[1:d, w] = (r[1:d, w] - median(r[1:d, w])) / std(r[1:d, w])
        end

        return r
    end

	##-----------------------------------------------------------------------------------
    function normalize_sp{T<:Number}(m::Array{T, 2}, column::Bool = true)
		r = column ? m : m'
		d = size(m, 1)

        @sync @parallel for w = 1:size(r, 2)
            r[1:d, w] = (r[1:d, w] - median(r[1:d, w])) / std(r[1:d, w])
        end

        return r
    end

    ##-----------------------------------------------------------------------------------
    function normalize_sps{T<:Number}(m::Array{T, 2}, column::Bool = true)
		r = convert(SharedArray, column ? m : m')
		d = size(r, 1)

        @inbounds @sync @parallel for w = 1:size(r, 2)
            r[1:d, w] = (r[1:d, w] - median(r[1:d, w])) / std(r[1:d, w])
        end

        return convert(Array, r)
    end


    ##===================================================================================
    ## rotation matrix
    ##===================================================================================
    export rotmat_3d

	##-----------------------------------------------------------------------------------
	rotmat_2d{T<:AbstractFloat}(angle::T = T(90)) = [cos(angle) -sin(angle); sin(angle) cos(angle)]

    ##-----------------------------------------------------------------------------------
    function rotmat_3d{T<:AbstractFloat}(axis::Array{T, 1}, angle::T = T(90))
        axis = axis'
        m = [ 0 -axis[3] axis[2]; axis[3] 0 -axis[1]; -axis[2] axis[1] 0 ]
        return eye(T, 3) + m * sind(alpha) + (1 - cosd(alpha)) * m^2
    end


    ##===================================================================================
    ## arithmetic mean column/row
    ##===================================================================================
    export mamean

    ##-----------------------------------------------------------------------------------
    function mamean{T<:AbstractFloat}(arr::Array{T, 2}, column::Bool = true)
        n = size(arr, (column ? 1 : 2))
        return BLAS.gemv((column ? 'N' : 'T'), 1/n, arr, ones(T, n))
    end

    ##-----------------------------------------------------------------------------------
    function mamean{T<:AbstractFloat}(arr::Array{T, 2}, weights::Array{T, 1}, column::Bool = true)
        n = size(arr, (column ? 1 : 2))
        return BLAS.gemv((column ? 'N' : 'T'), 1/n, weights.*arr, ones(T, n))
    end


    ##===================================================================================
    ## covariance
    ##===================================================================================
    export cov

    ##-----------------------------------------------------------------------------------
    function cov{T<:AbstractFloat, N<:Integer}(l::N, x::Array{T, 1}, n1::N, y::Array{T, 1}, n2::N)
        m1 = m2 = T(0)

		s = n1*l
		@assert(s <= size(x, 1), "out of bounds error")
        @inbounds for i = 1:n1:(n1*l)
            m1 = m1 + x[i]
        end

		s = n2*l
		@assert(s <= size(y, 1), "out of bounds error")
        @inbounds for i = 1:n2:(n2*l)
            m2 = m2 + y[i]
        end

        m1 = m1/l
        m2 = m2/l

        return (BLAS.dot(l, x, n1, y, n2)/l)-(m1*m2)
    end

    ##-----------------------------------------------------------------------------------
    cov{T<:AbstractFloat, N<:Integer}(l::N, x::Array{T, 1}, n::N, d::N = N(1)) = cov(l, x, n, x[d+1:end], n)     # d = delay


    ##===================================================================================
    ## covariance matrices from observations
    ##===================================================================================
    export covp, covs

    ##-----------------------------------------------------------------------------------
    function covp{T<:AbstractFloat}(m::Array{T, 2}, column::Bool = true)                # cov population
        s = size(x, (column ? 1 : 2))
        v = BLAS.gemv((column ? 'T' : 'N'), m, ones(T, s))
		return column ? BLAS.gemm('T', 'N', 1/s, m, m) - (bnrm(v)/v)^2 : BLAS.gemm('N', 'T', 1/s, m, m) - (bnrm(v)/v)^2
    end

    ##-----------------------------------------------------------------------------------
    function covs{T<:AbstractFloat}(m::Array{T, 2}, column::Bool = true)                # cov sample
        s = size(m, (column ? 1 : 2))
        v = BLAS.gemv((column ? 'T' : 'N'), m, ones(T, s))
        return column ? BLAS.gemm('T', 'N', 1/(s-1), m, m) - (bdot(v, v) / (v*(v-1))) : BLAS.gemm('N', 'T', 1/(s-1), m, m) - (bdot(v, v) / (v*(v-1)))
    end


    ##===================================================================================
    ## cross covariance
    ##===================================================================================
    export covc

    ##-----------------------------------------------------------------------------------
    function covc{T<:AbstractFloat}(x::Array{T, 1}, y::Array{T, 1})
        xs = size(x, 1); xm = gamean(x, xs, 1)
		ys = size(y, 1); ym = gamean(y, ys, 1)

        r = zeros(T, xs, ys)
		sc = 1/(xs*ys)

		@inbounds for xi = 1:xs, yi = 1:ys
            r[xi, yi] = cov(x[xi], xm, y[yi], ym, sc)
        end

        return r
    end


    ##===================================================================================
    ## cross covariance sumed (with delay)
    ##===================================================================================
    export covcs

    ##-----------------------------------------------------------------------------------
    covcs{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, u::Array{T, 1}, t::N = N(1)) = bdot((l-t), (v-gamean(size(v, 1), v, 1)), (circshift(u, t)-gamean(size(u, 1), u, 1)))/(l-t)


    ##===================================================================================
    ## cross correlation (with delay)
    ##===================================================================================
    export ccor

    ##-----------------------------------------------------------------------------------
    ccor{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, u::Array{T, 1}, t::N = N(1)) = covcs(v, u, t)./(std(v)*std(u))


    ##===================================================================================
    ## supp (support)
    ##===================================================================================
    export supp

    ##-----------------------------------------------------------------------------------
    function supp{T<:AbstractFloat}(v::Array{T, 1}, f::Function = (x::T) -> x)
        set_zero_subnormals(true)
        u = Array{T, 1}

        @inbounds for i = 1:size(v, 1)
			if abs(f(x)) == 0
				push!(u, v[i])
			end
		end

        return u
    end

    ##-----------------------------------------------------------------------------------
    function supp{T<:AbstractFloat}(vl::Array{Array{T, 1}, 1}, f::Function = (x::T) -> x)# supp for vector lists
        ul = Array{Array{T, 1}, 1}
        set_zero_subnormals(true)

        @inbounds for i = 1:size(vl, 1)
			if AND(abs(f(x)) .== 0)
				push!(ul, v[i])
			end
		end

		return ul
    end


	##===================================================================================
	##	shannon index
	##===================================================================================
	export shai

	##-----------------------------------------------------------------------------------
	shai{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, l::N, n::N) = BLAS.dot(l, v, n, log.(v[1:n:(n*l)]), 1)


	##===================================================================================
	##	Gini–Simpson index
	##===================================================================================
	export gishi

	##-----------------------------------------------------------------------------------
	gishi{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, l::N, n::N) = 1 - BLAS.dot(l, v, n, v, n)


	##===================================================================================
	##	Renyi Entropy
	##===================================================================================
	export reepy

	##-----------------------------------------------------------------------------------
	function reepy{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, p::N, l::N, n::N)
		s = n * l
		u = v[i]

		@assert(s <= size(v, 1), "out of bounds error")
		@inbounds for i = 1+n:n:s
			u += v[i]^p
		end

		return log(u)/(1-p)
	end


	##===================================================================================
	##	check status of the nth bit
	##===================================================================================
	export nbit_on, nbit_off

	##-----------------------------------------------------------------------------------
	nbit_on{T<:Integer}(x::T, i::Integer) = (x >> (i-1)) & T(1) == T(1)

	##-----------------------------------------------------------------------------------
	nbit_off{T<:Integer}(x::T, i::Integer) = xor((x >> (i-1)), T(1)) == T(1)


	##===================================================================================
	##	print binary representation
	##===================================================================================
	export bprint

	##-----------------------------------------------------------------------------------
	function bprint{T<:Integer}(x::T, bigendian::Bool = true)
		println(bigendian ? reverse(bits(x)) : bits(x))
	end


	##===================================================================================
	##	sizeof (in bits)
	##===================================================================================
	export sizeofb

	##-----------------------------------------------------------------------------------
	sizeofb(x::DataType) = sizeof(x)*8


	##===================================================================================
	##	invert bit range
	##===================================================================================
	export ibit_range

	##-----------------------------------------------------------------------------------
	ibit_range{T<:Unsigned}(x::T, lb::Integer, ub::Integer) = T(xor(sum([T(1) << x for x = (lb-1):(ub-1)]), x))


	##===================================================================================
	##	exchange bit range
	##===================================================================================
	export ebit_range

	##-----------------------------------------------------------------------------------
	function ebit_range{T<:Unsigned}(x::T, y::T, lb::Integer, ub::Integer)
		c = sum([T(1) << i for i = (lb-1):(ub-1)])
		d = ~c
		return (T(xor((x & d), (y & c))), T(xor((y & d), (x & c))))
	end

	##===================================================================================
	##	flip bit
	##===================================================================================
	export fbit

	##-----------------------------------------------------------------------------------
	fbit{T<:Unsigned}(x::T, i::Integer) = xor(x, T(1 << (i-1)))


	##===================================================================================
	##	set bit
	##===================================================================================
	export sbit

	##-----------------------------------------------------------------------------------
	sbit{T<:Unsigned}(x::T, i::Integer, v::Bool) = (nbit_on(x, i) == v) ? x : xor(x, T(1 << (i-1)))


	##===================================================================================
	##	column wise binary merge
	##===================================================================================
	export cb_merge

	##-----------------------------------------------------------------------------------
	function cb_merge{T<:Unsigned}(v::Array{T, 1})
		s = sizeofb(T)																	# r = A B C D E F G H I J K L M N O
		c = UInt8(1)																	# 	v[1] = A D G J M
		f = T(1)																		# 	v[2] = B E H K N
		r = T(0)																		# 	v[3] = C F I L O

		for i = 1:s, j = 1:size(v, 1)
			if nbit_on(v[j], i)
				r = xor(r, f)
			end

			if c == s
				return r
			end

			f <<= 1
			c += 1
		end
	end


	##===================================================================================
	##	split column wise binary merge again (cb_merge reverse)
	##===================================================================================
	export cb_split

	##-----------------------------------------------------------------------------------
	function cb_split{T<:Unsigned}(x::T, d::Integer)										# d = number of dimensions
		r = Array{T, 1}(zeros(d))
		s = sizeofb(T)
		c = UInt8(1)
		f = T(1)

		for i = 1:s
			for j = 1:d
				if nbit_on(x, c)
					r[j] = xor(r[j], f)
				end

				if c == s
					return r
				end
				c += 1
			end
			f <<= 1
		end
	end


	##===================================================================================
	##	gray codes (forward/backward)
	##===================================================================================
	export grayf, grayb

	##-----------------------------------------------------------------------------------
	grayf{T<:Unsigned}(x::T) = xor(x, (x >> 1))

	##-----------------------------------------------------------------------------------
	function grayb{T<:Unsigned}(x::T)
		y = x; r = x

		while y != T(0)
			y >>= 1
			r = xor(r, y)
		end

		return r
	end


	##===================================================================================
	##	hilbert curve (forward/backward)
	##===================================================================================
	export hilbert_cf, hilbert_cb

	##-----------------------------------------------------------------------------------
	function hilbert_cf{T<:Unsigned}(v::Array{T, 1})									# TODO support for different output types is needed
		b = sizeof(T)*8
		r = deepcopy(v)

		for i = (b-1):-1:1, j = 1:size(r, 1)
			if nbit_on(r[j], i)
				r[1] = ibit_range(r[1], i+1, b-1)
			else
				s = ebit_range(r[1], r[j], i+1, b-1)
				r[1] = s[1]
				r[j] = s[2]
			end
			println(r)
		end

		return grayf(cb_merge(r))
	end

	##-----------------------------------------------------------------------------------
	function hilbert_cb{T<:Unsigned}(v::T, d::Integer)
		r = cb_split(grayb(v), d)
		b = sizeof(T)*8

		println(r)

		for i = 1:b, j = d:-1:1
			if nbit_on(r[j], i)
				r[1] = ibit_range(r[1], i+1, b)
			else
				s = ebit_range(r[1], r[j], i+1, b)
				r[1] = s[1]
				r[j] = s[2]
			end
			println(r)
		end

		return r
	end


	##===================================================================================
	##	Sawtooth wave
	##===================================================================================
	export saww

	##-----------------------------------------------------------------------------------
	saww{T<:AbstractFloat}(x::T, a::T, p::T, q::T) = (-2*a/pi)*atan(cot((x-q)*pi/p))	# a = amplitude, p = period, q = phase


	##===================================================================================
	##	Square wave
	##===================================================================================
	export sqw

	##-----------------------------------------------------------------------------------
	sqw{T<:AbstractFloat}(x::T, a::T, p::T, q::T) = a*sign(sin((x-q)*q))				# a = amplitude, p = period, q = phase


	##===================================================================================
	##	Triangle wave
	##===================================================================================
	export triw

	##-----------------------------------------------------------------------------------
	function triw{T<:AbstractFloat}(x::T, a::T, p::T, q::T)
		s1 = 2/p
		s2 = floor(s1*(x+q)+.5)
		return a*2*s1*((x+q)-s1*s2)*(s2 % 2 == 0 ? 1 : -1)
	end


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
	## roots of polynomial
	##===================================================================================
	export rop

	##-----------------------------------------------------------------------------------
	function rop{T<:AbstractFloat}(c::Array{T, 1})
		s = size(c, 1)
		reverse!(c)

		@inbounds for i = s:-1:1
			if c[i] != 0
				break
			end
			s -= 1
		end

		m = shift(s-1, false)
		m[1, :] = (-c[2:end])/c[1]
		return map((x) -> 1/x, eigvals(m))
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
    ## roots of unity
    ##===================================================================================
	export rou

	##-----------------------------------------------------------------------------------
	rou(n::Integer) = [exp((2*pi*i*im)/n) for i = 0:(n-1)]


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
    ## samples
    ##===================================================================================
    export samples

    ##-----------------------------------------------------------------------------------
    function samples{T<:Any}(data::Array{T, 1}, size::Integer)
        L = length(data)
        @assert(size < L, "The number of samples musn't be bigger than the data!")
        return shuffle(getindex(data, sort(sample(1:L, size, replace = false))))
    end


    ##===================================================================================
    ## checks
    ##===================================================================================
    export iszero, levi_civita_tensor, index_permutations_count

    ##-----------------------------------------------------------------------------------
    function iszero(v)
		set_zero_subnormals(true)
		sumabs(v) == 0
	end

    ##-----------------------------------------------------------------------------------
    lecit{T<:Number}(v::Array{T, 1}) = 0 == index_permutations_count(v) % 2 ? 1 : -1

    ##-----------------------------------------------------------------------------------
    function index_permutations_count{T<:Any}(v::Array{T, 1})                           # [3,4,5,2,1] -> [1,2,3,4,5]
        s = size(v, 1)                                                            		# 3 inversions needed
        t = linspace(1, s, s)
		c = 0

		while v != t
            @inbounds for i = 1:size(v, 1)
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
    ## rm column/row
    ##===================================================================================
    export rm, rms

    ##-----------------------------------------------------------------------------------
    function rm{T<:Any}(m::Array{T, 2}, i::Integer, column::Bool = true)
		r = column ? m : m'
        return hcat(r[:, 1:(i-1)], r[:, (i+1):end])
    end

    ##-----------------------------------------------------------------------------------
    function rms{T<:Any}(m::Array{T, 2}, i::Array{Any, 1}, column::Bool = true)
		r = column ? m : m'

		@inbounds for x in i
            r = hcat(r[:, 1:(x-1)], r[:, (x+1):end])
            i .-= 1
        end

		return r
    end

    ##-----------------------------------------------------------------------------------
    function rm{T<:Any}(m::Array{T, 2}, i::Array{Any, 1}, column::Bool = true)
		r = column ? m : m'

		@inbounds for x in sort(i)
            r = hcat(r[:, 1:(x-1)], r[:, (x+1):end])
            i .-= 1
        end

        return r
    end

    ##-----------------------------------------------------------------------------------
    function rm{T<:Any, N<:Integer}(m::Array{T, 2}, ub::N, lb::N, column::Bool = true)
		column::Bool = true
        return hcat(r[:, 1:(lb-1)], r[:, (ub+1):end])
    end


    ##===================================================================================
    ## rm
    ##===================================================================================
    export rm, rms

    ##-----------------------------------------------------------------------------------
    function rm{T<:Any, N<:Int}(v::Array{T, 1}, i::Array{N, 1})
        i = sort(i)

        @inbounds for j=1:length(i)
            v = cat(1, v[1:(i[j]-1)], v[(i[j]+1):end])
            i .-= 1
        end

        return v
    end

    ##-----------------------------------------------------------------------------------
    function rms{T<:Any, N<:Int}(v::Array{T, 1}, i::Array{N, 1})
        @inbounds for j=1:length(i)
            v = cat(1, v[1:(i[j]-1)], v[(i[j]+1):end])
            i .-= 1
        end

        return v
    end


    ##===================================================================================
    ## union overload
    ##===================================================================================
    export union

    ##-----------------------------------------------------------------------------------
    function union{T<:Any}(vl::Array{Array{T, 1}, 1})
        v = vl[1]

        @inbounds for i=2:size(vl, 1)
            v = union(v, vl[i])
        end

		return v
    end


    ##===================================================================================
    ## intersect overload
    ##===================================================================================
    export intersect

    ##-----------------------------------------------------------------------------------
    function intersect{T<:Any}(vl::Array{Array{T, 1}, 1})
        v = vl[1]

        @inbounds for i=2:size(vl, 1)
            v = intersect(v, vl[i])
        end

        return v
    end


    ##===================================================================================
    ## prepend
    ##===================================================================================
    export prepend, prepend!

    ##-----------------------------------------------------------------------------------
    prepend{T<:Any}(v::Array{T, 1}, w) = cat(1, [w], v)

    ##-----------------------------------------------------------------------------------
    prepend{T<:Any}(v::Array{T, 1}, w::Array{T, 1}) = cat(1, w, v)

    ##-----------------------------------------------------------------------------------
    prepend!{T<:Any}(v::Array{T, 1}, w) = v = cat(1, [w], v)

    ##-----------------------------------------------------------------------------------
    prepend!{T<:Any}(v::Array{T, 1}, w::Array{T, 1}) = v = cat(1, w, v)


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
    ## map (overload)
    ##===================================================================================
    export map

    ##===================================================================================
    function map{T<:Any}(f::Function, vl::Array{Array{T, 1}, 1})
        ul = Array{Array{T, 1}, 1}

        @inbounds @simd for i = 1:length(ul)
            push!(ul, f(ul[i]))
        end

		return ul
    end


	##===================================================================================
    ## min (overload)
    ##===================================================================================
	export min

	##-----------------------------------------------------------------------------------
	min{T<:Number}(x::Tuple{T, T}) = x[1] < x[2] ? x[1] : x[2]


	##===================================================================================
    ## max (overload)
    ##===================================================================================
	export max

	##-----------------------------------------------------------------------------------
	max{T<:Number}(x::Tuple{T, T}) = x[1] > x[2] ? x[1] : x[2]


    ##===================================================================================
    ## apply
    ##===================================================================================
    export apply, apply_parallel, apply_parallel_shared, apply_tri_upper, apply_tri_lower

    ##-----------------------------------------------------------------------------------
    function apply(g::Function, m)
        @inbounds for i in eachindex(m)
            m[i] = g(m[i])
        end

        return m
    end

    ##-----------------------------------------------------------------------------------
    function apply_p(g::Function, m)
        m = convert(SharedArray, m)

		@inbounds @sync @parallel for i in eachindex(m)
            m[i] = g(m[i])
        end

        return convert(Array, m)
    end

    ##-----------------------------------------------------------------------------------
    function apply_ps(g::Function, m)
        @inbounds @sync @parallel for i in eachindex(m)
            m[i] = g(m[i])
        end

        return m
    end

    ##------------------------------------------------------------------------------------
    function apply_tri_upper{T<:Number}(g::Function, m::Array{T, 2})
        @inbounds for j = 2:size(m, 2), i = 1:j-1
            m[i, j] = g(m[i, j])
        end

        return m
    end

    ##------------------------------------------------------------------------------------
    function apply_tri_lower{T<:Number}(g::Function, m::Array{T, 2})
        @inbounds for i = 2:size(m, 2), j = 1:i-1
            m[i, j] = g(m[i, j])
        end

        return m
    end

	##-----------------------------------------------------------------------------------
	function apply_dia{T<:Number}(g::Function, m::Array{T, 2})
		for i = min(size(m))
			m[i, i] = g(m[i, i])
		end

		return m
	end
end
