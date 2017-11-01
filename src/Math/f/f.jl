@everywhere module f
    ##===================================================================================
    ##  using Directives
    ##===================================================================================
	using StatsBase
	using cnv
    using op


    ##===================================================================================
    ##  types
    ##===================================================================================
    type t_ncbd																			# n dimensional cuboid
		alpha::Array{Float64, 1}														# infimum 									(point)
		delta::Array{Float64, 1}														# diference between supremum and infimum	(s-i)
		n::Int64																		# n
	end


    ##===================================================================================
    ## hyperbolic tangent
    ##===================================================================================
    export tanh, d_tanh

    ##-----------------------------------------------------------------------------------
    tanh(x, eta) = @. Base.tanh(x-eta)

    ##-----------------------------------------------------------------------------------
    d_tanh(x, eta = 0) = @. 1-(tanh(x, eta)^2)


	##===================================================================================
	## std (standart deviation (overload))
	##		m = median
	##===================================================================================
	import Base.std
	export std

    ##-----------------------------------------------------------------------------------
    function std(v::Array{Float64, 1}, l::Int64, n::Int64 = 1)
        m = Float64(0)

        @inbounds for i = 1:n:(n*l)
            m = m + v[i]
        end

    	return sqrt(BLAS.dot(l, v, n, v, n)-(m/l)^2)
    end

    ##-----------------------------------------------------------------------------------
    function std(v::Array{Float64, 1}, m::Float64, l::Int64, n::Int64 = 1)
    	s = size(v, 1)
    	return sqrt(BLAS.dot(l, v, n, v, n)-m^2)
    end


    ##===================================================================================
    ##  householder reflection
    ##      reflects v about a hyperplane given by u
    ##===================================================================================
    export hh_rfl, hh_mat

    ##-----------------------------------------------------------------------------------
    hh_rfl(v::Array{Float64, 1}, u::Array{Float64, 1}) = u-(v*(2.0*bdot(u, v)))         # hh reflection (u = normal vector)

    ##-----------------------------------------------------------------------------------
    function hh_mat(v::Array{Float64, 1})
        s = size(v, 1)
        m = Array{Float64, 2}(s, s)

        for i=1:s, j=1:s
            if i == j
                m[i, j]= 1-(2*v[i]*v[j])
            else
                m[i, j]= -2*v[i]*v[j]
            end
        end

        return m
    end


    ##===================================================================================
    ##  QR (faster than vanilla)
    ##===================================================================================
    export qrd_sq, qrd

    ##-----------------------------------------------------------------------------------
    function qrd_sq(m::Array{Float64, 2})
    	s = size(m, 1)
    	t = zeros(s, s)
    	v = zeros(s)
    	r = copy(m)
    	w = Float64(0)

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
	function qrd(m::Array{Float64, 2})
		s = size(m, 1)
		t = zeros(s, s)
		v = zeros(s)
		r = copy(m)
		w = 0.

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
    function ul_x_expand(m::Array{Float64, 2}, s::Tuple{Int64, Int64}, x::Float64 = 1.0)# ul = upper left
        d = (s[1]-size(m, 1), s[2]-size(m,2))
        r = zeros(s)

        for i = 1:s[1], j = 1:s[2]
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
    function minor(m::Array{Float64, 2}, p::Tuple{Int64, Int64} = (1, 1))
        s = size(m)
        r = Array{Float64, 2}(s[1]-p[1], s[2]-p[1])

        for i=(1+p[1]):s[1], j=(1+p[2]):s[2]
            r[i-p[1], j-p[2]] = m[i, j]
        end

        return r
    end


	##===================================================================================
	##	generalized mean
	##===================================================================================
	export gnmn, gnmn_n1, gnmn_0

	##-----------------------------------------------------------------------------------
	function gnmn(v::Array{Float64, 1}, p::Int64, l::Int64, n::Int64)
		u = Float64(0)

		for i = 1:n:l u += v[i]^p end
		u = u / floor(l/n)

		return u^(1/p)
	end

	##-----------------------------------------------------------------------------------
	function gnmn_n1(v::Array{Float64, 1}, l::Int64, n::Int64)							# harmonic mean
		u = Float64(0)

		for i = 1:n:l
            u += 1/v[i]
        end

		return (floor(l/n)/u)
	end

	##-----------------------------------------------------------------------------------
	function gnmn_0(v::Array{Float64, 1}, l::Int64, n::Int64)							# geometric mean
		u = v[1]

		for i = 1+n:n:l
			u = u * v[i]
		end

		return u^(1/floor(l/n))
	end


    ##===================================================================================
    ##  outer product implementation (faster than vanila)
    ##===================================================================================
    export otr

    ##-----------------------------------------------------------------------------------
    function otr(v::Array{Float64, 1}, w::Array{Float64, 1})
        s = (size(v, 1), size(w, 1))
        m = Array{Float64, 2}(s)

        @inbounds for i=1:s[1], j=1:s[2]
            m[i, j]=v[i]*w[j]
        end

        return m
    end


    ##===================================================================================
    ##  boolean operators
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
    function /(x::Float64, v::Array{Float64,1})
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
    function var(v::Array{Float64, 1}, m::Float64)                                      # faster implementation
        l = size(v, 1)
        return (soq(l, v)/l) - (m^2)
    end

    ##-----------------------------------------------------------------------------------
    function var(v::Array{Float64, 1})                                                  # faster implementation
        l = size(v, 1)
        return (soq(l, v)/l) - (vmed(l, v)^2)
    end

    ##-----------------------------------------------------------------------------------
    var(v::Array{Float64, 1}, m::Float64, l::Int64) = (soq(l, v)/l) - (m^2)

    ##-----------------------------------------------------------------------------------
    var(v::Array{Float64, 1}, l::Int64) = (soq(l, v)/l) - (vmed(l, v)^2)


    ##===================================================================================
    ## BLAS wrapper
    ##===================================================================================
    export bdot, bdotu, bdotc, bnrm

    ##-----------------------------------------------------------------------------------
    bdot(l::Int64, v::Array{Float64, 1}, u::Array{Float64, 1}) = BLAS.dot(l, v, 1, u, 1)            # l = length of the vectors

    ##-----------------------------------------------------------------------------------
    bdot(v::Array{Float64, 1}, u::Array{Float64, 1}) = BLAS.dot(size(v, 1), v, 1, u, 1)

    ##-----------------------------------------------------------------------------------
    bdotu(l::Int64, v::Array{Complex128, 1}, u::Array{Complex128, 1}) = BLAS.dotu(l, v, 1, u, 1)    # l = length of the vectors

    ##-----------------------------------------------------------------------------------
    bdotu(v::Array{Complex128, 1}, u::Array{Complex128, 1}) = BLAS.dotu(size(v, 1), v, 1, u, 1)

    ##-----------------------------------------------------------------------------------
    bdotc(l::Int64, v::Array{Complex128, 1}, u::Array{Complex128, 1}) = BLAS.dotc(l, v, 1, u, 1)    # l = length of the vectors

    ##-----------------------------------------------------------------------------------
    bdotc(v::Array{Complex128, 1}, u::Array{Complex128, 1}) = BLAS.dotc(size(v, 1), v, 1, u, 1)

    ##-----------------------------------------------------------------------------------
    bnrm(l::Int64, v::Array{Float64, 1}) = BLAS.nrm2(l, v, 1)                                      # l = length of the vector

    ##-----------------------------------------------------------------------------------
    bnrm(v::Array{Float64, 1}) = BLAS.nrm2(size(v, 1), v, 1)


    ##===================================================================================
    ##  soq (sum of squares)
    ##===================================================================================
    export soq, soqu, soqc

    ##-----------------------------------------------------------------------------------
    soq(l::Int64, v::Array{Float64, 1}, n::Int64 = 1) = BLAS.dot(l, v, n, v, n)

    ##-----------------------------------------------------------------------------------
    soq(v::Array{Float64, 1}) = bdot(v, v)

    ##-----------------------------------------------------------------------------------
    soqu(l::Int64, v::Array{Complex128, 1}, n::Int64 = 1) = BLAS.dot(l, v, n, v, n)

    ##-----------------------------------------------------------------------------------
    soqu(v::Array{Complex128, 1}) = bdotu(v, v)

    ##-----------------------------------------------------------------------------------
    soqc(l::Int64, v::Array{Complex128, 1}, n::Int64 = 1) = bdotc(l, v, n, v, n)

    ##-----------------------------------------------------------------------------------
    soqc(v::Array{Complex128, 1}) = bdotc(v, v)


    ##===================================================================================
    ##  collatz conjecture
    ##===================================================================================
    export collatz

    ##-----------------------------------------------------------------------------------
    function collatz(x::Int64)
        c = 0

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
    function veconomy_core(v::Array{Float64, 1}, cc::Float64 = 0.4)
        lumV = norm(v) / MAX_LUM
        o = prison(rotation_matrix(rand_orthonormal_vec(v), 90)*(v-127.5), -127.5, 127.5)
        while abs(lumv-(norm(o)/MAX_LUM)) < cc; o = map((x) -> x*(lumV>0.5?.5:1.5), o) end
        return map((x) -> prison(round(x), 0, 255), o)
    end


    ##===================================================================================
    ##  dirac delta/impulse
    ##===================================================================================
    export dcd

    ##-----------------------------------------------------------------------------------
    function dcd(x::Float64)
        set_zero_subnormals(true)
        return x == 0 ? Inf : 0
    end


    ##===================================================================================
    ## kronecker delta
    ##===================================================================================
    export ked

    ##-----------------------------------------------------------------------------------
    ked(x, y) = x == y ? 1 : 0


	##===================================================================================
	##	kronecker product
	##===================================================================================
	export kep

	##-----------------------------------------------------------------------------------
	function kep(m::Array{Float64, 2}, n::Array{Float64, 2})
		sm = size(m); sn = size(n)
		r = zeros(sm[1]*sm[1], sn[1]*sn[1])
		px = 0; py = 0

		for i = 1:sm[1], j = 1:sm[2]

			px = (i-1)*sm[1]
			py = (j-1)*sm[2]

			for k = 1:sn[1], l = 1:sn[2]
				r[(px+k), (py+l)] = m[i,j]*n[k,l]
			end
		end

		return r
	end


	##===================================================================================
	##	Cofactor Matrix of a Matrix
	##===================================================================================
	export cof

	##-----------------------------------------------------------------------------------
	function cof(m::Array{Float64, 2})													# TODO needs better performance
		s = size(m)

		n = zeros(s[1]-1, s[2]-1)
		r = zeros(s)

		for i = 1:s[1], j = 1:s[2]
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
	##	Adjungate of a Matrix
	##===================================================================================
	export adj

	##-----------------------------------------------------------------------------------
	adj(m::Array{Float64, 2}) = return cof(m)'


	##===================================================================================
	##	Nullspace of a matrix
	##===================================================================================
	export nul

	##-----------------------------------------------------------------------------------
	function nul(m::Array{Float64, 2})
		s = size(m)

		if s[1] >= s[2]
			return zeros(s[2])
		end

		r = zeros(s[2], s[2]-s[1])

		for i = 1:s[1]
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
	king(p::Float64, ca::Float64, cs::Float64, c::Int64) = ((p^(sqrt(2*(c+1))))/(1-p))x*(0.5)*((ca^2)+(cs^2))


    ##===================================================================================
    ## sigmoid
    ##===================================================================================
    export sigmoid, d_sigmoid

    ##-----------------------------------------------------------------------------------
    sigmoid(x, eta = 0) = @. 1/(1+exp(-(x-eta)))

    ##-----------------------------------------------------------------------------------
    d_sigmoid(x, eta = 0) =  @. sigmoid(x, eta) * (1-sigmoid(x, eta))


    ##===================================================================================
    ## norm derivate
    ##===================================================================================
    export norm_d

    ##-----------------------------------------------------------------------------------
    nrom_d(v::Array{Float64, 1}, p::Int64 = 2) = @. sign(v)*(abs(v)/ifelse(iszero(v), 1, norm(v, p)))^(p-1)


    ##===================================================================================
    ## radial basis functions
    ##===================================================================================
    export rbf_gaussian, rbf_gaussian_d_lambda, rbf_gaussian_d_delta, rbf_triang, rbf_cos_decay,
        rbf_multi_quad, rbf_inv_multi_quad, rbf_inv_quad, rbf_poly_harm, rbf_thin_plate_spline

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
    rbf_multi_quad(delta, lambda::Float64 = 1) = @. sqrt(1+(lambda*delta)^2)

    ##-----------------------------------------------------------------------------------
    rbf_inv_multi_quad(delta, lambda::Float64 = 1) = @. 1/sqrt(1+(lambda*delta)^2)

    ##-----------------------------------------------------------------------------------
    rbf_inv_quad(delta, lambda::Float64 = 1) = @. 1/(1+(lambda*delta)^2)

    ##-----------------------------------------------------------------------------------
    rbf_poly_harm(delta, expt::Float64 = 2) = delta.^expt

    ##-----------------------------------------------------------------------------------
    rbf_thin_plate_spline(delta, expt::Float64 = 2) = @. (delta^expt)*log(delta)


    ##===================================================================================
    ## ramp
    ##===================================================================================
    export ramp, d_ramp

    ##-----------------------------------------------------------------------------------
    ramp(x, eta) = @. max(0, x-eta)

    ##-----------------------------------------------------------------------------------
    d_ramp(x::Float64, eta::Float64) = x-eta > 0 ? 1. : 0.

    ##-----------------------------------------------------------------------------------
    d_ramp(x::Array{Float64, 1}, eta::Float64) = AND(x.-eta .> 0) ? ones(x) : zeros(x)

    ##-----------------------------------------------------------------------------------
    function d_ramp(x::Array{Float64, 1}, eta::Array{Float64, 1})
        r = zeros(x)

        for i = 1:size(x, 1)
            if x[i]-eta[i] > 0
                r[i] = 1.
            end
        end

        return r
    end

    # TODO continue refactoring from here

    ##===================================================================================
    ## semi linear
    ##===================================================================================
    export semi_lin, d_semi_lin

    ##-----------------------------------------------------------------------------------
    semi_lin(x, eta, sigma = 0.5) = prison(x, (x) -> x-eta+sigma, eta-sigma, eta+sigma)

    ##-----------------------------------------------------------------------------------
    d_semi_lin(x, eta, sigma = 0.5) = ifelse(x > eta+sigma || x < eta-sigma, 0., 1.)


    ##===================================================================================
    ## semi linear
    ##===================================================================================
    export sine_saturation, d_sine_saturation

    ##-----------------------------------------------------------------------------------
    sine_saturation(x, eta, sigma = pi/2) = prison(x, (x) -> (sin(x-eta)+1)/2, eta-sigma, eta+sigma)

    ##-----------------------------------------------------------------------------------
    d_sine_saturation(x, eta, sigma = pi/2) = ifelse(x > eta+sigma || x < eta-sigma, 0, cos(x-eta)/2)


    ##===================================================================================
    ## softplus
    ##===================================================================================
    export softplus, d_softplus

    ##-----------------------------------------------------------------------------------
    softplus(x, eta) = log(1+exp(x-eta))

    ##-----------------------------------------------------------------------------------
    d_softplus(x, eta) = 1/(exp(x-eta)+1)


    ##===================================================================================
    ## orthogonal projection (only returning the projection matrices)
    ##===================================================================================
    export op, op_gen

    ##-----------------------------------------------------------------------------------
    op{T<:Real}(b::Array{T, 2}, complex = false) = b*(complex ? b.' : b')               # b is a orthonormal basis

    ##-----------------------------------------------------------------------------------
    op{T<:Real}(b::Array{Array{T, 1}, 1}) = op(vl_to_mat(b), T<:Complex)                # b is a orthonormal basis

    ##-----------------------------------------------------------------------------------
    function op_gen{T<:Real}(b::Array{T, 2}, complex = false)                           # b mustn't be a orthonormal basis
        bt = ifelse(complex, bt.', bt')
        return b\(bt*b)*bt
    end

    ##-----------------------------------------------------------------------------------
    op_gen{T<:Number}(b::Array{Array{T, 1}, 1}) = op_gen(vl_to_mat(b), T<:Complex)


    ##===================================================================================
    ## mutual incoherence
    ##===================================================================================
    export mut_incoherent

    ##-----------------------------------------------------------------------------------
    function mut_incoherent{T<:Number}(m::Array{T, 2}, rows = true, p::Int = 2)         # the lower the better the mutual incoherence property
        inf = 0; m = rows ? m : m'
        for x = 2:size(m, 1), y = 1:(x-1)
            inf = max(norm(bdot(m[x, :], m[y, :]), p), inf)
        end
        return inf
    end

    ##-----------------------------------------------------------------------------------
    function mut_incoherent{T<:Number}(vl::Array{Array{T, 1}}, p::Int = 2)
        inf = 0
        for x = 2:lenght(vl), y = 1:(x-1)
            inf = max(norm(bdot(m[x], m[y]), p), inf)
        end
        return inf
    end


    ##===================================================================================
    ## step
    ##===================================================================================
    export step, d_step

    ##-----------------------------------------------------------------------------------
    step(x, eta = 0.5) = ifelse(x >= eta, 1, 0)

    ##-----------------------------------------------------------------------------------
    d_step{T<:Any}(x::T, eta) = ifelse(x == eta, typemax(T), 0)


    ##===================================================================================
    ## trigonometric
    ##===================================================================================
    export sin2, cos2, versin, aversin, vercos, avercos, coversin, acoversin, covercos, acovercos,
        havsin, ahavsin, havcos, ahavcos, hacoversin, hacovercos

    ##-----------------------------------------------------------------------------------
    sin2(alpha) = sin(alpha)^2

    ##-----------------------------------------------------------------------------------
    cos2(alpha) = cos(alpha)^2

    ##-----------------------------------------------------------------------------------
    versin(alpha) = 1-cos(alpha)

    ##-----------------------------------------------------------------------------------
    aversin(alpha) = acos(1-alpha)

    ##-----------------------------------------------------------------------------------
    vercos(alpha) = 1+cos(alpha)

    ##-----------------------------------------------------------------------------------
    avercos(alpha) = acos(1+alpha)

    ##-----------------------------------------------------------------------------------
    coversin(alpha) = 1-sin(alpha)

    ##-----------------------------------------------------------------------------------
    acoversin(alpha) = asin(1-alpha)

    ##-----------------------------------------------------------------------------------
    covercos(alpha) = 1+sin(alpha)

    ##-----------------------------------------------------------------------------------
    acovercos(alpha) = asin(1+alpha)

    ##-----------------------------------------------------------------------------------
    havsin(alpha) = versin(alpha)/2

    ##-----------------------------------------------------------------------------------
    ahavsin(alpha) = 2*asin(sqrt(alpha))

    ##-----------------------------------------------------------------------------------
    havcos(alpha) = vercos(alpha)/2

    ##-----------------------------------------------------------------------------------
    ahavcos(alpha) = 2*acos(sqrt(alpha))

    ##-----------------------------------------------------------------------------------
    hacoversin(alpha) = coversin(alpha)/2

    ##-----------------------------------------------------------------------------------
    hacovercos(alpha) = covercos(alpha)/2


    ##===================================================================================
    ## angle
    ##===================================================================================
    export angle, acos_central_angle, asin_central_angle, atan_central_angle, central_angle,
        haversine_central_angle, vincenty_central_angle

    ##-----------------------------------------------------------------------------------
    angle(u::Array{Float64, 1}, v::Array{Float64, 1}, bias = .0) = acosd((abs(bdot(v, u))/(bnrm(v)*bnrm(u)))+bias)

    ##-----------------------------------------------------------------------------------
    acos_central_angle(u::Array{Float64, 1}, v::Array{Float64, 1}) = acos(bdot(u,v))             # returns radians | u&v = normal vectors on the circle

    ##-----------------------------------------------------------------------------------
    asin_central_angle(u::Array{Float64, 1}, v::Array{Float64, 1}) = asin(bnrm(cross(u, v)))     # returns radians | u&v = normal vectors on the circle

    ##-----------------------------------------------------------------------------------
    atan_central_angle(u::Array{Float64, 1}, v::Array{Float64, 1}) = atan(bnrm(cross(u, v))/bdot(u, v))  # returns radians | u&v = normal vectors on the circle

    ##-----------------------------------------------------------------------------------
    central_angle(pla, plo, sla, slo) = acos((sin(pla)*sin(sla))+(cos(pla)*cos(sla)*cos(abs(plo-slo)))) # returns radians | pla/sla = primary/secondary latitude / plo/slo = primary/secondary longitude

    ##-----------------------------------------------------------------------------------
    haversine_central_angle(pla, plo, sla, slo) = 2*asin(sqrt(havsin(abs(pla-sla))+cos(pla)*cos(sla)*havsin(abs(plo-slo)))) # returns radians | pla/sla = primary/secondary latitude / plo/slo = primary/secondary longitude

    ##-----------------------------------------------------------------------------------
    function vincenty_central_angle(pla, plo, sla, slo)
        longitude_delta = abs(plo-slo)                                                                                      # returns radians | pla/sla = primary/secondary latitude / plo/slo = primary/secondary longitude
        return atan2(sqrt((cos(sla)*sin(longitude_delta))^2+((cos(pla)*sin(sla))-(sin(pla)*cos(sla)*cos(longitude_delta)))^2), (sin(pla)*sin(sla)+cos(pla)*cos(sla)*cos(longitude_delta)))
    end

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
        n = size(arr, (column ? 1 : 2))
        return BLAS.gemv((column ? 'N' : 'T'), 1/n, arr, ones(n))
    end

    ##-----------------------------------------------------------------------------------
    function mmed{T<:Float64}(arr::Array{T, 2}, weights::Array{T, 1}, column::Bool = true)
        n = size(X, (column ? 1 : 2))
        return BLAS.gemv((column ? 'N' : 'T'), 1/n, weights.*arr, ones(n))
    end

    ##-----------------------------------------------------------------------------------
    vmed(l::Int64, v::Array{Float64, 1}) = BLAS.dot(l, v, 1, [1.0], 0)/l

    ##-----------------------------------------------------------------------------------
    function vmed(v::Array{Float64, 1})
        l = size(v, 1)
        return bdot(l, v, ones(l))/l
    end


    ##===================================================================================
    ## covariance
    ##===================================================================================
    export cov

    ##-----------------------------------------------------------------------------------
    function cov(l::Int64, x::Array{Float64, 1}, n1::Int64, y::Array{Float64, 1}, n2::Int64)
        m1 = Float64(0)
        m2 = Float64(0)

        for i = 1:n1:(n1*l)
            m1 = m1 + x[i]
        end

        for i = 1:n2:(n2*l)
            m2 = m2 + y[i]
        end

        m1 = m1/l
        m2 = m2/l

        return (BLAS.dot(l, x, n1, y, n2)/l)-(m1*m2)
    end


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
    function ccovs(v::Array{Float64, 1}, u::Array{Float64, 1}, t::Int64 = 1)    		# ccov sumed
        return BLAS:dot((l-t), (v-vmed(v)), 1, (circshift(u, t)-vmed(u)), 1)/(l-t)
    end


    ##===================================================================================
    ## cross correlation (with delay)
    ##===================================================================================
    export ccor

    ##-----------------------------------------------------------------------------------
    ccor(v::Array{Float64, 1}, u::Array{Float64, 1}, t::Int64 = 1) = ccovs(v, u, t)./(std(v)*std(u))

	##-----------------------------------------------------------------------------------
	mcor(v::Array{Float64, 1}, u::Array{Float64, 1}, t::Int64 = 1) = ccov(v, u, t)./(std(v)*std(u))


    ##===================================================================================
    ## supp (support)
    ##===================================================================================
    export supp

    ##-----------------------------------------------------------------------------------
    function supp(v::Array{Float64, 1}, f::Function = (x) -> x)
        set_zero_subnormals(true)
        u = Array{Float64, 1}

        @inbounds for i = 1:size(v, 1)
			if abs(f(x)) == 0
				push!(u, v[i])
			end
		end
        return u
    end

    ##-----------------------------------------------------------------------------------
    function supp(vl::Array{Array{T, 1}, 1}, f::Function = (x) -> x)		# supp for vector lists
        ul = Array{Array{Float64, 1}, 1}
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
	shai(v::Array{Float64, 1}, l::Int64, n::Int64) = BLAS.dot(l, v, n, Int64(floor(l/n)), log.(v[1:n:l]), 1)


	##===================================================================================
	##	Gini–Simpson index
	##===================================================================================
	export gishi

	##-----------------------------------------------------------------------------------
	gishi(v::Array{Float64, 1}, l::Int64, n::Int64) = 1 - BLAS.dot(l, v, n, l, v, n)


	##===================================================================================
	##	Renyi Entropy
	##===================================================================================
	export reepy

	##-----------------------------------------------------------------------------------
	function reepy(v::Array{Float64, 1}, p::Int64, l::Int64, n::Int64)
		u = v[i]

		for i = 1+n:n:l
			u += v[i]^p
		end

		return log(u)/(1-p)
	end


	##===================================================================================
	##	Sawtooth wave
	##===================================================================================
	export saww

	##-----------------------------------------------------------------------------------
	saww(x::Float64, p::Float64) = 2*((t/p)-floor(.5+(t/p)))							# p = period

	##-----------------------------------------------------------------------------------
	saww(x::Float64, p::Float64, a::Float64) = (-2*a/pi)*atan(cot(x*pi/p))				# a = amplitude


	##===================================================================================
	##	Square wave
	##===================================================================================
	export sqw

	##-----------------------------------------------------------------------------------
	sqw(x::Float64, a::Float64, p::Float64, q::Float64) = a*sign(sin((x+q)*q))			# a = amplitude, p = period, q = phase


	##===================================================================================
	##	Triangle wave
	##===================================================================================
	export triw

	##-----------------------------------------------------------------------------------
	function triw(x::Float64, a::Float64, p::Float64, q::Float64)
		s1 = 2/p
		s2 = floor(s1*(x+q)+.5)

		return a*2*s1*((x+q)-s1*s2)*(s2 % 2 == 0 ? 1 : -1)
	end


    ##===================================================================================
    ## random stochastic matrix
    ##===================================================================================
    export rand_sto_mat

    ##-----------------------------------------------------------------------------------
    function rand_sto_mat(sx::Int, sy::Int)
        m = APL.rand_sto_vec(sy)'

        for i = 2:sx
			m = vcat(m, APL.rand_sto_vec(sy)')
		end

        return m
    end

    ##-----------------------------------------------------------------------------------
    function rand_sto_mat(s::Int)
        return rand_sto_mat(s, s)
    end


    ##===================================================================================
    ## random vl
    ##===================================================================================
    export vl_rand

    ##-----------------------------------------------------------------------------------
    function vl_rand(l::Int64, w::Int64)
        vl = Array{Any, 1}
        for i = 1:l push!(vl, rand(w)) end
        return vl
    end

    ##-----------------------------------------------------------------------------------
	function vl_rand(ncbd::t_ncbd, l::Int64)
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
    function samples{T<:Any}(data::Array{T, 1}, size::Int)
        L = length(data)
        @assert size < L ["The number of samples musn't be bigger than the data!"]
        return shuffle(getindex(data, sort(sample(1:L, size, replace = false))))
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
    #export map

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
