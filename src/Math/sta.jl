@everywhere module sta
	##===================================================================================
	## using directives
	##===================================================================================
	using LinearAlgebra.BLAS
	using SpecialFunctions
	using Distributed

	
	##===================================================================================
	## bernoulli trial probability
	##===================================================================================
	export bernoulli

	##-----------------------------------------------------------------------------------
	function bernoulli(n::Z, k::Z, p::R) where R<:Real where Z<:Integer
		return comb(n, k) * (p^k) * ((1 - p)^(n - k))
	end


	##===================================================================================
	## c4 correction for the standard deviation of normally distributed data
	##	n = 343 is the current upper computable limit
	##===================================================================================
	export c4

	##-----------------------------------------------------------------------------------
	function c4(n::Z) where Z<:Integer
		@assert(n > 1, "n has to be greater than 1")
		a = (n - 1) / 2
		
		return (a^-.5) * (gamma(n / 2) / gamma(a))
	end


	##===================================================================================
	## number of combinations (selecting k out of n)
	##===================================================================================
	export comb

	##-----------------------------------------------------------------------------------
	function comb(n::Z, k::Z) where Z<:Integer
		k = k > floor(n / 2) ? n - k : k
		return perm(n, k)/factorial(k)
	end


	##===================================================================================
	## number of permutations (selecting k out of n)
	##	- rep = array where each element indicates gow often an element is present in the
	##			set that will be permuted
	##===================================================================================
	export perm, permr

	##-----------------------------------------------------------------------------------
	function perm(n::Z, k::Z) where Z<:Integer
		i = n - k + 1
		r = Z(1)

		@inbounds while i <= n
			r = r * i
			i = i + 1
		end

		return r
	end

	##-----------------------------------------------------------------------------------
	function permr(n::Z, rep::Array{Z, 1}) where Z<:Integer
		l = size(rep, 1)
		i = 1
		a = 1
		b = 1

		@inbounds while i <= n
			b = b * (i > l ? 1 : factorial(rep[i]))
			a = a * i
			i = i + 1
		end
	
		return Z(a/b)
	end 


	##===================================================================================
	## std (standart deviation (overload))
	##		m = median
	##===================================================================================
	export xstd

	##-----------------------------------------------------------------------------------
	function xstd(v::Array{R, 1}, l::Z, n::Z = 1) where R<:AbstractFloat where Z<:Integer
		s = n * l
		m = R(0)

		@assert(s <= size(v, 1), "out of bounds error")
		@inbounds for i = 1:n:(n*l)
			m = m + v[i]
		end

		return sqrt(BLAS.dot(l, v, n, v, n)-(m/l)^2)
	end

	##-----------------------------------------------------------------------------------
	function std(v::Array{R, 1}, m::R, l::Z, n::Z = 1) where R<:AbstractFloat where Z<:Integer 
		return sqrt(BLAS.dot(l, v, n, v, n) - m^2)
	end


	##===================================================================================
	## number of derangements
	##===================================================================================
	export derange

	##-----------------------------------------------------------------------------------
	function derange(x::Z) where Z<:Integer
		return round(AbstractFloat(factorial(x))/e)
	end


	##===================================================================================
	## mad (median average deviation)
	##===================================================================================
	export mad

	##-----------------------------------------------------------------------------------
	function mad(v::Array{R, 1}, l::Z, inx::Z) where R<:AbstractFloat where Z<:Integer
		m = median(v)
		r = zeros(l)
		s = inx * l
		j = Z(1)

		@assert(s <= size(v, 1), "out of bonds error");
		@inbounds for i = 1:inx:(inx*l)
			r[j] = v[i]
			j += 1
		end

		return median(r)
	end

	##-----------------------------------------------------------------------------------
	mad(v::Array{T, 1}) where T<:AbstractFloat = mad(v, size(v, 1), 1)


	##===================================================================================
	## poission distribution
	##	l needs to be positive 
	##===================================================================================
	export pois

	##-----------------------------------------------------------------------------------
	pois(k::T, l::T) where T<:Number = exp(k*log(l)-l-lgamma(k+1))


	##===================================================================================
	## nth order difference
	##	t: time series
	## 	d: delay
	##	l: size of the observation
	## 	inc: index increment
	##===================================================================================
	export nod

	##-----------------------------------------------------------------------------------
	function nod(t::Array{R, 1}, p::Z, l::Z, inc::Z = 1) where R<:AbstractFloat where Z<:Integer
		r = zeros(l)

		for i = (p+1):inc:(p+l)
			r[i-p] = t[i] - t[i-p]
		end

		return r
	end


	##===================================================================================
	##  z score
	##===================================================================================
	export zscore

	##-----------------------------------------------------------------------------------
	zscore(v::Array{R, 1}) where R<:AbstractFloat = (v-(sum(v)/size(v, 1)))/std(v)


	##===================================================================================
	## dice sampling
	##	n must be positive	
	##===================================================================================
	export w6, nw6

	##-----------------------------------------------------------------------------------
	w6() = Int(ceil(rand()*6))

	##-----------------------------------------------------------------------------------
	nw6(n::Z) where Z<:Integer = sum([w6() for i = 1:n])


	##===================================================================================
	## auto regressive model
	##===================================================================================
	export ar

	##-----------------------------------------------------------------------------------
	function ar(q::Z, v::Array{R, 1}) where R<:AbstractFloat where Z<:Integer
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
	## difference operator
	##===================================================================================
	export diff, ndiff

	##-----------------------------------------------------------------------------------
	function diff(t::Array{R, 1}, tau::Z = 1) where R<:AbstractFloat where Z<:Integer
		l = size(t, 1)
		r = zeros(R, l)

		@inbounds for i = (tau+1):l
			r[i] = t[i] - t[i-tau]
		end

		return r
	end

	##-----------------------------------------------------------------------------------
	function ndiff(t::Array{R, 1}, tau::Z = 1, ord::Z = 1) where R<:AbstractFloat where Z<:Integer
		l = size(t, 1)
		r0 = deepcopy(t)
		r1 = zeros(R, l)

		@inbounds for j = 1:ord
			for i = (tau+1):l
				r1[i] = r0[i] - r0[i-tau]
			end
			r0 = deepcopy(r1)
		end

		return r0
	end

	##===================================================================================
	## [augumented] dickey fuller test for stationarity
	##===================================================================================
	export difut, adifut

	##-----------------------------------------------------------------------------------
	difut(v::Array{R, 1}, p::R = .01) where R<:AbstractFloat = ar(1, v)[2] <= p

	##-----------------------------------------------------------------------------------
	difut(v::Array{R, 1}, p::R, d::R) where R<:AbstractFloat = ar(1, v-[d*x for x = 1:size(v, 1)])[2] <= p

	##-----------------------------------------------------------------------------------
	difut(v::Array{R, 1}, p::R, d::R, t::R) where R<:AbstractFloat = ar(1, v-[(d*x)+(t*sum(x)) for x = 1:size(v, 1)])[2] <= p

	##-----------------------------------------------------------------------------------
	function adifut(v::Array{R, 1}, q::Z, p::R = .01) where R<:AbstractFloat where Z<:Integer
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
	adifut(v::Array{R, 1}, q::Z, p::R, d::Z) where R<:AbstractFloat where Z<:Integer = adifut((v-[d*x for x = 1:size(v, 1)]), q, p)

	##-----------------------------------------------------------------------------------
	adifut(v::Array{R, 1}, q::Z, p::R, d::R, t::R) where R<:AbstractFloat where Z<:Integer = adifut((v-[(d*x)+(t*sum(x)) for x = 1:size(v, 1)]), q, p)


	##===================================================================================
	## angle granger test for cointegration
	##===================================================================================
	export angrat, aangrat

	##-----------------------------------------------------------------------------------
	function angrat(x::Array{R, 1}, y::Array{R, 1}, p::R = .01) where R<:AbstractFloat 
		return difut(y - (((soq(x))\dot(x, y))*x), p)
	end

	##-----------------------------------------------------------------------------------
	function angrat(x::Array{R, 1}, y::Array{R, 1}, p::R, d::R) where R<:AbstractFloat
		return difut(y - (((soq(x))\dot(x, y))*x), p, d)
	end

	##-----------------------------------------------------------------------------------
	function aangrat(x::Array{R, 1}, y::Array{R, 1}, p::R, d::R, t::R) where R<:AbstractFloat 
		return difut(y - (((soq(x))\dot(x, y))*x), p, d, t)
	end

	##-----------------------------------------------------------------------------------
	function aangrat(x::Array{R, 1}, y::Array{R, 1}, q::Z, p::R = .01) where R<:AbstractFloat where Z<:Integer
		return difut(y - (((soq(x))\dot(x, y))*x), q, p)
	end

	##-----------------------------------------------------------------------------------
	function aangrat(x::Array{R, 1}, y::Array{R, 1}, q::Z, p::R, d::R) where R<:AbstractFloat where Z<:Integer
		return difut(y - (((soq(x))\dot(x, y))*x), q, p, d)
	end 

	##-----------------------------------------------------------------------------------
	function aangrat(x::Array{R, 1}, y::Array{R, 1}, q::Z, p::R, d::R, t::R) where R<:AbstractFloat where Z<:Integer 
		return difut(y - (((soq(x))\dot(x, y))*x), q, p, d, t)
	end


	##===================================================================================
    ## mutual incoherence
	##	the lower the value the better
    ##===================================================================================
    export mut_inch

    ##-----------------------------------------------------------------------------------
    function mut_inch(m::Array{T, 2}, rows::Bool = true, p::Z = 2) where T<:Number where Z<:Integer
        m = rows ? m : m'
		inf = 0

		for x = 2:size(m, 1), y = 1:(x-1)
            inf = max(norm(bdot(m[x, :], m[y, :]), p), inf)
        end

		return inf
    end

    ##-----------------------------------------------------------------------------------
    function mut_inch(vl::Array{Array{T, 1}}, p::Z = 2) where T<:Number where Z<:Integer
        inf = 0

        @inbounds for x = 2:lenght(vl), y = 1:(x-1)
            inf = max(norm(bdot(m[x], m[y]), p), inf)
        end

		return inf
    end


	##===================================================================================
    ## normalize (statiscally)
	##	sets variance to 1 and mean to 0
    ##===================================================================================
    export normalize_s, normalize_sp, normalize_sps

    ##-----------------------------------------------------------------------------------
    function normalize_s(m::Array{T, 2}, column::Bool = true) where T<:Number
		r = column ? m : m'
		d = size(r, 1)

        @inbounds for w = 1:size(r, 2)
            r[1:d, w] = (r[1:d, w] - median(r[1:d, w])) / std(r[1:d, w])
        end

        return r
    end

	##-----------------------------------------------------------------------------------
    function normalize_sp(m::Array{T, 2}, column::Bool = true) where T<:Number
		r = column ? m : m'
		d = size(m, 1)

        @sync @distributed for w = 1:size(r, 2)
            r[1:d, w] = (r[1:d, w] - median(r[1:d, w])) / std(r[1:d, w])
        end

        return r
    end

    ##-----------------------------------------------------------------------------------
    function normalize_sps(m::Array{T, 2}, column::Bool = true) where T<:Number
		r = convert(SharedArray, column ? m : m')
		d = size(r, 1)

        @inbounds @sync @distributed for w = 1:size(r, 2)
            r[1:d, w] = (r[1:d, w] - median(r[1:d, w])) / std(r[1:d, w])
        end

        return convert(Array, r)
    end


	##===================================================================================
    ## covariance
	##	d = delay
    ##===================================================================================
    export cov

    ##-----------------------------------------------------------------------------------
    function cov(l::Z, x::Array{R, 1}, n1::Z, y::Array{R, 1}, n2::Z) where R<:AbstractFloat where Z<:Integer
        m1 = m2 = R(0)

		s = n1 * l
		@assert(s <= size(x, 1), "out of bounds error")
        @inbounds for i = 1:n1:(n1*l)
            m1 = m1 + x[i]
        end

		s = n2 * l
		@assert(s <= size(y, 1), "out of bounds error")
        @inbounds for i = 1:n2:(n2*l)
            m2 = m2 + y[i]
        end

        m1 = m1 / l
        m2 = m2 / l

        return (BLAS.dot(l, x, n1, y, n2)/l)-(m1*m2)
    end

    ##-----------------------------------------------------------------------------------
    cov(l::Z, x::Array{R, 1}, n::Z, d::Z = 1) where R<:AbstractFloat where Z<:Integer = cov(l, x, n, x[d+1:end], n)


    ##===================================================================================
    ## covariance matrices of observation matrix
	##	m: matrix
	##	t: indicator whether or not m is transposed
	##	p: indicator whether or not the population covariance shall be computed
    ##===================================================================================
    export mcov

    ##-----------------------------------------------------------------------------------
    function mcov(m::Array{R, 2}, t::B= false, p::B = true) where R<:AbstractFloat where B<:Bool
		s = size(m)

		if t
			x = m .- BLAS.gemm('N', 'T', 1/s[2], ones(T, s[2], s[2]), m)
			return (x*x')/(p ? s[2] : s[2] - 1)
		else
			x = m .- BLAS.gemm('N', 'N', 1/s[1], ones(T, s[1], s[1]), m)
			return (x*x')/(p ? s[1] : s[2] - 1)
		end
	end


    ##===================================================================================
    ## cross covariance
    ##===================================================================================
    export covc

    ##-----------------------------------------------------------------------------------
    function covc(x::Array{R, 1}, y::Array{R, 1}) where R<:AbstractFloat
        xs = size(x, 1); xm = gamean(x, xs, 1)
		ys = size(y, 1); ym = gamean(y, ys, 1)

        r = zeros(T, xs, ys)
		sc = 1 / (xs * ys)

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
	function covcs(v::Array{R, 1}, u::Array{R, 1}, l::Z, t::Z) where R<:AbstractFloat where Z<:Integer 
		return bdot((l-t), (v-gamean(v, size(v, 1), 1)), (circshift(u, t)-gamean(u, size(v, 1), 1)))/(l-t)
	end

	##-----------------------------------------------------------------------------------
    function covcs(v::Array{R, 1}, u::Array{R, 1}, t::Z = 1) where R<:AbstractFloat where Z<:Integer 
		return covcs(v, u, size(v, 1), t)
	end


    ##===================================================================================
    ## cross correlation (with delay)
    ##===================================================================================
    export ccor

    ##-----------------------------------------------------------------------------------
    function ccor(v::Array{R, 1}, u::Array{R, 1}, t::Z = 1) where R<:AbstractFloat where Z<:Integer
		return covcs(v, u, t)./(std(v)*std(u))
	end


	##===================================================================================
	##	shannon index
	##===================================================================================
	export shai

	##-----------------------------------------------------------------------------------
	function shai(v::Array{R, 1}, l::Z, n::Z) where R<:AbstractFloat where Z<:Integer 
		return BLAS.dot(l, v, n, log.(v[1:n:(n*l)]), 1)
	end


	##===================================================================================
	##	Gini–Simpson index
	##===================================================================================
	export gishi

	##-----------------------------------------------------------------------------------
	gishi(v::Array{R, 1}, l::Z, n::Z) where R<:AbstractFloat where Z<:Integer = 1 - BLAS.dot(l, v, n, v, n)


	##===================================================================================
	##	Renyi Entropy
	##===================================================================================
	export reepy

	##-----------------------------------------------------------------------------------
	function reepy(v::Array{R, 1}, p::Z, l::Z, n::Z) where R<:AbstractFloat where Z<:Integer
		s = n * l
		u = v[i]

		@assert(s <= size(v, 1), "out of bounds error")
		@inbounds for i = 1+n:n:s
			u += v[i]^p
		end

		return log(u)/(1-p)
	end
end
