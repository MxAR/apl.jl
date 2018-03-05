@everywhere module sta
	##===================================================================================
	##	import directives
	##===================================================================================
	import Base.std


	##===================================================================================
	##	include	directives
	##===================================================================================
	include("mean.jl")
	include("bla.jl")


	##===================================================================================
	##	using directives
	##===================================================================================
	using mean
	using bla


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
	##  chi distribution
	##===================================================================================
	export chi

	##-----------------------------------------------------------------------------------
	chi(k::Integer) = norm(randn(k))


	##===================================================================================
	## nth order difference
	##	t: time series
	## 	d: delay
	##	l: size of the observation
	## 	inc: index increment
	##===================================================================================
	export nod

	##-----------------------------------------------------------------------------------
	function nod{T<:AbstractFloat, N<:Integer}(t::Array{T, 1}, p::N, l::N, inc::N = 1)
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
	zscore{T<:AbstractFloat}(v::Array{T, 1}) = (v-(sum(v)/size(v, 1)))/std(v)


	##===================================================================================
	##  dice sampling
	##===================================================================================
	export w6, nw6

	##-----------------------------------------------------------------------------------
	w6() = Int(ceil(rand()*6))

	##-----------------------------------------------------------------------------------
	nw6(n::Integer) = sum([w6() for i = 1:n])


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
    ## mutual incoherence
    ##===================================================================================
    export mut_inch

    ##-----------------------------------------------------------------------------------
    function mut_inch{T<:Number, N<:Integer}(m::Array{T, 2}, rows = true, p::N = 2)    	# the lower the better the mutual incoherence property
        m = rows ? m : m'
		inf = 0

		for x = 2:size(m, 1), y = 1:(x-1)
            inf = max(norm(bdot(m[x, :], m[y, :]), p), inf)
        end

		return inf
    end

    ##-----------------------------------------------------------------------------------
    function mut_inch{T<:Number, N<:Integer}(vl::Array{Array{T, 1}}, p::N = 2)
        inf = 0

        @inbounds for x = 2:lenght(vl), y = 1:(x-1)
            inf = max(norm(bdot(m[x], m[y]), p), inf)
        end

		return inf
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
	covcs{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, u::Array{T, 1}, l::N, t::N) = bdot((l-t), (v-gamean(v, size(v, 1), 1)), (circshift(u, t)-gamean(u, size(v, 1), 1)))/(l-t)

	##-----------------------------------------------------------------------------------
    covcs{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, u::Array{T, 1}, t::N = N(1)) = covcs(v, u, size(v, 1), t)

    ##===================================================================================
    ## cross correlation (with delay)
    ##===================================================================================
    export ccor

    ##-----------------------------------------------------------------------------------
    ccor{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, u::Array{T, 1}, t::N = N(1)) = covcs(v, u, t)./(std(v)*std(u))


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
    ## samples
    ##===================================================================================
    export samples

    ##-----------------------------------------------------------------------------------
    function samples{T<:Any}(data::Array{T, 1}, size::Integer)
        L = length(data)
        @assert(size < L, "The number of samples musn't be bigger than the data!")
        return shuffle(getindex(data, sort(sample(1:L, size, replace = false))))
    end
end
