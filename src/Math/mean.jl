@everywhere module mean
	##===================================================================================
	##	generalized mean
	##		ga = arthimetic mean
	##		gh = harmonic mean
	##		gg = geometric mean
	##		gp = power mean
	##		gr = root squared mean
	##		gf = f mean
	##===================================================================================
	export gamean, ghmean, ggmean, gpmean, gfmean, grmean

	##-----------------------------------------------------------------------------------
	function gamean(v::Array{R, 1}, l::N, n::N) where R<:AbstractFloat where N<:Integer 
		return gfmean(v, (x) -> x, (x) -> x, l, n)
	end

	##-----------------------------------------------------------------------------------
	function gamean(v::Array{R, 1}) where R<:AbstractFloat 
		s = size(v, 1)
		r = R(0)

		@inbounds for i = 1:s
			r += v[i]
		end

		return r/R(s)
	end

	##-----------------------------------------------------------------------------------
	function ghmean(v::Array{R, 1}, l::N, n::N) where R<:AbstractFloat where N<:Integer
		return gfmean(v, (x) -> 1/x, (x) -> 1/x, l, n)
	end

	##-----------------------------------------------------------------------------------
	function ggmean(v::Array{R, 1}, l::N, n::N) where R<:AbstractFloat where N<:Integer
		return gfmean(v, (x) -> log(x), (x) -> exp(x), l, n)
	end

	##-----------------------------------------------------------------------------------
	function gpmean(v::Array{R, 1}, l::N, n::N, p::R) where R<:AbstractFloat where N<:Integer 
		return gfmean(v, (x) -> x^p, (x) -> x^(1/p), l, n)
	end

	##-----------------------------------------------------------------------------------
	function grmean(v::Array{R, 1}, l::N, n::N, p::R) where R<:AbstractFloat where N<:Integer
		return gfmean(v, (x) -> x^2, (x) -> sqrt(x), l, n)
	end

	##-----------------------------------------------------------------------------------
	function gfmean(v::Array{R, 1}, g::F, gi::F, l::N, n::N) where R<:AbstractFloat where N<:Integer where F<:Function
		@assert(size(v, 1) >= (n*l), "out of bounds error")
		u = R(0)

		@inbounds for i = 1:n:(n*l)
			u += g(v[i])
		end

		return gi(u/l)
	end
	

	##===================================================================================
    ## arithmetic mean column/row
    ##===================================================================================
    export mamean

    ##-----------------------------------------------------------------------------------
    function mamean(arr::Array{R, 2}, column::Bool = true) where R<:AbstractFloat
        n = size(arr, (column ? 1 : 2))
        return BLAS.gemv((column ? 'N' : 'T'), 1/n, arr, ones(R, n))
    end

    ##-----------------------------------------------------------------------------------
    function mamean(arr::Array{R, 2}, weights::Array{R,1}, column::Bool = true) where R<:AbstractFloat
        n = size(arr, (column ? 1 : 2))
        return BLAS.gemv((column ? 'N' : 'T'), 1/n, weights.*arr, ones(R, n))
    end


	##===================================================================================
	##	sma (simple moving average)
	##===================================================================================
	export sma

	##-----------------------------------------------------------------------------------
	function sma(v::Array{R, 1}, p::N, n::N, start::N, stop::N) where R<:AbstractFloat where N<:Integer
		@assert(start <= stop && stop <= size(v, 1), "out of bounds error")
		@assert(start > 0 && stop > 0, "out of bounds error")
		r = zeros(R, stop-start+1)
		np = n * p
		s = N(0)

		for i = start:stop
			s = s + 1
			for j = i:(-n):(i-np+1)
				r[s] = r[s] + v[j]
			end
			r[s] = r[s]/(np)
		end

		return r
	end


	##===================================================================================
	##	lma (linear moving average (weighted average))
	##===================================================================================
	export lma

	##-----------------------------------------------------------------------------------
	function lma(v::Array{R, 1}, pivot_weight::R, slope::R, p::N, n::N, start::N, stop::N) where R<:AbstractFloat where N<:Integer
		@assert(start <= stop && stop <= size(v, 1), "out of bounds error")
		@assert(start > 0 && stop > 0, "out of bounds error")
		r = zeros(T, stop-start+1)
		np = n * p
		s = N(0)

		for i = start:stop
			s = s + 1
			for j = i:(-n):(i-np+1)
				r[s] = r[s] + (v[j] * max((pivot_weight + slope*(i-j)), 0.))
			end
		end

		return r
	end


	##===================================================================================
	##	ema (exponential moving average (weighted average))
	##===================================================================================
	export ema

	##-----------------------------------------------------------------------------------
	function ema(v::Array{R, 1}, pivot_weight::R, slope::R, p::N, n::N, start::N, stop::N) where R<:AbstractFloat where N<:Integer
		@assert(start <= stop && stop <= size(v, 1), "out of bounds error")
		@assert(start > 0 && stop > 0, "out of bounds error")
		r = zeros(T, stop-start+1)
		np = n * p
		s = N(0)

		for i = start:stop
			s = s + 1
			for j = i:(-n):(i-np+1)
				r[s] = r[s] + (v[j] * pivot_weight * exp(-(slope*(i-j))^2))
			end
		end

		return r
	end
end
