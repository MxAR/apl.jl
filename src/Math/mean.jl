@everywhere module mean
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
	##	sma (simple moving average)
	##===================================================================================
	export sma

	##-----------------------------------------------------------------------------------
	function sma{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, p::N, n::N, start::N, stop::N)
		@assert(start <= stop && stop <= size(v, 1), "out of bounds error")
		@assert(start > 0 && stop > 0, "out of bounds error")
		r = zeros(T, stop-start+1)
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
	function lma{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, pivot_weight::T, slope::T, p::N, n::N, start::N, stop::N)
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
	function ema{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, pivot_weight::T, slope::T, p::N, n::N, start::N, stop::N)
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
