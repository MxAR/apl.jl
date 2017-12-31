@everywhere module mavg
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
