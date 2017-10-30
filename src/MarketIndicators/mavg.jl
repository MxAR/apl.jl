@everywhere module mavg
	##===================================================================================
	##	sma (simple moving average)
	##===================================================================================
	export sma

	##-----------------------------------------------------------------------------------
	function sma(v::Array{Float64, 1}, p::Int64, n::Int64, start::Int64, stop::Int64)
		r = zeros(stop-start+1)
		s = Int64(0)
		np = n * p

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
	function lma(v::Array{Float64, 1}, pivot_weight::Float64, slope::Float64, p::Int64, n::Int64, start::Int64, stop::Int64)
		r = zeros(stop-start+1)
		s = Int64(0)
		np = n * p

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
	function ema(v::Array{Float64, 1}, l::Int64, n::Int64 = 1)
		u = zeros(l)

		u[1] = 1.
		u[2] = (l-1)/(l+1)

		for i = 3:(l)
			u[i] = u[i-1]*u[2]
		end

		return (2.0*BLAS.dot(l, u, 1, v, n))/(l+1)
	end

	##-----------------------------------------------------------------------------------
	function emai(v::Array{Float64, 1}, emav::Float64, l::Int64)						# calculates current ema based on last ema
		return ((2*v[1])+((l-1)*emav))/(l+1)
	end
end
