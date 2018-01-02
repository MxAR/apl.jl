@everywhere module macd
	##===================================================================================
	##	using directives
	##===================================================================================
	using mean


	##===================================================================================
	##	smacd (simple moving average convergence divergence)
	##===================================================================================
	export smacd

	##-----------------------------------------------------------------------------------
	function smacd{T<:Float64, N<:Integer}(v::Array{T, 1}, p::Tuple{N, N}, n::N, start::N, stop::N)
		@assert(start <= stop && stop <= size(v, 1), "out of bounds error")
		@assert(start > 0 && stop > 0, "out of bounds error")
		r = q = zeros(T, stop-start+1)
		np = (n * p[1], n * p[2])
		s = N(0)

		@inbounds for i = start:stop
			s = s + 1

			for j = i:(-n):(i-np[1]+1)
				r[s] = r[s] + v[j]
			end

			r[s] = r[s]/(np[1])

			for j = i:(-n):(i-np[2]+1)
				q[s] = q[s] + v[j]
			end

			r[s] = r[s] - (q[s]/np[2])
		end

		return r
	end


	##===================================================================================
	##	lmacd (linear moving average convergence divergence)
	##===================================================================================
	export lmacd

	##-----------------------------------------------------------------------------------
	function lmacd{T<:Float64, N<:Integer}(v::Array{T, 1}, pivot_weight::T, slope::T, p::Tuple{N, N}, n::N, start::N, stop::N)
		d = N(abs(p[1] - p[2]))
		return -1*lma(v, (pivot_weight + slope*d), slope, minimum(p), n, start-d, stop-d)
	end


	##===================================================================================
	##	emacd (exponential moving average convergence divergence)
	##===================================================================================
	export emacd

	##-----------------------------------------------------------------------------------
	function emacd{T<:Float64, N<:Integer}(v::Array{T, 1}, pivot_weight::T, slope::T, p::Tuple{N, N}, n::N, start::N, stop::N)
		@assert(start <= stop && stop <= size(v, 1), "out of bounds error")
		@assert(start > 0 && stop > 0, "out of bounds error")
		np = n * (p[1] < p[2] ? p[1] : p[2])
		r = zeros(T, stop-start+1)
		d = N(abs(p[1] - p[2]))
		start = start - d
		stop = stop - d
		s = N(0)

		@inbounds for i = start:stop
			s = s + 1
			for j = i:(-n):(i-np+1)
				r[s] = r[s] + (-v[j] * pivot_weight * exp(-(slope*(i-j+d))^2))
			end
		end

		return r
	end
end
