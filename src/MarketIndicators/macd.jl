@everywhere module macd
	##===================================================================================
	##	using directives
	##===================================================================================
	using mavg


	##===================================================================================
	##	smacd (simple moving average convergence divergence)
	##===================================================================================
	export smacd

	##-----------------------------------------------------------------------------------
	function smacd(v::Array{Float64, 1}, p::Tuple{Int64, Int64}, n::Int64, start::Int64, stop::Int64)
		np = (n * p[1], n * p[2])
		s = stop-start+1
		r = zeros(s)
		q = zeros(s)
		s = 0

		for i = start:stop
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
	function lmacd(v::Array{Float64, 1}, pivot_weight::Float64, slope::Float64, p::Tuple{Int64, Int64}, n::Int64, start::Int64, stop::Int64)
		d = Int64(abs(p[1] - p[2]))
		return lma(v, (pivot_weight + slope*d), slope, minimum(p), n, start-d, stop-d)
	end
end
