@everywhere module macd
	##===================================================================================
	##	using directives
	##===================================================================================
	using mavg


	##===================================================================================
	##	smacd (simple moving average convergence divergence)
	##===================================================================================
	function smacd(v::Array{Float64, 1}, p::Tuple{Int64, Int64}, n::Int64, start::Int64, stop::Int64)
		v = sma(v, p[1], n, start, stop)
		u = sma(v, p[2], n, start, stop)

		for i = 1:(stop-start)
			v[i] = v[i] - u[i]
		end

		return v
	end
end
