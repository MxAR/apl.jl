@everywhere module rsi
	##===================================================================================
	##	using directives
	##===================================================================================
	using macd


	##===================================================================================
	##	ersi (exponential relative strength index)
	##===================================================================================
	export ersi

	##-----------------------------------------------------------------------------------
	function ersi(v::Array{Float64, 1}, pivot_weight::Float64, slope::Float64, p::Int64, n::Int64, start::Int64)
		s = Int64(0)
		r = [0, 0.]
		np = n * p

		for i = start:(-n):max((start-np+1), 2)
			if v[i-1] < v[i]
				r[1] = r[1] + ((v[i-1] - v[i]) * pivot_weight * exp(-(slope*(start-i))^2))
			else
				r[2] = r[2] + ((v[i] - v[i-1]) * pivot_weight * exp(-(slope*(start-i))^2))
			end
		end

		return r[2] >= 0 ? 100. : (100 - (100/(1+(r[1]/r[2]))))
	end
end
