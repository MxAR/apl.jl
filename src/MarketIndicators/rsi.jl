@everywhere module rsi
	##===================================================================================
	##	srsi (simple relative strength index)
	##===================================================================================
	export srsi

	##-----------------------------------------------------------------------------------
	function srsi(v::Array{Float64, 1}, pivot_weight::Float64, slope::Float64, p::Int64, n::Int64, start::Int64)
		r = [0, 0.]
		np = n * p

		for i = start:(-n):max((start-np+1), 2)
			if v[i-1] < v[i]
				r[1] = r[1] + (v[i-1] - v[i])
			else
				r[2] = r[2] + (v[i] - v[i-1])
			end
		end

		return r[2] >= 0 ? 100. : (100. - (100./(1+(r[1]/r[2]))))
	end


	##===================================================================================
	##	lrsi (linear relative strength index)
	##===================================================================================
	export lrsi

	##-----------------------------------------------------------------------------------
	function lrsi(v::Array{Float64, 1}, pivot_weight::Float64, slope::Float64, p::Int64, n::Int64, start::Int64)
		r = [0, 0.]
		np = n * p

		for i = start:(-n):max((start-np+1), 2)
			if v[i-1] < v[i]
				r[1] = r[1] + ((v[i-1] - v[i]) * max((pivot_weight + slope*(start-i)), 0.))
			else
				r[2] = r[2] + ((v[i] - v[i-1]) * max((pivot_weight + slope*(start-i)), 0.))
			end
		end

		return r[2] >= 0 ? 100. : (100 - (100/(1+(r[1]/r[2]))))
	end


	##===================================================================================
	##	ersi (exponential relative strength index)
	##===================================================================================
	export ersi

	##-----------------------------------------------------------------------------------
	function ersi(v::Array{Float64, 1}, pivot_weight::Float64, slope::Float64, p::Int64, n::Int64, start::Int64)
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
