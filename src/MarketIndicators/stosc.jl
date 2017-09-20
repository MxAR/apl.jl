@everywhere module stosc
	##===================================================================================
	##	sstosc (simple stochastic oscilator)
	##===================================================================================
	export sstosc

	##-----------------------------------------------------------------------------------
	function sstosc(v::Array{Float64, 1}, current_price::Float64, p::Int64, n::Int64, start::Int64, stop::Int64)
		r = zeros(stop-start+1)
		b = [-Inf, Inf]
		s = Int64(0)
		np = n * p

		for i = start:stop
			s = s + 1
			for j = i:(-n):(i-np+1)
				if v[j] > b[1]
					b[1] = v[j]
				end

				if v[j] < b[2]
					b[2] = v[j]
				end
			end

			r[s] = r[s] + ((current_price - b[2])/(b[1] - b[2]))
			b = [-Inf, Inf]
		end

		return r/(stop-start+1)
	end
end
