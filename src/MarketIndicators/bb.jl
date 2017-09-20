@everywhere module bb
	##===================================================================================
	##	sbb (simple bollinger bands)
	##===================================================================================
	export sbb

	##-----------------------------------------------------------------------------------
	function sbb(v::Array{Float64, 1}, k::Float64, p::Int64, n::Int64, start::Int64, stop::Int64)
		s = stop-start+1
		r1 = zeros(s)
		r2 = zeros(s)
		r3 = zeros(s)
		np = n * p
		s = 0

		for i = start:stop
			sr = i:(-n):(i-np+1)
			s = s + 1
			for j = sr
				r1[s] = r1[s] + v[j]
			end
			r1[s] = r1[s]/(np)
			r2[s] = r1[s] + k * f.std(v[sr], r1[s])
			r3[s] = r1[s] - k * f.std(v[sr], r1[s])
		end

		return (r1, r2, r3)
	end


	##===================================================================================
	##	lbb (linear bollinger bands)
	##===================================================================================
end
