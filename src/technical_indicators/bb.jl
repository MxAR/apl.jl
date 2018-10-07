@everywhere module bb
	##===================================================================================
	##	sbb (simple bollinger bands)
	##===================================================================================
	export sbb

	##-----------------------------------------------------------------------------------
	function sbb(v::Array{R, 1}, k::R, p::Z, n::Z, start::Z, stop::Z) where R<:AbstractFloat where Z<:Integer
		@assert(start <= stop && stop <= size(v, 1), "out of bounds error")
		@assert(start > 0 && stop > 0, "out of bounds error")
		s = stop-start+1
		r1 = zeros(T, s)
		r2 = zeros(T, s)
		r3 = zeros(T, s)
		np = n * p

		@inbounds for i = start:stop
			sr = i:n:(min(i+np, stop))
			r1[i] = sum(v[sr])/np
			s = k * sta.std(v[sr])
			s = isnan(s) ? 0 : s
			r2[i] = r1[i] + s
			r3[i] = r1[i] - s
		end

		return (r1, r2, r3)
	end


	##===================================================================================
	##	lbb (linear bollinger bands)
	##===================================================================================
	export lbb

	##-----------------------------------------------------------------------------------
	function lbb(v::Array{R, 1}, k::R, pivot_weight::R, slope::R, p::Z, n::Z, start::Z, stop::Z) where R<:AbstractFloat where Z<:Integer
		@assert(start <= stop && stop <= size(v, 1), "out of bounds error")
		@assert(start > 0 && stop > 0, "out of bounds error")
		r1 = r2 = r3 = zeros(T, stop-start+1)
		np = n * p
		s = 0

		@inbounds for i = start:stop
			sr = i:(-n):(i-np+1)
			s = s + 1

			for j = sr
				r1[s] = r1[s] + (v[j] * max((pivot_weight + slope*(i-j)), 0.))
			end

			r2[s] = r1[s] + k * f.std(v[sr], r1[s])
			r3[s] = r1[s] - k * f.std(v[sr], r1[s])
		end

		return (r1, r2, r3)
	end


	##===================================================================================
	##	ebb (exponential bollinger bonds)
	##===================================================================================
	export ebb

	##-----------------------------------------------------------------------------------
	function ebb(v::Array{R, 1}, k::R, pivot_weight::R, slope::R, p::Z, n::Z, start::Z, stop::Z) where R<:AbstractFloat where Z<:Integer
		@assert(start <= stop && stop <= size(v, 1), "out of bounds error")
		@assert(start > 0 && stop > 0, "out of bounds error")
		r1 = r2 = r3 = zeros(T, stop-start+1)
		np = n * p
		s = 0

		@inbounds for i = start:stop
			sr = i:(-n):(i-np+1)
			s = s + 1

			for j = sr
				r1[s] = r1[s] + (v[j] * pivot_weight * exp(-(slope*(i-j))^2))
			end

			r2[s] = r1[s] + k * f.std(v[sr], r1[s])
			r3[s] = r1[s] - k * f.std(v[sr], r1[s])
		end

		return (r1, r2, r3)
	end
end
