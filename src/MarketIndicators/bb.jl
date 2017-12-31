@everywhere module bb
	##===================================================================================
	##	sbb (simple bollinger bands)
	##===================================================================================
	export sbb

	##-----------------------------------------------------------------------------------
	function sbb{T<:Float64, N<:Integer}(v::Array{T, 1}, k::T, p::N, n::N, start::N, stop::N)
		@assert(start <= stop && stop <= size(v, 1), "out of bounds error")
		@assert(start > 0 && stop > 0, "out of bounds error")
		r1 = r2 = r3 = zeros(T, stop-start+1)
		np = n * p
		s = 0

		@inbounds for i = start:stop
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
	export lbb

	##-----------------------------------------------------------------------------------
	function lbb{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, k::T, pivot_weight::T, slope::T, p::N, n::N, start::N, stop::N)
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
	function ebb{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, k::T, pivot_weight::T, slope::T, p::N, n::N, start::N, stop::N)
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
