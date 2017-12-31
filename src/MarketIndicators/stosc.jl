@everywhere module stosc
	##===================================================================================
	##	sstosc (simple stochastic oscilator)
	##===================================================================================
	export sstosc

	##-----------------------------------------------------------------------------------
	function sstosc{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, current_price::T, p::N, n::N, start::N, stop::N)
		@assert(start <= stop && stop <= size(v, 1), "out of bounds error")
		@assert(start > 0 && stop > 0, "out of bounds error")
		r = zeros(T, stop-start+1)
		b = [T(-Inf), T(Inf)]
		np = n * p
		s = N(0)

		@inbounds for i = start:stop
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
			b = [T(-Inf), T(Inf)]
		end

		return r/(stop-start+1)
	end


	##===================================================================================
	##	lstosc (linear stochastic oscilator)
	##===================================================================================
	export lstosc

	##-----------------------------------------------------------------------------------
	function lstosc{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, current_price::T, pivot_weight::T, slope::T, p::N, n::N, start::N, stop::N)
		@assert(start <= stop && stop <= size(v, 1), "out of bounds error")
		@assert(start > 0 && stop > 0, "out of bounds error")
		r = zeros(T, stop-start+1)
		b = [T(-Inf), T(Inf)]
		np = n * p
		s = N(0)

		@inbounds for i = start:stop
			s = s + 1
			for j = i:(-n):(i-np+1)
				if v[j] > b[1]
					b[1] = v[j]
				end

				if v[j] < b[2]
					b[2] = v[j]
				end
			end

			r[s] = r[s] + (((current_price - b[2])/(b[1] - b[2])) * max((pivot_weight + slope*(s-1)), 0.))
			b = [T(-Inf), T(Inf)]
		end

		return r
	end


	##===================================================================================
	##	estosc (exponential stochastic oscilator)
	##===================================================================================
	export estosc

	##-----------------------------------------------------------------------------------
	function estosc{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, current_price::T, pivot_weight::T, slope::T, p::N, n::N, start::N, stop::N)
		@assert(start <= stop && stop <= size(v, 1), "out of bounds error")
		@assert(start > 0 && stop > 0, "out of bounds error")
		r = zeros(T, stop-start+1)
		b = [T(-Inf), T(Inf)]
		np = n * p
		s = N(0)

		@inbounds for i = start:stop
			s = s + 1
			for j = i:(-n):(i-np+1)
				if v[j] > b[1]
					b[1] = v[j]
				end

				if v[j] < b[2]
					b[2] = v[j]
				end
			end

			r[s] = r[s] + (((current_price - b[2])/(b[1] - b[2])) * pivot_weight * exp(-(slope*(s-1))^2))
			b = [T(-Inf), T(Inf)]
		end

		return r
	end
end
