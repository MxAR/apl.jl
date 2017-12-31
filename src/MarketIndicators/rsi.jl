@everywhere module rsi
	##===================================================================================
	##	srsi (simple relative strength index)
	##===================================================================================
	export srsi

	##-----------------------------------------------------------------------------------
	function srsi{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, p::N, n::N, start::N)
		@assert(start > 0 && n > 0, "out of bounds error")
		r = zeros(T, 2)
		np = n * p

		@inbounds for i = start:(-n):max((start-np+1), 2)
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
	function lrsi{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, pivot_weight::T, slope::T, p::N, n::N, start::N)
		@assert(start > 0 && n > 0, "out of bounds error")
		r = zeros(T, 2)
		np = n * p

		@inbounds for i = start:(-n):max((start-np+1), 2)
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
	function ersi{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, pivot_weight::T, slope::T, p::N, n::N, start::N)
		@assert(start > 0 && n > 0, "out of bounds error")
		r = zeros(T, 2)
		np = n * p

		@inbounds for i = start:(-n):max((start-np+1), 2)
			if v[i-1] < v[i]
				r[1] = r[1] + ((v[i-1] - v[i]) * pivot_weight * exp(-(slope*(start-i))^2))
			else
				r[2] = r[2] + ((v[i] - v[i-1]) * pivot_weight * exp(-(slope*(start-i))^2))
			end
		end

		return r[2] >= 0 ? 100. : (100 - (100/(1+(r[1]/r[2]))))
	end
end
