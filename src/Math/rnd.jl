@everywhere module rnd
	##===================================================================================
	##	expontential distriubtion
	##===================================================================================
	export rexp

	##-----------------------------------------------------------------------------------
	function rexp{R<:AbstractFloat, N<:Integer}(a::R, d::N...)
		r = isempty(d)?rand():rand(d)
		return -a*log.(r)
	end


	##===================================================================================
	##	hypoexponential distribution
	##		a: exponential scalar
	##		n: number sums taken
	##===================================================================================
	export rhexp
	
	##-----------------------------------------------------------------------------------
	function rhexp{R<:AbstractFloat, N<:Integer}(a::R, n::N, d::N...)
		if isempty(d)
			@inbounds return -a*sum([log(rand()) for i = 1:n])	
		else
			r = zeros(R, d)
			@inbounds for i = 1:n
				r += log.(rand(d))
			end
			return -a*r
		end 
	end 
end
