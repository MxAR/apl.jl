@everywhere module rnd
	##===================================================================================
	##	expontential distriubtion
	##		a: expoential scalar
	##===================================================================================
	export rexp

	##-----------------------------------------------------------------------------------
	function rexp{R<:AbstractFloat, N<:Integer}(a::R, d::N...)
		return -a*log.(isempty(d)?rand():rand(d))
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


	##===================================================================================
	##	standard triangular distribution
	##===================================================================================
	export rstri

	##-----------------------------------------------------------------------------------
	function rstri{N<:Integer}(d::N...)
		return isempty(d)?(rand()-rand()):(rand(d)-rand(d))
	end


	##===================================================================================
	##	standard power distribution
	##		a: power exponent
	##===================================================================================
	export rspow

	##-----------------------------------------------------------------------------------
	function rspow{R<:AbstractFloat, N<:Integer}(a::R, d::N...)
		return (isempty(d)?rand():rand(d))^(1/a)
	end


	##===================================================================================
	##	exponential power distribution
	##===================================================================================
	export rexpp

	##-----------------------------------------------------------------------------------
	function rexpp{R<:AbstractFloat, N<:Integer}(a::R, b::R, d::N...)
		if isempty(d)
			return (log(1-log(1-rand()))/a)^(1/b)
		else
			return (log.(1-log.(1-rand(d)))).^(1/b)
		end
	end
end
