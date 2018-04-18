@everywhere module rnd
	##===================================================================================
	##	expontential distriubtion
	##		a: expoential scalar
	##===================================================================================
	export rexp

	##-----------------------------------------------------------------------------------
	function rexp{R<:Number, N<:Integer}(a::R, d::N... = (1))
		return -a*log.(rand(d))
	end


	##===================================================================================
	##	hypoexponential distribution
	##		a: exponential scalar
	##		n: number sums taken
	##===================================================================================
	export rhexp
	
	##-----------------------------------------------------------------------------------
	function rhexp{R<:Number, N<:Integer}(a::R, n::N, d::N... = (1))
		r = zeros(R, d)
		@inbounds for i = 1:n
			r += log.(rand(d))
		end
		return -a*r
	end 


	##===================================================================================
	##	standard triangular distribution
	##===================================================================================
	export rstri

	##-----------------------------------------------------------------------------------
	function rstri{N<:Integer}(d::N... = (1))
		return rand(d).-rand(d)
	end


	##===================================================================================
	##	standard power distribution
	##		a: power exponent
	##===================================================================================
	export rspow

	##-----------------------------------------------------------------------------------
	function rspow{R<:Number, N<:Integer}(a::R, d::N... = (1))
		return rand(d).^(1/a)
	end


	##===================================================================================
	##	exponential power distribution
	##		a: exponential divisor
	##		b: power exponent
	##===================================================================================
	export rexpp

	##-----------------------------------------------------------------------------------
	function rexpp{R<:Number, N<:Integer}(a::R, b::R, d::N... = (1))
			return (log.(1-log.(1-rand(d)))/a).^(1/b)
	end


	##===================================================================================
	##	compertz distribution
	##===================================================================================
	export rcompertz

	##-----------------------------------------------------------------------------------
	function rcompertz{R<:Number, N<:Integer}(a::R, b::R, d::N... = (1))
		return log.(1-(log.(rand(d))*log(b)./a))./log(b)
	end


	##===================================================================================
	##	pareto distribution
	##===================================================================================
	export rpareto

	##-----------------------------------------------------------------------------------
	function rpareto{R<:Number, N<:Integer}(a::R, b::R, d::N... = (1))
		return a.*(rand(d)^(-1/b))
	end


	##===================================================================================
	##	uniform distribution
	##		a: left interval border
	##		b: right interval border
	##===================================================================================
	export runi

	##-----------------------------------------------------------------------------------
	function runi{R<:Number, N<:Integer}(a::R, b::R, d::N... = (1))
		return a.+((b-a).*rand(d))
	end


	##===================================================================================
	##	benford distribution
	##===================================================================================
	export rbenford

	##-----------------------------------------------------------------------------------
	function rbenford{N<:Integer}(d::N... = (1))
		return floor((10).^rand(d))
	end


	##===================================================================================
	##	log logistic distribution
	##===================================================================================
	export rllog

	##-----------------------------------------------------------------------------------
	function rllog{R<:Number, N<:Integer}(a::R, b::R, d::N... = (1))
		r = rand(d)
		return ((((1).-r)./r).^(1/b))/a
	end

	
	##===================================================================================
	##	logistic distribution
	##===================================================================================
	export rlog

	##-----------------------------------------------------------------------------------
	function rlog{R<:Number, N<:Integer}(a::R, b::R, d::N... = (1))
		return log.rllog(a, b, d)
	end


	##===================================================================================
	##	standard cauchy distribution
	##===================================================================================
	export rscauchy

	##-----------------------------------------------------------------------------------
	function rscauchy{N<:Integer}(d::N... = (1))
		return randn(d)./randn(d)
	end


	##===================================================================================
	##	cauchy distribution
	##===================================================================================
	export rcauchy

	##-----------------------------------------------------------------------------------
	function rcauchy{R<:Number, N<:Integer}(a::R, b::R, d::N... = (1))
		return a.+(b.*(randn(d)./randn(d)))
	end


	##===================================================================================
	##	hyperbolic secant distribution
	##===================================================================================
	export rhsec

	##-----------------------------------------------------------------------------------
	function rhsec{N<:Integer}(d::N... = (1))
		return log.(abs.(randn(d)./randn(d)))
	end


	##===================================================================================
	##	chi distribution
	##		a: number of additions
	##===================================================================================
	export rchi

	##-----------------------------------------------------------------------------------
	function rchi{N<:Integer}(a::N, d::N... = (1))
		r = zeros(d)

		@inbounds for i = 1:a
			r .+= randn(d).^2
		end

		return sqrt.(r)
	end
end
