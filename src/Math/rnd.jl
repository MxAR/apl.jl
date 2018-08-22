#@everywhere module rnd
	##===================================================================================
	##	normal distribution
	##		m: mean
	##		s: standard deviation
	##===================================================================================
	export rnormal

	##-----------------------------------------------------------------------------------
	function rnormal(m::R, s::R, d::N... = (1)) where R<:Number where N<:Integer
		return (randn(R, d).-m)./s
	end 


	##===================================================================================
	##	expontential distriubtion
	##		a: expoential scalar
	##===================================================================================
	export rexp

	##-----------------------------------------------------------------------------------
	function rexp(a::R, d::N... = (1)) where R<:Number where N<:Integer
		return -a*log.(rand(R, d))
	end


	##===================================================================================
	##	hypoexponential distribution
	##		a: exponential scalar
	##		n: number sums taken
	##===================================================================================
	export rhexp
	
	##-----------------------------------------------------------------------------------
	function rhexp(a::R, n::N, d::N... = (1)) where R<:Number where N<:Integer
		r = zeros(R, d)
		@inbounds for i = 1:n
			r += log.(rand(R, d))
		end
		return -a*r
	end 


	##===================================================================================
	##	standard triangular distribution
	##===================================================================================
	export rstri

	##-----------------------------------------------------------------------------------
	function rstri(d::N... = (1)) where N<:Integer
		return rand(d).-rand(d)
	end


	##===================================================================================
	##	standard power distribution
	##		a: power exponent
	##===================================================================================
	export rspow

	##-----------------------------------------------------------------------------------
	function rspow(a::R, d::N... = (1)) where R<:Number where N<:Integer
		return rand(R, d).^(1/a)
	end


	##===================================================================================
	##	exponential power distribution
	##		a: exponential divisor
	##		b: power exponent
	##===================================================================================
	export rexpp

	##-----------------------------------------------------------------------------------
	function rexpp(a::R, b::R, d::N... = (1)) where R<:Number where N<:Integer
		return (log.(1-log.(1-rand(R, d)))/a).^(1/b)
	end


	##===================================================================================
	##	compertz distribution
	##===================================================================================
	export rcompertz

	##-----------------------------------------------------------------------------------
	function rcompertz(a::R, b::R, d::N... = (1)) where R<:Number where N<:Integer
		return log.(1-(log.(rand(R, d))*log(b)./a))./log(b)
	end


	##===================================================================================
	##	pareto distribution
	##===================================================================================
	export rpareto

	##-----------------------------------------------------------------------------------
	function rpareto(a::R, b::R, d::N... = (1)) where R<:Number where N<:Integer
		return a.*(rand(R, d)^(-1/b))
	end


	##===================================================================================
	##	uniform distribution
	##		a: left interval border
	##		b: right interval border
	##===================================================================================
	export runi

	##-----------------------------------------------------------------------------------
	function runi(a::R, b::R, d::N... = (1)) where R<:Number where N<:Integer
		return a.+((b-a).*rand(R, d))
	end


	##===================================================================================
	##	benford distribution
	##===================================================================================
	export rbenford

	##-----------------------------------------------------------------------------------
	function rbenford(d::N... = (1)) where N<:Integer
		return floor((10).^rand(d))
	end


	##===================================================================================
	##	log logistic distribution
	##===================================================================================
	export rllog

	##-----------------------------------------------------------------------------------
	function rllog(a::R, b::R, d::N... = (1)) where R<:Number where N<:Integer
		r = rand(R, d)
		return ((((1).-r)./r).^(1/b))/a
	end

	
	##===================================================================================
	##	logistic distribution
	##===================================================================================
	export rlog

	##-----------------------------------------------------------------------------------
	function rlog(a::R, b::R, d::N... = (1)) where R<:Number where N<:Integer
		return log.rllog(a, b, d)
	end


	##===================================================================================
	##	standard cauchy distribution
	##===================================================================================
	export rscauchy

	##-----------------------------------------------------------------------------------
	function rscauchy(d::N... = (1)) where N<:Integer
		return randn(d)./randn(d)
	end


	##===================================================================================
	##	cauchy distribution
	##===================================================================================
	export rcauchy

	##-----------------------------------------------------------------------------------
	function rcauchy(a::R, b::R, d::N... = (1)) where R<:Number where N<:Integer
		return a.+(b.*(randn(R, d)./randn(R, d)))
	end


	##===================================================================================
	##	hyperbolic secant distribution
	##===================================================================================
	export rhsec

	##-----------------------------------------------------------------------------------
	function rhsec(d::N... = (1)) where N<:Integer
		return log.(abs.(randn(d)./randn(d)))
	end


	##===================================================================================
	##	central chi distribution
	##		a: number of additions
	##===================================================================================
	export rchi

	##-----------------------------------------------------------------------------------
	function rchi(a::N, d::N... = (1)) where N<:Integer
		r = zeros(d)

		@inbounds for i = 1:a
			r .+= randn(d).^2
		end

		return sqrt.(r)
	end


	##===================================================================================
	##	noncentral chi distribution
	##		a: number of sums
	##		m: mean
	##		s: standard deviation
	##===================================================================================
	export rnchi

	##-----------------------------------------------------------------------------------
	function rnchi(a::N, m::R, s::R, d::N... = (1)) where R<:Number where N<:Integer
		r = zeros(d)

		@inbounds for i = 1:a
			r .+= ((randn(R, d).-m)./s).^2 
		end 

		return sqrt.(r)
	end


	##===================================================================================
	##	central squared chi distribution
	##		a: number of sums
	##===================================================================================
	export rschi

	##-----------------------------------------------------------------------------------
	function rschi(a::N, d::N... = (1)) where N<:Integer
		r = zeros(d)

		@inbounds for i = 1:a
			r .+= randn(d).^2
		end

		return r
	end

	
	##===================================================================================
	##	noncentral squared chi distribution
	##		a: number of sums
	##		m: mean
	##		s: standard deviation
	##===================================================================================
	export rnschi

	##-----------------------------------------------------------------------------------
	function rnschi(a::N, m::R, s::R, d::N... = (1)) where R<:Number where N<:Integer
		r = zeros(d)

		@inbounds for i = 1:a
			r .+= ((randn(R, d).-m)./s).^2
		end 

		return r
	end


	##===================================================================================
	##	central F distribution
	##===================================================================================
	export rcf

	##-----------------------------------------------------------------------------------
	function rcf(d1::R, d2::R, a::N, d::N... = (1)) where R<:Number where N<:Integer
		r1 = zeros(d)
		r2 = zeros(d)

		@inbounds for i = 1:a
			r1 .+= randn(d, d1).^2
			r2 .+= randn(d, d2).^2
		end 

		return (r1./d1)./(r2./d2)
	end


	##===================================================================================
	##	noncentral F distribution
	##===================================================================================
	export rnf
	
	##----------------------------------------------------------------------------------
	function rnf(d1::R, d2::R, a::N, m::R, s::R, d::N... = (1)) where R<:Number where N<:Integer
		r1 = zeros(d)
		r2 = zeros(d)

		@inbounds for i = 1:a
			r1 .+= ((randn(R, d).-m)./s).^2
			r2 .+= ((randn(R, d).-m)./s).^2
		end

		return (r1./d1)./(r2./d2)
	end
#end
