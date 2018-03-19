@everywhere module f
	##===================================================================================
	##  using directives
	##===================================================================================
	using StatsBase
	using cnv
    using gen
	using op


	##===================================================================================
	##  import directives
	##===================================================================================
	import Base.map, Base.std


    ##===================================================================================
    ##  greatest common devisor
    ##===================================================================================
    export gcd

    ##-----------------------------------------------------------------------------------
    function gcd{T<:Integer}(x::T, y::T)
        z0 = max(x, y)
        z1 = min(x, y)
        s = T(0);

        while z1 != 0
            s = z0 % z1
            z0 = z1
            z1 = s
        end

        return z0
    end


    ##===================================================================================
    ##  number of perumtations (selecting k out of n)
    ##===================================================================================
    export perm, permr

    ##-----------------------------------------------------------------------------------
    function perm{N<:Integer}(n::N, k::N)
        r = N(1)

        @inbounds for i = n:-1:(n-k+1)
            r *= i
        end

        return r
    end

    ##-----------------------------------------------------------------------------------
    ##  rep = array where each element indicates how often an element is present in the
    ##          set that will be permutated | rep length need to be equal to n
    ##-----------------------------------------------------------------------------------
    function permr{N<:Integer}(n::N, rep::Array{N, 1})
        r = [N(1), N(factorial(rep[1]))]
        l = size(rep, 1)

        @inbounds for i = n:-1:2
            r[1] *= i
            r[2] *= i > l ? 1 : rep[i]
        end

        return N(r[1]/r[2])
    end


    ##===================================================================================
    ##  number of combinatons (selecting k out of n)
    ##===================================================================================
    export comb

    ##-----------------------------------------------------------------------------------
    function comb{N<:Integer}(n::N, k::N)
        k = k > n >> 1 ? n-k : k
        return N(perm(n, k)/factorial(k))
    end


    ##===================================================================================
    ##  collatz conjecture
    ##===================================================================================
    export collatz

    ##-----------------------------------------------------------------------------------
    function collatz{T<:Integer}(x::T)
		@assert(x > 0, "out of bounds error")
		c = UInt(0)

        while x != 4
            if x & 1 == 1
                x = 3*x +1
            end

            x = x >> 1
            c = c + 1
        end

        return c
    end


    ##===================================================================================
    ##  dirac delta/impulse
    ##===================================================================================
    export dcd

    ##-----------------------------------------------------------------------------------
    function dcd{T<:AbstractFloat}(x::T)
        set_zero_subnormals(true)
        return x == 0 ? Inf : 0
    end


    ##===================================================================================
    ## kronecker delta
    ##===================================================================================
    export ked

    ##-----------------------------------------------------------------------------------
    ked{T<:Number}(x::T, y::T) = x == y ? 1 : 0


	##===================================================================================
	##	kingsman equation
	##===================================================================================
	export king

	##-----------------------------------------------------------------------------------
	king{T<:AbstractFloat, N<:Integer}(p::T, ca::T, cs::T, c::N) = ((p^(sqrt(2*(c+1))))/(1-p))x*(0.5)*((ca^2)+(cs^2))


    ##===================================================================================
    ## sigmoid
    ##===================================================================================
    export sigmoid, sigmoidd

    ##-----------------------------------------------------------------------------------
    sigmoid(x, eta = 0) = @. 1/(1+exp(-(x-eta)))

    ##-----------------------------------------------------------------------------------
    sigmoidd(x, eta = 0) =  @. sigmoid(x, eta) * (1-sigmoid(x, eta))


    ##===================================================================================
    ## norm derivate
    ##===================================================================================
    export normd

    ##-----------------------------------------------------------------------------------
    nromd(v::Array{Float64, 1}, p::Int64 = 2) = @. sign(v)*(abs(v)/ifelse(iszero(v), 1, norm(v, p)))^(p-1)


    ##===================================================================================
    ## radial basis functions
    ##===================================================================================
    export rbf_gauss, rbf_gaussdl, rbf_gaussdd, rbf_triang, rbf_cos_decay,
        rbf_psq, rbf_inv_psq, rbf_inv_sq, rbf_exp, rbf_tps

    ##-----------------------------------------------------------------------------------
    rbf_gauss(delta, lambda::Float64 = 1) = @. exp(-(delta/(2*lambda))^2)

    ##-----------------------------------------------------------------------------------
    function rbf_gaussdl(delta, lambda::Float64 = 1)
        @. delta ^= 2
        lam = lambda^2
        return @. (delta/(lam*lambda))*exp(-delta/lam)
    end

    ##-----------------------------------------------------------------------------------
    function rbf_gaussdd(delta, lambda::Float64 = 1)
        @. lambda ^= 2
        return @. (delta./lambda) .* exp(-(delta.^2)./(2*lambda))
    end

    ##-----------------------------------------------------------------------------------
    rbf_triang(delta::Float64, lambda::Float64 = 1) = delta > lambda ? 0. : (1 - (delta/lambda))

    ##-----------------------------------------------------------------------------------
    function rbf_triang(delta::Array{Float64, 1}, lambda::Float64 = 1)
        if AND(delta .> lambda)
            return zeros(delta)
        else
            return (1.-(delta./lambda))
        end
    end

    ##-----------------------------------------------------------------------------------
    rbf_cos_decay(delta::Float64, lambda::Float64 = 1) = delta > lambda ? 0. : ((cos((pi*delta)/(2*lambda)))+1)/2

    ##-----------------------------------------------------------------------------------
    function rbf_cos_decay(delta::Array{Float64, 1}, lambda::Float64 = 1)
        if AND(delta .> lambda)
            return zeros(delta)
        else
            return @. ((cos((pi*delta)/(2*lambda))).+1)/2
        end
    end

    ##-----------------------------------------------------------------------------------
    rbf_psq(delta, lambda::Float64 = 1) = @. sqrt(1+(lambda*delta)^2)

    ##-----------------------------------------------------------------------------------
    rbf_inv_psq(delta, lambda::Float64 = 1) = @. 1/sqrt(1+(lambda*delta)^2)

    ##-----------------------------------------------------------------------------------
    rbf_inv_sq(delta, lambda::Float64 = 1) = @. 1/(1+(lambda*delta)^2)

    ##-----------------------------------------------------------------------------------
    rbf_exp(delta, expt::Float64 = 2) = delta.^expt

    ##-----------------------------------------------------------------------------------
    rbf_tps(delta, expt::Float64 = 2) = @. (delta^expt)*log(delta)


    ##===================================================================================
    ## ramp
    ##===================================================================================
    export ramp, rampd

    ##-----------------------------------------------------------------------------------
    ramp(x, eta) = @. max(0, x-eta)

    ##-----------------------------------------------------------------------------------
    rampd{T<:AbstractFloat}(x::T, eta::T) = x-eta > 0 ? T(1) : T(0)

    ##-----------------------------------------------------------------------------------
    rampd{T<:AbstractFloat}(x::Array{T, 1}, eta::T) = AND(x.-eta .> 0) ? ones(T, x) : zeros(T, x)

    ##-----------------------------------------------------------------------------------
    function rampd{T<:AbstractFloat}(x::Array{T, 1}, eta::Array{T, 1})
        r = zeros(T, x)

        @inbounds for i = 1:size(x, 1)
            if x[i]-eta[i] > 0
                r[i] = 1.
            end
        end

        return r
    end


    ##===================================================================================
    ## semi linear
    ##===================================================================================
    export semilin, semilind

    ##-----------------------------------------------------------------------------------
    semilin(x, eta, sigma = 0.5) = prison(x, (x) -> x-eta+sigma, eta-sigma, eta+sigma)

    ##-----------------------------------------------------------------------------------
    semilind(x, eta, sigma = 0.5) = (x > eta+sigma || x < eta-sigma) ? 0. : 1.


    ##===================================================================================
    ## sine saturation
    ##===================================================================================
    export sinesat, sinesatd

    ##-----------------------------------------------------------------------------------
    sinesat(x, eta, sigma = pi/2) = prison(x, (x) -> (sin(x-eta)+1)/2, eta-sigma, eta+sigma)

    ##-----------------------------------------------------------------------------------
    sinesatd(x, eta, sigma = pi/2) = (x > eta+sigma || x < eta-sigma) ? 0 : cos(x-eta)/2


    ##===================================================================================
    ## softplus
    ##===================================================================================
    export softplus, softplusd

    ##-----------------------------------------------------------------------------------
    softplus(x, eta) = @. log(1+exp(x-eta))

    ##-----------------------------------------------------------------------------------
    softplusd(x, eta) = @. 1/(exp(x-eta)+1)


    ##===================================================================================
    ## step
    ##===================================================================================
    export step, stepd

    ##-----------------------------------------------------------------------------------
    step(x, eta = 0.5) = @. ifelse(x >= eta, 1, 0)

    ##-----------------------------------------------------------------------------------
    stepd{T<:Number}(x::T, eta) = @. ifelse(x == eta, typemax(T), 0)


    ##===================================================================================
    ## supp (support)
    ##===================================================================================
    export supp

    ##-----------------------------------------------------------------------------------
    function supp{T<:AbstractFloat}(v::Array{T, 1}, f::Function = (x::T) -> x)
        set_zero_subnormals(true)
        u = Array{T, 1}

        @inbounds for i = 1:size(v, 1)
			if abs(f(x)) == 0
				push!(u, v[i])
			end
		end

        return u
    end

    ##-----------------------------------------------------------------------------------
    function supp{T<:AbstractFloat}(vl::Array{Array{T, 1}, 1}, f::Function = (x::T) -> x)# supp for vector lists
        ul = Array{Array{T, 1}, 1}
        set_zero_subnormals(true)

        @inbounds for i = 1:size(vl, 1)
			if AND(abs(f(x)) .== 0)
				push!(ul, v[i])
			end
		end

		return ul
    end


	##===================================================================================
	##	Sawtooth wave
	##===================================================================================
	export saww

	##-----------------------------------------------------------------------------------
	saww{T<:AbstractFloat}(x::T, a::T, p::T, q::T) = (-2*a/pi)*atan(cot((x-q)*pi/p))	# a = amplitude, p = period, q = phase


	##===================================================================================
	##	Square wave
	##===================================================================================
	export sqw

	##-----------------------------------------------------------------------------------
	sqw{T<:AbstractFloat}(x::T, a::T, p::T, q::T) = a*sign(sin((x-q)*q))				# a = amplitude, p = period, q = phase


	##===================================================================================
	##	Triangle wave
	##===================================================================================
	export triw

	##-----------------------------------------------------------------------------------
	function triw{T<:AbstractFloat}(x::T, a::T, p::T, q::T)
		s1 = 2/p
		s2 = floor(s1*(x+q)+.5)
		return a*2*s1*((x+q)-s1*s2)*(s2 % 2 == 0 ? 1 : -1)
	end


	##===================================================================================
	## roots of polynomial
	##===================================================================================
	export rop

	##-----------------------------------------------------------------------------------
	function rop{T<:AbstractFloat}(c::Array{T, 1})
		s = size(c, 1)
		reverse!(c)

		@inbounds for i = s:-1:1
			if c[i] != 0
				break
			end
			s -= 1
		end

		m = shift(s-1, false)
		m[1, :] = (-c[2:end])/c[1]
		return map((x) -> 1/x, eigvals(m))
	end


	##===================================================================================
    ## roots of unity
    ##===================================================================================
	export rou

	##-----------------------------------------------------------------------------------
	rou(n::Integer) = [exp((2*pi*i*im)/n) for i = 0:(n-1)]


    ##===================================================================================
    ## levi civita tensor
    ##===================================================================================
    export lecit, ipc

    ##-----------------------------------------------------------------------------------
    lecit{T<:Number}(v::Array{T, 1}) = 0 == index_permutations_count(v) % 2 ? 1 : -1

    ##-----------------------------------------------------------------------------------
    function ipc{T<:Any}(v::Array{T, 1})                                                # index permutations count [3,4,5,2,1] -> [1,2,3,4,5]
        s = size(v, 1)                                                            		# 3 permutations needed
        t = linspace(1, s, s)
		c = 0

		while v != t
            @inbounds for i = 1:size(v, 1)
                if v[i] != i
                    s = find(v .== i)
                    if s != i
                        v[s] = v[i]
                        v[i] = i
                        c += 1
                    end
                end
            end
        end

		return c
    end
end
