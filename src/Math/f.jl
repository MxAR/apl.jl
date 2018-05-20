@everywhere module f
	##===================================================================================
	##	orthorgonal complement
	##		(each row represents a vector)
	##===================================================================================
	export orthc

	##-----------------------------------------------------------------------------------
	function orthc{N<:Number}(m::Array{N, 2})
		x = Array{(N<:Real ? Float64 : Complex128), 2}(lufact(m)[:U])
		s = size(m)
		d = s[2]-s[1]

		i = 0
		@inbounds while x[s[1]-i, s[2]] == 0
			d = d + 1 
			i = i + 1
		end

		i = s[2] - d
		@inbounds while i > 1
			j = i

			while j <= s[2]
				x[i, j] = x[i, j] / x[i, i]

				k = i-1
				while k >= 1
					x[i-k, j] = x[i-k, j] - x[i-k, i] * x[i, j] 	
					k = k - 1
				end 
				
				j = j + 1
			end

			i = i - 1
		end

		@inbounds while i <= s[2]
			x[1, i] = x[1, i] / x[1, 1]
			i = i + 1
		end 

		n = zeros(d, s[2])
		i = s[1]+1
		j = 1

		@inbounds while j <= d
			n[j, i] = 1
			i = i + 1
			j = j + 1
		end

		i = s[1]+1
		j = 1
		k = 1 
		l = 1

		@inbounds while j <= s[2]-d
			while i <= s[2]
				n[k, l] = -x[j, i]
				i = i + 1
				k = k + 1
			end
			i = s[1]+1
			k = 1
			j = j + 1
			l = l + 1
		end

		return n
	end 


	##===================================================================================
	##	nth root of matrix
	##		m: must be a square matrix
	##===================================================================================
	export nroot, nroot_posdef

	##-----------------------------------------------------------------------------------
	function nroot{C<:Number, Z<:Integer}(m::Array{C, 2}, n::Z = 2)
		T = Complex128
		g = Base.LinAlg.Eigen{T,T,Array{T,2},Array{T,1}}(eigfact(m; permute=false, scale=false))
		s = size(m, 1)
		p = T(1/n)
		
		r = Array{T, 2}(s, s)
		i = 1

		@inbounds while i <= s
			j = 1

			while j < i 
				r[i, j] = T(0)
				r[j, i] = r[i, j]
				j = j + 1
			end 

			r[i, i] = g[:values][i]^p
			i = i + 1
		end
		
		r = g[:vectors]*r*inv(g[:vectors])
		return r
	end 

	##-----------------------------------------------------------------------------------
	function nroot_posdef{C<:Number, Z<:Integer}(m::Array{C, 2}, n::Z = 2)
		T = Float64
		g = Base.LinAlg.Eigen{T,T,Array{T,2},Array{T,1}}(eigfact(m; permute=false, scale=false))
		s = size(m, 1)
		p = T(1/n)

		r = Array{T, 2}(s, s)
		i = 1

		@inbounds while i <= s
			j = 1

			while j < i
				r[i, j] = T(0)
				r[j, i] = r[i, j]
				j = j + 1
			end

			@fastmath r[i, i] = g[:values][i]^p
			i = i + 1
		end

		r = g[:vectors]*r*inv(g[:vectors])
		return r
	end


	##===================================================================================
	##	cleaning of covariance matrices (RIE)
	##		m: covariance matrix (remember m has to be positive definite)
	##		t: number of observations
	##		d: flag whether or not double or single precision complex numbers shall be used
	##===================================================================================
	export ccovm

	##-----------------------------------------------------------------------------------
	function ccovm{T<:Number, N<:Integer}(m::Array{T, 2}, t::N)
		C = Complex128
		R = Float64

		s = size(m, 1)
		q = s / t
		c = 1 - q

		g = Base.LinAlg.Eigen{R,R,Array{R,2},Array{R,1}}(eigfact(m; permute=false, scale=false))
		@inbounds ln = g[:values][1]
		@fastmath vr = ln / (1 - q^0.5)^2
		@fastmath lp = vr * (1 + q^0.5)^2
		@fastmath a = im / (s^.5)
		d = 2 * vr

		z = Array{C, 1}(s)
		x = Array{R, 1}(s)
		i = 1

		while i <= s
			@inbounds z[i] = g[:values][i] - a
			i = i + 1
		end

		b = C(0)
		h = R(0)
		i = 1

		while i <= s
			j = 1
			while j <= i-1
				@inbounds b = b + 1 / (z[i] - g[:values][j])
				j = j + 1
			end

			j = j + 2
			while j <= s
				@inbounds b = b + 1 / (z[i] - g[:values][j])
				j = j + 1
			end

			@inbounds x[i] = g[:values][i] / abs2(c + q * z[i] * (b / s))
			@inbounds @fastmath h = abs2(c + (z[i] - vr * c - ((z[i] - ln)*(z[i] - lp))^.5) / d)
			@inbounds h = h * vr / g[:values][i]

			if h > 1
				@inbounds x[i] = x[i] * h
			end

			i = i + 1
			b = C(0)
			h = R(0)
		end

		@inbounds r = x[1] * g[:vectors][:, 1] * g[:vectors][:, 1]'
		i = 2

		while i <= s
			@inbounds r = r + x[i] * g[:vectors][:, i] * g[:vectors][:, i]'
			i = i + 1
		end

		return r
	end

    ##===================================================================================
	##	convert price matrix of assets to return matrix
	##		npmtrm: normalizes the return matrix
	##===================================================================================
	export pmtrm, npmtrm

	##-----------------------------------------------------------------------------------
	function pmtrm{R<:Number}(m::Array{R, 2})
		s = size(m)
		l = s[1] - 1
		n = s[2]
		
		r = Array{R, 2}(l, n)
		i = 1
		j = 1

		while i <= l
			while j <= n
				@inbounds r[i, j] = m[i + 1, j] / m[i, j] - 1
				j = j + 1
			end
			j = 1
			i = i + 1
		end

		return r
	end 

	##-----------------------------------------------------------------------------------
	function npmtrm{R<:Number}(m::Array{R, 2})
		s = size(m)
		l = s[1] - 1
		n = s[2]

		r = Array{R, 2}(l, n)
		d = zeros(R, n)
		a = zeros(R, n)

		i = 1
		j = 1

		while i <= l
			while j <= n
				@inbounds begin 
					r[i, j] = m[i + 1, j] / m[i, j] - 1
					a[j] = a[j] + r[i, j]
				end 
				j = j + 1
			end
			j = 1
			i = i + 1
		end

		i = 1

		while i <= n
			@inbounds a[i] = a[i] / l
			i = i + 1 
		end

		i = 1

		while i <= l
			while j <= n
				@inbounds begin 
					r[i, j] = r[i, j] - a[j]		
					d[j] = d[j] + r[i, j]^2
				end
				j = j + 1
			end
			j = 1
			i = i + 1
		end

		i = 1

		while i <= n
			@inbounds @fastmath d[i] = d[i]^.5
			i = i + 1
		end

		i = 1

		while i <= l
			while j <= n
				@inbounds r[i, j] = r[i, j] / d[j]
				j = j +  1
			end
			j = 1
			i = i + 1
		end

		return r
	end 


	##===================================================================================
	##	random cnn matrix
	##===================================================================================
	export rcnnm

	##-----------------------------------------------------------------------------------
	function rcnnm{N<:Integer, R<:AbstractFloat}(inrn::N, onrn::N, trh::R = -.5, hnrn::Array{N, 1} = Array{N, 1}())
		s = size(hnrn, 1)
		if s == 0
			r = zeros(R, onrn+1, inrn+1)
			r[1:onrn, 1:inrn] = randn(R, onrn, inrn)./3
			r[1:onrn, end] = trh
			r[end, end] = R(1)
			return r
		else
			append!(hnrn, onrn)
			op = hnrn[1]
			ip = inrn

			r = zeros(R, op+1, ip+1)
			r[1:op, 1:ip] = randn(R, op, ip)./3
			r[1:op, end] = trh
			r[end, end] = R(1)

			@inbounds for i = 2:(s+1)
				ip = op
				op = hnrn[i]
				
				t = zeros(R, op+1, ip+1)
				t[1:op, 1:ip] = randn(R, op, ip)./3
				t[1:op, end] = trh
				t[end, end] = R(1)

				r = t*r
			end

			return r
		end
	end 


	##===================================================================================
	##	nth fibonacci number
	##===================================================================================
	export fib

	##-----------------------------------------------------------------------------------
	function fib{T<:Integer}(n::T)
		p = 1.61803398874989484820458683437
		@fastmath return round(((p^n)-((-p)^(-n)))/(2.23606797749978969640917366873))
	end 


	##===================================================================================
	##	dct denosing
	##===================================================================================
	export dctdnoise

	##-----------------------------------------------------------------------------------
	function dctdnoise{R<:AbstractFloat}(v::Array{R, 1}, k = R(3), lowpass::Bool = true, highpass::Bool = true)
		s = size(v, 1)
		t = Array{R, 1}(s)
		p = plan_dct(v)

		A_mul_B!(t, p, v)

		m = R(0)
		d = R(0)
		i = 1

		while i <= s
			@inbounds m = m + t[i]
			i = i + 1
		end 

		m /= s
		i = 1

		while i <= s
			@inbounds d = d + (t[i]-m)^2
			i = i + 1
		end

		@fastmath d = k*(d/s)^.5
		i = 1

		while i <= s
			@inbounds t[i] = t[i] - m
			i = i + 1
		end 
		
		i = 1

		if highpass
			@inbounds while i <= s
				if t[i] <= -d
					t[i] = R(0)
				end
				i = i + 1
			end
			i = 1
		end

		if lowpass
			@inbounds while i <= s
				if t[i] >= d
					t[i] = R(0)
				end 
				i += 1
			end
		end

		r = Array{R, 1}(s)
		A_ldiv_B!(r, p, t)
		return r
	end

	##===================================================================================
	##	fft denosing
	##===================================================================================
	export fftdnoise

	##-----------------------------------------------------------------------------------
	function fftdnoise{R<:AbstractFloat}(v::Array{R, 1}, k = R(3))
		s = size(v, 1)
		a = zeros(R, s)
		t = fft(v)
		i = Int(1)
		m = R(0)
		d = R(0)		

		@inbounds while i <= s
			a[i] = abs(t[i])
			m += a[i]
			i += 1
		end 

		i = Int(1)
		m /= s

		@inbounds while i <= s
			d += (a[i]-m)^2
			i += 1
		end

		i = Int(1)
		d = k*sqrt(d/s)

		@inbounds while i <= s
			if abs(a[i]-m) >= d
				t[i] = Complex{R}(0)
			end
			i += 1
		end

		return real.(ifft(t))
	end 


	##===================================================================================
    ##  greatest common devisor
    ##===================================================================================
    export gcd, coprime

    ##-----------------------------------------------------------------------------------
    function gcd{T<:Integer}(x::T, y::T)
        z0 = T(0)
        z1 = T(0)
        s = T(0)
        
        if x < y
            z0 = y
            z1 = x
        else
            z0 = x
            z1 = y
        end
 
        @inbounds while z1 != 0
            s = z0 % z1
            z0 = z1
            z1 = s
        end

        return z0
    end

    ##-----------------------------------------------------------------------------------
    coprime{T<:Integer}(x::T, y::T) = gcd(x, y) == 1


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
		c = UInt(0)

        while x != 1
            x = x & 1 == 1 ? 3*x + 1: x >> 1
            c = c + 1
        end

        return c
    end
	

	##===================================================================================
	##	extended binomial coefficient calculation
	##===================================================================================
	export binom

	##-----------------------------------------------------------------------------------
	function binom{T<:Number}(n::T, k::T)
		if T <: Integer 
			if k < 0
				return 0
			elseif n < 0
				return (k%2==0?1:-1)*(factorial(k-n-1))/(factorial(k)*factorial(-n-1))		
			end
		end
		
		if n < 0
			r0 = T(1)
			r1 = T(2)

			@inbounds for i = 0:(n-1)
				r0 *= (n-i)
			end 

			@inbounds for i = 3:k
				r1 *= i
			end

			return r0/r1
		end 

		if n > 0
			return ggamma(n+T(1))/(ggamma(k+T(1))ggamma(n-k+T(1)))
		else
			return T(1)
		end
	end 


	##===================================================================================
	##	generalized gamma function
	##===================================================================================
	export ggamma
	
	##-----------------------------------------------------------------------------------
	function ggamma{T<:Number}(x::T)
		if x >= 0
			if T <: Integer
				return factorial(x-1)
			else
				return gamma(x)
			end
		else
			d = Int(ceil(abs(x)))
			r = gamma(x+d)

			@inbounds for i = 0:(d-1)
				r /= (x+i)
			end

			return r
		end
	end 


	##===================================================================================
	##	partition norm
	##		assuming that the partition is already sorted
	##===================================================================================
	export partnorm

	##-----------------------------------------------------------------------------------
	function partnorm{R<:Number}(v::Array{R, 1})
		b = -R(Inf)
		s = R(0)

		@inbounds for i = 2:size(v, 1)
			s = v[i]-v[i-1]
			if s > b
				b = s
			end 
		end

		return b
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
    ## e^p mod m
    ##===================================================================================
    export mpow

    ##-----------------------------------------------------------------------------------
    function mpow{N<:Integer}(bse::N, exp::N, mod::N)
        if mod == 1
            return 0
        else
            c = 1
            @inbounds for i = 1:exp
                c = (c*bse)%mod
            end
            return c
        end
    end


	##===================================================================================
	##	kingsman equation
	##===================================================================================
	export king

	##-----------------------------------------------------------------------------------
	king{T<:AbstractFloat, N<:Integer}(p::T, ca::T, cs::T, c::N) = ((p^(sqrt(2*(c+1))))/(1-p))x*(0.5)*((ca^2)+(cs^2))


    ##===================================================================================
	##	divisibility test
	##===================================================================================
	export div

	##-----------------------------------------------------------------------------------
	div{T<:Integer}(x::T, y::T) = y%x==0


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
    ## trapezoidal rule
    ##===================================================================================
    export trapezr

    ##-----------------------------------------------------------------------------------
    function trapezr{R<:AbstractFloat, N<:Integer}(a::R, b::R, f::Function, n::N)
        d = (b-a)/n
        ck = a+d
        lk = a
        r = R(0)

        @inbounds for i = 1:n
            r += f(lk)+f(ck)
            lk = ck
            ck += d
        end

        return (d*r)/2
    end


    ##===================================================================================
    ## midpoint rule
    ##===================================================================================
    export midptr

    ##-----------------------------------------------------------------------------------
    function midptr{R<:AbstractFloat, N<:Integer}(a::R, b::R, f::Function, n::N)
        d = (b-a)/n
        k = a+(d/2)
        r = R(0)

        @inbounds for i = 1:n
            r += f(k)
            k += d
        end

        return d*r
    end


    ##===================================================================================
    ## simpson's rule
    ##===================================================================================
    export simpsonr

    ##-----------------------------------------------------------------------------------
    function simpsonr{R<:AbstractFloat, N<:Integer}(a::R, b::R, f::Function, n::N)
        @assert(n%2==0, "n is not even")
        d = (b-a)/n; di = 2*d; r = R(0)
        j = Array{R, 1}([a, a+d, a+di])

        @inbounds for i = 1:N(n/2)
            r += f(j[1])+4*f(j[2])+f(j[3])
            j += di
        end

        return (d*r)/3
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
	rou{N<:Integer}(n::N) = [exp((2*pi*i*im)/n) for i = 0:(n-1)]


	##===================================================================================
	##	cross product
	##===================================================================================
	export crossp

	##-----------------------------------------------------------------------------------
	function crossp{T<:Number}(u::Array{T, 1}, v::Array{T, 1})
		return [(u[2]*v[3])-(u[3]-v[2]), (u[3]*v[1])-(u[1]*v[3]), (u[1]*v[2])-(u[2]*v[1])]
	end


    ##===================================================================================
    ## ln
    ##===================================================================================
	export ln

	##-----------------------------------------------------------------------------------
	ln{T<:Number}(n::T) = log(n)


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
