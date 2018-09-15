@everywhere module f
	##===================================================================================
	## using directives
	##===================================================================================
	using LinearAlgebra
	using FFTW


	##===================================================================================
	## prime factorisation 
	##	- not determinitistic
	##	- using an augumented Pollard-Strassen method
	##===================================================================================
	export pfactor

	##-----------------------------------------------------------------------------------
	function pfactor(x::Z) where Z<:Integer
		r = Array{Z, 1}()
		n = UInt(abs(x))
		T = UInt

		if x < 0
			append!(r, -1)
		end

		while true
			if n % 2 == 0
				append!(r, 2)
				n = n >> 1
			else
				a = T(rand(1:99999))
				b = T(rand(1:99999))
				c = T(1)

				while c == 1
					a = T((1 + a^2) % n)
					b = T((1 + ((1 + b^2) % n)^2) % n)
					c = f.gcd(abs(a - b), n)
				end

				if c == n
					append!(r, n)
					break
				else
					append!(r, c)
					n = T(n / c)
				end
			end
		end

		return r
	end


	##===================================================================================
	##	orthorgonal complement
	##		(each row represents a vector)
	##===================================================================================
	export orthc

	##-----------------------------------------------------------------------------------
	function orthc(m::Array{T, 2}) where T<:Number
		x = Array{(T<:Real ? Float64 : Complex128), 2}(lufact(m)[:U])
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
	##	cleaning of covariance matrices (RIE)
	##		m: covariance matrix (remember m has to be positive definite)
	##		t: number of observations
	##		d: flag whether or not double or single precision complex numbers shall be used
	##===================================================================================
	export ccovm

	##-----------------------------------------------------------------------------------
	function ccovm(m::Array{T, 2}, t::Z) where T<:Number where Z<:Integer
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
	function pmtrm(m::Array{T, 2}) where T<:Number
		s = size(m)
		l = s[1] - 1
		n = s[2]
		
		r = Array{T, 2}(l, n)
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
	function npmtrm(m::Array{T, 2}) where T<:Number
		s = size(m)
		l = s[1] - 1
		n = s[2]

		r = Array{T, 2}(l, n)
		d = zeros(T, n)
		a = zeros(T, n)

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
	##	random cnn matrix (not yet optimized)
	##===================================================================================
	export rcnnm

	##-----------------------------------------------------------------------------------
	function rcnnm(inrn::Z, onrn::Z, trh::R = -.5, hnrn::T = T()) where T<:Array{Z, 1} where Z<:Integer where R<:AbstractFloat
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
	function fib(n::Z) where Z<:Integer
		p = 1.61803398874989484820458683437
		@fastmath return round(((p^n)-((-p)^(-n)))/(2.23606797749978969640917366873))
	end 


	##===================================================================================
	##	dct denosing
	##===================================================================================
	export dctdnoise

	##-----------------------------------------------------------------------------------
	function dctdnoise(v::Array{R, 1}, k::Z, lowpass::Bool, highpass::Bool) where R<:AbstractFloat where Z<:Integer
		s = size(v, 1)
		p = plan_dct(v)
		t =  p * v

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
	function fftdnoise(v::Array{R, 1}, k::R) where R<:AbstractFloat
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
    function gcd(x::Z, y::Z) where Z<:Integer
        z0 = Z(0)
        z1 = Z(0)
        s = Z(0)
        
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
    coprime(x::Z, y::Z) where Z<:Integer = gcd(x, y) == 1


    ##===================================================================================
    ##  collatz conjecture
    ##===================================================================================
    export collatz

    ##-----------------------------------------------------------------------------------
    function collatz(x::Z) where Z<:Integer
		c = UInt(0)

        while x != 1
            x = x & 1 == 1 ? 3*x + 1 : x >> 1
            c = c + 1
        end

        return c
    end
	

	##===================================================================================
	##	extended binomial coefficient calculation
	##===================================================================================
	export binomf, binomi

	##-----------------------------------------------------------------------------------
	function binomf(n::R, k::R) where R<:AbstractFloat
		if n < 0
			r0 = R(1)
			r1 = R(2)

			@inbounds for i = 0:(n-1)
				r0 *= (n-i)
			end 

			@inbounds for i = 3:k
				r1 *= i
			end

			return r0/r1
		end 

		if n > 0
			return ggamma(n+R(1))/(ggamma(k+R(1))ggamma(n-k+R(1)))
		else
			return R(1)
		end
	end

	##-----------------------------------------------------------------------------------
	function binom(n::Z, k::Z) where Z<:Integer
		r = Z(NaN)

		if k < 0
			r = 0
		elseif n > 0
			r = ggamma(n+Z(1))/(ggamma(k+Z(1))*ggamma(n-k+Z(1)))
		elseif n < 0
			b = -n + k -1
			i = -n
			r = 1

			while i <= b
				r = r * i
				i = i + 1
			end

			b = 1
			i = 2

			while i <= k
				b = b * i
				i = i + 1
			end

			r = (k%2==0 ? 1 : -1) * Z(r / b)
		end

		return r
	end


	##===================================================================================
	##	generalized gamma function
	##===================================================================================
	export ggamma
	
	##-----------------------------------------------------------------------------------
	function ggamma(x::T) where T<:Number
		if x >= 0
			if T <: Integer
				return AbstractFloat(factorial(x - 1))
			else
				return gamma(x)
			end
		else
			d = Int(ceil(abs(x)))
			r = gamma(x + d)

			@inbounds for i = 0:(d-1)
				r = r / (x + i)
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
	function partnorm(v::Array{T, 1}) where T<:Number
		b = -T(Inf)
		s = T(0)

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
    function dcd(x::R) where R<:AbstractFloat
        set_zero_subnormals(true)
		return R(x == 0 ? Inf : 0)
	end


    ##===================================================================================
    ## kronecker delta
    ##===================================================================================
    export ked

    ##-----------------------------------------------------------------------------------
	ked(x::T, y::T) where T<:Number = T(x == y ? 1 : 0)


    ##===================================================================================
    ## e^p mod m
    ##===================================================================================
    export mpow

    ##-----------------------------------------------------------------------------------
    function mpow(bse::Z, exp::Z, mod::Z) where Z<:Integer
        if mod == 1
            return 0
        else
            c = 1
            
			@inbounds for i = 1:exp
                c = (c * bse) % mod
            end

            return c
        end
    end


	##===================================================================================
	##	kingsman equation
	##===================================================================================
	export king

	##-----------------------------------------------------------------------------------
	function king(x::R, p::R, ca::R, cs::R, c::Z) where R<:AbstractFloat where Z<:Integer
		@fastmath return (p^sqrt(2*c+2))*x*((ca^2)+(cs^2))/(2-2*p)
	end


    ##===================================================================================
	##	divisibility test
	##===================================================================================
	export divt

	##-----------------------------------------------------------------------------------
	divt(x::Z, y::Z) where Z<:Integer = y % x == 0


    ##===================================================================================
    ## sigmoid
    ##===================================================================================
    export sigmoid, sigmoid_derivate

    ##-----------------------------------------------------------------------------------
	sigmoid(x::T, eta::T = T(0)) where T<:Number = 1 / (1 + exp(-(x - eta)))

    ##-----------------------------------------------------------------------------------
	sigmoid_derivate(x::T, eta::T = T(0)) where T<:Number =  sigmoid(x, eta) * (1 - sigmoid(x, eta))


    ##===================================================================================
    ## norm derivate
    ##===================================================================================
    export normd

    ##-----------------------------------------------------------------------------------
	function normd(v::Array{T, 1}, p::Z = Z(2)) where T<:Number where Z<:Integer 
		s = size(v, 1)
		r = Array{T, 1}(s)
		i = 1

		while i <= s
			@inbounds begin 
				r[i] = v[i]
				if r[i] != T(0)
					r[i] = r[i] / norm(v, p)
				end
				@fastmath r[i] = r[i]^(p - 1)
			end
			i = i + 1
		end 

		return r		
	end 

    ##===================================================================================
    ## radial basis functions
    ##===================================================================================
    export rbf_gauss, rbf_gaussdl, rbf_gaussdd, rbf_triang, rbf_cos_decay
    export rbf_psq, rbf_inv_psq, rbf_inv_sq, rbf_tps

    ##-----------------------------------------------------------------------------------
	function rbf_gauss(delta::T, lambda::T = T(1)) where T<:Number
		return exp(-(delta / (2 * lambda))^2)
	end 

    ##-----------------------------------------------------------------------------------
	function rbf_gaussdl(delta::T, lambda::T = T(1)) where T<:Number
        d = delta^2
        l = lambda^2
        return (d / (l * lambda))*exp(-d / l)
    end

    ##-----------------------------------------------------------------------------------
	function rbf_gaussdd(delta::T, lambda::T = T(1)) where T<:Number
        l = lambda^2
        return (delta / l) * exp(-(delta^2) / (2 * l))
    end

    ##-----------------------------------------------------------------------------------
	function rbf_triang(delta::T, lambda::T = T(1)) where T<:Number
		return delta > lambda ? T(0) : T(1 - (delta / lambda))
	end

    ##-----------------------------------------------------------------------------------
	function rbf_cos_decay(delta::T, lambda::T = (1)) where T<:Number
		@fastmath return delta > lambda ? T(0) : T(((cos((pi * delta) / (2 * lambda))) + 1) / 2)
	end

    ##-----------------------------------------------------------------------------------
	function rbf_psq(delta::T, lambda::T = T(1)) where T<:Number
		@fastmath return sqrt(1 + (lambda * delta)^2)
	end 

    ##-----------------------------------------------------------------------------------
	function rbf_inv_psq(delta::T, lambda::T = T(1)) where T<:Number 
		@fastmath return (1 + (lambda * delta)^2)^-.5
	end 

    ##-----------------------------------------------------------------------------------
	function rbf_inv_sq(delta::T, lambda::T = T(1)) where T<:Number
		@fastmath return 1/(1 + (lambda * delta)^2)
	end

    ##-----------------------------------------------------------------------------------
	function rbf_tps(delta::T, expt::T = T(2)) where T<:Number
		@fastmath return (delta^expt)*log(delta)
	end


    ##===================================================================================
    ## ramp
    ##===================================================================================
    export ramp, rampd

    ##-----------------------------------------------------------------------------------
	ramp(x::T, eta::T) where T<:Number = max(T(0), x-eta)

    ##-----------------------------------------------------------------------------------
    rampd(x::T, eta::T) where T<:Number = x-eta > 0 ? T(1) : T(0)


    ##===================================================================================
    ## trapezoidal rule
    ##===================================================================================
    export trapezr

    ##-----------------------------------------------------------------------------------
    function trapezr(a::R, b::R, f::Function, n::Z) where R<:AbstractFloat where Z<:Integer
        d = (b - a) / n
        ck = a + d
        lk = a

        r = R(0)

        @inbounds for i = 1:n
            r = r + f(lk) + f(ck)
            lk = ck
            ck = ck + d
        end

        return (d * r) / 2
    end


    ##===================================================================================
    ## midpoint rule
    ##===================================================================================
    export midptr

    ##-----------------------------------------------------------------------------------
    function midptr(a::R, b::R, f::Function, n::Z) where R<:AbstractFloat where Z<:Integer
        d = (b - a) / n
        k = a + (d / 2)
        r = R(0)

        @inbounds for i = 1:n
            r = r + f(k)
            k = k + d
        end

        return d * r
    end


    ##===================================================================================
    ## simpson's rule
	##	n must be even
    ##===================================================================================
    export simpsonr

    ##-----------------------------------------------------------------------------------
    function simpsonr(a::R, b::R, f::Function, n::Z) where R<:AbstractFloat where Z<:Integer
        d = (b - a) / n 
		di = 2 * d 
		r = R(0)
        
		j = Array{R, 1}([a, a+d, a+di])

        @inbounds for i = 1:N(n/2)
			r = r + f(j[1]) + (4 * f(j[2])) + f(j[3])
            j = j + di
        end

        return (d * r) / 3
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
    stepd(x::T, eta) where T<:Number = @. ifelse(x == eta, typemax(T), 0)


    ##===================================================================================
    ## supp (support)
	## second supp is for vector lists
    ##===================================================================================
    export supp

    ##-----------------------------------------------------------------------------------
    function supp(v::Array{R, 1}, f::Function = (x::R) -> x) where R<:AbstractFloat
        set_zero_subnormals(true)
        u = Array{R, 1}

        @inbounds for i = 1:size(v, 1)
			if abs(f(x)) == 0
				push!(u, v[i])
			end
		end

        return u
    end

    ##-----------------------------------------------------------------------------------
    function supp(vl::Array{Array{R, 1}, 1}, f::Function = (x::R) -> x) where R<:AbstractFloat
        ul = Array{Array{R, 1}, 1}
        set_zero_subnormals(true)

        @inbounds for i = 1:size(vl, 1)
			if AND(abs(f(x)) .== 0)
				push!(ul, v[i])
			end
		end

		return ul
    end


	##===================================================================================
	## Sawtooth wave
	##	a = amplitude, p = period, q = phase
	##===================================================================================
	export saww

	##-----------------------------------------------------------------------------------
	saww(x::R, a::R, p::R, q::R) where R<:AbstractFloat = (-2 * a / pi) * atan(cot((x - q) * pi / p))


	##===================================================================================
	## Square wave
	##	a = amplitude, p = period, q = phase
	##===================================================================================
	export sqw

	##-----------------------------------------------------------------------------------
	sqw(x::R, a::R, p::R, q::R) where R<:AbstractFloat = a * sign(sin((x - q) * q))


	##===================================================================================
	## Triangle wave
	##	a = amplitude, p = period, q = phase	
	##===================================================================================
	export triw

	##-----------------------------------------------------------------------------------
	function triw(x::R, a::R, p::R, q::R) where R<:AbstractFloat
		s1 = 2 / p
		s2 = floor(s1 * (x + q) + .5)
		return a * 2 * s1 * ((x + q) - s1 * s2) * (s2 % 2 == 0 ? 1 : -1)
	end


	##===================================================================================
	## roots of polynomial
	##===================================================================================
	export rop

	##-----------------------------------------------------------------------------------
	function rop(c::Array{R, 1}) where R<:AbstractFloat
		s = size(c, 1)
		reverse!(c)

		@inbounds for i = s:-1:1
			if c[i] != 0
				break
			end
			s -= 1
		end

		m = shift(s - 1, false)
		m[1, :] = (-c[2:end]) / c[1]
		return map((x) -> 1/x, eigvals(m))
	end


	##===================================================================================
    ## roots of unity
    ##===================================================================================
	export rou

	##-----------------------------------------------------------------------------------
	rou(n::Z) where Z<:Integer = [exp((2 * pi * i * im) / n) for i = 0:(n-1)]


	##===================================================================================
	##	cross product
	##===================================================================================
	export crossp

	##-----------------------------------------------------------------------------------
	function crossp(u::Array{T, 1}, v::Array{T, 1}) where T<:Number
		return [(u[2]*v[3])-(u[3]-v[2]), (u[3]*v[1])-(u[1]*v[3]), (u[1]*v[2])-(u[2]*v[1])]
	end


    ##===================================================================================
    ## ln
    ##===================================================================================
	export ln

	##-----------------------------------------------------------------------------------
	ln(n::T) where T<:Number = log(n)


    ##===================================================================================
    ## levi civita tensor
	## 	index permutations count [3,4,5,2,1] -> [1,2,3,4,5] (3 permutation)
	##===================================================================================
    export lecit, index_permutations_count

    ##-----------------------------------------------------------------------------------
    lecit(v::Array{Z, 1}) where Z<:Integer = 0 == index_permutations_count(v) % 2 ? 1 : -1

    ##-----------------------------------------------------------------------------------
    function index_permutations_count(v::Array{Z, 1}) where Z<:Integer                                                
        s = size(v, 1)
        t = linspace(1, s, s)
		c = 0

		while v != t
            @inbounds for i = 1:size(v, 1)
                if v[i] != i
                    s = find(v .== i)
                    if s != i
                        v[s] = v[i]
                        v[i] = i
                        c = c + 1
                    end
                end
            end
        end

		return c
    end
end
