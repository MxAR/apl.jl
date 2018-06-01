@everywhere module bin
	##===================================================================================
	##	bit dot (dot product of the binary representation of two integers)
	##===================================================================================
	export bit_dot

	##-----------------------------------------------------------------------------------
	function bit_dot{N<:Unsigned}(a::N, b::N)
		p = bits(a)
		q = bits(b)
		c = 0
		i = 1

		s = length(q)
		@inbounds while i <= s
			if p[i] == q[i] && p[i] == '1'
				c = c + 1
			end
			i = i + 1
		end

		return c
	end 


	##===================================================================================
	##	check status of the nth bit
	##===================================================================================
	export nbit_on, nbit_off

	##-----------------------------------------------------------------------------------
	function nbit_on{T<:Integer}(x::T, i::T) 
		return (x >> (i-1)) & T(1) == T(1)
	end

	##-----------------------------------------------------------------------------------
	function nbit_off{T<:Integer}(x::T, i::T) 
		return xor((x >> (i-1)), T(1)) == T(1)
	end


	##===================================================================================
	##	print binary representation
	##===================================================================================
	export bprint

	##-----------------------------------------------------------------------------------
	function bprint{T<:Integer}(x::T, bigendian::Bool = true)
		r = bits(x)

		if bigendian
			r = reverse(r)
		end

		println(STDOUT, r)
	end


	##===================================================================================
	##	sizeof (in bits)
	##===================================================================================
	export sizeofb

	##-----------------------------------------------------------------------------------
	function sizeofb(x::DataType) 
		return Base.mul_int(sizeof(x), 8)
	end


	##===================================================================================
	##	invert bit range
	##===================================================================================
	export ibit_range

	##-----------------------------------------------------------------------------------
	function ibit_range{N<:Unsigned, Z<:Integer}(x::N, lb::Z, ub::Z)
		i = lb - 1
		r = N(0)

		@inbounds while i <= (ub - 1)
			r = r + (N(1) << i)
			i = i + 1
		end

		r = Base.xor_int(r, x)
		return r
	end


	##===================================================================================
	##	exchange bit range
	##===================================================================================
	export ebit_range

	##-----------------------------------------------------------------------------------
	function ebit_range{N<:Unsigned, Z<:Integer}(x::N, y::N, lb::Z, ub::Z)
		i = lb - 1
		c = N(0)

		@inbounds while i <= (ub - 1)
			c = c + (N(1) << i)
			i = i + 1
		end

		d = Base.not_int(c)
		return (Base.xor_int((x & d), (y & c)), Base.xor_int((x & c), (y & d)))
	end
	

	##===================================================================================
	##	flip bit
	##===================================================================================
	export fbit

	##-----------------------------------------------------------------------------------
	function fbit{N<:Unsigned, Z<:Integer}(x::N, i::Z)
		return Base.xor_int(x, N(1 << (i - 1)))
	end


	##===================================================================================
	##	set bit
	##===================================================================================
	export sbit

	##-----------------------------------------------------------------------------------
	function sbit{N<:Unsigned, Z<:Integer}(x::N, i::Z, v::Bool) 
		return ((x >> (i - 1)) & N(1) == v) ? x : Base.xor_int(x, N(1 << (i - 1)))
	end


	##===================================================================================
	##	column wise binary merge
	##		r = A B C D E F G H I J K L M N O
	##			v[1] = A D G J M
	##			v[2] = B E H K N
	##			v[3] = C F I L O
	##===================================================================================
	export cb_merge

	##-----------------------------------------------------------------------------------
	function cb_merge{N<:Unsigned}(v::Array{N, 1})
		s = Base.mul_int(sizeof(N), 8)
		l = size(v, 1)
		c = UInt8(1)
		f = N(1)
		r = N(0)

		i = 1
		@inbounds while i <= s && c <= s
			j = 1
			while j <= l && c <= s
				if Bool((v[j] >> (i - 1)) & N(1))
					r = Base.xor_int(r, f)
				end

				c = c + UInt8(1)
				f = f << 1
				j = j + 1
			end
			i = i + 1
		end

		return r
	end


	##===================================================================================
	##	split column wise binary merge again (cb_merge reverse)
	##		d = number of dimensions
	##===================================================================================
	export cb_split

	##-----------------------------------------------------------------------------------
	function cb_split{N<:Unsigned, Z<:Integer}(x::N, d::Z)
		s = Base.mul_int(sizeof(N), 8)
		r = Array{N, 1}(d)
		c = UInt8(1)
		f = N(1)

		i = 1
		@inbounds while i <= d
			r[i] = N(0)
			i = i + 1
		end


		i = 1
		@inbounds while i <= s && c <= s
			j = 1
			while j <= d && c <= s
				if Bool((x >> (c - 1)) & N(1))
					r[j] = xor(r[j], f)
				end
				
				c = c + UInt8(1)
				j = j + 1
			end
			f = f << 1
			i = i + 1
		end

		return r
	end


	##===================================================================================
	##	gray codes (forward/backward)
	##===================================================================================
	export grayf, grayb

	##-----------------------------------------------------------------------------------
	grayf{T<:Unsigned}(x::T) = xor(x, (x >> 1))

	##-----------------------------------------------------------------------------------
	function grayb{T<:Unsigned}(x::T)
		y = x; r = x

		while y != T(0)
			y >>= 1
			r = xor(r, y)
		end

		return r
	end


	##===================================================================================
	##	hilbert curve (forward/backward)
	##===================================================================================
	export hilbert_cf, hilbert_cb

	##-----------------------------------------------------------------------------------
	function hilbert_cf{T<:Unsigned}(v::Array{T, 1})									# TODO support for different output types is needed
		b = sizeof(T)*8
		r = deepcopy(v)

		for i = (b-1):-1:1, j = 1:size(r, 1)
			if nbit_on(r[j], i)
				r[1] = ibit_range(r[1], i+1, b-1)
			else
				s = ebit_range(r[1], r[j], i+1, b-1)
				r[1] = s[1]
				r[j] = s[2]
			end
			println(r)
		end

		return grayf(cb_merge(r))
	end

	##-----------------------------------------------------------------------------------
	function hilbert_cb{T<:Unsigned}(v::T, d::Integer)
		r = cb_split(grayb(v), d)
		b = sizeof(T)*8

		println(r)

		for i = 1:b, j = d:-1:1
			if nbit_on(r[j], i)
				r[1] = ibit_range(r[1], i+1, b)
			else
				s = ebit_range(r[1], r[j], i+1, b)
				r[1] = s[1]
				r[j] = s[2]
			end
			println(r)
		end

		return r
	end
end
