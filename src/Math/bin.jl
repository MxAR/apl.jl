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
	##===================================================================================
	export cb_merge

	##-----------------------------------------------------------------------------------
	function cb_merge{T<:Unsigned}(v::Array{T, 1})
		s = sizeofb(T)																	# r = A B C D E F G H I J K L M N O
		c = UInt8(1)																	# 	v[1] = A D G J M
		f = T(1)																		# 	v[2] = B E H K N
		r = T(0)																		# 	v[3] = C F I L O

		for i = 1:s, j = 1:size(v, 1)
			if nbit_on(v[j], i)
				r = xor(r, f)
			end

			if c == s
				return r
			end

			f <<= 1
			c += 1
		end
	end


	##===================================================================================
	##	split column wise binary merge again (cb_merge reverse)
	##===================================================================================
	export cb_split

	##-----------------------------------------------------------------------------------
	function cb_split{T<:Unsigned}(x::T, d::Integer)										# d = number of dimensions
		r = Array{T, 1}(zeros(d))
		s = sizeofb(T)
		c = UInt8(1)
		f = T(1)

		for i = 1:s
			for j = 1:d
				if nbit_on(x, c)
					r[j] = xor(r[j], f)
				end

				if c == s
					return r
				end
				c += 1
			end
			f <<= 1
		end
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
