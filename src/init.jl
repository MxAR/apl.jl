function t2a(v::Array{UInt64, 1}, b::UInt)
	l = size(v, 1)

	t = UInt64(v[l] >> 1)
	for i = l:-1:2
		v[i] = xor(v[i], v[i-1])
	end
	v[1] = t

	q = UInt64(2)
	while q != (2 << (b-1))
		p = UInt64(q - UInt64(1))
		for i = l:-1:1
			if v[i] & q == 0
				t = UInt64(xor(v[1], v[i]) & p)
				v[1] = xor(v[1], t)
				v[i] = xor(v[i], t)
			else
				v[1] = xor(v[1], p)
			end
		end
		q = UInt64(q << 1)
	end

	return v
end

function a2t(v::Array{UInt64, 1}, b::UInt)
	m = UInt64(1 << (b-1))
	l = size(v, 1)
	q = m

	while q > UInt64(1)
		p = UInt64(q - UInt64(1))
		for i = 1:l
			if v[i] & q == 0
				t = UInt64(xor(v[1], v[i]) & p)
				v[1] = xor(v[1], t)
				v[i] = xor(v[i], t)
			else
				v[1] = xor(v[1], p)
			end
		end
		q = UInt64(q >> 1)
	end

	for i = 2:l v[i] = xor(v[i], v[i-1]) end
	t = UInt64(0)
	q = m

	while q > UInt64(1)
		if !(v[l-1] & q == 0)
			t = UInt64(xor(t, q - UInt64(1)))
		end
		q = UInt64(q >> 1)
	end

	for i = 1:l
		v[i] = xor(v[i], t)
	end

	return v
end

grayf{T<:Unsigned}(x::T) = xor(x, (x >> 1))
function grayb{T<:Unsigned}(x::T)
	y = x
	while y != 0
		y =  y >> 1
		x = xor(x, y)
	end
	return x
end
