using mavg
using macd
using rsi

#v = 1 + (0.1*(0.5-rand(10)))
v = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.]
println("test vector: ", v)


function g(v::Array{Float64, 1})
	s = size(v, 1)
	m = sum(v) / s

	@inbounds for i = 1:s
		v[i] = v[i] - m
	end

	return sqrt(BLAS.dot(s, v, 1, v, 1)/s)
end

println(srsi(v, 0.5, 0.2, 4, 1, 3))
