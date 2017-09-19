using mavg
using macd
using rsi

v = 1 + (0.1*(0.5-rand(10)))
#v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10.]
println("test vector: ", v)


function lrsi(v::Array{Float64, 1}, pivot_weight::Float64, slope::Float64, p::Int64, n::Int64, start::Int64)
	s = Int64(0)
	r = [0, 0.]
	np = n * p

	for i = start:(-n):max((start-np+1), 2)
		if v[i-1] < v[i]
			r[1] = r[1] + ((v[i-1] - v[i]) * max((pivot_weight + slope*(start-i)), 0.))
		else
			r[2] = r[2] + ((v[i] - v[i-1]) * max((pivot_weight + slope*(start-i)), 0.))
		end
	end

	return r[2] >= 0 ? 100. : (100 - (100/(1+(r[1]/r[2]))))
end

println(lrsi(v, 0.5, 0.2, 4, 1, 4))
