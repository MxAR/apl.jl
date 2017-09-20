using mavg
using macd
using rsi
using f

#v = 1 + (0.1*(0.5-rand(10)))
v = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.]
println("test vector: ", v)


function g(v::Array{Float64, 1}, k::Float64, pivot_weight::Float64, slope::Float64, p::Int64, n::Int64, start::Int64, stop::Int64)
	s = stop-start+1
	r1 = zeros(s)
	r2 = zeros(s)
	r3 = zeros(s)
	np = n * p
	s = 0

	for i = start:stop
		sr = i:(-n):(i-np+1)
		s = s + 1

		for j = sr
			r1[s] = r1[s] + (v[j] * max((pivot_weight + slope*(i-j)), 0.))
		end

		r2[s] = r1[s] + k * f.std(v[sr], r1[s])
		r3[s] = r1[s] - k * f.std(v[sr], r1[s])
	end

	return (r1, r2, r3)
end


println(g(v, 2., .5, .1, 2, 1, 2, 10))
println(@code_warntype g(v, 2., .5, .1, 2, 1, 2, 10))
