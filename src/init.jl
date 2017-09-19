v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10.]
using mavg
using macd

function g(v::Array{Float64, 1}, pivot_weight::Float64, slope::Float64, p::Int64, n::Int64, start::Int64, stop::Int64)
	r = zeros(stop-start+1)
	s = Int64(0)
	np = n * p

	for i = start:stop
		s = s + 1
		for j = i:(-n):(i-np+1)
			r[s] = r[s] + (v[j] * max((pivot_weight + slope*(i-j)), 0.))
		end
	end

	return r
end

function emacd(v::Array{Float64, 1}, pivot_weight::Float64, slope::Float64, p::Tuple{Int64, Int64}, n::Int64, start::Int64, stop::Int64)
	np = n * (p[1] < p[2] ? p[1] : p[2])
	d = Int64(abs(p[1] - p[2]))
	r = zeros(stop-start+1)
	start = start - d
	stop = stop - d
	s = Int64(0)

	for i = start:stop
		s = s + 1
		for j = i:(-n):(i-np+1)
			r[s] = r[s] + (-v[j] * pivot_weight * exp(-(slope*(i-j+d))^2))
		end
	end

	return r
end

println(g(v, 0.5, -0.2, 2, 1, 4, 10)-g(v, 0.5, -0.2, 4, 1, 4, 10))
println(@code_warntype emacd(v, 0.5, -0.2, (4, 2), 1, 4, 10))
