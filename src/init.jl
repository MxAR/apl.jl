v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10.]


function sma(v::Array{Float64, 1}, p::Tuple{Int64, Int64}, n::Int64, start::Int64, stop::Int64)
	s = stop-start+1
	r = zeros(s)
	q = zeros(s)
	np = (n * p[1], n * p[2])
	s = 0

	for i = start:stop
		s = s + 1
		for j = i:(-n):(i-np+1)
			r[s] = r[s] + v[j]
		end
		r[s] = r[s]/(np)
	end

	return r
end

println(lma(v, 0.5, -0.1, 3, 1, 3, 10))
