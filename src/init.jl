m = rand(3,5)

function g(m::Array{Float64, 2})
	s = size(m)

	if s[1] >= s[2]
		return zeros(s[2])
	end

	r = zeros(s[2], s[2]-s[1])

	for i = 1:s[1]
		for j = 1:s[1]
			if i != j
				m[j,:] -= (m[j,i]/m[i,i]) * m[i,:]
			end
		end
		m[i,:] /= m[i,i]
	end

	r[1:s[1], :] = -m[:,(s[1]+1):s[2]]
	r[(s[1]+1):end, :] = eye(s[2]-s[1])

	return r
end

println(size(m))
println(m)
println(g(m))
#println(@code_warntype g(m))
