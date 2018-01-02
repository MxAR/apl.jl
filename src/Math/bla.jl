@everywhere module bla
	##===================================================================================
	## BLAS wrapper
	##===================================================================================
	export bdot, bdotu, bdotc, bnrm

	##-----------------------------------------------------------------------------------
	bdot{T<:AbstractFloat, N<:Integer}(l::N, v::Array{T, 1}, u::Array{T, 1}) = BLAS.dot(l, v, 1, u, 1)	# l = length of the vectors

	##-----------------------------------------------------------------------------------
	bdot{T<:AbstractFloat}(v::Array{T, 1}, u::Array{T, 1}) = BLAS.dot(size(v, 1), v, 1, u, 1)

	##-----------------------------------------------------------------------------------
	bdotu{C<:Complex, N<:Integer}(l::N, v::Array{C, 1}, u::Array{C, 1}) = BLAS.dotu(l, v, 1, u, 1)		# l = length of the vectors

	##-----------------------------------------------------------------------------------
	bdotu{C<:Complex}(v::Array{C, 1}, u::Array{C, 1}) = BLAS.dotu(size(v, 1), v, 1, u, 1)

	##-----------------------------------------------------------------------------------
	bdotc{C<:Complex, N<:Integer}(l::N, v::Array{C, 1}, u::Array{C, 1}) = BLAS.dotc(l, v, 1, u, 1)		# l = length of the vectors

	##-----------------------------------------------------------------------------------
	bdotc{C<:Complex}(v::Array{C, 1}, u::Array{C, 1}) = BLAS.dotc(size(v, 1), v, 1, u, 1)

	##-----------------------------------------------------------------------------------
	bnrm{T<:AbstractFloat, N<:Integer}(l::N, v::Array{T, 1}) = BLAS.nrm2(l, v, 1)                       # l = length of the vector

	##-----------------------------------------------------------------------------------
	bnrm{T<:AbstractFloat}(v::Array{T, 1}) = BLAS.nrm2(size(v, 1), v, 1)


	##===================================================================================
	##  soq (sum of squares)
	##===================================================================================
	export soq, soqu, soqc

	##-----------------------------------------------------------------------------------
	soq{T<:AbstractFloat, N<:Integer}(l::N, v::Array{T, 1}, n::N = 1) = BLAS.dot(l, v, n, v, n)

	##-----------------------------------------------------------------------------------
	soq{T<:AbstractFloat}(v::Array{T, 1}) = bdot(v, v)

	##-----------------------------------------------------------------------------------
	soqu{C<:Complex, N<:Integer}(l::N, v::Array{C, 1}, n::N = N(1)) = BLAS.dot(l, v, n, v, n)

	##-----------------------------------------------------------------------------------
	soqu{C<:Complex}(v::Array{C, 1}) = bdotu(v, v)

	##-----------------------------------------------------------------------------------
	soqc{C<:Complex, N<:Integer}(l::N, v::Array{C, 1}, n::N = 1) = bdotc(l, v, n, v, n)

	##-----------------------------------------------------------------------------------
	soqc{C<:Complex}(v::Array{C, 1}) = bdotc(v, v)


	##===================================================================================
	##  QR (faster than vanilla)
	##===================================================================================
	export qrd_sq, qrd

	##-----------------------------------------------------------------------------------
	function qrd_sq{T<:AbstractFloat}(m::Array{T, 2})
		s = size(m, 1)
		t = zeros(T, s, s)
		v = zeros(T, s)
		r = copy(m)
		w = T(0)

		@inbounds for i=1:(s-1)
			w = 0.
			for j=i:s
				v[j] = r[j, i]
				w += v[j]*v[j]
			end

			v[i] += (r[i, i] >= 0 ? 1. : -1.)*sqrt(w)
			w = 0.

			for j=i:s w += v[j]*v[j] end
			w = 2.0/w

			for j=1:s, k=1:s
				t[j, k] = k == j ? 1. : 0.
				if j>=i && k>=i
					t[j, k] -= w*v[j]*v[k]
				end
			end

			for j=1:s
				for k=1:s
					v[k] = r[k, j]
				end

				for l=1:s
					w = 0.
					for h=1:s
						w += v[h]*t[l, h]
					end
					r[l, j] = w
				end
			end
		end

		for j=1:(s-1), k=(j+1):s
			r[k, j] = 0.
		end

		return (m*inv(r), r)
	end

	##-----------------------------------------------------------------------------------
	function qrd{T<:AbstractFloat}(m::Array{T, 2})
		s = size(m, 1)
		t = zeros(T, s, s)
		v = zeros(T, s)
		r = copy(m)
		w = T(0)

		for i=1:(s-1)
			w = 0.
			for j=i:s
				v[j] = r[j, i]
				w += v[j]*v[j]
			end

			v[i] += (r[i, i] >= 0 ? 1. : -1.)*sqrt(w)
			w = 0.

			for j=i:s w += v[j]*v[j] end
			w = 2.0/w

			for j=1:s, k=1:s
				t[j, k] = k == j ? 1. : 0.
				if j>=i && k>=i
				    t[j, k] -= w*v[j]*v[k]
				end
			end

			if i == 1
				r .= t*r
			else
				for j=1:s
					for k=1:s
						v[k] = r[k, j]
					end

					for l=1:s
						w = 0.
						for h=1:s
							w += v[h]*t[l, h]
						end
						r[l, j] = w
					end
				end
			end
		end

		for j=1:(s-1), k=(j+1):s
			r[k, j] = 0.
		end

		return (m/r, r)
	end


	##===================================================================================
	## diagonal expansion of matrices
	##===================================================================================
	export ul_x_expand

	##-----------------------------------------------------------------------------------
	function ul_x_expand{T<:AbstractFloat, N<:Integer}(m::Array{T, 2}, s::Tuple{N, N}, x::T = 1.0)# ul = upper left
		d = (s[1]-size(m, 1), s[2]-size(m,2))
		r = zeros(T, s)

		@inbounds for i = 1:s[1], j = 1:s[2]
			if i>d[1] && j>d[2]
				r[i, j] = m[i-d[1], j-d[2]]
			elseif i == j
				r[i, j] = x
			end
		end

		return r
	end


	##===================================================================================
	##  minor of matrix (lower submatrix)
	##===================================================================================
	export minor

	##-----------------------------------------------------------------------------------
	function minor{T<:AbstractFloat, N<:Integer}(m::Array{T, 2}, p::Tuple{N, N} = (1, 1))
		s = size(m)
		r = Array{T, 2}(s[1]-p[1], s[2]-p[1])

		@inbounds for i=(1+p[1]):s[1], j=(1+p[2]):s[2]
			r[i-p[1], j-p[2]] = m[i, j]
		end

		return r
	end


	##===================================================================================
	##  outer product implementation (faster than vanila)
	##===================================================================================
	export otr

	##-----------------------------------------------------------------------------------
	function otr{T<:AbstractFloat}(v::Array{T, 1}, w::Array{T, 1})
		s = (size(v, 1), size(w, 1))
		m = Array{T, 2}(s)

		@inbounds for i=1:s[1], j=1:s[2]
			m[i, j] = v[i] * w[j]
		end

		return m
	end


	##===================================================================================
	##  gram schmidt proces
	##===================================================================================
    export grsc, grscn

	##-----------------------------------------------------------------------------------
	function grsc{T<:AbstractFloat}(m::Array{T, 2})
    	s = size(m, 2)
    	d = zeros(T, s)
    	ob = []

    	@inbounds for i = 1:s
        		push!(ob, m[:, i])
        		for j = 1:(i-1)
            		ob[i] -= (dot(ob[j], ob[i])/d[j])*ob[j]
        		end
     		d[i] = dot(ob[i], ob[i])
   		end

    	return ob
	end

    ##-----------------------------------------------------------------------------------
   	function grscn{T<:AbstractFloat}(m::Array{T, 2})
    	ob = []

    	@inbounds for i = 1:size(m, 2)
        		push!(ob, m[:, i])
        		for j = 1:(i-1)
            		ob[i] -= dot(ob[j], ob[i])*ob[j]
        		end
        		normalize!(ob[i])
    	end

    	return ob
    end


    ##===================================================================================
    ##  orthogonal projection
    ##===================================================================================
    export proj, projn

    ##-----------------------------------------------------------------------------------
    function proj{T<:AbstractFloat}(v::Array{T, 1}, m::Array{T, 2})
    	r = zeros(size(v))
    	@inbounds for i = 1:size(m, 2)
        	r += m[:, i]*(bdot(v, m[:, i])/bdot(m[:, i], m[:, i]))
    	end
    	return r
    end

	##-----------------------------------------------------------------------------------
    function proj{T<:Complex}(v::Array{T, 1}, m::Array{T, 2})
    	r = zeros(size(v))
    	@inbounds for i = 1:size(m, 2)
        	r += m[:, i]*(bdotc(v, m[:, i])/bdotc(m[:, i], m[:, i]))
    	end
    	return r
    end

    ##-----------------------------------------------------------------------------------
    projn{T<:AbstractFloat}(v::Array{T, 1}, m::Array{T, 2}) = m*m'*v

	##-----------------------------------------------------------------------------------
    projn{T<:Complex}(v::Array{T, 1}, m::Array{T, 2}) = m*m.'*v

	##===================================================================================
    ##	Cofactor Matrix of a Matrix
    ##===================================================================================
    export cof

    ##-----------------------------------------------------------------------------------
    function cof{T<:AbstractFloat}(m::Array{T, 2})										# TODO needs better performance
		s = size(m)
		n = zeros(T, s[1]-1, s[2]-1)
		r = zeros(T, s)

		@inbounds for i = 1:s[1], j = 1:s[2]
			for x = 1:s[1], y = 1:s[2]
				if x != i && y != j
					n[(x > i ? x-1 : x), (y > j ? y-1 : y)] = m[x, y]
				end
			end

			r[i, j] = det(n)*(i+j % 2 == 0 ? 1. : -1)
		end

		return r
    end


	##===================================================================================
	##	Adjugate of a Matrix
	##===================================================================================
	export adj

	##-----------------------------------------------------------------------------------
	adj{T<:AbstractFloat}(m::Array{T, 2}) = return cof(m)'


	##===================================================================================
	##	Nullspace of a matrix
	##===================================================================================
	export nul

	##-----------------------------------------------------------------------------------
	function nul{T<:AbstractFloat}(m::Array{T, 2})
		s = size(m)

		if s[1] >= s[2]
			return zeros(T, s[2])
		end

		r = zeros(T, s[2], s[2]-s[1])

		@inbounds for i = 1:s[1]
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

	##===================================================================================
    ## split
    ##===================================================================================
    export msplit, msplit_half

    ##-----------------------------------------------------------------------------------
    function msplit{T<:Number, N<:Integer}(m::Array{T, 2}, i::N, lrows::Bool = true)
        r = lrows ? m : m'
        @assert(i < 0 || i >= size(r, 1))
        return (r[1:i, :], r[(i+1):end, :])
    end

    ##-----------------------------------------------------------------------------------
    function msplit_half{T<:Number}(m::Array{T, 2}, lrows::Bool = true)
        r = lrows ? m : m'
        l = convert(Int, round(size(r, 1)/2))
        return (r[1:l, :], r[(l+1):end, :])
    end
end
