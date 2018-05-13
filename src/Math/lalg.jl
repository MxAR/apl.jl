@everywhere module lalg
	##===================================================================================
	##	using directives
	##===================================================================================
	using mean
	using bla
	using op


	##===================================================================================
	##	normalize matrix columns
	##===================================================================================
	export normalize

	##-----------------------------------------------------------------------------------
	function normalize!{T<:Number}(m::Array{T, 2})
		s = size(m)
		r = zeros(s[1], s[2])

		@inbounds for i = 1:s[2]
			a = 0.
			@inbounds for j = 1:s[1]
				a = a + m[j, i]^2
			end 

			a = a^.5
			@inbounds for j = 1:s[1]
				r[j, i] = m[j, i] / a
			end
		end

		return r
	end

	##===================================================================================
	##	phi (transform n dimensional points into their polar form)
	##		- first column represents the radius
	##===================================================================================
	export phi, iphi

	##-----------------------------------------------------------------------------------
	function phi{T<:Real}(m::Array{T, 2})
		s = size(m)
		r = zeros(s[1], s[2])

		@inbounds for i = 1:s[1]
			r[i, 1] = BLAS.nrm2(s[2], m[i, :], 1)
		end

		@inbounds for i = 1:s[1]
			a = 1.
			@inbounds for j = 2:s[2]
				r[i, j] = acosd(m[i, j-1]/(a*r[i, 1]))
				a = a * sind(r[i, j])
			end
		end

		return r
	end

	##-----------------------------------------------------------------------------------
	function iphi{T<:Real}(m::Array{T, 2})
		s = size(m)
		r = zeros(s[1], s[2])

		@inbounds for i = 1:s[1]
			a = 1.
			@inbounds for j = 1:(s[2]-1)
				r[i, j] = m[i, 1]*a*cosd(m[i, j+1])
				a  = a * sind(m[i, j+1])
			end
			r[i, s[2]] = m[i, 1]*a
		end

		return r
	end 


	##===================================================================================	
	##	scalar projection 
	##		get the length of the projection of a onto b
	##===================================================================================
	export scal

	##-----------------------------------------------------------------------------------
	function scal{T<:Number}(a::Array{T, 1}, b::Array{T, 1})
		l = size(a, 1)
		c = b ./ BLAS.nrm2(l, b, 1) 
		return BLAS.dot(l, a, 1, c, 1)
	end


	##===================================================================================
	##	vector projection (vector -> vector)
	##===================================================================================
	export proj

	##-----------------------------------------------------------------------------------
	function proj{T<:Number}(a::Array{T, 1}, b::Array{T, 1})
		l = size(a, 1)
		c = b ./ BLAS.nrm2(l, b, 1)
		return c.*BLAS.dot(l, a, 1, c, 1)
	end


	##===================================================================================
	##	get the diagonal of matrix
	##===================================================================================
	export diagonal

	##-----------------------------------------------------------------------------------
	function diagonal{T<:Number}(m::Array{T, 2})
		s = op.min(size(m))
		r = zeros(s)

		@inbounds for i = 1:s
			r[i] = m[i, i]
		end

		return r
	end 


    ##===================================================================================   	
    ##	logistic regression
    ##		last column is y the rest are x's
	##===================================================================================
    export rg_log

    ##-----------------------------------------------------------------------------------
    function rg_log{T<:AbstractFloat}(X::Array{T, 2})                                      #
        reve = zeros(T, 2, 1)
        coeff = zeros(T, 2, 2)
        rows = size(X, 1)
        L = ceil(maximum(X[1:rows,2]))

        coeff[1, 1] = size(X, 1)
        coeff[1, 2] = sum(X[1:rows,1])
        coeff[2, 2] = sumabs2(X[1:rows,1])
        coeff[2, 1] = coeff[1, 2]

        X[1:rows,2] = map((x) -> log((L - x) / x), X[1:rows,2])

        reve[2, 1] = bdot(X[1:rows,1], X[1:rows,2])
        reve[1, 1] = sum(X[1:rows,2])

        S = coeff \ reve
        return (x) -> L / (1 + exp(S[1] + (x*S[2])))
    end


    ##===================================================================================
    ## linear statistic regression
    ##===================================================================================
    export rg_sta

    ##-----------------------------------------------------------------------------------
    function rg_sta{T<:AbstractFloat}(X::Array{T, 2})
        m = mamean(X)
        a = cov(X[:, 1], X[:, 2])/var(X[:, 1])
        return [(m[2] - (a * m[1])), a]
    end
    

    ##===================================================================================
    ## multi linear qr decomposition regression
    ##===================================================================================
    export rg_qr

    ##-----------------------------------------------------------------------------------
    function rg_qr{T<:AbstractFloat}(X::Array{T, 2})
        QR = qr(hcat(ones(size(X, 1)), X[:, 1:end-1]))
        return QR[2] \ QR[1]' * X[:, end]
    end


	##===================================================================================	
	## pca (principal component analysis)												
	##		mat: matrix of data points where each row presents a point
	## 		t: wether or not the matrix is transposed
	##===================================================================================
	export pca

	##-----------------------------------------------------------------------------------
	function pca{T<:AbstractFloat}(m::Array{T, 2}, t::Bool = false)
		if t; m = m' end 
		s = size(m, 1)
		d = svd(m)

		@inbounds for i = 1:s, j = 1:s
			d[1][j, i] *= d[2][i] 	
		end
		
		return d[1]
	end
end
