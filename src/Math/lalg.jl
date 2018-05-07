@everywhere module lalg
	##===================================================================================
	##	using directives
	##===================================================================================
	using mean
	using bla
	using op

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
	##		mat: matrix of data points where each column presents a point
	##		len: number of data points that shall be considered
	##		inc: index distnance of the data points
	## 		trn: wether or not the matrix is transposed
	##===================================================================================
	export pca

	##-----------------------------------------------------------------------------------
	function pca{T<:AbstractFloat, N<:Integer}(m::Array{T, 2}, l::N, inc::N = 1)
		s = size(m)
		x = Array{T}(s[1], l)
		k = l*inc
		j = N(1)

		@assert(k <= s[2], "out of bounds error")
		@inbounds for i = 1:inc:k
			x[:, j] = m[:, i]
			j += 1
		end

		k = copy(x[:, 1])
		@inbounds for i = 2:l
			k .+= x[:, i]
		end

		k /= l
		@inbounds for i = 1:l
			x[:, i] .-= k 
		end
		
		k = eig(mcov(x))[2]
		@inbounds for i = 1:l
			k[:, i] /= BLAS.nrm2(j[1], k[:, 1], 1)
		end 

		return k
	end
end
