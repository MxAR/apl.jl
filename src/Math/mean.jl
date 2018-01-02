@everywhere module mean
	##===================================================================================
	##	generalized mean
	##===================================================================================
	export gamean, ghmean, ggmean, gpmean, gfmean, grmean

	##-----------------------------------------------------------------------------------
	gamean{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, l::N, n::N) = gfmean(v, (x) -> x, (x) -> x, l, n)					# arithmetic mean

	##-----------------------------------------------------------------------------------
	ghmean{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, l::N, n::N) = gfmean(v, (x) -> 1/x, (x) -> 1/x, l, n)				# harmonic mean

	##-----------------------------------------------------------------------------------
	ggmean{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, l::N, n::N) = gfmean(v, (x) -> log(x), (x) -> exp(x), l, n)		# geometric mean

	##-----------------------------------------------------------------------------------
	gpmean{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, l::N, n::N, p::T) = gfmean(v, (x) -> x^p, (x) -> x^(1/p), l, n)	# power mean

	##-----------------------------------------------------------------------------------
	grmean{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, l::N, n::N, p::T) = gfmean(v, (x) -> x^2, (x) -> sqrt(x), l, n)  		# root squared mean

	##-----------------------------------------------------------------------------------
	function gfmean{T<:AbstractFloat, N<:Integer}(v::Array{T, 1}, g::Function, g_inv::Function, l::N, n::N) 				# generalized f mean
		@assert(size(v, 1) >= (n*l), "out of bounds error")
		u = Float64(0)

		@inbounds for i = 1:n:(n*l)
			u += g(v[i])
		end

		return g_inv(u/l)
	end

	##===================================================================================
    ## arithmetic mean column/row
    ##===================================================================================
    export mamean

    ##-----------------------------------------------------------------------------------
    function mamean{T<:AbstractFloat}(arr::Array{T, 2}, column::Bool = true)
        n = size(arr, (column ? 1 : 2))
        return BLAS.gemv((column ? 'N' : 'T'), 1/n, arr, ones(T, n))
    end

    ##-----------------------------------------------------------------------------------
    function mamean{T<:AbstractFloat}(arr::Array{T, 2}, weights::Array{T, 1}, column::Bool = true)
        n = size(arr, (column ? 1 : 2))
        return BLAS.gemv((column ? 'N' : 'T'), 1/n, weights.*arr, ones(T, n))
    end
end
