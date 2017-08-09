@everywhere module mpa                                                                  # matching putsuit algorithms
    ##===================================================================================
    ##  using directives
    ##===================================================================================
    using f

    ##===================================================================================
    ## orthogonal matching pursuit algorithm
    ##===================================================================================
    export omp

    ##-----------------------------------------------------------------------------------
    function omp{T<:Number, N<:Number}(y::Array{T, 1}, X::Array{N, 2}, min_eps)         # columns of X should be normalized
        ly = length(y); re = y
        @assert ly == size(X, 1)
        gv = mat_to_vec_ls(X, false)
        sv = Array{Array{N, 1}, 1}
        coeff = Array{T, 1}

        while abs(re) > min_eps
            # find variable with biggest inner product
            sup = 0; alpha = 0
            for i = 1:length(gv)
                s = abs(bdot(ly, gv[i], re))
                if sup < s alpha = i; sup = s end
            end
            push!(sv, gv[alpha])

            # add next coefficient
            x = gv[end]
            push!(coeff, (x*x')/(bdot(ly, x, x)))

            # update residual
            re = (eye(ly)-coeff[end])*y
        end

        return (coeff, sv)
    end


    ##===================================================================================
    ## matching pursuit algorithm
    ##===================================================================================
    export mp

    ##-----------------------------------------------------------------------------------
    function mp{T<:Number, N<:Number}(y::Array{T, 1}, X::Array{N, 2}, min_eps)          # columns of X should be normalized
        ly = length(y); re = y
        @assert ly == size(X, 1)
        gv = mat_to_vec_ls(X, false)
        sv = Array{Array{N, 1}, 1}
        coeff = Array{T, 1}

        while abs(re) > min_eps
            # find variable with biggest inner product
            sup = 0; alpha = 0
            for i = 1:length(gv)
                s = abs(bdot(ly, gv[i], re))
                if sup < s alpha = i; sup = s end
            end
            push!(sv, gv[alpha])

            # add next coefficient
            push!(coeff, bdot(ly, re, sv[end]/abs(sv[end])))

            # update residual
            re -= gv[end]*coeff[end]
        end

        return (coeff, sv)
    end
end
