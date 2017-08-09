@everywhere module rg
    ##===================================================================================
    ##  using directives
    ##===================================================================================
    using f


    ##===================================================================================
    ## logistic regression
    ##===================================================================================
    export llog_rg

    ##-----------------------------------------------------------------------------------
    function llog_rg{T<:Real}(Points::Array{T, 2})                                      # last element in a row is y the rest are x's
        reve = zeros(2, 1)
        coeff = zeros(2, 2)
        rows = size(Points, 1)
        L = ceil(maximum(Points[1:rows,2]))

        coeff[1, 1] = size(Points, 1)
        coeff[1, 2] = sum(Points[1:rows,1])
        coeff[2, 2] = sumabs2(Points[1:rows,1])
        coeff[2, 1] = coeff[1, 2]

        Points[1:rows,2] = map((x) -> log((L - x) / x), Points[1:rows,2])

        reve[2, 1] = bdot(Points[1:rows,1], Points[1:rows,2])
        reve[1, 1] = sum(Points[1:rows,2])

        S = coeff \ reve
        return (x) -> L / (1 + exp(S[1] + (x*S[2])))
    end


    ##===================================================================================
    ## linear statistic regression
    ##===================================================================================
    export lsta_rg

    ##-----------------------------------------------------------------------------------
    function lsta_rg{T<:Real}(X::Array{T, 2})
        m = APL.mmed(X)
        a = cov(X[:, 1], X[:, 2]) / var(X[:, 1])
        return [(m[2] - (a * m[1])), a]
    end


    ##===================================================================================
    ## multi linear regression
    ##===================================================================================
    export mla_rg

    ##-----------------------------------------------------------------------------------
    function mla_rg{T<:Real}(X::Array{T, 2})
        y = X[:, end]
        X = hcat(ones(size(X, 1)), X[:, 1:end-1])
        Xt = X'
        return (Xt * X) \ (Xt * y)
    end

    ##===================================================================================
    ## multi linear qr decomposition regression
    ##===================================================================================
    export mlqr_rg

    ##-----------------------------------------------------------------------------------
    function mlqr_rg{T<:Real}(X::Array{T, 2})
        QR = qr(hcat(ones(size(X, 1)), X[:, 1:end-1]))
        return QR[2] \ QR[1]' * X[:, end]
    end

end
