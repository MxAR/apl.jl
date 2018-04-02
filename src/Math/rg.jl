@everywhere module rg
    ##===================================================================================
    ## logistic regression
    ##===================================================================================
    export rg_log

    ##-----------------------------------------------------------------------------------
    function rg_log{T<:AbstractFloat}(X::Array{T, 2})                                      # last column is y the rest are x's
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
        m = f.mamean(X)
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

end
