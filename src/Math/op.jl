@everywhere module op
    ##===================================================================================
    ## aggregations
    ##===================================================================================
    export PI, imp_add, lr_imp_add, imp_sub, lr_imp_sub

    ##-----------------------------------------------------------------------------------
    function PI{T<:Real}(N::Array{T, 1})
        p = N[1]
        for i = 2:length(N) p *= N[i] end
        return p
    end

    ##-----------------------------------------------------------------------------------
    function imp_add(V1, V2)
        L = [length(V1), length(V1)]
        V = zeros(maximum(L))
        V[1:L[1]] += V1
        V[1:L[2]] += V2
        return V
    end

    ##-----------------------------------------------------------------------------------
    function lr_imp_add(m1::Array{Float64, 2}, m2::Array{Float64, 2})
        s = (size(m1), size(m2))
        d = (s[1][1]-s[2][1], s[1][2]-s[2][2])
        r = Array{Float64, 2}(s[1])
        for i = 1:s[1][1], j = 1:s[1][2]
            if i>d[1] && j>d[2]
                r[i, j] = m1[i, j] + m2[i-d[1], j-d[2]]
            else
                r[i, j] = m1[i, j]
            end
        end
        return r
    end

    ##-----------------------------------------------------------------------------------
    function imp_subt{T<:Float64}(V1::Array{Float64, 1}, V2::Array{Float64, 1})
        L = [length(V1), length(V1)]
        V = zeros(maximum(L))
        V[1:L[1]] += V1
        V[1:L[2]] -= V2
        return V
    end

    ##-----------------------------------------------------------------------------------
    function lr_imp_sub(m1::Array{Float64, 2}, m2::Array{Float64, 2})
        s = (size(m1), size(m2))
        d = (s[1][1]-s[2][1], s[1][2]-s[2][2])
        r = Array{Float64, 2}(s[1])
        for i = 1:s[1][1], j = 1:s[1][2]
            if i>d[1] && j>d[2]
                r[i, j] = m1[i, j] - m2[i-d[1], j-d[2]]
            else
                r[i, j] = m1[i, j]
            end
        end
        return r
    end


    ##===================================================================================
    ## prison
    ##===================================================================================
    export prison

    ##-----------------------------------------------------------------------------------
    prison(value, infimum, supremum) = min(max(value, infimum), supremum)

    ##-----------------------------------------------------------------------------------
    prison(x, f::Function, infimum, supremum) = ifelse(x < infimum, 0, ifelse(x > supremum, 1, f(x)))
end
