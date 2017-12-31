@everywhere module op
    ##===================================================================================
    ## aggregations
    ##===================================================================================
    export PI, imp_add, lr_imp_add, imp_sub, lr_imp_sub

    ##-----------------------------------------------------------------------------------
    function PI{T<:Number}(N::Array{T, 1})
        p = N[1]

        @inbounds for i = 2:size(N, 1)
            p *= N[i]
        end

        return p
    end

    ##-----------------------------------------------------------------------------------
    function imp_add{T<:Number}(v1::Array{T, 1}, v2::Array{T, 1})
        l = (size(v1, 1), size(v1, 1))
        v = zeros(max(l))
        v[1:l[1]] = v1
        v[1:l[2]] += v2
        return v
    end

    ##-----------------------------------------------------------------------------------
    function imp_add{T<:Number}(m1::Array{T, 2}, m2::Array{T, 2})
        s = (size(m1), size(m2))
        b = (max(s[1][1], s[2][1]), max(s[1][2], s[2][2]))
        r = Array{T, 2}(i)

        for i = 1:b[1], j = 1:b[2]
            if i>s[1][1] && j>s[1][2]
                r[i, j] = m1[i, j]
            end

            if i>s[2][1] && j>s[2][2]
                r[i, j] += m2[i, j]
            end
        end

        return r
    end

    ##-----------------------------------------------------------------------------------
    function imp_sub{T<:Number}(V1::Array{T, 1}, V2::Array{T, 1})
        l = (length(v1), length(v1))
        v = zeros(max(l))
        v[1:l[1]] = v1
        v[1:l[2]] -= v2
        return v
    end

    ##-----------------------------------------------------------------------------------
    function imp_sub{T<:Number}(m1::Array{T, 2}, m2::Array{T, 2})
        s = (size(m1), size(m2))
        b = (max(s[1][1], s[2][1]), max(s[1][2], s[2][2]))
        r = Array{T, 2}(i)

        for i = 1:b[1], j = 1:b[2]
            if i>s[1][1] && j>s[1][2]
                r[i, j] = m1[i, j]
            end

            if i>s[2][1] && j>s[2][2]
                r[i, j] -= m2[i, j]
            end
        end

        return r
    end


    ##===================================================================================
    ## prison
    ##===================================================================================
    export prison

    ##-----------------------------------------------------------------------------------
    prison{T<:Number}(value::T, infimum::T, supremum::T) = min(max(value, infimum), supremum)

    ##-----------------------------------------------------------------------------------
    prison{T<:Number}(x::T, f::Function, infimum::T, supremum::T) = x < infimum ? 0 : (x > supremum ? 1 : f(x))
end
