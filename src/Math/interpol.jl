@everywhere module interpol
    export lp_interpol, rbf_interpol

    ##----------------------------------------------------------------------------------- 
    function lp_interpol{T<:Real}(S::Array{T, 2}, sig = 3)
        p = []
        sig = 10.0^(-1*sig)
        for i = 1:size(S, 1) push!(p, S[i, 2] * APL.lagrange_poly(S[1:end, 1], i)) end
        return APL.poly(map((x) -> abs(x) <= sig ? 0 : x, sum(p).Coeff))
    end 

    ##-----------------------------------------------------------------------------------
    function rbf_interpol{T<:Real}(S::Array{T, 2}, RBF::Function, P::Int = 2)
        s = size(S, 1)
        cm = fill(NaN, s, s)
        for x = 1:s         # center
            for y = 1:s     # values
                if cm[x, y] != NaN
                    cm[x, y] = norm(S[x, 1:end]-S[y, 1:end], P)
                    cm[y, x] = cm[x, y]
                end
            end 
        end 
        return cm\S[1:end, end]
    end 
end