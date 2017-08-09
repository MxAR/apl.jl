@everywhere module func
    import Base.deepcopy
    import Base.string
    import Base.+
    import Base.*

    type achain                                 # additive chain
        body::Array{Any, 1}                     # Any -> Union{atom, mchain}
    end

    type mchain                                 # multiplicative chain
        body::Array{Any, 1}                     # Any -> Union{atom, achain}
        scalar::Number

        mchain(body, scalar = 1) = new(body, scalar)
    end

    type atom
        basis::Union{String, Number}
        expo::Union{achain, mchain, Number, String}
    end


    ##===================================================================================
    ## internal functions
    ##===================================================================================
    is_num(x) = typeof(x)<:Number

    function deepcopy(c::achain)
        return achain(
                map(deepcopy, c.body)
            )
    end

    function deepcopy(c::mchain)
        return mchain(
                map(deepcopy, c.body)
            )
    end

    function deepcopy(a::atom)
        return atom(
                deepcopy(t.basis),
                deepcopy(t.expo)
            )
    end

    function string(c::achain)
        s = ["("]
        for n in c.body
            push!(s, *(string(n), "+"))
        end
        s[end] = ")"
        return join(s)
    end

    function string(c::mchain)
        s = ["("]; @assert 0<length(c.body)
        for n in c.body
            push!(s, (string(n))
            push!(s)
        end

        s[end] = ")"
        return join(s)
    end

    function string(a::atom)
        a = r(a)
        s = ["(", string(a.basis)]
        if a.expo != 1
            push!(s, *("^(", string(a.expo), ")"))
        end 
        return join(push!(s, ")"))
    end

    ##===================================================================================
    ## arithmetic operators
    ##===================================================================================

    r(a::atom) = is_num(a.basis) && is_num(a.expo) ? atom(a.basis^a.expo, 1) : a  # reduce

    function +(a::atom, b::atom)
        a = r(a); b = r(b)
        if a.expo == b.expo 
            if is_num(a.basis) && is_num(b.basis)
                return atom(a.basis+b.basis, 1)
            end
            if a.basis == b.basis
                return mchain(deepcopy(a), 2)
            end 
        end
        return achain([deepcopy(a), deepcopy(b)])
    end
end