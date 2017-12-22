@everywhere module pda
    ##===================================================================================
    ##  types
    ##===================================================================================
    type tpda{T<:Int}
        is::Tuple{T, T}                                                                 # initial state (x) and stack symbol (y)
        fs::Array{T, 1}                                                                 # final states
        cs::T                                                                           # current state
        st::Array{T, 1}                                                                 # stack
        il::Dict{Any, T}                                                                # input lookup table
        tm::Array{Tuple{T, T}, 3}                                                       # transition matrix x = input | y = state | z = top of stack || result x = state transition | y = stack operation (1 push | -1 pop)
    end


    ##===================================================================================
    ##  inputs
    ##===================================================================================
    export ais!, aiw!

    ##-----------------------------------------------------------------------------------
    function ais!(pda::tpda, symbol::Any)                                               # input a symbol
        r = pda.tm[pda.il[symbol], pda.cs, pda.st[end]]
        pda.cs = r[1]

        if r[2] == 1
            push!(pda.st, pda.il[symbol])
        elseif r[2] == -1
            pop!(pda.st)
        end

        return pda
    end

    ##-----------------------------------------------------------------------------------
    function aiw!(pda::tpda, word::String, is::Bool = true)                             # input a word
        if is; pda.cs = pda.is[1]; pda.st = [pda.is[2]] end
        for s in word ais!(pda, s) end
        return (pda, csif(pda))
    end


    ##===================================================================================
    ## checks
    ##===================================================================================
    export csif

    ##-----------------------------------------------------------------------------------
    csif(pda::tpda) = !isempty(find(pda.fs .== pda.cs))                                 # current state is one of the final ones
end
