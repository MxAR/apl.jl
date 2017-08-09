@everywhere module pda
    ##===================================================================================
    ##  types
    ##===================================================================================
    type tpda
        is::Tuple{Int, Int}                                                             # initial state (x) and stack symbol (y)
        fs::Array{Int, 1}                                                               # final states
        cs::Int                                                                         # current state
        st::Array{Int, 1}                                                               # stack
        il::Dict{Any, Int}                                                              # input lookup table
        tm::Array{Tuple{Int,Int},3}                                                     # transition matrix x = input | y = state | z = top of stack || result x = state transition | y = stack operation
    end


    ##===================================================================================
    ##  inputs
    ##===================================================================================
    export ais!, aiw!

    ##-----------------------------------------------------------------------------------
    function ais!(pda::tpda, symbol::Any)                                               # input a symbol
        r = pda.tm[pda.il[symbol], pda.cs, pda.st[end]]
        pda.cs = r[1]
        if r[2] == 1        # push symbol on top of the stack
            push!(pda.st, pda.il[symbol])
        elseif r[2] == -1   # delete first element of the stack
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
