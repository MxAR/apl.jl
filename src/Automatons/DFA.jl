@everywhere module dfa
    ##===================================================================================
    ##  types
    ##===================================================================================
    type tdfa
        is::Int             # initial state
        fs::Array{Int, 1}   # final states
        cs::Int             # current state
        il::Dict{Any, Int}  # input lookup table
        tm::Array{Int, 2}   # transition matrix
    end


    ##===================================================================================
    ##  inputs
    ##===================================================================================
    export ais!, aiw!

    ##-----------------------------------------------------------------------------------
    function ais!(dfa::tdfa, symbol::Any)                                               # input a symbol
        dfa.cs = dfa.tm[dfa.cs, dfa.il[symbol]]
        return dfa
    end

    ##-----------------------------------------------------------------------------------
    function aiw!(dfa::tdfa, word::String, is::Bool = true)                             # input a word
        dfa.cs = ifelse(is, dfa.is, dfa.cs)
        for s in word ais!(dfa, s) end
        return (dfa, csif(dfa))
    end


    ##===================================================================================
    ## checks
    ##===================================================================================
    export csif

    ##-----------------------------------------------------------------------------------
    csif(dfa::tdfa) = !isempty(find(dfa.fs .== dfa.cs))                                 # current state is one of the final ones
end
