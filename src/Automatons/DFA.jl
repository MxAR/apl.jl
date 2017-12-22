@everywhere module dfa
    ##===================================================================================
    ##  types
    ##===================================================================================
    type tdfa{T<:Int}
        is::T               # initial state
        fs::Array{T, 1}     # final states
        cs::T               # current state
        il::Dict{Any, T}    # input lookup table
        tm::Array{T, 2}     # transition matrix
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
        dfa.cs = is ? dfa.is : dfa.cs

        for s in word
            ais!(dfa, s) 
        end

        return (dfa, csif(dfa))
    end


    ##===================================================================================
    ## checks
    ##===================================================================================
    export csif

    ##-----------------------------------------------------------------------------------
    csif(dfa::tdfa) = !isempty(find(dfa.fs .== dfa.cs))                                 # current state is one of the final ones
end
