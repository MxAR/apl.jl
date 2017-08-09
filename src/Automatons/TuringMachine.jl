@everywhere module tuma
    ##===================================================================================
    ##  types
    ##===================================================================================
    type ttuma
        cs::Int                                                                         # current state
        fs::Array{Int, 1}                                                               # final states
        is::Int                                                                         # initial state
        cts::Int                                                                        # current tape symbol
        lst::Array{Int, 1}                                                              # left side of the tape
        rst::Array{Int, 1}                                                              # right side of the tape
        il::Dict{Any, Int}                                                              # input lookup table
        tm::Array{Tuple{Int, Int, Int}, 3}                                              # transition matrix x = input | y = state | z = current tape symbol || result x = state transition | y = symbol to be written | z = tape movement
    end


    ##===================================================================================
    ##  inputs
    ##===================================================================================
    export ais!, aiw!

    ##-----------------------------------------------------------------------------------
    function ais!(tuma::ttuma, symbol::Any) # input a symbol
        r = tuma.tm[tuma.il[symbol], tuma.cs, tuma.cts]
        tuma.cts = r[2]
        tuma.cs = r[1]
        if r[3] == 1    # move one to the right
            push!(tuma.lst, tuma.cs)
            tuma.cs = pop!(tuma.rst)
        elseif r[3] == -1    # move one to the left
            push!(tuma.rst, tuma.cs)
            tuma.cs = pop!(tuma.lst)
        end
        return tuma
    end

    ##-----------------------------------------------------------------------------------
    function aiw!(tuma::ttuma, word::String, is::Bool = true) # input a word
        tuma.cs = ifelse(is, tuma.is, tuma.cs)
        for s in word ais!(tuma, s) end
        return (tuma, csif(tuma))
    end

    ##===================================================================================
    ## checks
    ##===================================================================================
    export csif

    ##-----------------------------------------------------------------------------------
    csif(tuma::ttuma) = !isempty(find(tuma.fs .== tuma.cs))                             # current state is one of the final ones
end
