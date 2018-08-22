@everywhere module tuma
    ##===================================================================================
    ##  types
    ##===================================================================================
    struct ttuma{T<:Int}
        cs::T                                                                           # current state
        fs::Array{T, 1}                                                                 # final states
        is::T                                                                           # initial state
        cts::T                                                                          # current tape symbol
        lst::Array{T, 1}                                                                # left side of the tape
        rst::Array{T, 1}                                                                # right side of the tape
        il::Dict{Any, T}                                                                # input lookup table
        tm::Array{Tuple{T, T, T}, 3}                                                    # transition matrix x = input | y = state | z = current tape symbol || result x = state transition | y = symbol to be written | z = tape movement (1 right | -1 left)
    end


    ##===================================================================================
    ##  inputs
    ##===================================================================================
    export ais!, aiw!

    ##-----------------------------------------------------------------------------------
    function ais!(tuma::ttuma, symbol::Any)                                             # input a symbol
        r = tuma.tm[tuma.il[symbol], tuma.cs, tuma.cts]
        tuma.cts = r[2]
        tuma.cs = r[1]

        if r[3] == 1
            push!(tuma.lst, tuma.cs)
            tuma.cs = pop!(tuma.rst)
        elseif r[3] == -1
            push!(tuma.rst, tuma.cs)
            tuma.cs = pop!(tuma.lst)
        end

        return tuma
    end

    ##-----------------------------------------------------------------------------------
    function aiw!(tuma::ttuma, word::String, is::Bool = true)                           # input a word
        tuma.cs = ifelse(is, tuma.is, tuma.cs)

        for s in word
            ais!(tuma, s)
        end

        return (tuma, csif(tuma))
    end

    ##===================================================================================
    ## checks
    ##===================================================================================
    export csif

    ##-----------------------------------------------------------------------------------
    csif(tuma::ttuma) = !isempty(find(tuma.fs .== tuma.cs))                             # current state is one of the final ones
end
