@everywhere module graph
    ##===================================================================================
    ##  types
    ##===================================================================================
    struct tgraph
        vtx::Array{Int, 1}
        edge::Dict{Tuple{Int, Int}, Number}
    end

    ##===================================================================================
    ##  Bellman Ford
    ##===================================================================================
    export bellman_ford

    ##-----------------------------------------------------------------------------------
    function bellman_ford(g::tgraph, start_vtx::Int)
        delta = Dict{Int, Number}
        pre_vtx = Dict{Int, Int}

        @simd for x in g.vtx
            delta[x] = Inf
            pre_vtx[x] = nothing
        end

        delta[start_vtx] = 0

        for i = 1:length(g.vtx)-1, ed in keys(g.edge)
            if delta[ed[1]] + g.edge[ed] < delta[ed[2]]
                delta[ed[2]] = delta[ed[1]] + g.edge[ed]
                pre_vtx[ed[2]] = ed[1]
            end
        end

        for ed in keys(g.edge)
            @assert delta[ed[1]] + g.edge[ed] >= delta[ed[2]]   # grap contains negative weight loop
        end
    end
end
