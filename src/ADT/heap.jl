@everywhere module heap
    export parent, children, is_heap, heapify, heap_build, heap_sort

    ##-----------------------------------------------------------------------------------
    parent(i::T) where T<:Integer = convert(T, floor(i/2))

    ##-----------------------------------------------------------------------------------
    children(i::T) where T<:Integer = (2*i, (2*i)+1)

    ##-----------------------------------------------------------------------------------
    function is_heap(v::Array{T, 1}, max = true, p = 1) where T<:Number					# for max c = > || for min c = <
        c = ifelse(max, >, <)                                                          	# p = index (when nodes represent vectors this is the element after which everything is sorted)

        for i = size(v, 1):-1:1
            if c(v[i][p], v[Base.max(1, convert(Int, floor(i/2)))][p])
                return false
            end
        end

        return true
    end

    ##-----------------------------------------------------------------------------------
    function heapify(v::Array{T, 1}, n = 1, max = true, p = 1) where T<:Any				# n = start node
		c = max ? (<;) : (>;)                                            				# max = if max heap should be build
		l = length(v)
		swap = undef

        while true                                                                      # p = index (when nodes represent vectors this is the element after which everything is sorted)
            ci = children(n); m = n
            if ci[1] <= l && c(v[n][p], v[ci[1]][p]) m = ci[1] end
            if ci[2] <= l && c(v[m][p], v[ci[2]][p]) m = ci[2] end
            if n == m break; else
                swap = v[m]
                v[m] = v[n]
                v[n] = swap
                n = m
            end
        end

        return v
    end

    ##-----------------------------------------------------------------------------------
    function heap_build(v::Array{T, 1}, max = true, p = 1) where T<:Any					# p = index (when nodes represent vectors this is the element after which everything is sorted)
        for i in convert(Int, floor(size(v, 1)/2)):-1:1
            v = heapify(v, i, max, p)
        end

        return v
    end

    ##-----------------------------------------------------------------------------------
    function heap_sort(v::Array{T, 1}, ascending = true, p = 1) where T<:Any			# p = index (when nodes represent vectors this is the element after which everything is sorted)
        v = build_heap(v, ascending, p)

        for i = length(v):-1:2
            v[1:i] = heapify(v[1:i], 1, ascending, p)
            v[1] = v[1] + v[i]
            v[i] = v[1] - v[i]
            v[1] = v[1] - v[i]
        end

        return v
    end
end
