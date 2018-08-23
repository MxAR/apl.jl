@everywhere module op
    ##===================================================================================
	## using directives
	##===================================================================================
	using Distributed


	##===================================================================================
    ## aggregations
    ##===================================================================================
    export mul, imp_add, lr_imp_add, imp_sub, lr_imp_sub

    ##-----------------------------------------------------------------------------------
    function mul(v::Array{T, 1}) where T<:Number
        r = v[1]

        @inbounds for i = 2:size(v, 1)
            r = r * v[i]
        end

        return r
    end

    ##-----------------------------------------------------------------------------------
    function imp_add(v1::Array{T, 1}, v2::Array{T, 1}) where T<:Number
        l = (size(v1, 1), size(v1, 1))
        v = zeros(max(l))
        v[1:l[1]] = v1
        v[1:l[2]] += v2
        return v
    end

    ##-----------------------------------------------------------------------------------
    function imp_add(m1::Array{T, 2}, m2::Array{T, 2}) where T<:Number
        s = (size(m1), size(m2))
        b = (max(s[1][1], s[2][1]), max(s[1][2], s[2][2]))
        r = Array{T, 2}(i)

        for i = 1:b[1], j = 1:b[2]
            if i>s[1][1] && j>s[1][2]
                r[i, j] = m1[i, j]
            end

            if i>s[2][1] && j>s[2][2]
                r[i, j] += m2[i, j]
            end
        end

        return r
    end

    ##-----------------------------------------------------------------------------------
    function imp_sub(V1::Array{T, 1}, V2::Array{T, 1}) where T<:Number
        l = (length(v1), length(v1))
        v = zeros(max(l))
        v[1:l[1]] = v1
        v[1:l[2]] -= v2
        return v
    end

    ##-----------------------------------------------------------------------------------
    function imp_sub(m1::Array{T, 2}, m2::Array{T, 2}) where T<:Number
        s = (size(m1), size(m2))
        b = (max(s[1][1], s[2][1]), max(s[1][2], s[2][2]))
        r = Array{T, 2}(i)

        for i = 1:b[1], j = 1:b[2]
            if i>s[1][1] && j>s[1][2]
                r[i, j] = m1[i, j]
            end

            if i>s[2][1] && j>s[2][2]
                r[i, j] -= m2[i, j]
            end
        end

        return r
    end


    ##===================================================================================
    ## prison
    ##===================================================================================
    export prison

    ##-----------------------------------------------------------------------------------
    prison(value::T, infimum::T, supremum::T) where T<:Number = min(max(value, infimum), supremum)

    ##-----------------------------------------------------------------------------------
    prison(x::T, f::Function, infimum::T, supremum::T) where T<:Number = x < infimum ? 0 : (x > supremum ? 1 : f(x))


    ##===================================================================================
    ## rm column/row
    ##===================================================================================
    export rm, rms

    ##-----------------------------------------------------------------------------------
    function rm(m::Array{T, 2}, i::Integer, column::Bool = true) where T<:Any
		r = column ? m : m'
        return hcat(r[:, 1:(i-1)], r[:, (i+1):end])
    end

    ##-----------------------------------------------------------------------------------
    function rms(m::Array{T, 2}, i::Array{Any, 1}, column::Bool = true) where T<:Any
		r = column ? m : m'

		@inbounds for x in i
            r = hcat(r[:, 1:(x-1)], r[:, (x+1):end])
            i .-= 1
        end

		return r
    end

    ##-----------------------------------------------------------------------------------
    function rm(m::Array{T, 2}, i::Array{Any, 1}, column::Bool = true) where T<:Any
		r = column ? m : m'

		@inbounds for x in sort(i)
            r = hcat(r[:, 1:(x-1)], r[:, (x+1):end])
            i .-= 1
        end

        return r
    end

    ##-----------------------------------------------------------------------------------
    function rm(m::Array{T, 2}, ub::Z, lb::Z, column::Bool = true) where T<:Any where Z<:Integer
		column::Bool = true
        return hcat(r[:, 1:(lb-1)], r[:, (ub+1):end])
    end


    ##===================================================================================
    ## rm
    ##===================================================================================
    export rm, rms

    ##-----------------------------------------------------------------------------------
    function rm(v::Array{T, 1}, i::Array{Z, 1}) where T<:Any where Z<:Integer
        i = sort(i)

        @inbounds for j=1:length(i)
            v = cat(1, v[1:(i[j]-1)], v[(i[j]+1):end])
            i .-= 1
        end

        return v
    end

    ##-----------------------------------------------------------------------------------
    function rms(v::Array{T, 1}, i::Array{Z, 1}) where T<:Any where Z<:Integer
        @inbounds for j=1:length(i)
            v = cat(1, v[1:(i[j]-1)], v[(i[j]+1):end])
            i .-= 1
        end

        return v
    end


    ##===================================================================================
    ## union overload
    ##===================================================================================
    import Base.union
	export union

    ##-----------------------------------------------------------------------------------
    function union(vl::Array{Array{T, 1}, 1}) where T<:Any
        v = vl[1]

        @inbounds for i=2:size(vl, 1)
            v = union(v, vl[i])
        end

		return v
    end


    ##===================================================================================
    ## intersect overload
    ##===================================================================================
    import Base.intersect
	export intersect

    ##-----------------------------------------------------------------------------------
    function intersect(vl::Array{Array{T, 1}, 1}) where T<:Any
        v = vl[1]

        @inbounds for i=2:size(vl, 1)
            v = intersect(v, vl[i])
        end

        return v
    end


    ##===================================================================================
    ## prepend
    ##===================================================================================
    export prepend, prepend!

    ##-----------------------------------------------------------------------------------
    prepend(v::Array{T, 1}, w) where T<:Any = cat(1, [w], v)

    ##-----------------------------------------------------------------------------------
    prepend(v::Array{T, 1}, w::Array{T, 1}) where T<:Any = cat(1, w, v)

    ##-----------------------------------------------------------------------------------
    prepend!(v::Array{T, 1}, w) where T<:Any = v = cat(1, [w], v)

    ##-----------------------------------------------------------------------------------
    prepend!(v::Array{T, 1}, w::Array{T, 1}) where T<:Any = v = cat(1, w, v)


    ##===================================================================================
    ## map (overload)
    ##===================================================================================
    import Base.map
	export map

    ##-----------------------------------------------------------------------------------
    function map(f::Function, vl::Array{Array{T, 1}, 1}) where T<:Any
		ul = Array{Array{T, 1}, 1}()

        @inbounds @simd for i = 1:length(ul)
            push!(ul, f(ul[i]))
        end

		return ul
    end


	##===================================================================================
    ## min (overload)
    ##===================================================================================
	import Base.min
	export min

	##-----------------------------------------------------------------------------------
	min(x::Tuple{T, T}) where T<:Number = x[1] < x[2] ? x[1] : x[2]


	##===================================================================================
    ## max (overload)
    ##===================================================================================
	import Base.max
	export max

	##-----------------------------------------------------------------------------------
	max(x::Tuple{T, T}) where T<:Number = x[1] > x[2] ? x[1] : x[2]


    ##===================================================================================
    ## apply
    ##===================================================================================
    export apply, apply_parallel, apply_parallel_shared, apply_triu, apply_tril

    ##-----------------------------------------------------------------------------------
    function apply(g::Function, m)
        @inbounds for i in eachindex(m)
            m[i] = g(m[i])
        end

        return m
    end

    ##-----------------------------------------------------------------------------------
    function apply_p(g::Function, m)
        m = convert(SharedArray, m)

		@inbounds @sync @distributed for i in eachindex(m)
            m[i] = g(m[i])
        end

        return convert(Array, m)
    end

    ##-----------------------------------------------------------------------------------
    function apply_ps(g::Function, m)
        @inbounds @sync @distributed for i in eachindex(m)
            m[i] = g(m[i])
        end

        return m
    end

    ##------------------------------------------------------------------------------------
    function apply_triu(g::Function, m::Array{T, 2}) where T<:Number
        @inbounds for j = 2:size(m, 2), i = 1:j-1
            m[i, j] = g(m[i, j])
        end

        return m
    end

    ##------------------------------------------------------------------------------------
    function apply_tril(g::Function, m::Array{T, 2}) where T<:Number
        @inbounds for i = 2:size(m, 2), j = 1:i-1
            m[i, j] = g(m[i, j])
        end

        return m
    end

	##-----------------------------------------------------------------------------------
	function apply_dia(g::Function, m::Array{T, 2}) where T<:Number
		for i = min(size(m))
			m[i, i] = g(m[i, i])
		end

		return m
	end


    ##===================================================================================
    ##  boolean operators (aggregation)
    ##===================================================================================
    export and, or

    ##-----------------------------------------------------------------------------------
    function and(v::BitArray{1})
        @inbounds for i = 1:size(v, 1)
            if v[i] == false
                return false
            end
        end

        return true
    end

    ##-----------------------------------------------------------------------------------
    function or(v::BitArray{1})
        @inbounds for i = 1:size(v, 1)
            if v[i] == true
                return true
            end
        end

        return false
    end


    ##===================================================================================
    ##  / (overload)
    ##===================================================================================
	import Base./
	export /

    ##-----------------------------------------------------------------------------------
    function /(x::R, v::Array{R, 1}) where R<:AbstractFloat
        @inbounds for i = 1:size(v, 1)
            v[i] = x/v[i]
        end

        return v
    end


    ##===================================================================================
    ##  zero check
    ##===================================================================================
    export iszero

	##-----------------------------------------------------------------------------------
	function iszero(v)
		set_zero_subnormals(true)
		sumabs(v) == 0
	end
end
