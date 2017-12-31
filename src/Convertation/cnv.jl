@everywhere module cnv
    ##===================================================================================
    ## hex conversion
    ##===================================================================================
    export hex_to_vec, chex_to_vec, vec_to_hex

    ##-----------------------------------------------------------------------------------
    function hex_to_vec(str::String)
        if (isodd(length(str))) str = string(str, str) end
        return convert(Array{Float64, 1}, hex2bytes(str))
    end

    ##-----------------------------------------------------------------------------------
    function chex_to_vec(str::String)
        return hs2vec(strip(str, [ '"', '#' ]))
    end

    ##-----------------------------------------------------------------------------------
    function vec_to_hex(arr)
        str = ""

        str = string((arr[1] < 16 ? "0" : ""), str, hex(arr[1]))
        str = string((arr[2] < 16 ? "0" : ""), str, hex(arr[2]))
        str = string((arr[3] < 16 ? "0" : ""), str, hex(arr[3]))

        return str
    end


    ##===================================================================================
    ## coordinate (latitude/longitude) <> points in three dimensional space
    ##===================================================================================
    export coord_to_nvec, nvec_to_coord

    ##-----------------------------------------------------------------------------------
    coord_to_nvec{T<:Number}(la::T, lo::T) = [cos(la)*cos(lo), cos(la)*sin(lo), sin(la)]

    ##-----------------------------------------------------------------------------------
    nvec_to_coord{T<:Number}(v::Array{T, 1}) = [atan2(v[3], norm(v[1:2], 2)), atan2(v[2], v[1])]  # return: [la, lo]


    ##===================================================================================
    ## skew symmetric matrix
    ##===================================================================================
    export ssmm

    ##-----------------------------------------------------------------------------------
    function ssm{T<:Number}(v::Array{T, 1})
        l = size(v, 1); m = cross(v, b)
        b = insert!(zeros(l-1), 1, 1)

        @inbounds for i = 2:l
            b = circshift(b, 1)
            m = hcat(m, cross(v, b))
        end

        return m
    end


    ##===================================================================================
    ## pdrom_to_nvec (calculates the normal vector for a point on a sphere)
    ##===================================================================================
    export pdrom_to_nvec

    ##-----------------------------------------------------------------------------------
    pdrom_to_nvec{T<:Number}(v::Array{T, 1}, center::Array{T, 1} = zeros(3)) = normalize(v-center)


    ##===================================================================================
    ## join (vector list)
    ##===================================================================================
    export join

    ##-----------------------------------------------------------------------------------
    function join{T<:Any}(vl::Array{Array{T, 1}, 1})
        v = vl[1]

        @inbounds for i=2:length(vl)
            v = cat(1, v, vl[i])
        end

        return v
    end

    ##===================================================================================
    ## matrix -> vector list
    ##===================================================================================
    export mat_to_vl, vl_to_mat

    ##-----------------------------------------------------------------------------------
    function mat_to_vl{T<:Any}(m::Array{T, 2}, columns = true)
        m = columns ? m : m'
        vl = []

        @inbounds for i = 1:size(m, 2)
            push!(vl, m[:, i])
        end

        return vl
    end

    function vl_to_mat{T<:Any}(vl::Array{Array{T, 1}, 1}, columns = true)
        m = vl[1]

        @inbounds for i = 2:size(vl, 1)
            m = cat(2, m, vl[i])
        end

        return columns ? m : m'
    end
end
