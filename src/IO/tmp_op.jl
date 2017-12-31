@everywhere module tmp_op
    export mv_to_tmp, mv_to_tmp!, clr_tmp

    ##-----------------------------------------------------------------------------------
    function mv_to_tmp(src::AbstractString)
        name = *(tempdir(), "/", bytes2hex(sha256(open(src))), ".dat")
        (!isfile(name) && isfile(src)) && writetable(name, readtable(src))
        return name
    end

    ##-----------------------------------------------------------------------------------
    function mv_to_tmp!(src::AbstractString)
        name = *(tempdir(), "/", bytes2hex(sha256(open(src))), ".dat")
        if isfile(src) && !isfile(name)
            src = name; writetable(name, readtable(src))
        end
        return src
    end

    ##-----------------------------------------------------------------------------------
    function clr_tmp()
        rm(*(tempdir(), "/"); force=true, recursive=true)
    end
end
