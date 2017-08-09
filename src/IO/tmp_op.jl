@everywhere module tmp_op
    export mv_to_tmp, mv_to_tmp!, clr_tmp

    ##-----------------------------------------------------------------------------------
    function mv_to_tmp(Src::AbstractString)
        name = *(tempdir(), "/", bytes2hex(sha256(open(Src))), ".dat")
        (!isfile(name) && isfile(Src)) && writetable(name, readtable(Src))
        return name
    end

    ##-----------------------------------------------------------------------------------
    function mv_to_tmp!(Src::AbstractString)
        name = *(tempdir(), "/", bytes2hex(sha256(open(Src))), ".dat")
        if isfile(Src) && !isfile(name)
            Src = name; writetable(name, readtable(Src))
        end
        return Src
    end

    ##-----------------------------------------------------------------------------------
    function clr_tmp()
        rm(*(tempdir(), "/"); force=true, recursive=true)
    end
end
