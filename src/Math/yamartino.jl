@everywhere module yamartino
    import Base.deepcopy
    export IYamartinoResult, TYamartinoState, YamartinoResult,
    CalcYamartino, PCalcYamartino


    ##===================================================================================
    ##  types
    ##===================================================================================
    immutable IYamartinoResult
        Direction::Real
        Deviation::Real
    end

    ##-----------------------------------------------------------------------------------
    type TYamartinoState
        SampleNumber::Int
        Sa::Real
        Ra::Real
    end


    ##===================================================================================
    ##  main
    ##===================================================================================
    function YamartinoResult(TYS::TYamartinoState)
        Sa = TYS.Sa / TYS.SampleNumber
        Ra = TYS.Ra / TYS.SampleNumber
        theta = sqrt(1-(Sa^2 + Ra^2))
        return IYamartinoResult(atan2(Ra, Sa), (asind(theta) * (1 + ((2/sqrt(3)) - 1) * theta^3)))
    end

    ##-----------------------------------------------------------------------------------
    function CalcYamartino{T<:AbstractFloat}(arr::Array{T, 1}, PTYS::TYamartinoState = TYamartinoState(-1, 0, 0))
        PTYS.SampleNumber += PTYS.SampleNumber < 0 ? size(arr, 1) + 1 : size(arr, 1)
        PTYS.Sa += sum(map(sind, arr))
        PTYS.Ra += sum(map(cosd, arr))
        return PTYS
    end

    ##-----------------------------------------------------------------------------------
    function PCalcYamartino{T<:AbstractFloat}(arr::Array{T, 1}, PTYS::TYamartinoState = TYamartinoState(-1, 0, 0))
        SampleCount = size(arr, 1)
        PTYS.SampleNumber += PTYS.SampleNumber < 0 ? SampleCount + 1 : SampleCount

        @parallel (+) for i = 1:SampleCount
            PTYS.Sa += sind(arr[i])
            PTYS.Ra += cosd(arr[i])
        end

        return PTYS
    end


    ##===================================================================================
    ## internal functions
    ##===================================================================================
    function deepcopy(yr::IYamartinoResult)
        return IYamartinoResult(
            deepcopy(yr.Direction),
            deepcopy(yr.Deviation)
        )
    end

    ##-----------------------------------------------------------------------------------
    function deepcopy(ys::TYamartinoState)
        return TYamartinoState(
            deepcopy(ys.SampleNumber),
            deepcopy(ys.Sa),
            deepcopy(ys.Ra)
        )
    end
end
