@everywhere module yamartino
	##===================================================================================
	## using directives
	##===================================================================================
	using Distributed	

	##===================================================================================
	## export directives
	##===================================================================================
    export IYamartResult, TYamartState, YamartResult, CalcYamart, PCalcYamart


    ##===================================================================================
    ##  types
    ##===================================================================================
    struct IYamartResult
        Direction::Real
        Deviation::Real
    end

    ##-----------------------------------------------------------------------------------
    mutable struct TYamartState
        SampleNumber::Int
        Sa::Real
        Ra::Real
    end


    ##===================================================================================
    ##  main
    ##===================================================================================
    function YamartResult(TYS::TYamartState)
        Sa = TYS.Sa / TYS.SampleNumber
        Ra = TYS.Ra / TYS.SampleNumber
        theta = sqrt(1-(Sa^2 + Ra^2))
        return IYamartResult(atan2(Ra, Sa), (asind(theta) * (1 + ((2/sqrt(3)) - 1) * theta^3)))
    end

    ##-----------------------------------------------------------------------------------
    function CalcYamart(arr::Array{R, 1}, PTYS::TYamartState = TYamartState(-1, 0, 0)) where R<:AbstractFloat
        PTYS.SampleNumber += PTYS.SampleNumber < 0 ? size(arr, 1) + 1 : size(arr, 1)
        PTYS.Sa += sum(map(sind, arr))
        PTYS.Ra += sum(map(cosd, arr))
        return PTYS
    end

    ##-----------------------------------------------------------------------------------
    function PCalcYamart(arr::Array{R, 1}, PTYS::TYamartState = TYamartState(-1, 0, 0)) where R<:AbstractFloat
        SampleCount = size(arr, 1)
        PTYS.SampleNumber += PTYS.SampleNumber < 0 ? SampleCount + 1 : SampleCount

        @distributed (+) for i = 1:SampleCount
            PTYS.Sa += sind(arr[i])
            PTYS.Ra += cosd(arr[i])
        end

        return PTYS
    end


    ##===================================================================================
    ## internal functions
    ##===================================================================================
    function deepcopy(yr::IYamartResult)
        return IYamartResult(
            deepcopy(yr.Direction),
            deepcopy(yr.Deviation)
        )
    end

    ##-----------------------------------------------------------------------------------
    function deepcopy(ys::TYamartState)
        return TYamartState(
            deepcopy(ys.SampleNumber),
            deepcopy(ys.Sa),
            deepcopy(ys.Ra)
        )
    end
end
