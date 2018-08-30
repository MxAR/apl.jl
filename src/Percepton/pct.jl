@everywhere module pct # percepton
    ##===================================================================================
    ##  types
    ##===================================================================================
    mutable struct tpct{R<:AbstractFloat}                                                         # Percepton
        weights::Array{R, 1}
        threshold::R
        threshold_func::Function
    end


    ##===================================================================================
    ## constructors
	##	tpctv2 will approximate all weights
    ##===================================================================================
    export tpctv0, tpctv1, tpctv2

    ##-----------------------------------------------------------------------------------
    function tpctv0(weights::Array{R, 1}, threshold::R = .5, threshold_func::Function = t(x, y) = x >= y ? 1 : 0) where R<:AbstractFloat
        return tpct(weights, threshold, threshold_func)
    end

    ##-----------------------------------------------------------------------------------
    function tpctv1(Inputs::Z, Random::Bool = true, threshold::R = .5, threshold_func::Function = t(x, y) = x >= y ? 1 : 0) where R<:AbstractFloat where Z<:Integer
        @assert(Inputs > 0, "the numerb of inputs musn't be negative")
        return tpct(Random ? randn(Inputs) : zeros(Inputs), threshold, threshold_func)
    end

    ##-----------------------------------------------------------------------------------
    function tpctv2(Inputs::Array{R, 2}, ExpectedResults::Array{R, 2}, threshold::R = .5, threshold_func::Function = t(x, y) = x >= y ? 1 : 0) where R<:AbstractFloat 
		return tpct((Inputs \ ExpectedResults), threshold, threshold_func)
    end


    ##===================================================================================
    ##  integrate and fire
    ##===================================================================================
    export iaf

    ##-----------------------------------------------------------------------------------
    function iaf(PCT::tpct, Input::Array{R, 1}) where R<:AbstractFloat
        @assert(size(PCT.weights, 1) == size(Input, 1), "input array length must match weight length")
        return PCT.threshold_func(bdot(PCT.weights, Input), PCT.threshold);
    end


    ##===================================================================================
    ## gradient descent training
    ##===================================================================================
    export gdb!, gdo!

    ##-----------------------------------------------------------------------------------
    function gdo!(PCT::tpct, Inputs::Array{R, 2}, ExpectedResults::Array{R, 1}, LearningRate::R = .2, MaxCycles::Z = -1) where R<:AbstractFloat where Z<:Integer
        InputD2Size = size(Inputs, 2)                                                   # faster than Batch with tpctv2
        CurrentInput = []
        epsilon = delta = 10

        while epsilon > 0 && MaxCycles != 0
            epsilon = 0
            for i = 1:size(Inputs, 1)
                Input = Inputs[i, 1:InputD2Size]
                delta = ExpectedResults[i] - PCT.threshold_func(bdot(PCT.weights, Input), PCT.threshold)
                if delta != 0
                    PCT.weights = map((x, y) -> y + (LearningRate * x * delta), Input, PCT.weights)
                    PCT.threshold -= LearningRate * delta
                    epsilon += abs(delta)
                end
            end
            MaxCycles -= 1;
        end

        return PCT
    end

	##-----------------------------------------------------------------------------------
    function gdb!(PCT::tpct, Inputs::Array{R, 2}, ExpectedResults::Array{R, 1}, LearningRate::R = .2, MaxCycles::Z = -1) where R<:AbstractFloat where Z<:Integer
        InputD2Size = size(Inputs, 2)
        CurrentInput = Wc = []
        epsilon = delta = Tc = 10

        while epsilon > 0  && MaxCycles != 0
            epsilon = Tc = 0
            Wc = zeros(InputD2Size)
            for i = 1:size(Inputs, 1)
                Input = Inputs[i, 1:InputD2Size]
                delta = ExpectedResults[i] - PCT.threshold_func(bdot(PCT.weights, Input), PCT.threshold)
                if delta != 0
                    Wc = map((x, y) -> y + (LearningRate * x * delta), Input, Wc)
                    Tc -= LearningRate * delta
                    epsilon += abs(delta)
                end
            end
            PCT.threshold += Tc
            PCT.weights += Wc
            MaxCycles -= 1;
        end

        return PCT
    end


    ##===================================================================================
    ## internal functions
    ##===================================================================================
    function deepcopy(PCT::tpct)
        return tpct(
            deepcopy(PCT.weights),
            deepcopy(PCT.threshold),
            deepcopy(PCT.threshold_func)
        )
    end
end
