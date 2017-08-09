@everywhere module pct
    ##===================================================================================
    ##  using directives
    ##===================================================================================
    import Base.deepcopy
    using f


    ##===================================================================================
    ##  types
    ##===================================================================================
    type tpct                                                                           # Percepton
        weights::Array{Real, 1}
        threshold::Float64
        threshold_func::Function
    end


    ##===================================================================================
    ##  constructors
    ##===================================================================================
    export tpctv0, tpctv1, tpctv2

    ##-----------------------------------------------------------------------------------
    function tpctv0(weights, threshold = 0.5, threshold_func::Function = t(x, y) = x >= y ? 1 : 0)
        return tpct(weights, threshold, threshold_func)
    end

    ##-----------------------------------------------------------------------------------
    function tpctv1(Inputs::Int, Random = true, threshold = 0.5, threshold_func::Function = t(x, y) = x >= y ? 1 : 0)
        @assert Inputs > 0 ["the inputs musn't be negative"]
        return tpct(Random ? randn(Inputs) : zeros(Inputs), threshold, threshold_func)
    end

    ##-----------------------------------------------------------------------------------
    function tpctv2{T}(Inputs::Array{T, 2}, ExpectedResults::Array{T, 2}, threshold = 0.5, threshold_func::Function = t(x, y) = x >= y ? 1 : 0) # constructor which will approximate all weights
        return tpct((Inputs \ ExpectedResults), threshold, threshold_func)
    end


    ##===================================================================================
    ##  integrate and fire
    ##===================================================================================
    export iaf

    ##-----------------------------------------------------------------------------------
    function iaf{T}(PCT::tpct, Input::Array{T, 1})
        @assert length(PCT.weights) == length(Input) ["input array length must match weight length"]
        return PCT.threshold_func(bdot(PCT.weights, Input), PCT.threshold);
    end


    ##===================================================================================
    ## gradient descent training
    ##===================================================================================
    export gdb!, gdo!

    ##-----------------------------------------------------------------------------------
    function gdo!{T}(PCT::tpct, Inputs::Array{T, 2}, ExpectedResults::Array{T, 1}, LearningRate::T = 0.2, MaxCycles::Int = -1)
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

    function gdb!{T}(PCT::tpct, Inputs::Array{T, 2}, ExpectedResults::Array{T, 1}, LearningRate::T = 0.2, MaxCycles::Int = -1)
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
