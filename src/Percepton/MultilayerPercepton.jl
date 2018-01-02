@everywhere module mlp
    import Base.deepcopy
    import Base.print

    ##===================================================================================
    ##  using directives
    ##===================================================================================
    using f


    ##===================================================================================
    ##  types
    ##===================================================================================
    type tlayer{T<:AbstractFloat, N<:Int}
        LTWM::Array{T, 2}                                                            # Layer-Transition-Weight-Matrix | x = current y = preceding
        LTWMD::Array{T, 2}
        Threshold::Array{T, 1}
        ThresholdD::Array{T, 1}
        SubsequentLayer::N
        PreceedingLayer::N
    end

    type tmlp{T<:Int}                                                                           # MultilayerPercepton
        Layers::Array{Any, 1}
        ThresholdFunction::Function
        ThresholdFunctionDerivate::Function
        FirstHiddenLayer::T
        OutputLayer::T
    end


    ##===================================================================================
    ##  constructors
    ##===================================================================================
    export tmlpv0, disjuncfunc_to_tmlp

    ##-----------------------------------------------------------------------------------
    function tmlpv0{N<:Integer}(NumberOfInputs::N, NeuronsPerLayer::Array{N, 1}, ThresholdFunction::Function, ThresholdFunctionDerivate::Function, RandomWeights::Bool = true)
        MLP = tmlp([], ThresholdFunction, ThresholdFunctionDerivate, 1, length(NeuronsPerLayer))
        for (i, l) in enumerate(NeuronsPerLayer)
            push!(MLP.Layers, tlayer(
                RandomWeights == true ? rand(l, NumberOfInputs) * 0.5 : zeros(l, NumberOfInputs), zeros(l, NumberOfInputs),
                RandomWeights == true ? rand(l) * 0.5 : zeros(l), zeros(l), i+1, i-1))
            NumberOfInputs = l
        end
        return MLP
    end

    ##-----------------------------------------------------------------------------------
    function disj_tmlp(BooleanFunction::AbstractString)
        BooleanFunction = replace(uppercase(BooleanFunction), " ", "")
        # "([A-Z])((-([A-Z]|\([A-Z](\+[A-Z]|)*\)))*|)" | OR -> -   AND -> + | Regex for disjunctive normal form
        @assert(ismatch(r"([a-zA-Z])((-([a-zA-Z]|\([a-zA-Z](\+[a-zA-Z]|)*\)))*|)", BooleanFunction), "it would be great if the boolean function matches the disjunctive normal form or the syntax")

        or = 1; lambda = []
        for n in BooleanFunction
            if (ismatch(r"[a-zA-Z]", string(n)) && isempty(find(lambda .== n)))
                append!(lambda, [n])
            elseif n == '-' or += 1 end
        end

        t(x, y) = x >= y ? 1 : 0
        MLP = tmlp([
            tlayer(zeros(or, length(lambda)), [], fill(1.0, or), 2, 0),
            tlayer(fill(1.0, 1, or), [], [1.0], 3, 1)
        ], t, 1, 2)

        lambda = map((n) -> Char(n), sort(map((n) -> Int(Char(n[1])), lambda)))
        for n in split(BooleanFunction, ['-'])
            swap = split(replace(n, ['+', '(', ')'],""), "")
            weight = 1 / length(swap)
            for l in swap MLP.Layers[1].LTWM[or, find(lambda .== l[1])] = weight end
            or -= 1
        end

        return MLP
    end


    ##===================================================================================
    ##  integrate and fire
    ##===================================================================================
    export iaf

    ##-----------------------------------------------------------------------------------
    function iaf{T<:AbstractFloat}(MLP::tmlp, V::Array{T, 1})
        NextLayer = MLP.FirstHiddenLayer
        while true
            V = map(MLP.ThresholdFunction, *(MLP.Layers[NextLayer].LTWM, V), MLP.Layers[NextLayer].Threshold)
            NextLayer != MLP.OutputLayer ? NextLayer = MLP.Layers[NextLayer].SubsequentLayer : return V
        end
    end


    ##===================================================================================
    ## gradient descent training
    ##===================================================================================
    export gdb!

    ##-----------------------------------------------------------------------------------
    function gdb!{T<:AbstractFloat, N<:Integer}(MLP::tmlp, TrainingData::Array{Array{Array{T, 1}, 1}, 1}, LearningRate::T = 0.2, MaxEp::N = 2000, MaxErr::T = 0.01, Alpha::T = 0.2, Beta::T = 0.998)
        CVSize = convert(Int, round(length(TrainingData) / log(length(TrainingData))))  # TrainingData example:
        CVSize = 10
        NetworkLength = length(MLP.Layers)                                              # second subarray: output
                                                                                        # 4-element Array{Any,1}:
        for E = 1:MaxEp                                                                 #     Array{Float64,1}[[2.0,2.0],[2.0,2.0]]
            epsilon = 0.0                                                               #     Array{Float64,1}[[2.0,2.0],[2.0,2.0]]
            for TD in samples(TrainingData, CVSize)                                     #     Array{Float64,1}[[2.0,2.0],[2.0,2.0]]
                NextLayer = MLP.FirstHiddenLayer                                        #     Array{Float64,1}[[2.0,2.0],[2.0,2.0]]
                PH = Array{Any, 2}(NetworkLength, 2)                                    # MaxEp = maximal number of epochs
                PH[NextLayer, 1] = TD[1]                                                # MaxErr = maximal error that is allowed
                delta = []                                                              # Alpha = number by which the threshold derivate is lifted
                                                                                        # Beta  = weight decay factor
                # forward-propagation
                for i = 1:NetworkLength
                    PH[NextLayer, 2] = *(MLP.Layers[NextLayer].LTWM, PH[NextLayer, 1])
                    PH[NextLayer, 1] = map(MLP.ThresholdFunction, PH[NextLayer, 2], MLP.Layers[NextLayer].Threshold)
                    if NextLayer != MLP.OutputLayer
                        PH[MLP.Layers[NextLayer].SubsequentLayer, 1] = PH[NextLayer, 1]
                        NextLayer = MLP.Layers[NextLayer].SubsequentLayer
                    else break end
                end

                # vertex
                delta = (TD[2] - PH[NextLayer, 1]) .* (map((x, y) -> begin
                    z = MLP.ThresholdFunctionDerivate(x, y)
                    return z < 0 ? z - Alpha :  z + Alpha
                end, PH[NextLayer, 2], MLP.Layers[NextLayer].Threshold))
                epsilon += sum(abs, TD[2] - PH[NextLayer])

                CNL = [0, NextLayer]
                delta = [0, delta]

                # back-propagation
                for i = 1:NetworkLength
                    CNL = [CNL[2], MLP.Layers[CNL[2]].PreceedingLayer]
                    delta = [delta[2], (MLP.Layers[CNL[1]].LTWM' * delta[2])]

                    out = MLP.Layers[CNL[1]].PreceedingLayer == 0 ? TD[1] : PH[MLP.Layers[CNL[1]].PreceedingLayer]
                    MLP.Layers[CNL[1]].LTWMD = min(1., max(-1., MLP.Layers[CNL[1]].LTWMD + (delta[1] * out') * LearningRate))
                    MLP.Layers[CNL[1]].ThresholdD = min(1., max(-1., MLP.Layers[CNL[1]].ThresholdD - delta[1] * LearningRate))

                    if CNL[2] == 0 break; else
                        delta[2] .*= (map((x, y) -> begin
                            z = MLP.ThresholdFunctionDerivate(x, y)
                            return z < 0 ? z - Alpha :  z + Alpha
                        end, PH[CNL[2], 2], MLP.Layers[CNL[2]].Threshold))
                    end
                end
            end

            for i = 1:NetworkLength
                MLP.Layers[i].LTWM .*= Beta
                MLP.Layers[i].LTWM = min(1., max(-1., MLP.Layers[i].LTWM + MLP.Layers[i].LTWMD))
                MLP.Layers[i].LTWMD = 0.2 * (MLP.Layers[i].LTWMD)

                MLP.Layers[i].Threshold .*= Beta
                MLP.Layers[i].Threshold = min(1., max(-1., MLP.Layers[i].Threshold - MLP.Layers[i].ThresholdD))
                MLP.Layers[i].ThresholdD = 0.2 * (MLP.Layers[i].ThresholdD)
            end

            #println(epsilon/CVSize, " <> ", E)
            #println("-------------------")

            if epsilon <= MaxErr
                for TD in TrainingData
                    epsilon += sumabs2(TD[2] - iaf(MLP, TD[1]))
                end
                epsilon <= MaxErr && break
            end
        end
        #println(TrainingData)
        return MLP
    end


    ##===================================================================================
    ## internal functions
    ##===================================================================================
    function print(io::IO, MLP::tmlp)
        NL = MLP.FirstHiddenLayer
        println("layers:")
        while true
            println(io, MLP.Layers[NL])
            println("--------------------------------------------------------")
            NL != MLP.OutputLayer ? NL = MLP.Layers[NL].SubsequentLayer : break
        end
        println(io, "threshold function: ", MLP.ThresholdFunction)
        println(io, "threshold function derivate: ", MLP.ThresholdFunctionDerivate)
        println(io, "first hidden layer: ", MLP.FirstHiddenLayer)
        println(io, "output layer: ", MLP.OutputLayer)
    end

    ##-----------------------------------------------------------------------------------
    function print(io::IO, Layer::tlayer)
        println(io, "layer transition weight matrix: ", Layer.LTWM)
        println(io, "threshold: ", Layer.Threshold)
        println(io, "subsequent layer: ", Layer.SubsequentLayer)
        println(io, "preceeding layer: ", Layer.PreceedingLayer)
    end

    ##-----------------------------------------------------------------------------------
    function deepcopy(MLP::tmlp)
        return tmlp(
            map(deepcopy, MLP.Layers),
            deepcopy(MLP.ThresholdFunction),
            deepcopy(MLP.ThresholdFunctionDerivate),
            deepcopy(MLP.FirstHiddenLayer),
            deepcopy(MLP.OutputLayer)
        )
    end

    ##-----------------------------------------------------------------------------------
    function deepcopy(TLA::tlayer)
        return tlayer(
            deepcopy(TLA.LTWM),
            deepcopy(TLA.LTWMD),
            deepcopy(TLA.Threshold),
            deepcopy(TLA.ThresholdD),
            deepcopy(TLA.SubsequentLayer),
            deepcopy(TLA.PreceedingLayer)
        )
    end
end
