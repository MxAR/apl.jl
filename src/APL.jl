@everywhere module APL
    # Packages
    include("Packages/require.jl")
    include("Packages/pkg.jl")

    # IO
    include("IO/const.jl")
    include("IO/tmp_op.jl")

    # ADT
    include("ADT/graph.jl")
    include("ADT/heap.jl")

    # Automatons
    include("Automatons/TuringMachine.jl")
    include("Automatons/DFA.jl")
    include("Automatons/PDA.jl")

    # Convertation
    include("Convertation/cnv.jl")

    # Math
    include("Math/op.jl")
    include("Math/f/f.jl")
    include("Math/dist.jl")
    include("Math/interpol.jl")
    include("Math/rg.jl")
    include("Math/mpa.jl")
    include("Math/vq.jl")
    include("Math/yamartino.jl")
    #include("Math/func.jl") under construction...

    # MarketIndicators
    include("MarketIndicators/med.jl")
    include("MarketIndicators/mavg.jl")
    include("MarketIndicators/macd.jl")
    include("MarketIndicators/rsi.jl")
    include("MarketIndicators/bb.jl")
    include("MarketIndicators/stosc.jl")


    # ProcedualGeneration
    include("ProcedualGeneration/WaveFunctionCollapse.jl")

    # Percepton
    include("Percepton/MultilayerPercepton.jl")
    include("Percepton/Percepton.jl")

    # Eva
    include("Eva/eva.jl")

    # RBFNetworks
    include("RBFNetworks/RBFNetwork.jl")

    # init.jl
    #include("init.jl")
end
