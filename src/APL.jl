@everywhere module APL
    # Packages
    include("Packages/require.jl")
    include("Packages/pkg.jl")

    # ADT
    include("ADT/graph.jl")
    include("ADT/heap.jl")

    # Automatons
    include("Automatons/TuringMachine.jl")
    include("Automatons/DFA.jl")
    include("Automatons/PDA.jl")

    # Convertation
    include("Convertation/cnv.jl")

    # IO
    include("IO/tmp_op.jl")
    include("IO/const.jl")

    # Math
    include("Math/f.jl")
    include("Math/mat_op.jl")
    include("Math/dist.jl")
    include("Math/interpol.jl")
    include("Math/rg.jl")
    include("Math/mp.jl")
    include("Math/vq.jl")
    include("Math/yamartino.jl")
    include("Math/op.jl")
    #include("Math/func.jl") under construction...

    # ProcedualGeneration
    include("ProcedualGeneration/WaveFunctionCollapse.jl")

    # Percepton
    include("Percepton/MultilayerPercepton.jl")
    include("Percepton/Percepton.jl")

    # RBFNetworks
    include("RBFNetworks/RBFNetwork.jl")
end
