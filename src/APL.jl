@everywhere module APL
	# base imports
	import Base.deepcopy
	import Base.print

	# base usings
	using Base.LinAlg
	using Base.Test

    # atomic modules
	include("ADT/graph.jl")
	include("ADT/heap.jl")
	include("Automatons/dfa.jl")
	include("Automatons/pda.jl")
	include("Automatons/tuma.jl")
	include("Convertation/cnv.jl")
	include("IO/const.jl")
	include("IO/tmp_op.jl")
	include("MarketIndicators/rsi.jl")
	include("MarketIndicators/stosc.jl")
	include("Math/bin.jl")
	include("Math/bla.jl")
	include("Math/dist.jl")
	include("Math/gen.jl")
	include("Math/rnd.jl")
	include("Math/mean.jl")
	include("Math/op.jl")
	include("Math/yamartino.jl")
	include("Packages/pkg.jl")

	# atomic usings
	using graph
	using heap
	using dfa
	using pda
	using tuma
	using cnv
	using tmp_op
	using rsi
	using stosc
	using bin
	using bla
	using dist
	using gen
	using rnd
	using mean
	using op
	using yamartino
	using pkg
					

	# composite modules
	include("Math/trig.jl")					# bla
	include("Eva/eva.jl") 					# op/gen
	include("MarketIndicators/bb.jl") 		# sta
	include("MarketIndicators/macd.jl") 	# mean	
	include("Math/mpa.jl")					# f
	include("Math/sta.jl")					# mean/bla	
	include("Math/f.jl")					# cnv/gen/op
	include("Math/lalg.jl")					# mean/bla
	include("Math/vq.jl")					# f
	include("Percepton/mlp.jl")				# f
	include("Percepton/pct.jl")				# f
	include("ProcedualGeneration/wfc.jl")	# f/cnv 	
	include("RBFNetworks/rbfn.jl")			# f

	# composite usings
	using trig
	using sta
	using f
	using eva
	using bb
	using macd
	using mpa
	using lalg
	using vq
	using mlp
	using pct
	using wfc
	using rbfn
end
