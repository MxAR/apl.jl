# path to apl
path = "/home/mxar/Documents/Projects/"

# base imports
import Base.deepcopy
import Base.print

# special base modules
using SpecialFunctions

# base usings
using Base.LinAlg
using Base.Test

# atomic modules
include(string(path, "apl.jl/src/ADT/graph.jl"))
include(string(path, "apl.jl/src/ADT/heap.jl"))
include(string(path, "apl.jl/src/Automatons/dfa.jl"))
include(string(path, "apl.jl/src/Automatons/pda.jl"))
include(string(path, "apl.jl/src/Automatons/tuma.jl"))
include(string(path, "apl.jl/src/Convertation/cnv.jl"))
include(string(path, "apl.jl/src/IO/const.jl"))
include(string(path, "apl.jl/src/IO/tmp_op.jl"))
include(string(path, "apl.jl/src/MarketIndicators/rsi.jl"))
include(string(path, "apl.jl/src/MarketIndicators/stosc.jl"))
include(string(path, "apl.jl/src/Math/bin.jl"))
include(string(path, "apl.jl/src/Math/dist.jl"))
include(string(path, "apl.jl/src/Math/rnd.jl"))
include(string(path, "apl.jl/src/Math/mean.jl"))
include(string(path, "apl.jl/src/Math/op.jl"))
include(string(path, "apl.jl/src/Math/yamartino.jl"))
include(string(path, "apl.jl/src/Packages/pkg.jl"))

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
using dist
using rnd
using mean
using op
using yamartino
using pkg
			

# composite modules
include(string(path, "apl.jl/src/Math/gen.jl"))					# bin
include(string(path, "apl.jl/src/Eva/eva.jl"))					# op/gen
include(string(path, "apl.jl/src/MarketIndicators/bb.jl")) 		# sta
include(string(path, "apl.jl/src/MarketIndicators/macd.jl")) 	# mean	
include(string(path, "apl.jl/src/Math/mpa.jl"))					# f	
include(string(path, "apl.jl/src/Math/f.jl"))					# cnv/gen/op
include(string(path, "apl.jl/src/Math/bla.jl"))					# mean/op
include(string(path, "apl.jl/src/Math/sta.jl"))					# mean/bla
include(string(path, "apl.jl/src/Math/trig.jl"))				# bla
include(string(path, "apl.jl/src/Math/vq.jl"))					# f
include(string(path, "apl.jl/src/Percepton/mlp.jl"))			# f
include(string(path, "apl.jl/src/Percepton/pct.jl"))			# f
include(string(path, "apl.jl/src/ProcedualGeneration/wfc.jl"))	# f/cnv 	
include(string(path, "apl.jl/src/RBFNetworks/rbfn.jl"))			# f

# composite usings
using gen
using f
using eva
using bb
using macd
using mpa
using bla
using sta
using trig
using vq
using mlp
using pct
using wfc
using rbfn
