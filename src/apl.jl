# path to apl
path = "/home/mxar/Documents/projects/apl.jl/"

# start a worker for each cpu thread
include(string(path, "src/start_workers.jl"))

# import everywhere directive for the modules
import Distributed.@everywhere

# atomic modules
include(string(path, "src/ADT/graph.jl"))
include(string(path, "src/ADT/heap.jl"))
include(string(path, "src/Automatons/dfa.jl"))
include(string(path, "src/Automatons/pda.jl"))
include(string(path, "src/Automatons/tuma.jl"))
include(string(path, "src/Convertation/cnv.jl"))
include(string(path, "src/IO/const.jl"))
include(string(path, "src/IO/tmp_op.jl"))
include(string(path, "src/technical_indicators/rsi.jl"))
include(string(path, "src/technical_indicators/stosc.jl"))
include(string(path, "src/Math/bin.jl"))
include(string(path, "src/Math/constants.jl"))
include(string(path, "src/Math/dist.jl"))
include(string(path, "src/Math/rnd.jl"))
include(string(path, "src/Math/mean.jl"))
include(string(path, "src/Math/op.jl"))
include(string(path, "src/Math/yamartino.jl"))
include(string(path, "src/Packages/pkg.jl"))

# atomic usings
using .graph
using .heap
using .dfa
using .pda
using .tuma
using .cnv
using .tmp_op
using .rsi
using .stosc
using .bin
using .constants
using .dist
using .rnd
using .mean
using .op
using .yamartino
using .pkg

# composite modules
include(string(path, "src/Math/gen.jl"))					# bin
include(string(path, "src/Eva/eva.jl"))						# op/gen
include(string(path, "src/technical_indicators/bb.jl")) 	# sta
include(string(path, "src/technical_indicators/macd.jl"))	# mean	
include(string(path, "src/Math/mpa.jl"))					# f	
include(string(path, "src/Math/f.jl"))						# cnv/gen/op
include(string(path, "src/Math/bla.jl"))					# mean/op
include(string(path, "src/Math/sta.jl"))					# mean/bla
include(string(path, "src/Math/trig.jl"))					# bla
include(string(path, "src/Math/vq.jl"))						# f
include(string(path, "src/Percepton/mlp.jl"))				# f
include(string(path, "src/Percepton/pct.jl"))				# f
include(string(path, "src/procedual_generation/wfc.jl"))	# f/cnv 	
include(string(path, "src/rbf_networks/rbfn.jl"))			# f

# composite usings
using .gen
using .f
using .eva
using .bb
using .macd
using .mpa
using .bla
using .sta
using .trig
using .vq
using .mlp
using .pct
using .wfc
using .rbfn
