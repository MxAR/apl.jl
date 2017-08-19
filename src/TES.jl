@everywhere module TES
	include("APL.jl")

	include("Math/f/t_f.jl")
	using t_f
	t_f.main()

	include("Eva/t_eva.jl")
	using t_eva
	t_eva.main()
end
