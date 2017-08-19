@everywhere module t_eva
	##===================================================================================
	## using Directives
	##===================================================================================
	using Base.Test
	using eva

	##===================================================================================
	##	constants
	##===================================================================================
	const tol = 10.0^-3
	const t_alpha = [-5., -5.]
	const t_delta = [10., 10.]
	const t_n = 2
	const t_sp = t_ncbd(t_alpha, t_delta, t_n)
	const t_ff(v::Array{Float64, 1}) = 1/((x[1]^2 + x[2]^2)+1)
	const t_optp = t_opt_prb(t_sp, t_ff)
	const t_tc = (10000, .01)
	const t_gen_size = 500
	const t_mr = 1.


	##===================================================================================
	##	internal functions
	##===================================================================================
	function t_vl_rand()
		@testset "vl_rand function tests" begin
			vl = eva.vl_rand(t_sp, t_gen_size)
			@test size(vl, 1) == t_gen_size 	atol = 0
			@test size(vl[1], 1) == t_sp.n		atol = 0
			@test eltype(vl) == Array{Float64, 1}
		end
	end
end
