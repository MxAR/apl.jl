@everywhere module t_eva
	##===================================================================================
	## using Directives
	##===================================================================================
	using Base.Test
	using eva
	using f

	##===================================================================================
	##	constants
	##===================================================================================
	const tol = 10.0^-3
	const t_alpha = [-5., -5.]
	const t_delta = [10., 10.]
	const t_n = 2
	const t_sp = eva.t_ncbd(t_alpha, t_delta, t_n)
	const t_ff(v::Array{Float64, 1}) = v[1]^2 + v[2]^2
	const t_optp = eva.t_opt_prb(t_sp, t_ff)
	const t_tc = (10000, .01)
	const t_gen_size = 500
	const t_mr = 1.
	const t_max_delta = abs.(t_mr./t_sp.delta)


	##===================================================================================
	##	internal functions
	##===================================================================================
	function t_vl_rand()
		@testset "vl_rand function tests" begin
			vl = eva.vl_rand(t_sp, t_gen_size)
			@test size(vl, 1) == t_gen_size
			@test size(vl[1], 1) == t_sp.n
			@test eltype(vl) == Array{Float64, 1}
		end
	end

	##-----------------------------------------------------------------------------------
	function t_mut_default()
		v = eva.vl_rand(t_sp, 1)[1]; w = deepcopy(v)
		@testset "mut_default function tests" begin
			v = abs.(eva.mut_default(v, t_sp, t_max_delta).-w)
			@test f.AND(v.<=t_max_delta)
			@test sum(v) != 0.
		end
	end

	##-----------------------------------------------------------------------------------
	function t_eve()
		@time eva.eve(t_optp, t_tc, t_gen_size, eva.mut_default, t_mr)
	end

	##===================================================================================
	##	main function
	##===================================================================================
	function main()
		t_vl_rand()
		t_mut_default()
		t_eve()
	end
end
