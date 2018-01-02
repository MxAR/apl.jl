@everywhere module t_eva
	##===================================================================================
	## using Directives
	##===================================================================================
	using Base.Test
	using eva
	using op
	using f

	##===================================================================================
	##	constants
	##===================================================================================
	const t_tol = 10.0^-3
	const t_alpha = [-5., -5.]
	const t_delta = [10., 10.]
	const t_n = 2
	const t_sp = f.tncbd(t_alpha, t_delta, t_n)
	const t_ff(v::Array{Float64, 1}) = v[1]^2 + v[2]^2
	const t_optp = eva.topt_prb(t_sp, t_ff)
	const t_tc = (100000, .01)
	const t_gen_size = 500
	const t_mr = (x) -> 1.0
	const t_max_delta = [.1, .1]


	##===================================================================================
	##	internal functions
	##===================================================================================
	function t_vl_rand()
		@testset "vl_rand function tests" begin
			vl = f.vl_rand(t_sp, t_gen_size)
			@test size(vl, 1) == t_gen_size
			@test size(vl[1], 1) == t_sp.n
			@test eltype(vl) == Array{Float64, 1}
		end
	end

	##-----------------------------------------------------------------------------------
	function t_mut_default()
		v = f.vl_rand(t_sp, 1)[1]; w = deepcopy(v)
		@testset "mut_default function tests" begin
			v = abs.(eva.mut_default(v, v, t_sp, t_max_delta, t_sp.alpha + t_sp.delta).-w)
			@test op.AND(v.<=t_max_delta)
			@test sum(v) != 0.
		end
	end

	##-----------------------------------------------------------------------------------
	function t_eve()
		@testset "eve function test" begin
			v = eva.eve(t_optp, t_tc, f.vl_rand(t_sp, t_gen_size), eva.mut_default, t_mr)
			@test t_ff(v) <= t_tc[2]
		end
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
