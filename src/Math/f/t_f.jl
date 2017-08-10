@everywhere module t_f
	##===================================================================================
	##	constants
	##===================================================================================
	const tol = 10.0^-3


	##===================================================================================
	## using Directives
	##===================================================================================
	using Base.Test
	using f


	##===================================================================================
	##	internal functions
	##===================================================================================
	function t_tanh()
		@testset "hyperbolic tangent function tests" begin
			@testset "tanh overload tests" begin
				@test f.tanh(3, 3)		≈ 0 	atol = tol
				@test f.tanh(.55, 0)  	≈ .5	atol = tol
			end
			@testset "tanh derivate tests" begin
				@test d_tanh(3, 3) 	≈ 1 	atol = tol
				@test d_tanh(.5, 0) ≈ .7864 atol = tol
			end
		end
	end

	##-----------------------------------------------------------------------------------


	##===================================================================================
	##	main functions
	##===================================================================================
	function main()
		t_tanh()
	end
end
