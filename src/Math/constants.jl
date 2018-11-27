@everywhere module constants
	##===================================================================================
	## tau (2*pi)
	##===================================================================================
	const tau_f128 = BigFloat(2 * BigFloat(pi))
	const tau_f64 = convert(Float64, tau_f128)
	const tau_f32 = convert(Float32, tau_f128)
	const tau_f16 = convert(Float16, tau_f128)
	const tau = tau_f64

	##===================================================================================
	## eulers constant
	##===================================================================================
	const euler_f128 = BigFloat(.5772156649015328606065120900824024310421593359399235988)
	const euler_f64 = convert(Float64, euler_f128)
	const euler_f32 = convert(Float32, euler_f128)
	const euler_f16 = convert(Float16, euler_f128)
	const euler = euler_f64

	##===================================================================================
	## fibonacci phi
	##===================================================================================
	const fib_phi_f128 = BigFloat(.5) + sqrt(BigFloat(1.25))
	const fib_phi_f64 = convert(Float64, fib_phi_f128)
	const fib_phi_f32 = convert(Float32, fib_phi_f128)
	const fib_phi_f16 = convert(Float16, fib_phi_f128)
	const fib_phi = fib_phi_f64

	##===================================================================================
	## plastic number
	##===================================================================================
	const plastic_num_f128 = (0.5 + sqrt(BigFloat(69))/(18))^(1/BigFloat(3)) + (0.5 - sqrt(BigFloat(69))/(18))^(1/BigFloat(3))
	const plastic_num_f64 = convert(Float64, plastic_num_f128)
	const plastic_num_f32 = convert(Float32, plastic_num_f128)
	const plastic_num_f16 = convert(Float16, plastic_num_f128)
	const plastic_num = plastic_num_f64
end
