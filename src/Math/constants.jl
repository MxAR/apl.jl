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
end
