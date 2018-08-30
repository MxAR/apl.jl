@everywhere module trig
	##===================================================================================
	##	using directives
	##===================================================================================
	using ..bla

	##===================================================================================
	## basic
	##===================================================================================
	export sin2, cos2, versin, aversin, vercos, avercos, coversin, acoversin, covercos
	export acovercos, havsin, ahavsin, havcos, ahavcos, hacoversin, hacovercos, tanh, tanhd

	##-----------------------------------------------------------------------------------
	sin2(alpha) = @. sin(alpha)^2

	##-----------------------------------------------------------------------------------
	cos2(alpha) = @. cos(alpha)^2

	##-----------------------------------------------------------------------------------
	versin(alpha) = @. 1-cos(alpha)

	##-----------------------------------------------------------------------------------
	aversin(alpha) = @. acos(1-alpha)

	##-----------------------------------------------------------------------------------
	vercos(alpha) = @. 1+cos(alpha)

	##-----------------------------------------------------------------------------------
	avercos(alpha) = @. acos(1+alpha)

	##-----------------------------------------------------------------------------------
	coversin(alpha) = @. 1-sin(alpha)

	##-----------------------------------------------------------------------------------
	acoversin(alpha) = @. asin(1-alpha)

	##-----------------------------------------------------------------------------------
	covercos(alpha) = @. 1+sin(alpha)

	##-----------------------------------------------------------------------------------
	acovercos(alpha) = @. asin(1+alpha)

	##-----------------------------------------------------------------------------------
	havsin(alpha) = @. versin(alpha)/2

	##-----------------------------------------------------------------------------------
	ahavsin(alpha) = @. 2*asin(sqrt(alpha))

	##-----------------------------------------------------------------------------------
	havcos(alpha) = @. vercos(alpha)/2

	##-----------------------------------------------------------------------------------
	ahavcos(alpha) = @. 2*acos(sqrt(alpha))

	##-----------------------------------------------------------------------------------
	hacoversin(alpha) = @. coversin(alpha)/2

	##-----------------------------------------------------------------------------------
	hacovercos(alpha) = @. covercos(alpha)/2

	##-----------------------------------------------------------------------------------
	tanh(x, eta) = @. Base.tanh(x-eta)

	##-----------------------------------------------------------------------------------
	tanhd(x, eta = 0) = @. 1-(tanh(x, eta)^2)
	

	##===================================================================================
	##	radian conversion
	##===================================================================================
	export rad, irad

	##-----------------------------------------------------------------------------------
	rad(x::T) where T<:Number = @. x*0.0174532925199432957692369076849

	##-----------------------------------------------------------------------------------
	irad(x::T) where T<:Number = @. x*57.2957795130823208767981548141


	##===================================================================================
	## angle
	##===================================================================================
	export angle, ccntrl_angle, scntrl_angle, tcntrl_angle 
	export cntrl_angle, hcntrl_angle, vincenty_cntrl_angle

	##-----------------------------------------------------------------------------------
	function angle(u::Array{R, 1}, v::Array{R, 1}, bias::R = 0.) where R<:AbstractFloat 
		return acosd((abs(bdot(u, v))/(bnrm(v)*bnrm(u)))+bias)
	end

	##-----------------------------------------------------------------------------------
	function ccntrl_angle(u::Array{R, 1}, v::Array{R, 1}) where R<:AbstractFloat
		return @. acos(bdot(u, v))           			# returns radians | u&v = normal vectors on the circle
	end

	##-----------------------------------------------------------------------------------
	function scntrl_angle(u::Array{R, 1}, v::Array{R, 1}) where R<:AbstractFloat
		return @. asin(bnrm(cross(u, v)))     			# returns radians | u&v = normal vectors on the circle
	end

	##-----------------------------------------------------------------------------------
	function tcntrl_angle(u::Array{R, 1}, v::Array{R, 1}) where R<:AbstractFloat 
		return @. atan(bnrm(cross(u, v))/bdot(u, v))  	# returns radians | u&v = normal vectors on the circle
	end

	##-----------------------------------------------------------------------------------
	function cntrl_angle(pla::R, plo::R, sla::R, slo::R) where R<:AbstractFloat 
		return acos((sin(pla)*sin(sla))+(cos(pla)*cos(sla)*cos(abs(plo-slo))))				# returns radians | pla/sla = primary/secondary latitude / plo/slo = primary/secondary longitude
	end

	##-----------------------------------------------------------------------------------
	function hcntrl_angle(pla::R, plo::R, sla::R, slo::R) where R<:AbstractFloat 
		return 2*asin(sqrt(havsin(abs(pla-sla))+cos(pla)*cos(sla)*havsin(abs(plo-slo))))	# returns radians | pla/sla = primary/secondary latitude / plo/slo = primary/secondary longitude
	end

	##-----------------------------------------------------------------------------------
	function vcntrl_angle(pla::R, plo::R, sla::R, slo::R) where R<:AbstractFloat
		longitude_delta = abs(plo-slo)														# returns radians | pla/sla = primary/secondary latitude / plo/slo = primary/secondary longitude
		return atan2(sqrt((cos(sla)*sin(longitude_delta))^2+((cos(pla)*sin(sla))-(sin(pla)*cos(sla)*cos(longitude_delta)))^2), (sin(pla)*sin(sla)+cos(pla)*cos(sla)*cos(longitude_delta)))
	end
end
