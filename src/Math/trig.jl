@everywhere module trig
	##===================================================================================
	##	using directives
	##===================================================================================
	using bla

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
	rad{T<:Number}(x::T) = @. x*0.0174532925199432957692369076849

	##-----------------------------------------------------------------------------------
	irad{T<:Number}(x::T) = @. x*57.2957795130823208767981548141


	##===================================================================================
	## angle
	##===================================================================================
	export angle, ccntrl_angle, scntrl_angle, tcntrl_angle 
	export cntrl_angle, hcntrl_angle, vincenty_cntrl_angle

	##-----------------------------------------------------------------------------------
	function angle{T<:AbstractFloat}(u::Array{T, 1}, v::Array{T, 1}, bias::T = T(0)) 
		return acosd((abs(bdot(u, v))/(bnrm(v)*bnrm(u)))+bias)
	end

	##-----------------------------------------------------------------------------------
	ccntrl_angle{T<:AbstractFloat}(u::Array{T, 1}, v::Array{T, 1}) = @. acos(bdot(u, v))           				# returns radians | u&v = normal vectors on the circle

	##-----------------------------------------------------------------------------------
	scntrl_angle{T<:AbstractFloat}(u::Array{T, 1}, v::Array{T, 1}) = @. asin(bnrm(cross(u, v)))     			# returns radians | u&v = normal vectors on the circle

	##-----------------------------------------------------------------------------------
	tcntrl_angle{T<:AbstractFloat}(u::Array{T, 1}, v::Array{T, 1}) = @. atan(bnrm(cross(u, v))/bdot(u, v))  	# returns radians | u&v = normal vectors on the circle

	##-----------------------------------------------------------------------------------
	cntrl_angle{T<:AbstractFloat}(pla::T, plo::T, sla::T, slo::T) = acos((sin(pla)*sin(sla))+(cos(pla)*cos(sla)*cos(abs(plo-slo)))) 						# returns radians | pla/sla = primary/secondary latitude / plo/slo = primary/secondary longitude

	##-----------------------------------------------------------------------------------
	hcntrl_angle{T<:AbstractFloat}(pla::T, plo::T, sla::T, slo::T) = 2*asin(sqrt(havsin(abs(pla-sla))+cos(pla)*cos(sla)*havsin(abs(plo-slo)))) 				# returns radians | pla/sla = primary/secondary latitude / plo/slo = primary/secondary longitude

	##-----------------------------------------------------------------------------------
	function vcntrl_angle{T<:AbstractFloat}(pla::T, plo::T, sla::T, slo::T)
		longitude_delta = abs(plo-slo)                                                                                      								# returns radians | pla/sla = primary/secondary latitude / plo/slo = primary/secondary longitude
		return atan2(sqrt((cos(sla)*sin(longitude_delta))^2+((cos(pla)*sin(sla))-(sin(pla)*cos(sla)*cos(longitude_delta)))^2), (sin(pla)*sin(sla)+cos(pla)*cos(sla)*cos(longitude_delta)))
	end
end
