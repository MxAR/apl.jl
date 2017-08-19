@everywhere module eva
	##===================================================================================
	##	using directives
	##===================================================================================
	using op

	##===================================================================================
	##	types
	##===================================================================================
	type t_ncbd																			# n dimensional cuboid
		alpha::Array{Float64, 1}														# infimum 									(point)
		delta::Array{Float64, 1}														# diference between supremum and infimum	(s-i)
		n::Int64																		# n
	end

	##-----------------------------------------------------------------------------------
	type t_opt_prb																		# optimization problem
		sp::t_ncbd																		# search space
		ff::Function																	# fitness function
	end

	##===================================================================================
	##	eve
	##		evolve main function
	##		op = t_opt_prb
	##		tc = termination conditions
	##			tc[1] = maximal number of epochs
	##			tc[2] = minimal error where the search is aborted
	##		gen_size = size of each generation
	##		mut = mutation function
	##		mr = mutation rate
	##===================================================================================
	function eve(optp::t_opt_prb, tc::Tuple{Int64, Float64}, gen_size::Int64, mut::Function, mr::Float64)
		cc = [1, Inf, 0]; s = .0														# cc = current condition of the population (epoch, current error, best element)
		pop = vl_rand(optp.sp, gen_size)												# generate population from the given search space
		max_delta = abs(mr/ncbd.delta)													# calculate the maximal change that can occure through mutation
		while cc[1] <= tc[1] && cc[2] > tc[2]											# initiate the process
			@inbounds for i = 1:gen_size												# get the fittest subject
				s = optp.ff(pop[i])															# evaluate the fitness of the subject
				if s < cc[2]																# compare it to the current champion
					cc[2] = s																	# if better the the highscore gets updated
					cc[3] = i 																	# and the subject becomes the new champion
				end
			end

			pop[1] = pop[cc[3]]															# move champion to the first place
			@inbounds for i = 2:gen_size												# generate a new generation from the gens of the champion
				pop[i] = mut(pop[1], optp.sp, max_delta)									# (mutation)
			end

			c += 1																		# update epoche counter
		end
		return pop[1]																	# return overall champion
	end


	##===================================================================================
	##	vl_rand
	##		fills a vl with l vectors of dim n from a n dim cuboid
	##		ncbd = t_ncbd
	## 		l = length of vl
	##===================================================================================
	function vl_rand(ncbd::t_ncbd, l::Int64)
		vl = Array{Array{Float64, 1}, 1}(l)												# create an empty vl of length l
		@inbounds for i = 1:l															# fill the list
			vl[i] = ncbd.alpha+(rand(ncbd.n).*ncbd.delta)								# (filling)
		end
		return vl																		# return of the vl
	end


	##===================================================================================
	##	mut_default
	##		default mutation function for n dim vectors
	##		v = the vector to be mustated
	##		ncbd = t_ncbd
	##		max_delta = the maximal change that can occure through muation
	##===================================================================================
	function mut_default(v::Array{Float64, 1}, ncbd::t_ncbd, max_delta::Array{Float64, 1})
		supremum = ncbd.alpha .+ ncbd.delta												# calculate the supremum (for later checks)
		@inbound for i=1:ncbd.n															# mutate each value inside the vector
			v[i] += prison(randn()*max_delta[i], ncbd.alpha[i], ncbd.supremum[i])		# random mutation inside the given n dim intervall
		end
		return v																		# return the mutation
	end
end
