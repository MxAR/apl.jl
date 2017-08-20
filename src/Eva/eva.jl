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
		const max_delta = map((x) -> abs(mr/x), optp.sp.delta)							# calculate the maximal change that can occure through mutation
		supremum = optp.sp.alpha + optp.sp.delta										# supremum of the n dim cubiod (for mutation)
		pop = vl_rand(optp.sp, gen_size)												# generate population from the given search space
		swp = Float64(0)																# swap var, mainly for searching for the best element
		cbe = Int64(0)																	# current best element (champion)
		cep = Int64(1)																	# current epoch
		cer = Inf																		# current error

		while cep <= tc[1] && cer > tc[2]												# initiate the process
			@inbounds for i = 1:gen_size												# get the fittest subject
				swp = Float64(optp.ff(pop[i]))												# evaluate the fitness of the subject
				if swp < cer																# compare it to the current champion
					cer = swp																	# if better the the highscore gets updated
					cbe = i 																	# and the subject becomes the new champion
				end
			end

			pop[1] = deepcopy(pop[cbe])													# move champion to the first place
			@inbounds for i = 2:gen_size												# generate a new generation from the genes of the champion
				pop[i] .= mut(pop[i], pop[1], optp.sp, max_delta, supremum)					# (mutation)
			end
			cep += 1																	# update epoche counter
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
		set_zero_subnormals(true)														# to save computing time
		@inbounds for i = 1:l															# fill the list
			vl[i] = ncbd.alpha+(rand(ncbd.n).*ncbd.delta)								# (filling)
		end
		return vl																		# return of the vl
	end


	##===================================================================================
	##	mut_default
	##		default mutation function for n dim vectors
	##		child = the array that will store a mutated version of the parent
	##		parent = the original parent that from which a new generation is created
	##		ncbd = t_ncbd
	##		max_delta = the maximal change that can occure through muation
	## 		supremum = the supremum of the n dim cubiod represented in optp.sp
	##===================================================================================
	function mut_default(child::Array{Float64, 1}, parent::Array{Float64, 1}, ncbd::t_ncbd, max_delta::Array{Float64, 1}, supremum::Array{Float64, 1})
		@inbounds for i=1:ncbd.n																		# mutate each value inside the vector
			child[i] = parent[i] + op.prison(2*(rand()-.5)*max_delta[i], ncbd.alpha[i], supremum[i])	# random mutation inside the given n dim intervall
		end
		return child
	end
end
