@everywhere module eva
	##===================================================================================
	##	using directives
	##===================================================================================
	using op
	using f

	##===================================================================================
	##	types
	##===================================================================================
	type t_opt_prb																		# optimization problem
		sp::f.t_ncbd																		# search space
		ff::Function																	# fitness function
	end

	##===================================================================================
	##	eve
	##		evolve main function
	##		op = t_opt_prb
	##		tc = termination conditions
	##			tc[1] = maximal number of epochs
	##			tc[2] = minimal error where the search is aborted
	##		pop = initial population
	##		mut = mutation function
	##		mr = mutation rate
	##===================================================================================
	function eve(optp::t_opt_prb, tc::Tuple{Int64, Float64}, pop::Array{Array{Float64, 1}, 1}, mut::Function, mr::Function)
		supremum = optp.sp.alpha + optp.sp.delta										# supremum of the n dim cubiod (for mutation)
		gen_size = size(pop, 1)															# size of each generation
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

			max_delta = Base.map((x) -> abs(mr(cep)/x), optp.sp.delta)							# calculate the maximal change that can occure through mutation
			pop[1] = deepcopy(pop[cbe])													# move champion to the first place
			@inbounds for i = 2:gen_size												# generate a new generation from the genes of the champion
				pop[i] .= mut(pop[i], pop[1], optp.sp, max_delta, supremum)					# (mutation)
			end

			cep += 1																	# update epoche counter
		end
		return pop[1]																	# return overall champion
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
	function mut_default(child::Array{Float64, 1}, parent::Array{Float64, 1}, ncbd::f.t_ncbd, max_delta::Array{Float64, 1}, supremum::Array{Float64, 1})
		@inbounds for i=1:ncbd.n																		# mutate each value inside the vector
			child[i] = parent[i] + prison(2*(rand()-.5)*max_delta[i], ncbd.alpha[i], supremum[i])	# random mutation inside the given n dim intervall
		end
		return child
	end
end
