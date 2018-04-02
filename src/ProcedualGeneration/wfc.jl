@everywhere module wfc # wavefunctioncollapse
	##===================================================================================
	## main
	##===================================================================================
	export wfcc

	##-----------------------------------------------------------------------------------
	function wfcc{T<:Integer}(pairs::Dict{T, Array{T, 1}}, space::Tuple{T, T}, border_type::T, start::Tuple{T, T, T})
		plane = fill(-1, space[1], space[2])											# to be filled plane
		plane[start[1], start[2]] = start[3]											# set intial tile

		tbd = nigb(start[1:2], space, plane)[2]											# add the neighbouring tiles to the list (to be done (rbd)) of tiles to be processed

		# check if the first tile is vaild
		@assert(!isempty(intersect(nigbt(tbd, plane, pairs, border_type))), "given initial tile is invalid")

		# while there are indefinite tiles
		while !isempty(tbd)
			x = pop!(tbd)																# get an idf tile

			nb = nigb(x, space, plane)													# get the neighbours
			if isempty(nb[1]); continue end

			nt = nigbt(nb[1], plane, pairs, border_type)								# get all type pairs for the neighbours
			ni = intersect(nt)

			if isempty(ni)																# if there is no common type, then the neighbours must be changed (backtracking)
				u = union(nt)															# convert nt (vl) to matrix, so that find() works | get union of all neighbouring types
				d = Dict(zip(map((n) -> length(find(join(nt) .== n)), u), u))			# dictionary where the keys are the number of neighbours that share that type (value)

				while !isempty(d) && d[maximum(collect(keys(d)))] == border_type		# check if the maximum points to the border_type
					delete!(d, getindex(maximum(collect(keys(d)))))						# if so delete the entrie, since we don't want to create a border inside the plane
				end


				if isempty(d)															# if there are no remaing shares
					nt = find(nt .!= border_type)										# select all types that are not the border_type
					plane[x[1], x[2]] = nt[rand(1:1:length(nt))]						# and chose one randomly
				else
					plane[x[1], x[2]] = d[maximum(collect(keys(d)))]					# but if there are any left chose the one with the biggest share
				end

				for n in nb[1]															# all neighbours are now reseted to -1 and added to the tbd list
					plane[n[1], n[2]] = -1												# if so set tile back to -1
					push!(tbd, n)														# add tile to the tbd list
				end
			else																		# if there is an non empty intersection, then there is a solution for the idf tile
				plane[x[1], x[2]] = ni[rand(1:1:length(ni))]							# write solution
				tbd = vcat(nb[2], tbd)													# add all new neighbours that are idf to the tdb list
			end
			tbd = unique(tbd)
		end

		return plane																	# return solution
	end


	##===================================================================================
	##	internal functions
	##===================================================================================
	u{T<:Integer}(vl::Array{Array{Tuple{T, T}, 1}, 1}) = vcat(vl[1], vl[2])

	##-----------------------------------------------------------------------------------
	function nigb{T<:Integer}(p::Tuple{T, T}, s::Tuple{T, T}, plane::Array{T, 2})
		c = [(p[1]+1, p[2]), (p[1], p[2]+1), (p[1]-1, p[2]), (p[1], p[2]-1)]
		r = [Array{Tuple{T, T}, 1}(), Array{Tuple{T, T}, 1}()]
		@inbounds for i=1:4
			if c[i][1]>0 && c[i][2]>0 && c[i][1]<=s[1] && c[i][2]<=s[2]
				push!(r[(plane[c[i][1], c[i][2]] == -1) ? 2 : 1], c[i]) # 1 df | 2 idf
			end
		end
		return r
	end

	##-----------------------------------------------------------------------------------
	function nigbt{T<:Integer}(nb::Array{Tuple{T, T}, 1}, plane::Array{T, 2}, pairs::Dict{T, Array{T, 1}}, border_type::T)
		nt = map((x) -> pairs[plane[x[1], x[2]]], nb)
		if length(nt)<4; push!(nt, pairs[border_type]) end
		return nt
	end
end
