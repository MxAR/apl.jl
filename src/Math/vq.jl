@everywhere module vq 																	# vector quantization
	##===================================================================================
	##	using directives
	##===================================================================================
	using mat_op
	using f

	##===================================================================================
	##	learning vector quantization
	##===================================================================================
	export lvq

	##-----------------------------------------------------------------------------------
	function lvq{T<:Number}(vl::Array{Array{T, 1}, 1}, nn::Array{Array{T, 1}, 1}, eta::Function = (x) -> 0.2, max_epoch = 100)
		for i in 1:max_epoch															# vl = data as vector list
			bn = [0, inf]																# nn = probing points as vectorlist
			for v = 1:length(vl)														# eta = learn value function
				for n = 1:length(nn)
					s = norm(v-nn[n])
					if bn[2] > s
						bn[1] = n
						bn[2] = s
					end
				end
				nn[bn[1]] += eta(i)*(v-nn[bn[1]])
			end
		end
		return nn
	end


	##===================================================================================
	##	k-means
	##===================================================================================
	export kmeans

	##-----------------------------------------------------------------------------------
	function kmeans{T<:Real, N<:Int}(vl::Array{Array{T, 1}, 1}, k::N, max_iter::N = 100, dist::Function = (l, v) -> soq(l, v))
		j = length(vl); l = length(vl[1])
		mn = vl[1:k]; cl = fill(0, j)

		for ep = 1:max_iter
			for v = 1:j																	# assignment step
				i = [Inf, 0]
				for m = 1:k																# find nearest cluster for vec v
					d = dist(l, mn[m]-vl[v])
					if i[1] > d
						i[1] = d
						i[2] = m
					end
				end
				cl[v] = i[2] 															# assign vector to nearest cluster
			end

			hmn = mn
			mn = vl_zeros(k, l)
			c = zeros(k)
			for v = 1:j																	# update step
				mn[cl[v]] += vl[v]
				c[cl[v]] += 1
			end
			mn ./= c
			println(mn)
			hmn == mn && break
		end

		return mn
	end
end
