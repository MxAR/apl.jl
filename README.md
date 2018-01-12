# apl.jl
apl is an advanced processing library, that adds many (in my opinion) missing functions to julia

# function list
t = type
f = function
* = needs to be done

### graph
	t tgraph
	* dijkstra
	f bellman ford algorithm (shortest path)

### heap
	f parent 		(gets parent node)
	f children 		(gets children)
	f is_heap
	f heapify
	f heap_build	(builds a heap from a source array)
	f heap_sort

### dfa 			(deterministic finite state automaton)
	t tdfa
	f ais! 			(input a symbol into the automaton)
	f aiw! 			(input a word into the automaton)
	f csif 			(check if the automaton is in a final state)

### pda				(push down automaton)
	t tpda
	f ais! 			(input a symbol into the automaton)
	f aiw! 			(input a word into the automaton)
	f csif 			(check if the automaton is in a final state)

### tuma 			(turing machine)
	t ttuma
	f ais! 			(input a symbol into the automaton)
	f aiw! 			(input a word into the automaton)
	f csif 			(check if the automaton is in a final state)

### cnv				(convertion module)
	f hex_to_vec	(hex string to Float64 array)
	f chex_to_vec	(color hex string to Float64 array)
	f vec_to_hex	(Float64 vector to hexadecimal string)
	f coord_to_nvec	(converts latitude and longitude into 3d vectors (normalized))
	f nvec_to_coord	(convert normalized 3rd vector into latitude and longitude)
	f ssmm			(generates a skew symmetric matrix from a vector)
	f pdrom_to_nvec	(calculates the normal vector for a point on a sphere)
	f join			(joins a vector list)
	f mat_to_vl		(matrix to vector list)
	f vl_to_mat		(vector list to matrix)

### eva 			(evolutionary algorithm framework)
	t topt_prb		(optimization problem)
	f eve 			(evolutionary solver for the optimization problem)
	f mut_default	(a default mutator for the eve function)

### temp_op			(operations on the temp dir)
	f mv_to_tmp		(moves dataframe to temp dir)
	f mv_to_tmp! 	(moves dataframe to temp dir and returns path)
	f clr_tmp		(deletes contents of the temp dir)

### bb				(bollinger bands)
	f sbb 			(simple bollinger bands)
	f lbb 			(linear bollinger bands)
	f ebb 			(exponential bollinger bands)
	* bb % b

### macd			(moving average convergence divergence)
	f smacd 		(simple macd)
	f lmacd 		(linear macd)
	f emacd 		(exponential macd)

### rsi				(relative strength index)
	f srsi 			(simple rsi)
	f lrsi 			(linear rsi)
	f ersi 			(exponential rsi)

### stosc			(stochastic oscillator)
	f sstosc 		(simple stosc)
	f lstosc 		(linear stosc)
	f estosc 		(exponential stosc)

### f				(math function collection)
	t tncbd
	f collatz		(number of iterations till 4)
	f dcd			(dirac delta/impulse)
	f ked			(kronecker delta)
	f king 			(kingsman equation)
	f sigmoid
	f sigmoidd		(derivate)
	f normd 		(p norm derivate)
	f rbf_gauss		(gaussian radial basis function)
	f rbf_gaussdd	(gaussian radial basis function derived after delta)
	f rbf_gaussdl	(gaussian radial basis function derived after lambda)
	f rbf_triang	(triangular radial basis function)
	f rbf_triang	(triangular radial basis function (on vectors))
	f rbf_cos_decay	(cosine decay radial basis function)
	f rbf_cos_decay	(cosine decay radial basis function (on arrays))
	f rbf_psq		(product squared radial basis function)
	f rbf_inv_psq	(inverse product squared radial basis function)
	f rbf_inv_sq	(inverse squared radial basis function)
	f rbf_exp		(exponential radial basis function)
	f rbf_tps		(thin plate spline radial basis function)
	f ramp			(ramp activation function)
	f rampd 		(ramp activation function derivate)
	f rampd 		(ramp activation function derivate for multiple x and one eta)
	f rampd			(ramp activation function derivate for multiple x and multiple eta)
	f semilin		(semi linear function activation function)
	f semilind 		(semi linear function activation function derivate)
	f sinesat		(sine saturation activation function)
	f sinesatd 		(sine saturation activation function derivate)
	f softplus		(softplus activation function)
	f softplusd		(softplus activation function derivate)
	f step			(step function)
	f stepd			(step function derivate)
	f supp 			(support of a given function (vector))
	f supp 			(support of a given function (vector list))
	f saww			(sawtooth wave (period & amplitude & phase))
	f sqw			(square wave (period & amplitude & phase))
	f triw 			(triangle wave (period & amplitude & phase))
	f rop			(roots of a polynomial)
	f rou			(roots of unity)
	f lecit			(levi civita tensor)
	f ipc			(index permutations count (on an array))


### bin				(binary)
	f nbit_on		(check if the nth bit is on)
	f nbit_off		(check if the nth bit is off)
	f bprint		(print the binary representation of the number)
	f sizeofb		(returns the bit size)
	f ibit_range	(inverts a binary sequence in a given Integer)
	f ebit_range	(exchanges two binary sequences of two Integers)
	f fbit			(flip a bit in an Integer)
	f sbit			(set a bit in an Integer)
	f cb_merge		(column wise binary merge of Integers)
	f cb_split		(splits binary string into columns of Integers)
	f garyf			(gray encode)
	f grayb			(gray decode)
	f hilbert_cf	(point to length)
	f nilbert_cb	(length to point)

### sta				(statistics)
	f std 			(standard deviation)
	f std			(standard deviation with given mean)
	f chi			(distribution)
	f w6			(dice sampling (distribution))
	f nw6 			(cumulative dice sampling)
	f var 			(variance overload with given mean)
	f var 			(variance overload)
	f var 			(variance overload with given mean and length of the sample)
	f var 			(variance overload with given sample length)
	f ar			(autoregressive model)
	f difut 		(dickey fuller test for stationarity)
	f difut 		(dickey fuller test for stationarity (with drift))
	f difut 		(dickey fuller test for stationarity (with drift & trend))
	f adifut		(augmented dickey fuller test for stationarity)
	f adifut		(augmented dickey fuller test for stationarity (with drift))
	f adifut 		(augmented dickey fuller test for stationarity (with drift & trend))
	f angrat		(angle granger test for co-integration)
	f angrat		(angle granger test for co-integration (with drift))
	f angrat		(angle granger test for co-integration (with drift & trend))
	f angrat		(augmented angle granger test for co-integration)
	f angrat		(augmented angle granger test for co-integration (with drift))
	f angrat		(augmented angle granger test for co-integration (with drift & trend))
	f mut_inch		(mutual incoherence for a matrix)
	f mut_inch		(mutual incoherence for a vector list)
	f normalize_s	(normalizes matrix rows so that mean = 0 and variance = 1)
	f normalize_sp	(normalizes matrix rows so that mean = 0 and variance = 1 (parallel))
	f normalize_sps	(normalizes matrix rows so that mean = 0 and variance = 1 (parallel)(shared))
	f cov			(covariance of two vectors)
	f cov			(covariance of a vector with it past self)
	f covp			(population covariance of column/row vectors)
	f covs 			(sample covariance of column/row vectors)
	f covc 			(cross covariance of two vectors)
	f covcs 		(cross covariance of two vectors summed (with delay))
	f ccor 			(cross correlation with delay)
	f shai 			(shannon index)
	f gishi 		(gini-shannon index)
	f reepy 		(renyi entropy)
	f sample		(returns a sample set from an array)

### gen 			(generator module)
	f vandermonde	(vandermonde matrix)
	f zeros_sq		(square matrix filled with zeros)
	f ones_sq		(square matrix filled with ones)
	f fill_sq		(fill square matrix with a number)
	f rand_sq 		(random square matrix)
	f randn_sq		(normal random square matrix)
	f ones_dia		(fill a diagonal matrix with ones)
	f fill_dia		(fill a diagonal matrix with a number)
	f rand_dia		(fill a diagonal matrix with random numbers)
	f randn_dia 	(fill a diagonal matrix with (normal) random numbers)
	f fill_tri		(fill upper/lower triangular matrix with a number)
	f ones_tri		(fill upper/lower triangular matrix with ones)
	f rand_tri 		(fill upper/lower triangular matrix with random numbers)
	f randn_tri		(fill upper/lower triangular matrix with normal random numbers)
	f zeros_vl		(fill vector list with zeros)
	f ones_vl		(fill vector list with ones)
	f rand_vl		(fill vector list with random numbers)
	f randn_vl		(fill vector list with normal random numbers)
	f zeros_hs		(fill a ndim hypercube with zeros)
	f rand_rgb_vec	(random rgb or rgba color vector)
	f rand_sto_vec	(random stochastic vector)
	f rand_orth_vec	(random orthonormal vector)
	f rand_circ_mat	(random circulant matrix)
	f rand_sto_mat	(random stochastic matrix)
	f rand_sto_mat	(random stochastic square matrix)
	f vl_rand		(random vector list)
	f vl_rand		(random vector list constraint by a ndim hypercube)
	f pasc			(pascal matrix)
	f exm			(exchange matrix)
	f rotmat_2d		(real rotation matrix for 2d space)
	f rotmat_3d		(real rotation matrix for 3d space)
	f gevmat		(general evaluation matrix)

### trig			(trigonometric functions)
	f tanh			(...)
	f tanhd			(...)
	f sin2			(...)
	f cos2			(...)
	f versin		(...)
	f aversin		(...)
	f vercos		(...)
	f avercos		(...)
	f coversin		(...)
	f acoversin		(...)
	f covercos		(...)
	f acovercos		(...)
	f havsin		(...)
	f ahavsin		(...)
	f havcos		(...)
	f ahavcos		(...)
	f hacoversin	(...)
	f hacovercos	(...)
	f angle			(angle between some vectors plus bias)
	f ccntrl_angle	(central angle calculation from the position vectors through acos)
	f scntrl_angle	(central angle calculation from the position vectors through asin)
	f tcntrl_angle	(central angle calculation from the position vectors through atan)
	f cntrl_angle 	(central angle calculation from the latitude and longitude of two points)
	f hcntrl_angle	(central angle calculation from the latitude and longitude through haversine)
	f vcntrl_angle	(vincenty central angle calculation from the latitude and longitude)

### bla				(basic linear algebra)
	f bdot			(blas dot wrapper for float vectors with given length)
	f bdot			(blas dot wrapper for float vectors)
	f bdotu			(blas dot wrapper for complex vectors with given length)
	f bdotu 		(blas dot wrapper for complex vectors)
	f bodtc			(blas dot wrapper for complex vectors with given length (hermitian transpose))
	f bdotc			(blas dot wrapper for complex vector (hermitian transpose))
	f bnrm 			(blas nrm wrapper for float vector with given length)
	f bnrm 			(blas nrm wrapper for float vector)
	f soq 			(sum of float squares (given length))
	f soq			(sum of float squares)
	f soqu			(sum of complex squares (given length))
	f soqu 			(sum of complex squares)
	f soqc			(sum of complex squares (given length)(hermitian transpose))
	f soqc 			(sum of complex squares (hermitian transpose))
	f cof			(cofactor matrix of a matrix)
	f adj 			(adjugate of a matrix)
	f nul 			(null-space of a matrix)
	f qrd_sq		(qr decomposition of a square matrix (linAlg))
	f qrd			(qr decomposition of a matrix (linAlg))
	f ul_x_expand	(diagonal matrix expansion (linAlg))
	f minor			(minor of a matrix (linAlg))
	f otr			(outer product)
	f grsc 			(gram schmidt process)
	f grscn			(gram schmidt process for normal vectors)
	f proj 			(project vector onto a subspace spanned by a orthogonal matrix (not normal))
	f projn			(poject vector onto a subspace spanned by a orthogonal matrix)
	f msplit		(split matrix at row/column index i)
	f msplit_half	(split matrix at row/column middle index)
	f ols			(ordinary least squares (optimization))
	f hh_rfl		(householder reflection (linAlg))
	f hh_mat		(householder matrix (linAlg))
	f kep			(kronecker product)

### mean
	f gamean		(generalized arithmetic mean)
	f ghmean		(generalized harmonic mean)
	f ggmean		(generalized geometric mean)
	f gpmean		(generalized power mean)
	f grmean		(generalized root squared mean)
	f gfmean		(generalized f mean)
	f mamean		(column/row mean of matrix (arithmetic))
	f mamean		(column/row mean of matrix (weighted))
	f sma			(simple moving average)
	f lma 			(linear moving average)
	f ema			(exponential moving average)

### dist 			(distance function module)
	f gdist			(distance between each element in a set m)
	f gdist_p		(distance between each element in a set m (parallel))
	f hdist			(hamming distance of two Integers)
	f hdist 		(hamming distance of two strings)
	f odist 		(orthodromic distance of two points given by latitude and longitude)
	f odist 		(orthodromic distance of two points given by their position vectors)
	f mdist 		(mahalanobis distance)
	f mdist_dvec	(mahalanobis distance derivate after the position)
	f mdist_dcov	(mahalanobis distance derivate after the covariance)
	f mdist_sq		(mahalanobis distance squared)
	f mdist_sq_dvec	(mahalanobis distance squared derivate after the position)
	f mdist_sq_dcov	(mahalanobis distance squared derivate after the covariance)

### mpa				(matching pursuit module)
	f omp			(orthogonal matching pursuit algorithm)
	f mp			(matching pursuit algorithm)

### op				(operation module)
	f PI			(big capital PI)
	f imp_add		(adds two arrays of distinct length)
	f imp_add 		(adds two matrices of distinct size)
	f imp_sub 		(subtracts two arrays of distinct length)
	f imp_sub 		(subtracts two matrices of distinct size)
	f prison 		(min max constraint for a number)
	f prison 		(min max constraint for a number where a lambda is applied when the number is in between the borders)
	f rm			(removes column/row from matrix)
	f rms 			(removes rows/columns given by a sorted set of indexers)
	f rm			(removes rows/column given by a set of indexers)
	f rm			(removes a range of columns/rows)
	f rm			(removes elements given by a indexer array from an array)
	f rms			(removes elements given by a sorted indexer array from an array)
	f union 		(union on a vector list)
	f intersect		(intersect for vector lists)
	f prepend		(prepend value)
	f prepend		(prepend array)
	f prepend!		(prepend value (override))
	f prepend!		(prepend array (override))
	f map			(map overload for vector lists)
	f min 			(min overload for tuples)
	f max 			(max overload for tuples)
	f apply 		(apply lambda on all elements)
	f apply_p		(apply lambda in parallel on all elements)
	f apply_ps 		(apply lambda in parallel on all elements of an shared data-structure)
	f apply_triu	(apply lambda on all elements of an upper triangular matrix)
	f apply_tril	(apply lambda on all elements of an lower triangular matrix)
	f apply_dia		(apply lambda on all elements of diagonal matrix)
	f AND			(checks if all elements are true)
	f OR 			(checks if one element is true)
	f / 			(overload for vectors)
	f iszero		(checks if value is zero)

### rg 				(regression module)
	f rg_log 		(logistic regression)
	f rg_sta		(statistic regression)
	f rg_qr			(qr regression)

### vq 				(vector quantization)
	f lvq			(learning vector quantization)
	f kmeans		(k means clustering)

### yamartino		(yamartino algorithm)
	t IYamartResult	(...)
	t TYamartState	(...)
	f YamartResult	(calculates the result from the state)
	f CalcYamart	(calculates the best direction approximation with a single pass through)
	f PCalcYamart	(parallel calculation of the best direction approximation with a single pass through)

### pkg				(package list operations)
	f spkg_list		(save a list of all packages)
	f ipkg_list		(installs packages that are in a list generated by the above method)

### mlp				(Multilayer Percepton module)
	t tlayer		(a layer of a mlp)
	t tmlp 			(a mlp)
	f tmlpv0		(the default constructor for a mlp instance)
	f disj_tmlp		(creates a pretrained mlp which represents a a disjunctive boolean function)
	f iaf 			(integrate and fire function (propagates input through the network))
	f gdb! 			(gradient descent learning of a mlp)

### pct 			(single layer percepton module)
	t tpct			(stores a single layer percepton)
	f tpctv0		(tpct constructor which sets the given weights)
	f tpctv1		(tpct constructor which generates zero/random weights)
	f tpctv2		(tpct constructor which will approximate the best weights for a tpct)
	f iaf			(integrate and fire function (propagates input through the network))
	f gdb! 			(gradient descent learning batch)
	f gdo! 			(gradient descent learning online)

### wfc				(wave function collapse)
	f wfc			(wave function collapse (procedural generation))
	f nigb			(get all neighbors)
	f nigbt			(get the types of all neighbors)

### rbfn 			(radial basis networks)
	t trbfn 		(stores rbfn)
	f simple_rbfn	(returns an rbfn that inhibits every given training example)
	f init_rbfn		(initializes a rbfn)
	f iaf			(integrate and fire (propagates a given input through the network))
	f gdb!			(gradient descent learning for a rbfn)
