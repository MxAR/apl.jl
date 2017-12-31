@everywhere module rbfn
    ##===================================================================================
    ##  using directives
    ##===================================================================================
    using f

    ##===================================================================================
    ##  types
    ##===================================================================================
    type trbfn{T<:AbstractFloat}                                                        # radial basis function network
        hl_ltwm::Array{T, 2}                                                            # hidden layer-transition-weight-matrix
        hl_th::Array{T, 1}                                                              # hidden layer radiis
        ol_ltwm::Array{T, 2}                                                            # output layer-transition-weight-matrix
        rbf::Function                                                                   # radial basis function
        rbf_derivate_lamda::Function                                                    # radial basis function derivation after lambda
        rbf_derivate_dist::Function                                                     # radial basis function derivation after dist
        df::Function                                                                    # distance function
        df_derivate::Function                                                           # distance function derivate
    end


    ##===================================================================================
    ##  constructors
    ##===================================================================================
    export simple_rbfn, init_rbfn

    ##-----------------------------------------------------------------------------------
    function simple_rbfn(train_examp::Array{Any, 1})
        return init_rbfn(train_examp, length(train_examp))                              # returns an rbfn that inhibits every training example
    end

    ##-----------------------------------------------------------------------------------
    function init_rbfn{T<:AbstractFloat, N<:Integer}(train_examp::Array{Any, 1}, support_points::N, rbf::Function = rbf_gaussian, rbf_derivate_lamda::Function = rbf_gaussian_d_lambda, rbf_derivate_dist::Function = rbf_gaussian_d_delta, df::Function = (x, y) -> norm(x-y), df_derivate = nrm_d, precision::T = 2.)
        sp = APL.samples(train_examp, support_points)
        out = convert(Array{Float64, 2}, train_examp[1][2]')
        hl_ltwm = convert(Array{Float64, 2}, sp[1][1]')
        s = size(train_examp, 1)

        @inbounds for x = 2:support_points
             hl_ltwm = vcat(hl_ltwm, convert(Array{Float64, 2}, sp[x][1]'))
        end

        max_d = 0.0
        @inbounds for x = 1:s
            for y = (x+1):s
                d = BLAS.nrm2(train_examp[x][1] - train_examp[y][1])
                max_d = ifelse(d > max_d, d, max_d)
            end
            if x+1 <= s
                out = vcat(out, convert(Array{Float64, 2}, train_examp[x+1][2]'))
            end
        end

        A = zeros(T, s, support_points)
        hl_th = fill(max_d/sqrt(precision*s), support_points)

        for x = 1:s, y = 1:support_points
            A[x, y] = rbf(df(train_examp[x][1], hl_ltwm[y, :]), hl_th[y])
        end

        #println(hl_ltwm)
        #println(hl_th)
        #println((pinv(hcat(A, ones(s))) * out))
        return trbfn(hl_ltwm, hl_th, (pinv(hcat(A, ones(s))) * out)', rbf, rbf_derivate_lamda, rbf_derivate_dist, df, df_derivate)
    end

    ##===================================================================================
    ##  integrate and fire
    ##===================================================================================
    export iaf

    ##-----------------------------------------------------------------------------------
    function iaf{T<:AbstractFloat}(rbfn::trbfn, v::Array{T, 1})
        s = size(rbfn.hl_ltwm, 1)
        oh = zeros(T, eltype(rbfn.hl_ltwm), s)
        for i = 1:s oh[i] = rbfn.rbf(rbfn.df(v, rbfn.hl_ltwm[i, :]), rbfn.hl_th[i]) end
        return (rbfn.ol_ltwm * append!(oh, ones(T, 1)))
    end


    ##===================================================================================
    ## gradient descent training
    ##===================================================================================
    export gdb!


    # gradient descent batch
    # td = training data; lr = learning-rate
    # max_ep = maximum of training epoches
    # max_err = the error at which the training is aborted
    # alpha = evalation of the threshold derivate
    # beta = weight decay factor
    ##-----------------------------------------------------------------------------------
    function gdb!{T<:AbstractFloat, N<:Integer}(rbfn::trbfn, td::Array{Any, 1}, lr::N = 1, max_ep::N = 1000, max_err::T = .01, alpha::T = .2, beta = 1)
        ntt = (zeros(T, rbfn.hl_ltwm), zeros(T, rbfn.hl_th), zeros(T, rbfn.ol_ltwm))    # network trainings type
        cvs = convert(Int, round(1.5*length(td) / log(length(td))))                     # cross validation sample size
        dlf = (x) -> x + (alpha * sign(x))                                              # derivate lifting function
        hls = size(rbfn.hl_ltwm, 1)                                                     # hidden layer size
        lrs = lr * [0.01, 0.05, 0.01]                                                   # learning-rates
        elt = eltype(rbfn.hl_ltwm)                                                      # element-type of the ltwm

        for E = 1:max_ep
            epsilon = 0.0
            for t in samples(td, cvs)
                # forward propagation
                oh = zeros(elt, hls)
                dist = zeros(elt, hls)
                for i = 1:hls
                    dist[i] = rbfn.df(t[1], rbfn.hl_ltwm[i, :])
                    oh[i] = rbfn.rbf(dist[i], rbfn.hl_th[i])
                end
                oo = rbfn.ol_ltwm * vcat(oh, ones(1))

                # vertex
                d = (t[2] - oo)
                epsilon += sumabs2(d)

                # backpropagation
                ntt[3] .+= (d * vcat(oh, ones(1))') .* lrs[3]                                                                               # output layer weight changes
                ntt[2] .+= ((rbfn.ol_ltwm[:, 1:end-1]' * d) .* dlf(rbfn.rbf_derivate_lamda(oh))) * lrs[2]                                   # radii changes
                ntt[1] .+= ((rbfn.ol_ltwm[:, 1:end-1]' * d) .* (dlf(rbfn.rbf_derivate_dist(oh)) .* dlf(rbfn.df_derivate(dist)))) * lrs[1]   # hidden layer weight changes
            end

            #println("ntt: ")
            #for x in ntt println("  ", x) end

            #rbfn.hl_ltwm .*= beta
            rbfn.hl_ltwm += ntt[1]

            #rbfn.hl_th .*= beta
            rbfn.hl_th += ntt[2]

            #rbfn.ol_ltwm .*= beta
            rbfn.ol_ltwm += ntt[3]

            ntt = map(x -> x * 0, ntt)

            #println(epsilon, " <> ", E)
            #println("-------------------")

            if epsilon <= max_err
                epsilon = 0
                for t in td epsilon += sumabs2(t[2] - iaf(rbfn, t[1])) end
                if epsilon <= max_err break; end
            end
        end
        #println(rbfn)
        return rbfn
    end
end
