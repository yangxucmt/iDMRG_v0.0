module iDMRG

using BenchmarkTools
using ITensors, ITensorMPS
using LinearAlgebra
using KrylovKit
using Base.Threads
using Dates
using JLD2

export idmrg_general, iDMRG_init, lambdamodule


function lambdamodule(l1, l2)
    mat1 = array(l1)
    mat2 = array(l2)
    size1 = size(mat1)[1]
    size2 = size(mat2)[1]
    largel1 = zeros(Float64, max(size1, size2))
    largel2 = zeros(Float64, max(size1, size2))
    for i = 1:size1
        largel1[i] = mat1[i, i]
    end
    for i = 1:size2
        largel2[i] = mat2[i, i]
    end
    return dot(largel1, largel2)
end

mutable struct Heff
    LE::ITensor
    RE::ITensor
    MPO::ITensor  # Common MPO for both Heff1 and Heff2
    second_MPO::Union{ITensor,Nothing}  # Optional second MPO for Heff2
end

function product(p::Heff, v::ITensor)
    hv = v
    hv *= p.LE
    hv *= p.MPO
    if !isnothing(p.second_MPO)
        hv *= p.second_MPO  # Only apply if it exists (i.e., for Heff2)
    end
    hv *= p.RE
    return noprime(hv)
end

(p::Heff)(v::ITensor) = product(p::Heff, v::ITensor)

function update_iLPs!(iH::MPO, psi::ITensor, LPs::MPO, siteind)
    uclength = length(iH)
    newsiteind = mod1(siteind + 1, uclength)
    LPs[newsiteind] = copy(LPs[siteind])
    LPs[newsiteind] *= psi
    LPs[newsiteind] *= iH[siteind]
    LPs[newsiteind] *= prime(dag(psi))
end


function update_iRPs!(iH::MPO, psi::ITensor, RPs::MPO, siteind)
    uclength = length(iH)
    newsiteind = mod1(siteind - 1, uclength)
    RPs[newsiteind] = copy(RPs[siteind])
    RPs[newsiteind] *= psi
    RPs[newsiteind] *= iH[siteind]
    RPs[newsiteind] *= prime(dag(psi))
end

function update_iLP_disk!(iH::MPO, psi::ITensor, siteind)
    uclength = length(iH)
    newsiteind = mod1(siteind + 1, uclength)
    LP_temp = jldopen("LP_$siteind.jld2", "r") do file
        copy(file["LP_$siteind"])
    end
    LP_temp *= psi
    LP_temp *= iH[siteind]
    LP_temp *= prime(dag(psi))
    jldopen("LP_$newsiteind.jld2", "w") do file  # "r+" = read+write, don't destroy existing data
        file["LP_$newsiteind"] = LP_temp       # this will add or overwrite "LP_1", "LP_2", etc.
    end
end

function update_iRP_disk!(iH::MPO, psi::ITensor, siteind)
    uclength = length(iH)
    newsiteind = mod1(siteind - 1, uclength)
    RP_temp = jldopen("RP_$siteind.jld2", "r") do file
        copy(file["RP_$siteind"])
    end
    RP_temp *= psi
    RP_temp *= iH[siteind]
    RP_temp *= prime(dag(psi))
    jldopen("RP_$newsiteind.jld2", "w") do file  # "r+" = read+write, don't destroy existing data
        file["RP_$newsiteind"] = RP_temp       # this will add or overwrite "RP_1", "RP_2", etc.
    end
end

function noise_expand_left(u, s, v, lrp, mpo, noise; maxdim)
    M = u * s
    pi = noise * lrp
    pi *= M
    pi *= mpo
    pi = noprime(pi)
    lind = commonind(lrp, u)
    siteind = commonind(mpo, u)
    rind = commonind(s, v)
    prightinds = uniqueinds(pi, lind, siteind)
    prightcombiner = combiner(prightinds)
    prightleg = combinedind(prightcombiner)
    newpi = pi * prightcombiner
    newsize = dim(rind) + dim(prightleg)
    injectionmat1 = Matrix{Float64}(I, dim(rind), newsize)
    finalrightleg = Index(newsize)
    injection1 = ITensor(injectionmat1, rind, finalrightleg) #This can also be used for B
    injectionmat2 = zeros(Float64, dim(prightleg), newsize)
    for i = 1:dim(prightleg)
        injectionmat2[i, newsize-dim(prightleg)+i] = 1
    end
    injection2 = ITensor(injectionmat2, prightleg, finalrightleg)
    newAL = newpi * injection2 + M * injection1
    newB = injection1 * v
    uenlarged, senlarged, venlarged = svd(newAL, lind, siteind; maxdim)
    returnedM = newB
    returnedM *= venlarged
    returnedM *= senlarged
    return [uenlarged, venlarged * senlarged, newB]
end

function noise_expand_right(u, s, v, lrp, mpo, noise; maxdim)
    M = s * v
    pi = noise * lrp
    pi *= M
    pi *= mpo
    pi = noprime(pi)

    rind = commonind(lrp, v)
    siteind = commonind(mpo, v)
    lind = commonind(u, s)
    pleftinds = uniqueinds(pi, rind, siteind)
    pleftcombiner = combiner(pleftinds)
    pleftleg = combinedind(pleftcombiner)
    newpi = pi * pleftcombiner
    newsize = dim(lind) + dim(pleftleg)
    injectionmat1 = Matrix{Float64}(I, dim(lind), newsize)
    finalleftleg = Index(newsize)
    injection1 = ITensor(injectionmat1, lind, finalleftleg) #This can also be used for A
    injectionmat2 = zeros(Float64, dim(pleftleg), newsize)
    for i = 1:dim(pleftleg)
        injectionmat2[i, newsize-dim(pleftleg)+i] = 1
    end
    injection2 = ITensor(injectionmat2, pleftleg, finalleftleg)
    newLB = newpi * injection2 + M * injection1
    newA = injection1 * u
    uenlarged, senlarged, venlarged = svd(newLB, finalleftleg; maxdim)
    returnedM = newA
    returnedM *= uenlarged
    returnedM *= senlarged
    return [newA, uenlarged * senlarged, venlarged]
end


function update_bond!(iham, psi, LRP, sites; sweep, update_lrp, kryloverr, krylovdimmax, niter, maxdim, cutoff, verbose=0, noise="off", noisedir="L") #Let's not change the value during the process
    # Here, LPs, RPs, sitei, and MPOs, should be stored in the same record
    # We always update the wave-function in place, because this updates the Gamma in Vidal notation. We can return the new bond lambda value.
    LPs, RPs = LRP
    sitei1, sitei2 = sites
    initialguess0 = psi[sitei1] * psi[sitei2]
    ph = Heff(LPs[sitei1], RPs[sitei2], iham[sitei1], iham[sitei2])
    if sweep == true #Only when sweep==true, consider add noise
        if noise == "off"
            valslist, vecslist = eigsolve(ph, initialguess0, 1, :SR, krylovdim=krylovdimmax, tol=kryloverr, maxiter=niter, verbosity=0, ishermitian=true)
            u, s, v = svd(vecslist[1], uniqueinds(psi[sitei1], psi[sitei2]); maxdim, cutoff)
        elseif noise isa Number
            valslist, vecslist = eigsolve(ph, initialguess0, 1, :SR, krylovdim=krylovdimmax, tol=kryloverr, maxiter=niter, verbosity=0, ishermitian=true)
            u, s, v = svd(vecslist[1], uniqueinds(psi[sitei1], psi[sitei2]); maxdim, cutoff)
            if noisedir == "L"
                u, s, v = noise_expand_left(u, s, v, LPs[sitei1], iham[sitei1], noise; maxdim)
            elseif noisedir == "R"
                u, s, v = noise_expand_right(u, s, v, RPs[sitei2], iham[sitei2], noise; maxdim)
            end
        end
    else
        u, s, v = svd(initialguess0, uniqueinds(psi[sitei1], psi[sitei2]); maxdim, cutoff)
    end

    # Also modify the wave-function, always do it.
    if update_lrp[1] == true
        update_iLPs!(iham, u, LPs::MPO, sitei1)
    end

    if update_lrp[2] == true
        update_iRPs!(iham, v, RPs::MPO, sitei2)
    end
    psi[sitei1] = u
    psi[sitei2] = v
    if verbose == 0
        return s #Let's return lambda here.
    elseif verbose == 1
        return [s, valslist[1]]
    end
end

function update_bond_disk(iham, psi, LRP, sites; sweep, update_lrp, kryloverr, krylovdimmax, niter, maxdim, cutoff, verbose=0, noise="off", noisedir="L") #Let's not change the value during the process
    # Here, LPs, RPs, sitei, and MPOs, should be stored in the same record
    # We always update the wave-function in place, because this updates the Gamma in Vidal notation. We can return the new bond lambda value.
    LPs, RPs = LRP
    #Here LPs, RPs are just one single MPO
    sitei1, sitei2 = sites
    LPs = jldopen("LP_$sitei1.jld2", "r") do file
        file["LP_$sitei1"]
    end
    RPs = jldopen("RP_$sitei2.jld2", "r") do file
        file["RP_$sitei2"]
    end
    initialguess0 = psi[sitei1] * psi[sitei2]
    ph = Heff(LPs, RPs, iham[sitei1], iham[sitei2])
    if sweep == true #Only when sweep==true, consider add noise
        if noise == "off"
            valslist, vecslist = eigsolve(ph, initialguess0, 1, :SR, krylovdim=krylovdimmax, tol=kryloverr, maxiter=niter, verbosity=0, ishermitian=true)
            u, s, v = svd(vecslist[1], uniqueinds(psi[sitei1], psi[sitei2]); maxdim, cutoff)
        elseif noise isa Number
            valslist, vecslist = eigsolve(ph, initialguess0, 1, :SR, krylovdim=krylovdimmax, tol=kryloverr, maxiter=niter, verbosity=0, ishermitian=true)
            u, s, v = svd(vecslist[1], uniqueinds(psi[sitei1], psi[sitei2]); maxdim, cutoff)
            if noisedir == "L"
                u, s, v = noise_expand_left(u, s, v, LPs, iham[sitei1], noise; maxdim)
            elseif noisedir == "R"
                u, s, v = noise_expand_right(u, s, v, RPs, iham[sitei2], noise; maxdim)
            end
        end
    else
        u, s, v = svd(initialguess0, uniqueinds(psi[sitei1], psi[sitei2]); maxdim, cutoff)
    end

    # Also modify the wave-function, always do it.
    if update_lrp[1] == true
        update_iLP_disk!(iham, u, sitei1)
    end

    if update_lrp[2] == true
        update_iRP_disk!(iham, v, sitei2)
    end
    psi[sitei1] = u
    psi[sitei2] = v
    if verbose == 0
        return s #Let's return lambda here.
    elseif verbose == 1
        return [s, valslist[1]]
    end
end

function iDMRG_init(isites, impolinks)
    uc = size(isites)[1]
    l = 1

    sitetensor0 = zeros(ComplexF64, 1, 1, 2)
    sitetensor0[1, 1, 1] = 1 / sqrt(2)
    sitetensor0[1, 1, 2] = 1 / sqrt(2)
    bondtensor0 = zeros(ComplexF64, 1, 1)
    bondtensor0[1, 1] = 1
    # We want to creat psis0,LP1,RPl,L_1l/2,L_0l

    llinks = Array{Index}(undef, uc + 2)
    for i = 1:uc+2
        llinks[i] = Index(1)
    end

    psis0 = MPS(uc)

    for i = 1:l
        psis0[i] = ITensor(sitetensor0, llinks[i], llinks[i+1], isites[i])  #A1...Al
    end

    L_2 = delta(llinks[l+1], llinks[l+2])       #L2

    for i = l+1:uc
        psis0[i] = ITensor(sitetensor0, llinks[i+1], llinks[i+2], isites[i]) #Bl+1,...,B2l
    end

    L_4 = delta(llinks[uc+2], llinks[1])

    #We initialize LP1.

    initialmat = zeros(ComplexF64, dim(impolinks[1]), 1, 1)
    initialmat[dim(impolinks[1]), :, :] = Matrix{Float64}(I, 1, 1)
    LP10 = ITensor(initialmat, impolinks[1], prime(llinks[1]), llinks[1])
    initialmat = zeros(ComplexF64, dim(impolinks[1]), 1, 1)#Make sure this is correct
    initialmat[1, :, :] = Matrix{Float64}(I, 1, 1)
    RPuc = ITensor(initialmat, impolinks[1], prime(llinks[uc+2]), llinks[uc+2])
    return [psis0, LP10, RPuc, L_2, L_4]
end

function idmrg_general(iH::MPO, psi0::MPS, LPinput, RPinput, L_n2, L_np4; nsweeps, maxdim=400, cutoff=1e-8, krylovdimmax=3, niter=2, verbosity=1, kryloverr=1e-14, noise="off")
    #We define two temporary variables LP_temp, RP_temp to store the temps used in calculation
    L = size(iH)[1]
    gsenergy_ini = 0
    gsenergy_fin = 0
    save_to_disk = false
    if maxdim * maxdim * dim(commonind(iH[1], iH[2])) * L * 16 * 2 > 5e9
        save_to_disk = true
    end

    if save_to_disk
        LP_temp = LPinput
        RP_temp = RPinput

        filename = "LP_1.jld2"
        jldopen(filename, "w") do file
            file["LP_1"] = LP_temp
        end

        filename = "RP_$L.jld2"
        jldopen(filename, "w") do file
            file["RP_$L"] = RP_temp
        end

        lambda_2 = copy(L_n2)
        lambda_L = copy(L_np4)
        psis = copy(psi0)
        psis[2] *= lambda_2

        maxbonddim = 0

        # Delete the file

        for i = L:-1:3
            update_iRP_disk!(iH, psis[i], i)
        end

        #The first part: sweep from left to right
        for i = 1:L-2
            if i != 1
                lambdai = update_bond_disk(iH, psis, [LP_temp, RP_temp], [i, i + 1]; sweep=true, update_lrp=[true, false], cutoff, maxdim, krylovdimmax, niter, kryloverr, noise, noisedir="L") #This is updating the wave-function and the environment
            else
                lambdai, gsenergy_ini = update_bond_disk(iH, psis, [LP_temp, RP_temp], [i, i + 1]; sweep=true, update_lrp=[true, false], cutoff, maxdim, krylovdimmax, niter, kryloverr, verbose=1, noise, noisedir="L")
            end
            #Get the lambda_(L-1) from the last step
            psis[i+1] *= lambdai
        end

        lambda_lm1 = update_bond_disk(iH, psis, [LP_temp, RP_temp], [L - 1, L]; sweep=true, update_lrp=[true, true], cutoff, maxdim, krylovdimmax, niter, kryloverr) #This is updating the wave-function and the environment
        psis[L-1] *= lambda_lm1

        for i = L-2:-1:1
            lambdai = update_bond_disk(iH, psis, [LP_temp, RP_temp], [i, i + 1]; sweep=false, update_lrp=[false, true], cutoff, maxdim, krylovdimmax, niter, kryloverr)
            psis[i] *= lambdai
        end

        #At this point, the environment RP[1] is ready.
        #Insert one unit-cell
        psis[L] *= lambda_lm1 #This is for dmrg
        psis[L] *= ITensor(pinv(array(lambda_L), 1e-8), inds(lambda_L))

        lambda_temp = update_bond_disk(iH, psis, [LP_temp, RP_temp], [L, 1]; sweep=true, update_lrp=[true, false], cutoff, maxdim, krylovdimmax, niter, kryloverr, noise, noisedir="L")
        psis[1] *= lambda_temp
        lambda_temp, gsenergy_fin = update_bond_disk(iH, psis, [LP_temp, RP_temp], [1, 2]; sweep=true, update_lrp=[false, true], cutoff, maxdim, krylovdimmax, niter, kryloverr, verbose=1, noise, noisedir="R")
        psis[1] *= lambda_temp
        new_lambda_L = update_bond_disk(iH, psis, [LP_temp, RP_temp], [L, 1]; sweep=true, update_lrp=[true, true], cutoff, maxdim, krylovdimmax, niter, kryloverr)

        psis[L] *= new_lambda_L #This will be used as the right tensor in line 221, i.e., i=L-1:-1:2 first step
        psis[1] *= new_lambda_L

        #At this point, we have AL, lambda_L*B1, B2,..., BL-1
        #Now add one more unit-cell.

        for i = 1:L-2
            #We will use lambda_L later, probably don't need other stuff
            lambdai = update_bond_disk(iH, psis, [LP_temp, RP_temp], [i, i + 1]; sweep=false, update_lrp=[true, false], cutoff, maxdim, krylovdimmax, niter, kryloverr)
            psis[i+1] *= lambdai
        end

        psis[L-1] *= ITensor(pinv(array(lambda_lm1), 1e-8), inds(lambda_lm1)) #Use the lm1 obtained before

        for i = L-1:-1:2
            lambdai = update_bond_disk(iH, psis, [LP_temp, RP_temp], [i, i + 1]; sweep=true, update_lrp=[false, true], cutoff, maxdim, krylovdimmax, niter, kryloverr, noise, noisedir="R")
            psis[i] *= lambdai
        end

        new_lambda_1 = update_bond_disk(iH, psis, [LP_temp, RP_temp], [1, 2]; sweep=false, update_lrp=[false, false], cutoff, maxdim, krylovdimmax, niter, kryloverr)
        maxbonddim = size(new_lambda_1)[1]

        if verbosity == 1
            println("Maximum bond dimension is: ", maxbonddim)
            println("Energy is: ", (gsenergy_fin - gsenergy_ini) / (L))
            println("Lambda overlap is: ", lambdamodule(new_lambda_1, L_n2))
        end

        LP_temp = jldopen("LP_1.jld2", "r") do file
            file["LP_1"]
        end
        RP_temp = jldopen("RP_$L.jld2", "r") do file
            file["RP_$L"]
        end

        for i = 1:L
            rm("LP_$i.jld2")
            rm("RP_$i.jld2")
        end

        return [psis, LP_temp, RP_temp, new_lambda_1, new_lambda_L]

    else
        LPs = MPO(L)
        RPs = MPO(L)
        LPs[1] = copy(LPinput)
        RPs[L] = copy(RPinput)
        lambda_2 = copy(L_n2)
        lambda_L = copy(L_np4)
        psis = copy(psi0)
        psis[2] *= lambda_2
        #psis, lambda_L=idmrg_gauge(psi0)

        #After gauging, we have A1, (Lambda B2), B3, B4, ..., BL
        maxbonddim = 0

        #Prepare right environment
        for i = L:-1:3
            update_iRPs!(iH, psis[i], RPs, i)
        end

        #The first part: sweep from left to right
        for i = 1:L-2
            if i != 1
                lambdai = update_bond!(iH, psis, [LPs, RPs], [i, i + 1]; sweep=true, update_lrp=[true, false], cutoff, maxdim, krylovdimmax, niter, kryloverr, noise, noisedir="L") #This is updating the wave-function and the environment
            else
                lambdai, gsenergy_ini = update_bond!(iH, psis, [LPs, RPs], [i, i + 1]; sweep=true, update_lrp=[true, false], cutoff, maxdim, krylovdimmax, niter, kryloverr, verbose=1, noise, noisedir="L")
            end
            #Get the lambda_(L-1) from the last step
            psis[i+1] *= lambdai
        end

        lambda_lm1 = update_bond!(iH, psis, [LPs, RPs], [L - 1, L]; sweep=true, update_lrp=[true, true], cutoff, maxdim, krylovdimmax, niter, kryloverr) #This is updating the wave-function and the environment
        psis[L-1] *= lambda_lm1

        for i = L-2:-1:1
            lambdai = update_bond!(iH, psis, [LPs, RPs], [i, i + 1]; sweep=false, update_lrp=[false, true], cutoff, maxdim, krylovdimmax, niter, kryloverr)
            psis[i] *= lambdai
        end

        #At this point, the environment RP[1] is ready.
        #Insert one unit-cell
        psis[L] *= lambda_lm1 #This is for dmrg
        psis[L] *= ITensor(pinv(array(lambda_L), 1e-8), inds(lambda_L))

        lambda_temp = update_bond!(iH, psis, [LPs, RPs], [L, 1]; sweep=true, update_lrp=[true, false], cutoff, maxdim, krylovdimmax, niter, kryloverr, noise, noisedir="L")
        psis[1] *= lambda_temp
        lambda_temp, gsenergy_fin = update_bond!(iH, psis, [LPs, RPs], [1, 2]; sweep=true, update_lrp=[false, true], cutoff, maxdim, krylovdimmax, niter, kryloverr, verbose=1, noise, noisedir="R")
        psis[1] *= lambda_temp
        new_lambda_L = update_bond!(iH, psis, [LPs, RPs], [L, 1]; sweep=true, update_lrp=[true, true], cutoff, maxdim, krylovdimmax, niter, kryloverr)

        psis[L] *= new_lambda_L #This will be used as the right tensor in line 221, i.e., i=L-1:-1:2 first step
        psis[1] *= new_lambda_L

        #At this point, we have AL, lambda_L*B1, B2,..., BL-1
        #Now add one more unit-cell.

        for i = 1:L-2
            #We will use lambda_L later, probably don't need other stuff
            lambdai = update_bond!(iH, psis, [LPs, RPs], [i, i + 1]; sweep=false, update_lrp=[true, false], cutoff, maxdim, krylovdimmax, niter, kryloverr)
            psis[i+1] *= lambdai
        end

        psis[L-1] *= ITensor(pinv(array(lambda_lm1), 1e-8), inds(lambda_lm1)) #Use the lm1 obtained before

        for i = L-1:-1:2
            lambdai = update_bond!(iH, psis, [LPs, RPs], [i, i + 1]; sweep=true, update_lrp=[false, true], cutoff, maxdim, krylovdimmax, niter, kryloverr, noise, noisedir="R")
            psis[i] *= lambdai
        end

        new_lambda_1 = update_bond!(iH, psis, [LPs, RPs], [1, 2]; sweep=false, update_lrp=[false, false], cutoff, maxdim, krylovdimmax, niter, kryloverr)
        maxbonddim = size(new_lambda_1)[1]

        if verbosity == 1
            println("Maximum bond dimension is: ", maxbonddim)
            println("Energy is: ", (gsenergy_fin - gsenergy_ini) / (L))
            println("Lambda overlap is: ", lambdamodule(new_lambda_1, L_n2))
        end

        return [psis, LPs[1], RPs[L], new_lambda_1, new_lambda_L]
    end

end

end