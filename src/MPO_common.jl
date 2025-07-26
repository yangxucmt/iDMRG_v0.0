#=
1. Direct sum methods--done
2. Add projection function, which can read off certain blocks of MPO
3. Implement fermion
4. Wrap around the current MPO generation code
5. Implement symmetry
=#
module MPO_common

using BenchmarkTools
using ITensors, ITensorMPS
using LinearAlgebra
using KrylovKit
using Base.Threads
using Dates

export op_to_hm_inf, MPOsum, mpoadd!,expand_opstring

struct MPOterm
    coeff::Number
    ops::Vector{Tuple{Int, String}}

    function MPOterm(coeff::Number, ops::Vector{Tuple{Int, String}})
        new(coeff, sort(ops, by=x -> x[1]))  # Sort by site index
    end
end

struct MPOsum
    terms::Vector{MPOterm}
    
    function MPOsum()
        new([])
    end
end

function mpoadd!(sum::MPOsum, coeff::Number, ops::Vararg{Tuple{Int, String}})
    push!(sum.terms, MPOterm(coeff, collect(ops)))
end

function add_column(hmi, column_location; sites, siteid)
    sizel, sizer = size(hmi)
    columnvec = fill(0 * op("Id", sites, siteid), sizel)
    newhmi = [hmi[:, 1:column_location-1] columnvec hmi[:, column_location:end]]
    return newhmi
end

function add_row(hmi, row_location; sites, siteid)
    sizel, sizer = size(hmi)
    rowvec = reshape(fill(0 * op("Id", sites, siteid), sizer), 1, sizer)
    newhmi = [hmi[1:row_location-1, :]; rowvec; hmi[row_location:end, :]]
    return newhmi
end

function expand_opstring(opstring)
    first_site, last_site = opstring[1][1], opstring[end][1]
    expanded = fill("Id", last_site - first_site + 1)
    for (site, label) in opstring
        expanded[site - first_site + 1] = label
    end
    return expanded, first_site
end

function op_to_hm_inf(oplist::MPOsum, sites)
    N = length(sites)
    uc = only(size(sites))
    hm = Array{Any}(undef, N) #Each element is a matrix

    #Let's initialize hm
    for i = 1:N
        hmi = fill(0 * op("Id", sites, i), 2, 2)
        hmi[1, 1] += op("Id", sites, i)
        hmi[2, 2] += op("Id", sites, i)
        hm[i] = hmi
    end

    for term in oplist.terms
        coeff, opstring = term.coeff, term.ops
        expanded_opstring, start_site = expand_opstring(opstring)
        stringlength = length(expanded_opstring)

        if stringlength == 1
            label, siteindex = expanded_opstring[1], mod1(start_site, uc)
            operator = op(label, sites, siteindex)
            hm[mod1(siteindex, uc)][end, 1] += coeff * operator
        end

        if stringlength != 1
            label, siteindex = expanded_opstring[1], mod1(start_site, uc)
            operator = op(label, sites, siteindex)
            #First part: initialization part
            newcol = findfirst(x -> x == operator, hm[siteindex][end, :])
            if newcol == nothing
                (sizel, sizer) = size(hm[siteindex])
                hm[siteindex] = add_column(hm[siteindex], sizer; sites, siteid=siteindex)
                hm[mod1(siteindex + 1, uc)] = add_row(hm[mod1(siteindex + 1, uc)], sizer; sites, siteid=mod1(siteindex + 1, uc))
                newcol = sizer
                hm[siteindex][end, newcol] += operator
                #else do nothing
            end
            #Second part: 2-stringlength, need to see if it reaches the end
            for site_iterator = 2:stringlength
                label, siteindex = expanded_opstring[site_iterator], mod1(start_site + site_iterator - 1, uc)
                operator = op(label, sites, siteindex)

                if site_iterator != stringlength
                    if findfirst(x -> x == operator, hm[siteindex][newcol, :]) == nothing
                        (sizel, sizer) = size(hm[siteindex])
                        hm[siteindex] = add_column(hm[siteindex], sizer; sites, siteid=siteindex) #We always add at the final location
                        hm[mod1(siteindex + 1, uc)] = add_row(hm[mod1(siteindex + 1, uc)], sizer; sites, siteid=mod1(siteindex + 1, uc))
                        hm[siteindex][newcol, sizer] += operator
                        newcol = sizer

                        #else do nothing
                    else
                        newcol = findfirst(x -> x == operator, hm[siteindex][newcol, :])
                    end
                elseif site_iterator == stringlength #Then don't need to find, just add it.
                    hm[siteindex][newcol, 1] += coeff * operator
                end
            end
        end
    end

    iH = MPO(uc)
    leftinds = Vector{Index}(undef, uc)

    for i = 1:uc
        leftinds[i] = Index(size(hm[i])[1])
    end

    for i = 1:uc
        diml = size(hm[i])[1]
        dimr = size(hm[i])[2]
        Harray = zeros(ComplexF64, diml, dimr, 2, 2)
        for lindex = 1:diml
            for rindex = 1:dimr
                Harray[lindex, rindex, :, :] = array(hm[i][lindex, rindex], prime(sites[i]), sites[i])
            end
        end
        iH[i] = ITensor(Harray, leftinds[i], leftinds[mod1(i + 1, uc)], prime(sites[i]), sites[i])
    end

    return iH
end




end