# item: stencil constructors
# scenario:
#  - 1D and 2D nodes
#  - passing nothing (empty stencil)
#  - passing a dictionary
#  - passing a vector of nodes and a vector of weights
@test try
    p11 = asg.Node{1}((1,), (1,))
    p12 = asg.Node{1}((2,), (2,))
    p21 = asg.Node{2}((1,1), (1,1))
    p22 = asg.Node{2}((2,3), (2,1))

    # empty stencil
    _res = asg.LinearStencil{1}()
    _res = asg.LinearStencil{2}()

    # dictionary
    _res = asg.LinearStencil{1}(asg.Dictionary{asg.Node{1},Float64}(
        asg.Node{1}[p11, p12],
        Float64[1.0, 2.0]
    ))
    _res = asg.LinearStencil{2}(asg.Dictionary{asg.Node{2},Float64}(
        asg.Node{2}[p21, p22],
        Float64[1.0, 2.0]
    ))

    # vector of nodes and weights
    _res = asg.LinearStencil{1}([p11, p12], [1.0, 2.0])
    _res = asg.LinearStencil{2}([p21, p22], [1.0, 2.0])

    true
catch
    false
end


# item: stencil constructors (practical examples)
# scenario:
#  - 1D function
#  - examples that are mostly used in practice
#  - mimic forward, backward and central difference stencils
@test try
    pc   = asg.Node{1}((1,), (1,))
    pl1  = asg.Node{1}((2,), (0,))
    pr1  = asg.Node{1}((2,), (2,))
    Δxl1 = asg.get_dist_along(pl1, pc, 1)
    Δxr1 = asg.get_dist_along(pc, pr1, 1)

    # forward
    stc_f = asg.LinearStencil{1}(asg.Dictionary{asg.Node{1},Float64}(
        asg.Node{1}[pc, pr1],
        Float64[-1.0 / Δxr1, 1.0 / Δxr1]
    ))

    # backward
    stc_b = asg.LinearStencil{1}(asg.Dictionary{asg.Node{1},Float64}(
        asg.Node{1}[pl1, pc],
        Float64[-1.0 / Δxl1, 1.0 / Δxl1]
    ))

    # central
    stc_c = asg.LinearStencil{1}(asg.Dictionary{asg.Node{1},Float64}(
        asg.Node{1}[pl1, pc, pr1],
        Float64[-1.0 / (Δxl1 + Δxr1), 0.0, 1.0 / (Δxl1 + Δxr1)]
    ))

    true
catch
    false
end


# item: stencil arithmetic
# scenario:
#  - plus & minus:
#     - stencil     +/- stencil
#     - stencil     +/- real scalar
#     - real scalar +/- stencil
@test begin
    
    p1 = asg.Node{1}((1,), (1,))
    p2 = asg.Node{1}((2,), (2,))
    w1 = 2.0
    w2 = -114514
    a1 = 2.0
    a2 = 520

    stc1 = asg.LinearStencil{1}([p1,], Float64[w1,])
    stc2 = asg.LinearStencil{1}([p2,], Float64[w2,])
    stc3 = asg.LinearStencil{1}([p1, p2], Float64[w1, w2])

    flag_pass = true

    # stencil +/- stencil (all nodes are the same)
    flag_pass &= stc1 + stc1 == asg.LinearStencil{1}([p1,],Float64[w1 + w1,])
    flag_pass &= stc1 - stc1 == asg.LinearStencil{1}([p1,],Float64[w1 - w1,])

    # stencil +/- stencil (nodes are partially different)
    flag_pass &= stc1+stc3 == asg.LinearStencil{1}([p1,p2],Float64[w1+w1,w2])
    flag_pass &= stc1-stc3 == asg.LinearStencil{1}([p1,p2],Float64[w1-w1,-w2])

    # stencil +/- stencil (nodes are completely different)
    flag_pass &= stc1+stc2 == asg.LinearStencil{1}([p1,p2],Float64[w1,w2])
    flag_pass &= stc1-stc2 == asg.LinearStencil{1}([p1,p2],Float64[w1,-w2])

    # stencil +/- real scalar
    flag_pass &= stc1 + a1 == asg.LinearStencil{1}([p1,],Float64[w1+a1,])
    flag_pass &= stc1 - a1 == asg.LinearStencil{1}([p1,],Float64[w1-a1,])

    # real scalar +/- stencil
    flag_pass &= a1 + stc1 == asg.LinearStencil{1}([p1,],Float64[w1+a1,])
    flag_pass &= a1 - stc1 == asg.LinearStencil{1}([p1,],Float64[a1-w1,])

    flag_pass
end



# item: stencil arithmetic
# scenario:
#  - times & divide:
#     - stencil     * and / real scalar
#     - real scalar * and / stencil
#  - also test the undefined: stencil * and / stencil
@test begin
    
    p1 = asg.Node{1}((1,), (1,))
    p2 = asg.Node{1}((2,), (2,))
    w1 = 2.0
    w2 = -114514
    a1 = 2.0
    a2 = 520

    stc1 = asg.LinearStencil{1}([p1,], Float64[w1,])
    stc2 = asg.LinearStencil{1}([p2,], Float64[w2,])
    stc3 = asg.LinearStencil{1}([p1, p2], Float64[w1, w2])

    flag_pass = true

    # stencil * and / real scalar
    flag_pass &= stc1 * a1 == asg.LinearStencil{1}([p1,],Float64[w1*a1,])
    flag_pass &= stc1 / a1 == asg.LinearStencil{1}([p1,],Float64[w1/a1,])

    # real scalar * and / stencil
    flag_pass &= a1 * stc1 == asg.LinearStencil{1}([p1,],Float64[w1*a1,])
    flag_pass &= a1 / stc1 == asg.LinearStencil{1}([p1,],Float64[a1/w1,])

    # undefined operations: stencil * and / stencil
    flag_pass &= try
        stc1 * stc2
        false
    catch
        true
    end

    flag_pass
end



# item: stencil arithmetic - combination of all
# scenario:
#  - plus & minus & times & divide
#  - mimic weighted average
@test begin
    
    p1 = asg.Node{1}((1,), (1,))
    p2 = asg.Node{1}((2,), (2,))
    w1 = 2.0
    w2 = -3
    a1 = 2.0
    a2 = 3
    a3 = 4.0

    stc1 = asg.LinearStencil{1}([p1,], Float64[w1,])
    stc2 = asg.LinearStencil{1}([p2,], Float64[w2,])
    stc3 = asg.LinearStencil{1}([p1, p2], Float64[w1, w2])

    res1 = a1 * stc1 + stc2 * a2 - stc3 / a3 + a3 / stc3
    res2 = asg.LinearStencil{1}(
        [p1, p2],
        Float64[
            a1 * w1 - w1 / a3 + a3 / w1,
            a2 * w2 - w2 / a3 + a3 / w2
        ]
    )

    res1 == res2
end























