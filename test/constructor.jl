# ------------------------------------------------------------------------------
# Node{d}

# item: creating a default blank (invalid) node
@test asg.Node{1}() == asg.Node{1}()
@test asg.Node{2}()
@test asg.Node{100}()

# item: creating a node with a value
@test asg.Node{1}((1,),(1,))
@test asg.Node{2}((1,2),(1,0))
@test asg.Node{2}((1,2),(1,2))
@test asg.Node{5}(ntuple(i -> i,5),ntuple(i -> i,5))


# ------------------------------------------------------------------------------
# NodeValue{d}

# item: creating a default blank (invalid) node value
@test asg.NodeValue{1}()
@test asg.NodeValue{2}()
@test asg.NodeValue{100}()

# item: creating a node value with a value
@test asg.NodeValue{1}(0.5,0.5)
@test asg.NodeValue{2}(0.5,0.5)
@test asg.NodeValue{100}(0.5,0.5)


# ------------------------------------------------------------------------------
# AdaptiveSparseGrid{d}

@test asg.AdaptiveSparseGrid{1}(2)
@test asg.AdaptiveSparseGrid{2}(10)
@test asg.AdaptiveSparseGrid{100}(10)

@test asg.AdaptiveSparseGrid{1}(2, rtol = 1e-3)
@test asg.AdaptiveSparseGrid{2}(10, rtol = 1e-3)
@test asg.AdaptiveSparseGrid{100}(10, rtol = 1e-3)


















