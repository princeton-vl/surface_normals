require('./Residual')

local function lin2(numIn,numOut,inp)
    -- Apply 1x1 convolution, no stride, no padding
    local l_ = nn.Linear(numIn,numOut)(inp)
    return nnlib.ReLU(true)(nn.BatchNormalization(numOut)(l_))
end

function hourglass(n, numIn, numOut, inp)
    local d = 64

    -- Upper branch
    local up1 = Residual(numIn,d)(inp)
    local up2 = Residual(d,d)(up1)
    local up4 = Residual(d,numOut)(up2)

    -- Lower branch
    local pool = nnlib.SpatialMaxPooling(2,2,2,2)(inp)
    local low1 = Residual(numIn,d)(pool)
    local low2 = Residual(d,d)(low1)
    local low5 = Residual(d,d)(low2)
    local low6
    if n > 1 then
        low6 = hourglass(n-1,d,numOut,low5)
    else
        low6 = Residual(d,numOut)(low5)
    end
    local low7 = Residual(numOut,numOut)(low6)
    local up5 = nn.SpatialUpSamplingNearest(2)(low7)

    -- Bring two branches together
    return nn.CAddTable()({up4,up5})
end

function hourglass_inter_supervise(n, numIn, numOut, inp)
    local outNode

    local d = 64

    -- Upper branch
    local up1 = Residual(numIn,d)(inp)
    local up2 = Residual(d,d)(up1)
    local up4 = Residual(d,numOut)(up2)

    -- Lower branch
    local pool = nnlib.SpatialMaxPooling(2,2,2,2)(inp)
    local low1 = Residual(numIn,d)(pool)
    local low2 = Residual(d,d)(low1)
    local low5 = Residual(d,d)(low2)
    local low6
    if n > 1 then
        low6, outNode = hourglass(n-1,d,numOut,low5)
    else
        low6 = Residual(d,numOut)(low5)

        local flat = nn.View(-1,numOut*4*4):setNumInputDims(4)(low6)
        local l1 = lin2(numOut*4*4,1024,flat)
        local l2 = lin2(1024,1024,l1)
        outNode = nn.LogSoftMax()(nn.Linear(1024,g_nClass)(l2))
    end
    local low7 = Residual(numOut,numOut)(low6)
    local up5 = nn.SpatialUpSamplingNearest(2)(low7)

    -- Bring two branches together
    return nn.CAddTable()({up4,up5}), outNode
end

function hourglass_identity_skip(n, numInOut, inp)
    local d = 64

    -- Upper branch: Identity connections
    -- local up1 = Residual(numIn,d)(inp)
    -- local up2 = Residual(d,d)(up1)
    -- local up4 = Residual(d,numOut)(up2)

    -- Lower branch
    local pool = nnlib.SpatialMaxPooling(2,2,2,2)(inp)
    local low1 = Residual(numInOut,d)(pool)
    local low2 = Residual(d,d)(low1)
    local low5 = Residual(d,d)(low2)
    local low6
    if n > 1 then
        low6 = hourglass_identity_skip(n-1,d,low5)
    else
        low6 = Residual(d,d)(low5)
    end
    local low7 = Residual(d,numInOut)(low6)
    local up5 = nn.SpatialUpSamplingNearest(2)(low7)

    -- Bring two branches together
    return nn.CAddTable()({inp,up5})

end

function hg_module(n, numIn, numOut)
    local inp = nnlib.Identity()()
    local hg = hourglass(n, numIn, numOut, inp)
    return nn.gModule({inp}, {hg})
end

function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, no stride, no padding
    local l_ = nnlib.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return nnlib.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l_))
end

