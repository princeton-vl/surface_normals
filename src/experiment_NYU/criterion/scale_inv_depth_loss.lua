require 'cunn'

local scale_inv_depth_loss, parent = torch.class('nn.scale_inv_depth_loss', 'nn.Criterion')


function scale_inv_depth_loss:__init()
    parent.__init(self)
    self.buffer = torch.Tensor()
end



function scale_inv_depth_loss:updateOutput(input, target)
    -- The input[1] and target are both 4D tensors, [batchSize, 4, height, width], and represents the normal maps
    -- input[2] is a mask that denotes which locations are valid (1 is valid) [batchSize, 4, height, width]
    -- the loss is (d1 - d2)^2 / (d1 + d2)^2


    self.output = 0
    
    local denominator = torch.pow(input[1] - target, 2)
    local nominator = torch.pow(input[1] + target, 2)

    local zero_mask = nominator:eq(0)
    nominator[zero_mask] = 1e-7
    denominator:cdiv(nominator)
    
    denominator:cmul(input[2])
    self.output = torch.sum(denominator)
    

    return self.output / torch.sum(input[2])
end



function scale_inv_depth_loss:updateGradInput(input, target)    
    -- The input[1] and target are both 4D tensors, [batchSize, 4, height, width], and represents the normal maps
    -- input[2] is a mask that denotes which locations are valid (1 is valid) [batchSize, 4, height, width]
    -- the loss is (d1 - d2)^2 / (d1 + d2)^2

    -- pre-allocate memory and reset gradient to 0
    if self.gradInput then
        local nElement = self.gradInput:nElement()        
        if self.gradInput:type() ~= input[1]:type() then
            self.gradInput = self.gradInput:typeAs(input[1]);
        end
        self.gradInput:resizeAs(input[1])
    end

    self.gradInput:zero()


    self.gradInput:copy(input[1])
    self.gradInput:csub(target)
    local temp_sum_3 = torch.pow(input[1] + target, 3)
    
    self.gradInput:cmul(target)
    self.gradInput:mul(4)
    
    

    local zero_mask = temp_sum_3:eq(0)
    temp_sum_3[zero_mask] = 1e-7
    self.gradInput:cdiv(temp_sum_3)
    
    self.gradInput:cmul(input[2])
    
    -- print(self.gradInput[{1,2,1,1}])
    -- print(target[{1,2,1,1}])
    -- print(input[{1,2,1,1}])
    -- io.read()

    return self.gradInput:div(torch.sum(input[2]))
end