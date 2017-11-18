require 'cunn'

local normal_negative_cos_fast, parent = torch.class('nn.normal_negative_cos_fast', 'nn.Criterion')


function normal_negative_cos_fast:__init()
    parent.__init(self)
    self.buffer = torch.Tensor()
end



function normal_negative_cos_fast:updateOutput(input, target)
    -- The input is 4D tensor, [batchSize, 3, height, width], and represents the normal maps
    --      The 1st channle is the x component, 2nd is the y component, 3rd is the z component!!

    -- The target is a table of the form defined in DataLoader.lua, with 3 components {x, y, normal}. Each of the 3 components is a tensor 
    -- We assume that the input normal has all been normalized to be unit vector!!!!!

    -- the loss is the negative cos(angle)
    
    local n_point_total = input:size(1) * input:size(3) * input:size(4)

    self.output = - torch.sum( torch.cmul(input, target) )     -- dot product of normals , seems quite expensive move
       
    return self.output / n_point_total
end



function normal_negative_cos_fast:updateGradInput(input, target)    
    -- The input is 4D tensor, [batchSize, 3, height, width], and represents the normal maps
    --      The 1st channle is the x component, 2nd is the y component, 3rd is the z component!!

    -- The target is a table of the form defined in DataLoader.lua, with 3 components {x, y, normal}. Each of the 3 components is a tensor 
    -- We assume that the input normal has all been normalized to be unit vector!!!!!

    -- the loss is the negative cos(angle)



    -- pre-allocate memory and reset gradient to 0
    if self.gradInput then
        local nElement = self.gradInput:nElement()        
        if self.gradInput:type() ~= input:type() then
            self.gradInput = self.gradInput:typeAs(input);
        end
        self.gradInput:resizeAs(input)
    end

    self.gradInput:zero()



    local n_point_total = input:size(1) * input:size(3) * input:size(4)

    self.gradInput:copy(target)

    return self.gradInput:div( -n_point_total )
end