require 'cunn'

local normal_negative_cos, parent = torch.class('nn.normal_negative_cos', 'nn.Criterion')


function normal_negative_cos:__init()
    parent.__init(self)
    self.buffer = torch.Tensor()
end



function normal_negative_cos:updateOutput(input, target)
    -- The input is 4D tensor, [batchSize, 3, height, width], and represents the normal maps
    --      The 1st channle is the x component, 2nd is the y component, 3rd is the z component!!

    -- The target is a table of the form defined in DataLoader.lua, with 3 components {x, y, normal}. Each of the 3 components is a tensor 
    -- We assume that the input normal has all been normalized to be unit vector!!!!!

    -- the loss is the negative cos(angle)
    self.output = 0
    
    local n_point_total = 0
    local cpu_input = input

    for batch_idx = 1 , cpu_input:size(1) do

        n_point_total = n_point_total + target[batch_idx].n_point

        local x_arr = target[batch_idx].x            -- to check: the length of x vary with each sample!!!!! 
        local y_arr = target[batch_idx].y        

        local batch_input = cpu_input[{batch_idx, {}}]      -- batch_input is 3 dimension -- checked       

        local normal_arr = batch_input:index(3, x_arr):gather(2,  torch.repeatTensor(y_arr:view(1,-1),3,1):view(3,1,-1)  ):squeeze()        
        local ground_truth_arr = target[batch_idx].normal


        self.output = self.output - torch.sum( torch.cmul(normal_arr, ground_truth_arr) )     -- dot product of normals , seems quite expensive move
    end
       
    return self.output / n_point_total
end



function normal_negative_cos:updateGradInput(input, target)    
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



    local n_point_total = 0
    local cpu_input = input        -- is this necessary?  can it be gpu data??    to check

    for batch_idx = 1 , cpu_input:size(1) do

        n_point_total = n_point_total + target[batch_idx].n_point

        local x_arr = target[batch_idx].x
        local y_arr = target[batch_idx].y

        local batch_input = cpu_input[{batch_idx, {}}]      -- batch_input is 3 dimension -- checked       
        
        local unsqueeze = nn.Unsqueeze(2):forward( target[batch_idx].normal:double() ):cuda()

        local p2 = torch.Tensor(3, cpu_input:size()[3], target[batch_idx].n_point):zero():cuda()
        local p1 = torch.Tensor(batch_input:size(1), batch_input:size(2), batch_input:size(3)):zero():cuda()
        p2:scatter(2, torch.repeatTensor(y_arr:view(1,-1),3,1):view(3,1,-1), unsqueeze)        
        p1:indexAdd(3, x_arr, p2)

        self.gradInput[{batch_idx,{}}]:copy(p1)
    end

    return self.gradInput:div( -n_point_total )
end