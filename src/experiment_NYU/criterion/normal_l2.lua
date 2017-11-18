require 'cunn'

local normal_l2, parent = torch.class('nn.normal_l2', 'nn.Criterion')


function normal_l2:__init()
    print(">>>>>>>>>>>>>>>>>   normal loss = normal l2 loss")
    parent.__init(self)
    self.buffer = torch.Tensor()
end



function normal_l2:updateOutput(input, target)
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

        local x_arr = target[batch_idx].x             -- to check: the length of x vary with each sample!!!!! 
        local y_arr = target[batch_idx].y      

        local batch_input = cpu_input[{batch_idx, {}}]      -- batch_input is 3 dimension -- checked       

        local normal_arr = batch_input:index(3, x_arr):gather(2,  torch.repeatTensor(y_arr:view(1,-1),3,1):view(3,1,-1)  ):squeeze()        
        local ground_truth_arr = target[batch_idx].normal

        self.output = self.output + torch.sum( torch.pow(torch.csub(normal_arr, ground_truth_arr),2) )     -- dot product of normals , seems quite expensive move
    end
       
    return self.output / n_point_total
end



function normal_l2:updateGradInput(input, target)    
    -- The input is 4D tensor, [batchSize, 3, height, width], and represents the normal maps
    --      The 1st channle is the x component, 2nd is the y component, 3rd is the z component!!

    -- The target is a table of the form defined in DataLoader.lua, with 3 components {x, y, normal}. Each of the 3 components is a tensor 
    -- We assume that the input normal has all been normalized to be unit vector!!!!!

    -- the loss is the negative cos(angle)

    -- only accept one single point!!!!

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
    local cpu_input = input 

    for batch_idx = 1 , cpu_input:size(1) do

        n_point_total = n_point_total + target[batch_idx].n_point
        local x = target[batch_idx].x[{1}]
        local y = target[batch_idx].y[{1}]

        local batch_input = cpu_input[{batch_idx, {}}]

        local ground_truth_arr = target[batch_idx].normal

        self.gradInput[{batch_idx,{}, y, x}]:zero()
        self.gradInput[{batch_idx,{}, y, x}]:copy(batch_input[{{}, y, x}])
        self.gradInput[{batch_idx,{}, y, x}]:csub(ground_truth_arr)      
        self.gradInput[{batch_idx,{}, y, x}]:mul(2)                      
    end
    -- io.read()
    return self.gradInput:div( n_point_total )
end