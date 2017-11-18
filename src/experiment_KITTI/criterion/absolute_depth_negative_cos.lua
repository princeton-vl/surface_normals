-- require 'cunn'
require 'nn'

-- depth to normal
require '../models/world_coord_to_normal'
require '../models/img_coord_to_world_coord'

-- sub criterions
require './normal_negative_cos'

local absolute_depth_negative_cos, parent = torch.class('nn.absolute_depth_negative_cos', 'nn.Criterion')

function absolute_depth_negative_cos:__init(w_normal)
    print(string.format(">>>>>>>>>>>>>>>>>>>>>>Criterion: absolute_depth_negative_cos() "))
    parent.__init(self)
    self.depth_crit = nn.MSECriterion():cuda()
    self.normal_crit = nn.normal_negative_cos()    
    self.depth_to_normal = nn.Sequential()
    self.depth_to_normal:add(nn.img_coord_to_world_coord())
    self.depth_to_normal:add(world_coord_to_normal())
    self.depth_to_normal = self.depth_to_normal:cuda()
    self.w_normal = w_normal

    self.__loss_normal = 0
    self.__loss_absolute_depth = 0    
end

function absolute_depth_negative_cos:updateOutput(input, target)
    -- the input is tensor taht represents the depth map
    -- the target is a table, where the first component is a table that contains relative depth info, and the second component is a table that contains normal info.
    local n_depth = target[1].n_sample
    local n_normal = target[2].n_sample
    
    self.output = 0
    self.__loss_absolute_depth = 0
    self.__loss_normal = 0

    if n_depth > 0 then
        self.__loss_absolute_depth = self.depth_crit:forward(input:sub(1, n_depth), target[1].full_metric_depth)   -- to test
        self.output = self.output + self.__loss_absolute_depth
    end
    if n_normal > 0 then    -- to test
        -- first go through the depth->normal transormation:   ----   to test
        local normal = self.depth_to_normal:forward(input:sub(n_depth+1, -1))
        -- then go through the criterion
        self.__loss_normal = self.w_normal * self.normal_crit:forward( normal, target[2])  
        self.output = self.output + self.__loss_normal
    end

    return self.output
end

function absolute_depth_negative_cos:updateGradInput(input, target)
    -- the input is tensor taht represents the depth map
    -- the target is a table, where the first component is a table that contains relative depth info, and the second component is a table that contains normal info.
    
    -- pre-allocate memory and reset gradient to 0
    if self.gradInput then
        local nElement = self.gradInput:nElement()        
        if self.gradInput:type() ~= input:type() then
            self.gradInput = self.gradInput:typeAs(input);
        end
        self.gradInput:resizeAs(input)
    end

    local n_depth = target[1].n_sample
    local n_normal = target[2].n_sample

    assert( torch.type(target) == 'table' );

    if n_depth > 0 then
        self.gradInput:sub(1, n_depth):copy(self.depth_crit:backward(input:sub(1, n_depth), target[1].full_metric_depth))   -- to test
    end
    if n_normal > 0 then        -- to test      
        -- then go through the criterion
        self.gradInput:sub(n_depth+1, -1):copy(self.depth_to_normal:backward( input:sub(n_depth+1, -1), self.normal_crit:backward( self.depth_to_normal.output, target[2])) )
        self.gradInput:sub(n_depth+1, -1):mul(self.w_normal)
    end
   
    return self.gradInput
end