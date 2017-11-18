-- require 'cunn'
require 'nn'

-- depth to normal
require '../models/world_coord_to_normal'
require '../models/img_coord_to_world_coord'

-- depth variance loss
require './depth_var_loss'

-- sub criterions
require './relative_depth_margin'
require './normal_negative_cos'


local relative_depth_margin_negative_cos_var, parent = torch.class('nn.relative_depth_margin_negative_cos_var', 'nn.Criterion')

function relative_depth_margin_negative_cos_var:__init(w_normal, margin, d_var_thresh)
    print("\n>>>>>>>>>>>>>>>>>>>>>>Criterion: relative_depth_margin_negative_cos_var()")
    parent.__init(self)
    self.depth_crit = nn.relative_depth_crit(margin)
    self.normal_crit = nn.normal_negative_cos()    
    self.depth_to_normal = nn.Sequential()
    self.depth_to_normal:add(nn.img_coord_to_world_coord())
    self.depth_to_normal:add(world_coord_to_normal())
    self.depth_to_normal = self.depth_to_normal:cuda()
    self.w_normal = w_normal

    
    -- depth variance
    self.w_d_var = 1
    print(string.format("\t\tw_normal=%f, margin=%f, w_d_var=%f", w_normal, margin, self.w_d_var))
    self.nn_depth_var = nn.depth_var_loss(d_var_thresh):cuda()
    

    self.__loss_normal = 0
    self.__loss_relative_depth = 0    
end

function relative_depth_margin_negative_cos_var:updateOutput(input, target)
    -- the input is tensor taht represents the depth map
    -- the target is a table, where the first component is a table that contains relative depth info, and the second component is a table that contains normal info.
    local n_depth = target[1].n_sample
    local n_normal = target[2].n_sample
    
    assert( torch.type(target) == 'table' );
    
    self.output = 0
    self.__loss_relative_depth = 0
    self.__loss_normal = 0

    if n_depth > 0 then
        self.__loss_relative_depth = self.depth_crit:forward(input:sub(1, n_depth), target[1])   -- to test
        self.output = self.output + self.__loss_relative_depth

        -- the depth variance loss
        self.output = self.output + self.w_d_var * self.nn_depth_var:forward(input:sub(1, n_depth), nil)
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

function relative_depth_margin_negative_cos_var:updateGradInput(input, target)
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
        self.gradInput:sub(1, n_depth):copy(self.depth_crit:backward(input:sub(1, n_depth), target[1]))   -- to test

        -- the depth variance loss
        local temp = self.nn_depth_var:backward(input:sub(1, n_depth), nil)
        temp:mul(self.w_d_var)
        self.gradInput:sub(1, n_depth):add(temp)
    end
    if n_normal > 0 then        -- to test      
        -- then go through the criterion
        self.gradInput:sub(n_depth+1, -1):copy(self.depth_to_normal:backward( input:sub(n_depth+1, -1), self.normal_crit:backward( self.depth_to_normal.output, target[2])) )
        self.gradInput:sub(n_depth+1, -1):mul(self.w_normal)
    end
   
    return self.gradInput
end