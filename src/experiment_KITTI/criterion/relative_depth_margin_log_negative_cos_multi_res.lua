-- require 'cunn'
require 'nn'

-- depth to normal
require '../models/world_coord_to_normal'
require '../models/img_coord_to_world_coord_multi_res'

-- sub criterions
require './relative_depth_margin'
require './normal_negative_cos'

local relative_depth_margin_log_negative_cos_multi_res, parent = torch.class('nn.relative_depth_margin_log_negative_cos_multi_res', 'nn.Criterion')

function relative_depth_margin_log_negative_cos_multi_res:__init(w_normal, margin, num_scale)
    print(string.format(">>>>>>>>>>>>>>>>>>>>>>Criterion: relative_depth_margin_negative_cos()  w_normal:%f, margin:%f", w_normal, margin))
    parent.__init(self)
    self.depth_crit = nn.relative_depth_crit(margin)
    self.normal_crit = nn.normal_negative_cos()    

    self.depth_to_normal = {}

    for scale_idx = 1, num_scale do
        self.depth_to_normal[scale_idx] =  nn.Sequential()

        if scale_idx ~= 1 then
            -- to test
            self.depth_to_normal[scale_idx]:add(nn.SpatialAveragePooling(g_scales[scale_idx], g_scales[scale_idx], g_scales[scale_idx], g_scales[scale_idx]))      
        end

        self.depth_to_normal[scale_idx]:add(nn.img_coord_to_world_coord_multi_res(g_scales[scale_idx]))
        self.depth_to_normal[scale_idx]:add(world_coord_to_normal())

        self.depth_to_normal[scale_idx] = self.depth_to_normal[scale_idx]:cuda()
    end

    self.w_normal = w_normal

    self.num_scale = num_scale

    self.__loss_normal = 0
    self.__loss_relative_depth = 0    


end

local function visualize_normal(normal, filename)
    local _normal_img = normal:clone()
    _normal_img:add(1)
    _normal_img:mul(0.5)
    image.save(filename, _normal_img) 

    print("Done saving to ", filename)
end

function relative_depth_margin_log_negative_cos_multi_res:updateOutput(input, target)
    -- the input is tensor taht represents the depth map
    -- the target is a table, where the first component is a table that contains relative depth info, and the second component is a table that contains normal info.
    local n_depth = target[1].n_sample
    local n_normal = target[2].n_sample
    
    assert( torch.type(target) == 'table' );
    
    self.output = 0
    self.__loss_relative_depth = 0
    self.__loss_normal = 0

    if n_depth > 0 then
        self.__loss_relative_depth = self.depth_crit:forward(nn.Log():cuda():forward(input:sub(1, n_depth)), target[1])   -- to test
        self.output = self.output + self.__loss_relative_depth
    end
    if n_normal > 0 then    -- to test

        for scale_idx = 1, self.num_scale do
            -- first go through the depth->normal transormation: 
            local normal = self.depth_to_normal[scale_idx]:forward(input:sub(n_depth+1, -1))
            -- then go through the criterion
            self.__loss_normal = self.__loss_normal + self.w_normal * self.normal_crit:forward( normal, target[2][scale_idx])  

            -- -- debug
            -- for batch_idx = 1 , n_normal do
            --     local x_arr = target[2][scale_idx][batch_idx].x
            --     local y_arr = target[2][scale_idx][batch_idx].y

            --     local unsqueeze = nn.Unsqueeze(2):forward( target[2][scale_idx][batch_idx].normal:double() ):cuda()
            --     local p2 = torch.Tensor(3, normal:size(3), target[2][scale_idx][batch_idx].n_point):zero():cuda()
            --     p2:scatter(2, torch.repeatTensor(y_arr:view(1,-1),3,1):view(3,1,-1), unsqueeze)  
            --     local normal_map = torch.Tensor( 3, normal:size(3), normal:size(4)):zero():cuda()
            --     normal_map:indexAdd(3, x_arr, p2)   
            --     visualize_normal(normal_map, string.format("scale_%d.png", scale_idx))
            -- end
            -- io.read()

        end
        self.output = self.output + self.__loss_normal
        
    end

    return self.output
end

function relative_depth_margin_log_negative_cos_multi_res:updateGradInput(input, target)
    -- the input is tensor that represents the depth map
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
        self.gradInput:sub(1, n_depth):copy( self.depth_crit:backward(nn.Log():cuda():forward(input:sub(1, n_depth)), target[1]) )   -- to test
        self.gradInput:sub(1, n_depth):copy( nn.Log():cuda():backward(input:sub(1, n_depth), self.gradInput:sub(1, n_depth)) )
    end
    if n_normal > 0 then        -- to test      
        -- then go through the criterion

        self.gradInput:sub(n_depth+1, -1):zero()
        for scale_idx = 1, self.num_scale do
            local grad_wrt_depth_at_scale = self.normal_crit:backward( self.depth_to_normal[scale_idx].output, target[2][scale_idx]) 
            self.gradInput:sub(n_depth+1, -1):add(self.depth_to_normal[scale_idx]:backward( input:sub(n_depth+1, -1), grad_wrt_depth_at_scale))
        end

        self.gradInput:sub(n_depth+1, -1):mul(self.w_normal)
    end
   
    return self.gradInput
end