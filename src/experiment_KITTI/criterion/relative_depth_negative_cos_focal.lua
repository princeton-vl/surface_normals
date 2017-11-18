-- require 'cunn'
require 'nn'

-- depth to normal
require '../models/world_coord_to_normal'
require '../models/img_coord_to_world_coord_focal'

-- sub criterions
require './relative_depth'
require './normal_negative_cos'

local relative_depth_negative_cos_focal, parent = torch.class('nn.relative_depth_negative_cos_focal', 'nn.Criterion')

function relative_depth_negative_cos_focal:__init(w_normal, b_focal)
    print(">>>>>>>>>>>>>>>>>>>>>>Criterion: relative_depth_negative_cos_focal()")
    parent.__init(self)
    self.depth_crit = nn.relative_depth_crit()
    self.normal_crit = nn.normal_negative_cos()    
    self.depth_to_normal = nn.Sequential()
    self.depth_to_normal:add(nn.img_coord_to_world_coord_focal())
    self.depth_to_normal:add(world_coord_to_normal())
    self.depth_to_normal = self.depth_to_normal:cuda()
    self.w_normal = w_normal
    self.b_focal = b_focal          -- apply MSE criterion on focal length

    -- to do : should probably set the weight of the focal length

    self.__loss_normal = 0
    self.__loss_relative_depth = 0
    self.focal_length_loss = nn.MSECriterion():cuda()

    self.gradInput = {torch.Tensor(), torch.Tensor()}
end

function relative_depth_negative_cos_focal:updateOutput(input, target)
    -- The input is table,
    --   1. The first component is a tensor that represents the depth map
    --   2. The second component is a 2D tensor that contain the predicted focal length
    --      Note that all the predicted depth comes with a focal length!
    -- The target is a table, 
    --   1. The first component is a table that contains relative depth info,
    --   2. The second component is a table that contains normal info. 
    --      It also contains the groundtruth focal length of each sample, and is stored in a 2D tensor.
    --      But the size of this tensor is the same as the lenght of the normal info. 
    --   The sum of the number of relative depth info and the number of normal info should be equal
    --   to the batch size.


    local n_depth = target[1].n_sample
    local n_normal = target[2].n_sample
    
    assert( torch.type(input) == 'table' )
    assert( torch.type(target) == 'table' )
    

    self.output = 0
    self.__loss_relative_depth = 0
    self.__loss_normal = 0
    
    if n_depth > 0 then
        self.__loss_relative_depth = self.depth_crit:forward(input[1]:sub(1, n_depth), target[1])
        self.output = self.output + self.__loss_relative_depth
    end
    if n_normal > 0 then
        assert( target[2].focal_length:size(1) == n_normal )
        -- first go through the depth->normal transormation:
        local normal = self.depth_to_normal:forward( {input[1]:sub( n_depth + 1, -1 ), input[2]:sub(n_depth + 1, -1)} )
        -- then go through the criterion
        self.__loss_normal = self.w_normal * self.normal_crit:forward( normal, target[2])
        self.output = self.output + self.__loss_normal

        
        -- if we want to consider the loss on focal length
        if self.b_focal then
            -- then the loss on focal length, note that we only have a subset of ground truth focal length.
            -- note that only test the input[2]:sub(n_depth+1, -1) predicted focal length 
            self.output = self.output + self.focal_length_loss:forward(input[2]:sub( n_depth + 1, -1 ), target[2].focal_length:cuda())
        end
    end

    return self.output
end

function relative_depth_negative_cos_focal:updateGradInput(input, target)
    -- The input is table,
    --   1. The first component is a tensor that represents the depth map
    --   2. The second component is a 2D tensor that contain the predicted focal length
    --      Note that all the predicted depth comes with a focal length!
    -- The target is a table, 
    --   1. The first component is a table that contains relative depth info,
    --   2. The second component is a table that contains normal info. 
    --      It also contains the groundtruth focal length of each sample, and is stored in a 2D tensor.
    --      But the size of this tensor is the same as the lenght of the normal info. 
    --   The sum of the number of relative depth info and the number of normal info should be equal
    --   to the batch size.
    
    -- pre-allocate memory and reset gradient to 0
    if self.gradInput then
        if self.gradInput[1]:type() ~= input[1]:type() then
            self.gradInput[1] = self.gradInput[1]:typeAs(input[1]);
        end
        self.gradInput[1]:resizeAs(input[1])

        if self.gradInput[2]:type() ~= input[2]:type() then
            self.gradInput[2] = self.gradInput[2]:typeAs(input[2]);
        end
        self.gradInput[2]:resizeAs(input[2])
    end
    self.gradInput[2]:zero()


    local n_depth = target[1].n_sample
    local n_normal = target[2].n_sample

    assert( torch.type(input) == 'table' )
    assert( torch.type(target) == 'table' )
    

    if n_depth > 0 then
        self.gradInput[1]:sub(1, n_depth):copy(self.depth_crit:backward(input[1]:sub(1, n_depth), target[1]))   
    end
    if n_normal > 0 then
        assert( target[2].focal_length:size(1) == n_normal )
        -- then go through the criterion
        local _grad = self.depth_to_normal:backward( {input[1]:sub( n_depth + 1, -1 ), input[2]:sub(n_depth + 1, -1)}, self.normal_crit:backward( self.depth_to_normal.output, target[2]))

        self.gradInput[1]:sub(n_depth+1, -1):copy( _grad[1] )
        self.gradInput[1]:sub(n_depth+1, -1):mul(self.w_normal)

        self.gradInput[2]:sub(n_depth+1, -1):copy( _grad[2])
        self.gradInput[2]:sub(n_depth+1, -1):mul(self.w_normal)        

        -- if we want to consider the loss on focal length
        if self.b_focal then
            -- the focal length loss  
            -- note that only test the input[2]:sub(n_depth+1, -1) predicted focal length
            self.gradInput[2]:sub(n_depth+1, -1):add( self.focal_length_loss:backward( input[2]:sub(n_depth+1, -1), target[2].focal_length:cuda() ) )
        end
    end
   
    return self.gradInput
end