-- require 'cunn'
require 'nn'

-- depth from normal
require '../models/get_theoretical_depth_from_normal_v2'

-- sub criterions
require './scale_inv_depth_loss'
require './relative_depth_margin'
require 'hdf5'
require 'image'

local relative_depth_margin_log_normal_depth, parent = torch.class('nn.relative_depth_margin_log_normal_depth', 'nn.Criterion')


local function load_hdf5_z(h5_filename, field_name)
    local myFile = hdf5.open(h5_filename, 'r')
    local read_result = myFile:read(field_name):all()
    myFile:close()

    return read_result
end

local function visualize_mask(mask, filename)
    local _mask_img = mask:clone()
    image.save(filename, _mask_img:double()) 
    print("Done saving to ", filename)
end

local function visualize_depth(z, filename)
    local _z_img = z:clone()
    _z_img = _z_img:add( - torch.min(_z_img) )
    _z_img = _z_img:div( torch.max(_z_img) )
    image.save(filename, _z_img) 
    print("Done saving to ", filename)
end

local function visualize_normal(normal, filename)
    local _normal_img = normal:clone()
    _normal_img:add(1)
    _normal_img:mul(0.5)
    image.save(filename, _normal_img) 

    print("Done saving to ", filename)
end

function relative_depth_margin_log_normal_depth:__init(w_normal, margin)
    print(string.format(">>>>>>>>>>>>>>>>>>>>>>Criterion: relative_depth_margin_log_normal_depth()  w_normal:%f, margin:%f", w_normal, margin))
    parent.__init(self)
    self.depth_crit = nn.relative_depth_crit(margin)
    self.normal_crit = nn.scale_inv_depth_loss():cuda()         -- compare the depths
    self.normal_to_depth = get_theoretical_depth():cuda()
    self.shift_mask = get_shifted_mask():cuda()

    self.gt_normal_map = torch.Tensor():cuda()
    self.gt_normal_mask = torch.Tensor():cuda()
    self.replicated_depth = torch.Tensor():cuda()

    self.w_normal = w_normal

    self.__loss_normal = 0
    self.__loss_relative_depth = 0    
end

function relative_depth_margin_log_normal_depth:updateOutput(input, target)
    -- the input is tensor that represents the depth map
    -- the target is a table, where the first component is a table that contains relative depth info, 
    --      and the second component is a table that contains normal info.
    local n_depth = target[1].n_sample
    local n_normal = target[2].n_sample
    
    assert( torch.type(target) == 'table' );
    
    self.output = 0
    self.__loss_relative_depth = 0
    self.__loss_normal = 0

    if n_depth > 0 then
        -- for batch_idx = 1, n_depth do
        --     print(input:size())
        --     print(target[1].gt_depth[{batch_idx,{}}]:size())
        --     input[{batch_idx, 1, {}}]:copy(image.scale(target[1].gt_depth[{batch_idx,{}}], 320, 240))
        --     visualize_depth(input[{batch_idx, 1, {}}], string.format("depth_%d.png", batch_idx))
        -- end        

        self.__loss_relative_depth = self.depth_crit:forward(nn.Log():cuda():forward(input:sub(1, n_depth)), target[1])   -- to test
        self.output = self.output + self.__loss_relative_depth
    end
    if n_normal > 0 then

        -- -- local gtz = load_hdf5_z('/scratch/jiadeng_flux/wfchen/qual_depth/data/795_NYU_MITpaper_train_imgs_NO_CROP/all/1_gt_depth.h5', 'gt_depth')
        -- -- gtz = image.scale(gtz, 320, 240)
        -- local gtz = load_hdf5_z('/scratch/jiadeng_flux/wfchen/qual_depth/data/normal_from_depth_NYU/795_NYU_MITpaper_train_imgs_NO_CROP_myway_normal/1_gt_depth.h5', 'gt_depth')
        -- gtz = gtz:t()
        -- -- gtz:mul(5.0/torch.min(gtz))
        -- print("minval = ", torch.min(gtz))
        -- visualize_depth(gtz,'gtz.png')
        -- input:sub(n_depth+1, -1):copy(gtz)
        -- local gtnormal = load_hdf5_z('/scratch/jiadeng_flux/wfchen/qual_depth/data/normal_from_depth_NYU/795_NYU_MITpaper_train_imgs_NO_CROP_myway_normal/1_normal.h5', 'normal')
        -- gtnormal = gtnormal:permute(1,3,2)
        -- -- visualize_depth(input[{n_depth+1, 1, {}}], 'predicted.png')
        


        -- construct a groundtruth normal map from point samples in the groundtruth
        self.gt_normal_map:resize(n_normal, 3, input:size(3), input:size(4)):zero():cuda()
        local _gt_normal_mask = torch.Tensor(n_normal, 3, input:size(3), input:size(4)):zero():cuda()
        for batch_idx = 1 , n_normal do
            local x_arr = target[2][batch_idx].x
            local y_arr = target[2][batch_idx].y

            local unsqueeze = nn.Unsqueeze(2):forward( target[2][batch_idx].normal:double() ):cuda()
            local p2 = torch.Tensor(3, input:size(3), target[2][batch_idx].n_point):zero():cuda()
            p2:scatter(2, torch.repeatTensor(y_arr:view(1,-1),3,1):view(3,1,-1), unsqueeze)  
            self.gt_normal_map[{batch_idx, {}}]:indexAdd(3, x_arr, p2)      


            -- fill the mask
            unsqueeze:fill(1)
            p2:scatter(2, torch.repeatTensor(y_arr:view(1,-1),3,1):view(3,1,-1), unsqueeze)  
            _gt_normal_mask[{batch_idx, {}}]:indexAdd(3, x_arr, p2)   
        end


        -- self.gt_normal_map[{1, {}}]:copy(gtnormal)

        self.gt_normal_mask = self.shift_mask:forward(_gt_normal_mask[{{},{1,1},{}}])
        -- self.gt_normal_mask:zero()
        -- self.gt_normal_mask[{{},{},{3,238},{3,318}}]:fill(1)


        
        -- obtain the predicted depth from normals, it produce a 4 dimensional depth, the number of channel is 4 in the second dimension
        self.normal_to_depth:forward({self.gt_normal_map, input:sub(n_depth+1, -1)})

        -- replicate the depth 4 times in the second dimension
        self.replicated_depth:resize(n_normal, 4 * input:size(2), input:size(3), input:size(4))
        self.replicated_depth:copy(nn.Replicate(4,2):cuda():forward(input:sub(n_depth+1, -1)))


        -- then go through the criterion, multiplied by the weight of the normal term
        self.__loss_normal = self.w_normal * self.normal_crit:forward( {self.normal_to_depth.output, self.gt_normal_mask}, self.replicated_depth)
        self.output = self.output + self.__loss_normal

        -- print(self.__loss_normal)
        -- io.read()
    end

    return self.output
end

function relative_depth_margin_log_normal_depth:updateGradInput(input, target)
    -- the input is tensor taht represents the depth map
    -- the target is a table, where the first component is a table that contains relative depth info, and the second component is a table that contains normal info.
    
    -- pre-allocate memory and reset gradient to 0
    if self.gradInput then
        local nElement = self.gradInput:nElement()        
        if self.gradInput:type() ~= input:type() then
            self.gradInput = self.gradInput:typeAs(input);
        end
        self.gradInput:resizeAs(input)
        self.gradInput:zero()
    end

    local n_depth = target[1].n_sample
    local n_normal = target[2].n_sample

    assert( torch.type(target) == 'table' );

    if n_depth > 0 then
        self.gradInput:sub(1, n_depth):copy( self.depth_crit:backward(nn.Log():cuda():forward(input:sub(1, n_depth)), target[1]) )   -- to test
        self.gradInput:sub(1, n_depth):copy( nn.Log():cuda():backward(input:sub(1, n_depth), self.gradInput:sub(1, n_depth)) )
    end
    if n_normal > 0 then        -- to test 
        
        local grad_wrt_depth_from_normal = self.normal_crit:backward( {self.normal_to_depth.output, self.gt_normal_mask}, self.replicated_depth)
        -- grad_depth1 is going to be an array, where the second dimension is the gradient wrt input:sub(n_depth+1, -1)
        local grad_depth1 = self.normal_to_depth:backward({self.gt_normal_map, input:sub(n_depth+1, -1)}, grad_wrt_depth_from_normal)
        -- grad_depth2 is going to be the gradient of the replicated depth, we need to accumulate it along the 2nd dimension
        local grad_depth2 = self.normal_crit:backward( {self.replicated_depth, self.gt_normal_mask}, self.normal_to_depth.output)
        grad_depth2 = torch.sum(grad_depth2, 2)     -- 

        -- add the gradient from two source
        self.gradInput:sub(n_depth+1, -1):copy( grad_depth1[2] )
        self.gradInput:sub(n_depth+1, -1):add( grad_depth2 )

        -- finally multipled by the weight
        self.gradInput:sub(n_depth+1, -1):mul(self.w_normal)
    end
   
    return self.gradInput
end