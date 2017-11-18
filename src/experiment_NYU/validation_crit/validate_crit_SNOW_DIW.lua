require 'image'

local function _is_correct(z_A, z_B, ground_truth)
    assert( ground_truth ~= 0, 'Warining: The ground_truth is not supposed to be 0.')
    local _classify_res = 1;
    if z_A > z_B then
        _classify_res = 1
    elseif z_A < z_B then
        _classify_res = -1    
    end

    if _classify_res == ground_truth then
        return true
    else
        return false
    end
end

local function _count_correct(output, target)
    assert(output:size(1) == 1)
    y_A = target.y_A[1]
    x_A = target.x_A[1]
    y_B = target.y_B[1]
    x_B = target.x_B[1]    

    z_A = output[{1, 1, y_A, x_A}]
    z_B = output[{1, 1, y_B, x_B}]

    assert(x_A ~= x_B or y_A ~= y_B)

    ground_truth = target.ordianl_relation[1];    -- the ordinal_relation is in the form of 1 and -1


    if _is_correct(z_A, z_B, ground_truth) then
        return 1
    else
        return 0
    end
end


local function angle_diff(input, target)
    -- the input should have 3 dimension
    local x_arr = target.x
    local y_arr = target.y        

    local normal_arr = input:index(3, x_arr):gather(2,  torch.repeatTensor(y_arr:view(1,-1),3,1):view(3,1,-1)  ):squeeze()        
    local ground_truth_arr = target.normal

    -- to test
    local cos = torch.sum(torch.cmul(normal_arr, ground_truth_arr),1)      -- dot product of normals , seems quite expensive move
    local mask_gt1 = torch.gt(cos, 1)
    cos:maskedFill(mask_gt1, 1)
    local mask_lt_1 = torch.lt(cos, -1)
    cos:maskedFill(mask_lt_1, -1)

    local acos = cos:acos()

    if torch.sum(mask_gt1) > 0 then
        print(">>>>>>>> Greater than 1 cos!")
        print(cos:maskedSelect(mask_gt1))
    end
    if torch.sum(mask_lt_1) > 0 then
        print(">>>>>>>> Less than -1 cos!")
        print(cos:maskedSelect(mask_lt_1))
    end 
    
    return torch.sum(acos)
end


local function visualize_depth(z, filename)
    local _z_img = z:clone()
    _z_img = _z_img:add( - torch.min(_z_img) )
    _z_img = _z_img:div( torch.max(_z_img) )
    image.save(filename, _z_img) 
end


function evaluate( data_loader, model, criterion, max_n_sample ) 
    print('>>>>>>>>>>>>>>>>>>>>>>>>> Valid Crit SNOW DIW: Evaluating on validation set...');

    print("Evaluate() Switch  On!!!")
    model:evaluate()    -- this is necessary
    
    local total_depth_validation_loss = 0;
    local total_normal_validation_loss = 0;
    local total_angle_difference = 0;
    local n_depth_iters = math.min(data_loader.n_relative_depth_sample, max_n_sample);
    local n_normal_iters = math.min(data_loader.n_normal_sample, max_n_sample);
    local n_total_depth_point_pair = 0;    
    local n_total_normal_point = 0;    


    local correct_count = 0;
    
    print(string.format("Number of relative depth samples we are going to examine: %d", n_depth_iters))
    print(string.format("Number of normal samples we are going to examine: %d", n_normal_iters))

    for iter = 1, n_depth_iters do
         -- Get sample one by one
        local batch_input, batch_target = data_loader:load_indices(torch.Tensor({iter}), nil, false)   -- to do: now only assume that we check on relative depth, what about normal??         
        -- The first component is the relative depth target. Since there is only one sample, we just take its first sample.
        local relative_depth_target = batch_target[1][1]

        -- forward
        local batch_output = model:forward(batch_input)    
        local batch_loss = criterion:forward(batch_output, batch_target);   

        local output_depth = get_depth_from_model_output(batch_output)

        -- check this output
        local _n_point_correct = _count_correct(output_depth, relative_depth_target)
        
        -- get validation loss and correct ratio
        total_depth_validation_loss = total_depth_validation_loss + batch_loss 
        correct_count = correct_count + _n_point_correct
        n_total_depth_point_pair = n_total_depth_point_pair + 1

        collectgarbage()
    end   

    for iter = 1, n_normal_iters do
        -- Get sample one by one
        local batch_input, batch_target = data_loader:load_indices(nil, torch.Tensor({iter}), false)   -- now only assume that we check on normal
        -- The second component is the normal target. Since there is only one sample, we just take its first sample.
        local normal_target = batch_target[2][1]

        -- forward
        local batch_output = model:forward(batch_input)   

        --------------------------------------------------
        ------------------Normal Loss
        --------------------------------------------------
        local batch_loss = criterion:forward(batch_output, batch_target);   
        local normal = criterion.depth_to_normal.output
        
        -- get the sum of angle difference
        local sum_angle_diff = angle_diff(normal[{1,{}}], normal_target)
        total_angle_difference = total_angle_difference + sum_angle_diff

        -- get thel loss 
        total_normal_validation_loss = total_normal_validation_loss + batch_loss * normal_target.n_point        -- 

        -- update the number of point pair
        n_total_normal_point = n_total_normal_point + normal_target.n_point


        collectgarbage()        
    end   

    print("Evaluate() Switch Off!!!")
    model:training()

    local WHDR = 1 - correct_count / n_total_depth_point_pair
    print("Evaluation result: WHDR = ", WHDR)
    --Return the relative depth loss per point pair, ERROR ratio(WKDR), the average normal loss, and the average angle difference between predicted and ground-truth normal
    return total_depth_validation_loss / n_total_depth_point_pair, WHDR, total_normal_validation_loss / n_total_normal_point, total_angle_difference / n_total_normal_point, 0, 0, 0
end