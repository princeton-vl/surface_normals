require 'image'

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

local function _classify(z_A, z_B, ground_truth, thresh)
    local _classify_res = 1;
    if z_A - z_B > thresh then
        _classify_res = 1
    elseif z_A - z_B < -thresh then
        _classify_res = -1
    elseif z_A - z_B <= thresh and z_A - z_B >= -thresh then
        _classify_res = 0;
    end

    if _classify_res == ground_truth then
        return true
    else
        return false
    end
end



local function _count_correct(output, target, record)
    
    for point_idx = 1, target.n_point do

        x_A = target.x_A[point_idx]
        y_A = target.y_A[point_idx]
        x_B = target.x_B[point_idx]
        y_B = target.y_B[point_idx]

        z_A = output[{1, 1, y_A, x_A}]
        z_B = output[{1, 1, y_B, x_B}]

        if x_A == x_B and y_A == y_B then
            print("Warning: _count_correct() has point pair that have the same coordinate!")
        end


        ground_truth = target.ordianl_relation[point_idx];    -- the ordinal_relation is in the form of 1 and -1 and 0

        for tau_idx = 1 , record.n_thresh do
            if _classify(z_A, z_B, ground_truth, record.thresh[tau_idx]) then
                
                if ground_truth == 0 then
                    record.eq_correct_count[tau_idx] = record.eq_correct_count[tau_idx] + 1;
                elseif ground_truth == 1 or ground_truth == -1 then
                    record.not_eq_correct_count[tau_idx] = record.not_eq_correct_count[tau_idx] + 1;
                end

            end
        end

        if ground_truth == 0 then
            record.eq_count = record.eq_count + 1
        elseif ground_truth == 1 or ground_truth == -1 then
            record.not_eq_count = record.not_eq_count + 1
        end

    end
    
end

function normalize_with_mean_std( input, mean, std )
    
    local normed_input = input:clone()
    normed_input = normed_input - torch.mean(normed_input);
    normed_input = normed_input / torch.std(normed_input);
    normed_input = normed_input * std;
    normed_input = normed_input + mean;
    
    -- remove and replace the depth value that are negative
    if torch.sum(normed_input:lt(0)) > 0 then
        -- fill it with the minimum of the non-negative plus a eps so that it won't be 0
        normed_input[normed_input:lt(0)] = torch.min(normed_input[normed_input:gt(0)]) + 0.00001
    end

    return normed_input
end

local function visualize_depth(z, filename)
    local _z_img = z:clone()
    _z_img = _z_img:add( - torch.min(_z_img) )
    _z_img = _z_img:div( torch.max(_z_img) )
    image.save(filename, _z_img) 
end


local function metric_error(gt_z, z)
    local resize_z = image.scale( z, 640, 480 )
    -- the input gt_z and z should be 2-dimensional tensor
    local std_of_NYU_training = 0.6148231626
    local mean_of_NYU_training = 2.8424594402

    local _crop = 16
    local gt_z = gt_z:sub(_crop,480-_crop,_crop,640-_crop)
    local resize_z = resize_z:sub(_crop,480-_crop,_crop,640-_crop) 
    
    local normed_NYU_z = normalize_with_mean_std( resize_z, mean_of_NYU_training, std_of_NYU_training )

    local fmse = torch.mean(torch.pow(gt_z - normed_NYU_z, 2))
    local flsi = torch.mean(  torch.pow(torch.log(normed_NYU_z)  - torch.log(gt_z) + torch.mean(torch.log(gt_z) - torch.log(normed_NYU_z)) , 2 )  )


    local gt_mean = torch.mean(gt_z)
    local gt_std = torch.std(gt_z - gt_mean)
    local normed_gt_z = normalize_with_mean_std( resize_z, gt_mean, gt_std )
    local fmse_si = torch.mean(torch.pow(gt_z - normed_gt_z, 2))

    -- -- debug
    -- visualize_depth(gt_z, './gt_z.png')
    -- visualize_depth(torch.abs(gt_z - normed_NYU_z), './normed_NYU_z.png')
    -- visualize_depth(torch.abs(gt_z - normed_gt_z), './normed_gt_z.png')

    -- print(torch.max(torch.abs(gt_z - normed_NYU_z)), torch.min(torch.abs(gt_z - normed_NYU_z)), torch.mean(torch.abs(gt_z - normed_NYU_z)))
    -- print(torch.max(torch.abs(gt_z - normed_gt_z)), torch.min(torch.abs(gt_z - normed_gt_z)), torch.mean(torch.abs(gt_z - normed_gt_z)))
    -- print("fmse=", math.sqrt(fmse))
    -- print("fmse_si=", math.sqrt(fmse_si))
    -- io.read()
    return fmse, flsi, fmse_si
end


local _eval_record = {}
_eval_record.n_thresh = 140;
_eval_record.eq_correct_count = torch.Tensor(_eval_record.n_thresh )
_eval_record.not_eq_correct_count = torch.Tensor(_eval_record.n_thresh )
_eval_record.not_eq_count = 0;
_eval_record.eq_count = 0;
_eval_record.thresh = torch.Tensor(_eval_record.n_thresh )
_eval_record.WKDR = torch.Tensor(_eval_record.n_thresh , 4)
local WKDR_step = 0.01


for i = 1, _eval_record.n_thresh  do
    _eval_record.thresh[i] = i * WKDR_step + 0.1
end



print(string.format(">>>>>> Validation: margin = %f, WKDR Step = %f", g_args.margin, WKDR_step))
print(_eval_record.thresh)


local function reset_record(record)
    record.eq_correct_count:fill(0)
    record.not_eq_correct_count:fill(0)
    record.WKDR:fill(0)
    record.not_eq_count = 0
    record.eq_count = 0
    -- print(record.eq_correct_count)
    -- print(record.not_eq_correct_count)
    -- print(record.not_eq_count)
    -- print(record.eq_count)
    -- io.read()
end


function evaluate( data_loader, model, criterion, max_n_sample ) 
    print('>>>>>>>>>>>>>>>>>>>>>>>>> Valid Crit Threshed: Evaluating on validation set...');

    print("Evaluate() Switch  On!!!")
    model:evaluate()    -- this is necessary
    
    -- reset the record
    reset_record(_eval_record)

    local total_depth_validation_loss = 0;
    local total_normal_validation_loss = 0;
    local total_angle_difference = 0;
    local n_depth_iters = math.min(data_loader.n_relative_depth_sample, max_n_sample);
    local n_normal_iters = math.min(data_loader.n_normal_sample, max_n_sample);
    local n_total_depth_point_pair = 0;    
    local n_total_normal_point = 0;    
    

    local fmse = torch.Tensor(n_depth_iters):fill(0)
    local fmse_si = torch.Tensor(n_depth_iters):fill(0)
    local flsi = torch.Tensor(n_depth_iters):fill(0) 


    print(string.format("Number of relative depth samples we are going to examine: %d", n_depth_iters))
    print(string.format("Number of normal samples we are going to examine: %d", n_normal_iters))

    for iter = 1, n_depth_iters do
        -- Get sample one by one
        local batch_input, batch_target = data_loader:load_indices(torch.Tensor({iter}), nil, true)   -- now only assume that we check on relative depth
        -- The first component is the relative depth target. Since there is only one sample, we just take its first sample.
        local relative_depth_target = batch_target[1][1]

        -- forward
        local batch_output = model:forward(batch_input)    
        local batch_loss = criterion:forward(batch_output, batch_target);   

        local output_depth = get_depth_from_model_output(batch_output)

        --------------------------------------------------
        ------------------WKDR
        --------------------------------------------------

        -- count the number of correct point pairs.
        _count_correct(output_depth, relative_depth_target, _eval_record)        

        -- get relative depth loss
        total_depth_validation_loss = total_depth_validation_loss + batch_loss * relative_depth_target.n_point        -- 

        -- update the number of point pair
        n_total_depth_point_pair = n_total_depth_point_pair + relative_depth_target.n_point

        --------------------------------------------------
        ------------------Mertric error
        --------------------------------------------------
        fmse[iter], flsi[iter], fmse_si[iter] = metric_error(batch_target[1].gt_depth[{1,{}}], output_depth[{1,1,{}}]:double())


        collectgarbage()        
    end   

    if nn.img_coord_to_world_coord == nil then
        
        require '../models/img_coord_to_world_coord'
    end

    if world_coord_to_normal == nil then
        require '../models/world_coord_to_normal'
    end



    local _depth_to_normal = nn.Sequential()
    _depth_to_normal:add(nn.img_coord_to_world_coord())
    _depth_to_normal:add(world_coord_to_normal())
    _depth_to_normal = _depth_to_normal:cuda()

    for iter = 1, n_normal_iters do
        -- Get sample one by one
        local batch_input, batch_target = data_loader:load_indices(nil, torch.Tensor({iter}), true)   -- now only assume that we check on normal
        -- The second component is the normal target. Since there is only one sample, we just take its first sample.
        local normal_target 

        if g_args.n_scale == 1 then
           normal_target  = batch_target[2][1]
        else
            normal_target  = batch_target[2][1][1]
        end

        -- forward
        local batch_output = model:forward(batch_input)   

        --------------------------------------------------
        ------------------Normal Loss
        --------------------------------------------------
        local batch_loss = criterion:forward(batch_output, batch_target);   
        local normal = _depth_to_normal:forward(batch_output)
        
        -- get the sum of angle difference
        local sum_angle_diff = angle_diff(normal[{1,{}}], normal_target)
        total_angle_difference = total_angle_difference + sum_angle_diff

        -- get thel loss 
        total_normal_validation_loss = total_normal_validation_loss + batch_loss * normal_target.n_point        -- 

        -- update the number of point pair
        n_total_normal_point = n_total_normal_point + normal_target.n_point


        --------------------------------------------------
        ------------------Mertric error
        --------------------------------------------------
        if n_depth_iters == 0 then      
            fmse[iter], flsi[iter], fmse_si[iter] = metric_error(batch_target[2].gt_depth[{1,{}}], batch_output[{1,1,{}}]:double())
        end
        collectgarbage()        
    end   

    print("Evaluate() Switch Off!!!")
    model:training()



    local max_min = 0;
    local max_min_i = 1;
    for tau_idx = 1 , _eval_record.n_thresh do
        _eval_record.WKDR[{tau_idx, 1}] = _eval_record.thresh[tau_idx]
        _eval_record.WKDR[{tau_idx, 2}] = (_eval_record.eq_correct_count[tau_idx] + _eval_record.not_eq_correct_count[tau_idx]) / (_eval_record.eq_count + _eval_record.not_eq_count)
        _eval_record.WKDR[{tau_idx, 3}] = _eval_record.eq_correct_count[tau_idx] / _eval_record.eq_count
        _eval_record.WKDR[{tau_idx, 4}] = _eval_record.not_eq_correct_count[tau_idx] / _eval_record.not_eq_count
        
        if math.min(_eval_record.WKDR[{tau_idx,3}], _eval_record.WKDR[{tau_idx,4}]) > max_min then
            max_min = math.min(_eval_record.WKDR[{tau_idx,3}], _eval_record.WKDR[{tau_idx,4}])
            max_min_i = tau_idx;
        end
    end

    print(_eval_record.WKDR)
    print(_eval_record.WKDR[{{max_min_i}, {}}])        
    print(string.format("\tEvaluation Completed. Average Relative Depth Loss = %f, WKDR = %f", total_depth_validation_loss / n_total_depth_point_pair, 1 - max_min))
    print(string.format("\tEvaluation Completed. Average Normal Loss = %f, Average Angular Difference = %f", total_normal_validation_loss / n_total_normal_point, total_angle_difference / n_total_normal_point))
    local rmse = math.sqrt(torch.mean(fmse))
    local rmse_si = math.sqrt(torch.mean(fmse_si))
    local lsi = math.sqrt(torch.mean(flsi))
    print(string.format('\trmse:\t%f',rmse))
    print(string.format('\trmse_si:%f',rmse_si))
    print(string.format('\tlsi:\t%f',lsi))
    --Return the relative depth loss per point pair, ERROR ratio(WKDR), the average normal loss, and the average angle difference between predicted and ground-truth normal
    return total_depth_validation_loss / n_total_depth_point_pair, 1 - max_min, total_normal_validation_loss / n_total_normal_point, total_angle_difference / n_total_normal_point, rmse, rmse_si, lsi
end