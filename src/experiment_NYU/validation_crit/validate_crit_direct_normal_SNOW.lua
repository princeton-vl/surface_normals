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

local function visualize_depth(z, filename)
    local _z_img = z:clone()
    _z_img = _z_img:add( - torch.min(_z_img) )
    _z_img = _z_img:div( torch.max(_z_img) )
    image.save(filename, _z_img) 
end


local _eval_record = {}
_eval_record.n_thresh = 15;
_eval_record.eq_correct_count = torch.Tensor(_eval_record.n_thresh )
_eval_record.not_eq_correct_count = torch.Tensor(_eval_record.n_thresh )
_eval_record.not_eq_count = 0;
_eval_record.eq_count = 0;
_eval_record.thresh = torch.Tensor(_eval_record.n_thresh )
_eval_record.WKDR = torch.Tensor(_eval_record.n_thresh , 4)
local WKDR_step = 0.1
if g_args.margin <= 0.05 then
    WKDR_step = 0.01
else
    WKDR_step = 0.1
end

for i = 1, _eval_record.n_thresh  do
    _eval_record.thresh[i] = i * WKDR_step
end

print(string.format(">>>>>> Validation: margin = %f, WKDR Step = %f", g_args.margin, WKDR_step))

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
    print('>>>>>>>>>>>>>>>>>>>>>>>>> Valid Crit pure normal: Evaluating on validation set...');

    print("Evaluate() Switch  On!!!")
    model:evaluate()    -- this is necessary
    
    -- reset the record
    reset_record(_eval_record)


    local total_depth_validation_loss = 0;
    local total_normal_validation_loss = 0;
    local total_angle_difference = 0;
    local n_normal_iters = math.min(data_loader.n_normal_sample, max_n_sample);
    local n_total_depth_point_pair = 0;    
    local n_total_normal_point = 0;    
    

    local fmse = torch.Tensor(max_n_sample):fill(0)
    local fmse_si = torch.Tensor(max_n_sample):fill(0)
    local flsi = torch.Tensor(max_n_sample):fill(0) 


    print(string.format("Number of normal samples we are going to examine: %d", n_normal_iters))

    for iter = 1, n_normal_iters do
        -- Get sample one by one
        local batch_input, batch_target = data_loader:load_indices(nil, torch.Tensor({iter}), false)   -- now only assume that we check on normal
        -- The second component is the normal target. Since there is only one sample, we just take its first sample.
        local normal_target = batch_target[2]


        -- forward
        local batch_output = model:forward(batch_input)   
        --------------------------------------------------
        ------------------Normal Loss
        --------------------------------------------------
        local batch_loss = criterion:forward(batch_output, normal_target);  
        local normal = batch_output

        -- get the sum of angle difference
        local sum_angle_diff = angle_diff(normal[{1,{}}], normal_target[1])
        total_angle_difference = total_angle_difference + sum_angle_diff
        -- get the loss 
        total_normal_validation_loss = total_normal_validation_loss + batch_loss        -- 

        -- update the number of point pair
        n_total_normal_point = n_total_normal_point + 1


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