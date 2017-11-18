require 'nn'
require 'image'
require 'csvigo'
require 'hdf5'
require 'measure'

local network_input_height = 240
local network_input_width = 320

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

local function _read_data_handle( _filename )        
    -- the file is a csv file
    local csv_file_handle = csvigo.load({path = _filename, mode = 'large'})
    
    local _n_lines = #csv_file_handle - 1;  -- minus 1 because the first line is invalid


    local _data ={}
    local _line_idx = 2 --skip the first line
    local _sample_idx = 0
    while _line_idx <= _n_lines do

        _sample_idx = _sample_idx + 1
        

        _data[_sample_idx] = {};
        _data[_sample_idx].img_filename = csv_file_handle[ _line_idx ][ 1 ]    
        _data[_sample_idx].n_point = tonumber(csv_file_handle[ _line_idx ][ 3 ])



        _data[_sample_idx].y_A = {}
        _data[_sample_idx].y_B = {}
        _data[_sample_idx].x_A = {}
        _data[_sample_idx].x_B = {}
        _data[_sample_idx].ordianl_relation = {}       
        
        --print(string.format('n_point = %d',_data[_sample_idx].n_point ))        
        --io.read()

        _line_idx = _line_idx + 1 
        for point_idx = 1 , _data[_sample_idx].n_point do
            
            _data[_sample_idx].y_A[point_idx] = tonumber(csv_file_handle[ _line_idx ][ 1 ])
            _data[_sample_idx].x_A[point_idx] = tonumber(csv_file_handle[ _line_idx ][ 2 ])
            _data[_sample_idx].y_B[point_idx] = tonumber(csv_file_handle[ _line_idx ][ 3 ])
            _data[_sample_idx].x_B[point_idx] = tonumber(csv_file_handle[ _line_idx ][ 4 ])

             -- Important!
            if csv_file_handle[ _line_idx ][ 5 ] == '>' then
                _data[_sample_idx].ordianl_relation[point_idx] = 1;
            elseif csv_file_handle[ _line_idx ][ 5 ] == '<' then
                _data[_sample_idx].ordianl_relation[point_idx] = -1;    
            elseif csv_file_handle[_line_idx][ 5 ] == '=' then        -- important!
                _data[_sample_idx].ordianl_relation[point_idx] = 0;
            end

            --print(string.format('%d, %d, %d, %d, %d',_data[_sample_idx].y_A[point_idx], _data[_sample_idx].x_A[point_idx], _data[_sample_idx].y_B[point_idx], _data[_sample_idx].x_B[point_idx],_data[_sample_idx].ordianl_relation[point_idx]))
            --io.read()

            _line_idx = _line_idx + 1 
        end                 

        --print(string.format('line_idx = %d',_line_idx))
        --io.read()
    end
--[[------------------------------------------
            Need Debug

    print('_n_samples = ',#_data,'_sample_idx =',_sample_idx)
    io.read()   
]]--------------------------------------------

    return _sample_idx, _data;
end




local function _evaluate_correctness_eigen(_batch_output, _batch_target)

    local n_gt_correct = 0;
    local n_gt = 0;

    local n_lt_correct = 0;
    local n_lt = 0;

    local n_eq_correct = 0;
    local n_eq = 0;

    for point_idx = 1, _batch_target.n_point do

        x_A = _batch_target.x_A[point_idx]
        y_A = _batch_target.y_A[point_idx]
        x_B = _batch_target.x_B[point_idx]
        y_B = _batch_target.y_B[point_idx]

        z_A = _batch_output[{1, y_A, x_A}]
        z_B = _batch_output[{1, y_B, x_B}]
        
        ground_truth = _batch_target.ordianl_relation[point_idx];    -- the ordianl_relation is in the form of 1 and -1

        local _classify_res;
        if z_A / z_B > 1.02 then
            _classify_res = 1
        elseif z_B / z_A > 1.02 then
            _classify_res = -1
        elseif z_A / z_B <= 1.02 and z_B / z_A < 1.02 then
            _classify_res = 0;
        end

        if _classify_res == 0 and ground_truth == 0 then
            n_eq_correct = n_eq_correct + 1;
        elseif _classify_res == 1 and ground_truth == 1 then
            n_gt_correct = n_gt_correct + 1;
        elseif _classify_res == -1 and ground_truth == -1 then
            n_lt_correct = n_lt_correct + 1;
        end

        
        if ground_truth > 0 then
            n_gt = n_gt + 1;
        elseif ground_truth < 0 then
            n_lt = n_lt + 1;
        elseif ground_truth == 0 then
            n_eq = n_eq + 1;
        end
    end

    -- WKDR[i], WKDR_eq[i], WKDR_neq[i] 
    a = (n_eq_correct + n_gt_correct + n_lt_correct) / (n_eq + n_gt + n_lt)
    b = n_eq_correct / n_eq
    c = (n_gt_correct + n_lt_correct) / (n_gt + n_lt)
    -- print(string.format('error_sum_ie = %d, weight_sum_ie = %d, error_sum_eq = %d, weight_sum_eq = %d', (n_gt + n_lt) - (n_gt_correct + n_lt_correct), (n_gt + n_lt),  n_eq - n_eq_correct, n_eq ));
    return 1 - a, 1 - b, 1 - c
end

local function inpaint_pad_output_eigen(output, img_orig_w, img_orig_h)
    -- assert(output:size(2) == 109)
    -- assert(output:size(3) == 147)
    -- assert(img_orig_w == 640)
    -- assert(img_orig_h == 480)

    -- -- [11 .. 227) * 2 = [22 .. 454) = [22 .. 453]   height
    -- -- [13 .. 305) * 2 = [26 .. 610) = [26 .. 609]   width

    -- local resize_height = 453 - 22 + 1
    -- local resize_width = 609 - 26 + 1
    -- local resize_output = image.scale( output[{1,{}}]:double(), resize_width, resize_height)
    

    -- local padded_output1 = torch.Tensor(1, resize_height, img_orig_w);
    -- padded_output1[{{},{}, {26, 609}}]:copy(resize_output)
    -- -- pad left 
    -- for i = 1 , 25 do
    --     padded_output1[{1,{}, i}]:copy(resize_output[{{},1}])                
    -- end
    -- -- pad right
    -- for i = 610 , 640 do
    --     padded_output1[{1,{}, i}]:copy(resize_output[{{},resize_width}])                
    -- end


    -- local padded_output2 = torch.Tensor(1, img_orig_h, img_orig_w);
    -- padded_output2[{{},{22, 453}, {}}]:copy(padded_output1)
    -- -- pad top and down    
    -- for i = 1 , 21 do
    --     padded_output2[{1, i, {}}]:copy(padded_output1[{1, 1,{}}])        
    -- end
    -- for i = 454, 480 do
    --     padded_output2[{1, i, {}}]:copy(padded_output1[{1, resize_height,{}}])
    -- end


    -- return padded_output2


    local final_output = torch.Tensor(1, 480, 640);
    final_output[{1,{}}]:copy(image.scale(output, 640, 480))
    -- -- debug!!
    -- _pred_z_img = final_output:clone()
    -- _pred_z_img = _pred_z_img:add( - torch.min(_pred_z_img) )
    -- _pred_z_img = _pred_z_img:div( torch.max(_pred_z_img) )
    -- image.save( 'debug.png', _pred_z_img)
    -- io.read()
    return final_output
end


local function _evaluate_WKDR(_batch_output, _batch_target, WKDR, WKDR_eq, WKDR_neq)
    local n_gt_correct = torch.Tensor(n_thresh):fill(0);
    local n_gt = 0;

    local n_lt_correct = torch.Tensor(n_thresh):fill(0);
    local n_lt = 0;

    local n_eq_correct = torch.Tensor(n_thresh):fill(0);
    local n_eq = 0;

    for point_idx = 1, _batch_target.n_point do

        x_A = _batch_target.x_A[point_idx]
        y_A = _batch_target.y_A[point_idx]
        x_B = _batch_target.x_B[point_idx]
        y_B = _batch_target.y_B[point_idx]

        z_A = _batch_output[{y_A, x_A}]
        z_B = _batch_output[{y_B, x_B}]
        

        ground_truth = _batch_target.ordianl_relation[point_idx];    -- the ordianl_relation is in the form of 1 and -1

        for thresh_idx = 1, n_thresh do

            local _classify_res = 1;
            if z_A - z_B > thresh[thresh_idx] then
                _classify_res = 1
            elseif z_A - z_B < -thresh[thresh_idx] then
                _classify_res = -1
            elseif z_A - z_B <= thresh[thresh_idx] and z_A - z_B >= -thresh[thresh_idx] then
                _classify_res = 0;
            end

            if _classify_res == 0 and ground_truth == 0 then
                n_eq_correct[thresh_idx] = n_eq_correct[thresh_idx] + 1;
            elseif _classify_res == 1 and ground_truth == 1 then
                n_gt_correct[thresh_idx] = n_gt_correct[thresh_idx] + 1;
            elseif _classify_res == -1 and ground_truth == -1 then
                n_lt_correct[thresh_idx] = n_lt_correct[thresh_idx] + 1;
            end      
        end

        if ground_truth > 0 then
            n_gt = n_gt + 1;
        elseif ground_truth < 0 then
            n_lt = n_lt + 1;
        elseif ground_truth == 0 then
            n_eq = n_eq + 1;
        end
    end



    for i = 1 , n_thresh do        
        WKDR[{i}] = 1 - (n_eq_correct[i] + n_lt_correct[i] + n_gt_correct[i]) / (n_eq + n_lt + n_gt)
        WKDR_eq[{i}] = 1 - n_eq_correct[i]  / n_eq
        WKDR_neq[{i}] = 1 - (n_lt_correct[i] + n_gt_correct[i]) / (n_lt + n_gt)
    end
    -- -- for debugging and comparting result with the MIT code
    --print(string.format('threshold = %f', thresh[21]))
    --print(string.format('WKDR=%f, WKDR_eq=%f,WKDR_neq=%f', WKDR[{21}], WKDR_eq[{21}], WKDR_neq[{21}]));
    --print(string.format('n_eq_error=%d, neq=%d, n_ieq_error=%d,n_ieq=%f', n_eq - n_eq_correct[21], n_eq, n_lt + n_gt - n_lt_correct[21] - n_gt_correct[21], n_lt+n_gt));
    --io.read()
end

function smooth_depth_map(input)
    assert(input:nDimension() == 2)
    local gauss_kernel = image.gaussian(5,1,1,true)    
    local output = image.convolve(input:double(), gauss_kernel, 'same')
    return output:cuda()
end


function metric_error(gtz, z)
    local fmse = torch.mean(torch.pow(gtz - z, 2))
    local fmselog = torch.mean(torch.pow(torch.log(gtz) - torch.log(z), 2))
    local flsi = torch.mean(  torch.pow(torch.log(z)  - torch.log(gtz) + torch.mean(torch.log(gtz) - torch.log(z)) , 2 )  )
    local fabsrel = torch.mean( torch.cdiv(torch.abs( z - gtz ), gtz ))
    local fsqrrel = torch.mean( torch.cdiv(torch.pow( z - gtz ,2), gtz ))

    return fmse, fmselog, flsi, fabsrel, fsqrrel
end

function find_least_square_scale_shift_XYZ(src_XYZ, dst_XYZ)
    -- -- debug
    -- src_XYZ = torch.Tensor({{1,2,3}, {4,5,6}, {7,8,9}, {10,11,12}})
    -- dst_XYZ = torch.Tensor({{3,6,9}, {9,12,15}, {15,18,21}, {21,24,27}})

    -- src_XYZ is a n x 3 matrix, so is dst_XYZ
    local n_point = src_XYZ:size(1)

    if n_point == 1 then
        local transformed_XYZ = dst_XYZ:clone()
        return transformed_XYZ
    end

    local XTX_11 = torch.sum(torch.pow(src_XYZ, 2))
    local XTX_12 = torch.sum(src_XYZ[{{},1}])
    local XTX_13 = torch.sum(src_XYZ[{{},2}])
    local XTX_14 = torch.sum(src_XYZ[{{},3}])

    local XTY_1 = torch.sum(torch.cmul(src_XYZ, dst_XYZ))
    local XTY_2 = torch.sum(dst_XYZ[{{},1}])
    local XTY_3 = torch.sum(dst_XYZ[{{},2}])
    local XTY_4 = torch.sum(dst_XYZ[{{},3}])

    local XTX = torch.Tensor({{XTX_11, XTX_12, XTX_13, XTX_14},{XTX_12, n_point, 0, 0}, {XTX_13, 0, n_point, 0},{XTX_14, 0, 0, n_point}})
    local XTY = torch.Tensor({{XTY_1},{XTY_2},{XTY_3},{XTY_4}})

    local inv_XTX = torch.inverse(XTX)
    local solution = torch.mm(inv_XTX, XTY)



    local transformed_XYZ = src_XYZ:clone()
    transformed_XYZ[{{},1}]:mul(solution[{1,1}]):add(solution[{2,1}])
    transformed_XYZ[{{},2}]:mul(solution[{1,1}]):add(solution[{3,1}])
    transformed_XYZ[{{},3}]:mul(solution[{1,1}]):add(solution[{4,1}])

    return transformed_XYZ
end

function find_least_square_scale(src_z, dst_z)
    local transformed_z = src_z:clone()

    local src_mean = torch.mean(src_z)
    local ne_mask = torch.ne(src_z, src_mean)


    if torch.sum(ne_mask) < src_z:nElement() then
        local dst_mean = torch.mean(dst_z)
        transformed_z = transformed_z - src_mean + dst_mean
    else
        local s_T_s = torch.sum(torch.pow(src_z, 2))
        local s_T_d = torch.sum(torch.cmul(src_z, dst_z))
        local solution = s_T_d / s_T_s

        transformed_z:mul(solution)
    end

    return transformed_z
end

function find_least_square_scale_shift(src_z, dst_z)
    local transformed_z = src_z:clone()

    local src_mean = torch.mean(src_z)
    local ne_mask = torch.ne(src_z, src_mean)

    if torch.sum(ne_mask) < src_z:nElement() then
        local dst_mean = torch.mean(dst_z)
        transformed_z = transformed_z - src_mean + dst_mean
    else
        local s_T_s = torch.sum(torch.pow(src_z, 2))
        local s_T_1 = torch.sum(src_z)
        local s_T_d = torch.sum(torch.cmul(src_z, dst_z))
        local d_T_1 = torch.sum(dst_z)
        local n_element = src_z:nElement()

        local XTX = torch.Tensor({{s_T_s, s_T_1},{s_T_1, n_element}})
        local XTY = torch.Tensor({{s_T_d},{d_T_1}})

        local inv_XTX = torch.inverse(XTX)
        local solution = torch.mm(inv_XTX, XTY)
        
        
        transformed_z:mul(solution[{1,1}])
        transformed_z:add(solution[{2,1}])
    end

    return transformed_z
end

function _f_img_coord_to_world_coord_480_640(z)
    local _fx_rgb = 5.1885790117450188e+02;
    local _fy_rgb = -5.1946961112127485e+02;
    local _cx_rgb = 3.2558244941119034e+02;
    local _cy_rgb = 2.5373616633400465e+02;

    local mesh_grid_X = torch.Tensor(480, 640)
    local mesh_grid_Y = torch.Tensor(480, 640)
    for y = 1 , 480 do               -- to test
       for x = 1 , 640 do
          mesh_grid_X[{y,x}] = (x - _cx_rgb) / _fx_rgb
          mesh_grid_Y[{y,x}] = (y - _cy_rgb) / _fy_rgb
       end
    end 
    local XYZ = torch.Tensor(3,480,640)
    XYZ[{1,{}}]:copy(z)
    XYZ[{2,{}}]:copy(z)
    XYZ[{3,{}}]:copy(z)

    XYZ[{1,{}}]:cmul(mesh_grid_X)
    XYZ[{2,{}}]:cmul(mesh_grid_Y)

    return XYZ
end

function _select_masked_XYZ(XYZ, mask_2D)
    local n_points = XYZ[{1,{}}]:maskedSelect(mask_2D):nElement()
    local XYZ_masked = torch.Tensor(n_points, 3)

    XYZ_masked[{{},1}]:copy( XYZ[{1,{}}]:maskedSelect(mask_2D) )
    XYZ_masked[{{},2}]:copy( XYZ[{2,{}}]:maskedSelect(mask_2D) )
    XYZ_masked[{{},3}]:copy( XYZ[{3,{}}]:maskedSelect(mask_2D) )

    return XYZ_masked
end

function save_mesh_XYZ(filename, XYZ)
    local file = io.open(filename, "w")
    local n_point = XYZ:size(1)
    for i = 1, n_point do
        file:write(string.format("v %f %f %f\n", XYZ[{i, 1}], XYZ[{i, 2}], -XYZ[{i, 3}]))
    end    
    io.close(file)
end

function object_scale_invariant_error(gtz, pred_z, filename)
    -- get 3D point cloud
    local gt_XYZ = _f_img_coord_to_world_coord_480_640(gtz)
    local pred_XYZ = _f_img_coord_to_world_coord_480_640(pred_z)

    -- local temp = torch.Tensor(1,3,480,640)
    -- -- temp[{1,{}}]:copy(gt_XYZ)
    -- -- save_mesh("gt_XYZ.obj",temp)
    -- temp[{1,{}}]:copy(pred_XYZ)
    -- save_mesh("pred_XYZ.obj",temp:sub(1,1,1,3,16,480-16,16, 640-16))
    -- io.read()

    -- read in the masks for this test case
    local mask_h5filename = string.gsub(filename, '.png', '_object_masks.h5')
    local myFile = hdf5.open(mask_h5filename, 'r')
    local mask = myFile:read('/object_mask'):all()
    myFile:close()
    mask = mask:transpose(2,3);
    mask = torch.gt(mask, 0);

    local n_mask = mask:size(1)
    
    local _crop = 16
    local gt_XYZ_cropped = gt_XYZ:sub(1,3,45,471,41,601)
    local pred_XYZ_cropped = pred_XYZ:sub(1,3,45,471,41,601)

    
    local __transformed_XYZ = torch.Tensor({{0,0,0}})
    local __pred_XYZ = torch.Tensor({{0,0,0}})
    local __gt_XYZ = torch.Tensor({{0,0,0}})
    local s_i_error_sum = 0
    local pixel_count = 0
    for i = 1, n_mask do
        local mask_cropped = mask[{i,{}}]:sub(45,471,41,601)
        local n_points = gt_XYZ_cropped[{1,{}}]:maskedSelect(mask_cropped):nElement()
        -----------------------------------------------------------------------
        --              Find the least square error
        -----------------------------------------------------------------------
        if n_points > 0 then
            local gt_XYZ_obj = _select_masked_XYZ(gt_XYZ_cropped, mask_cropped)
            local pred_XYZ_obj = _select_masked_XYZ(pred_XYZ_cropped, mask_cropped)

            local transformed_XYZ_obj = find_least_square_scale_shift_XYZ(pred_XYZ_obj, gt_XYZ_obj)
            __transformed_XYZ = torch.cat(__transformed_XYZ, transformed_XYZ_obj, 1)
            __pred_XYZ = torch.cat(__pred_XYZ, pred_XYZ_obj, 1)
            __gt_XYZ = torch.cat(__gt_XYZ, gt_XYZ_obj, 1)

            -- get error
            pixel_count = pixel_count + n_points
            s_i_error_sum = s_i_error_sum + torch.sum(torch.pow(gt_XYZ_obj - transformed_XYZ_obj, 2)) 
        end
    end


                -- save_mesh_XYZ("gt_XYZ_obj.obj", __gt_XYZ)
                -- save_mesh_XYZ("pred_XYZ_obj.obj", __pred_XYZ)
                -- save_mesh_XYZ("transformed_XYZ_obj.obj", __transformed_XYZ)
                -- print("Done")
                -- io.read()


    return s_i_error_sum / pixel_count
end



function img_scale_invariant_error(gtz, pred_z)
    -- read in the masks for this test case
    local mask = torch.ByteTensor(gtz:size(1), gtz:size(2)):fill(1)
    local gtz_img = gtz:maskedSelect(mask) 
    local pred_z_img = pred_z:maskedSelect(mask)

    -----------------------------------------------------------------------
    --              Find the least square error
    -----------------------------------------------------------------------
    pred_z_img = find_least_square_scale_shift(pred_z, gtz_img)

    -- get error
    local s_i_error_sum = torch.sum(torch.pow(gtz_img - pred_z_img, 2))    

    return s_i_error_sum / gtz_img:nElement()
end

function read_normal(img_name)
    local n_point = 50000
    local normal_name = string.gsub(img_name, ".png", "_normal.bin")
    local file = torch.DiskFile(normal_name, 'r'):binary();    
    local normal = torch.DoubleTensor(file:readDouble(5 * n_point))
    file:close();                
    normal = torch.view(normal, n_point, 5)                 -- tested
    normal = normal:t()
    

    -- pay atteniton to the order!
    local record = {}
    record.x = torch.Tensor()
    record.y = torch.Tensor()
    record.x:resize(n_point):copy(normal[{1,{}}]:int())
    record.x = record.x:long()
    record.y:resize(n_point):copy(normal[{2,{}}]:int()):int()
    record.y = record.y:long()
    record.normal = normal[{{3,5},{}}]:clone()       -- To check.  Do we need to clone()? 
    record.n_point = n_point

    return record
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

    -- if torch.sum(mask_gt1) > 0 then
    --     print(">>>>>>>> Greater than 1 cos!")
    --     print(cos:maskedSelect(mask_gt1))
    -- end
    -- if torch.sum(mask_lt_1) > 0 then
    --     print(">>>>>>>> Less than -1 cos!")
    --     print(cos:maskedSelect(mask_lt_1))
    -- end 
    
    return torch.sum(acos)
end



function normalize_output_depth_with_NYU_mean_std( input )
    local std_of_NYU_training = 0.6148231626
    local mean_of_NYU_training = 2.8424594402
    
    local predicted_z_4D = input:clone()
    predicted_z_4D = predicted_z_4D - torch.mean(predicted_z_4D);
    predicted_z_4D = predicted_z_4D / torch.std(predicted_z_4D);
    predicted_z_4D = predicted_z_4D * std_of_NYU_training;
    predicted_z_4D = predicted_z_4D + mean_of_NYU_training;
    
    -- remove and replace the depth value that are negative
    if torch.sum(predicted_z_4D:lt(0)) > 0 then
        -- fill it with the minimum of the non-negative plus a eps so that it won't be 0
        predicted_z_4D[predicted_z_4D:lt(0)] = torch.min(predicted_z_4D[predicted_z_4D:gt(0)]) + 0.00001
    end

    return predicted_z_4D
end


function normalize_output_depth_with_gtz( input, gt_z)
    
    local gt_mean = torch.mean(gt_z)
    local gt_std = torch.std(gt_z - gt_mean)
    
    local predicted_z_4D = input:clone()
    predicted_z_4D = predicted_z_4D - torch.mean(predicted_z_4D);
    predicted_z_4D = predicted_z_4D / torch.std(predicted_z_4D);
    predicted_z_4D = predicted_z_4D * gt_std;
    predicted_z_4D = predicted_z_4D + gt_mean;
    
    -- remove and replace the depth value that are negative
    if torch.sum(predicted_z_4D:lt(0)) > 0 then
        -- fill it with the minimum of the non-negative plus a eps so that it won't be 0
        predicted_z_4D[predicted_z_4D:lt(0)] = torch.min(predicted_z_4D[predicted_z_4D:gt(0)]) + 0.00001
    end

    return predicted_z_4D
end


function save_mesh(filename, XYZ)
    local file = io.open(filename, "w")
    local img_height = XYZ:size(3)
    local img_width = XYZ:size(4)
    for y = 1, img_height do
        for x = 1, img_width do
            file:write(string.format("v %f %f %f\n", XYZ[{1, 1, y, x}], XYZ[{1, 2, y, x}], -XYZ[{1, 3, y, x}]))
        end
    end

    for y = 1, img_height - 1 do
        for x = 1, img_width - 1 do
            local this_index = ( y - 1 ) * img_width + x
            file:write(string.format("f %d %d %d\n", this_index, this_index + img_width, this_index + 1))
            file:write(string.format("f %d %d %d\n", this_index + img_width, this_index + img_width + 1, this_index + 1))
        end
    end

    io.close(file)
    print("Done saving mesh to " .. filename)
end

local function load_hdf5_z(h5_filename, field_name)
    local myFile = hdf5.open(h5_filename, 'r')
    local read_result = myFile:read(field_name):all()
    myFile:close()

    return read_result
end
----------------------------------------------[[
--[[

Main Entry

]]--
------------------------------------------------



cmd = torch.CmdLine()
cmd:text('Options')

cmd:option('-num_iter',1,'number of training iteration')
cmd:option('-prev_model_file','','Absolute / relative path to the previous model file. Resume training from this file')
cmd:option('-vis', false, 'visualize output')
cmd:option('-mesh', false, 'visualize output')
cmd:option('-output_folder','./output_imgs','image output folder')
cmd:option('-mode','validate','mode: test or validate')
cmd:option('-valid_set', '45_NYU_validate_imgs_points_resize_240_320.csv', 'validation file name');
-- cmd:option('-test_set','654_NYU_test_imgs_orig_size_points.csv', 'test file name');
cmd:option('-test_set','654_NYU_MITpaper_test_imgs_orig_size_points.csv', 'test file name');
cmd:option('-model','ours', 'eigen15 ,Chakrabarti16 or ours')
cmd:option('-si_obj_err', false, 'evaluate scale invaraint error at object level')
cmd_params = cmd:parse(arg)



if cmd_params.mode == 'test' then
    csv_file_name = '../../data/' .. cmd_params.test_set       -- test set
elseif cmd_params.mode == 'validate' then
    csv_file_name = '../../data/' .. cmd_params.valid_set        -- validation set 1001 images       
end
preload_t7_filename = string.gsub(csv_file_name, "csv", "t7")




f=io.open(preload_t7_filename,"r")
if f == nil then
    print('loading csv file...')
    n_sample, data_handle = _read_data_handle( csv_file_name )
    torch.save(preload_t7_filename, data_handle)
else
    io.close(f)
    print('loading pre load t7 file...')
    data_handle = torch.load(preload_t7_filename)
    n_sample = #data_handle
end



print("Hyper params: ")
print("csv_file_name:", csv_file_name);
print("N test samples:", n_sample);

n_iter = math.min( n_sample, cmd_params.num_iter )
print(string.format('n_iter = %d',n_iter))



if true then

    require 'cudnn'
    require 'cunn'
    require 'cutorch'
    -- depth to normal
    require './models/world_coord_to_normal'
    require './models/img_coord_to_world_coord'
    require './models/img_coord_to_world_coord_focal'

    local depth_to_world_coord_network = nn.img_coord_to_world_coord():cuda()
    local depth_to_world_coord_network_focal = nn.img_coord_to_world_coord_focal():cuda()
    local world_coord_to_normal_network = world_coord_to_normal():cuda()

    local measures = {}
    measures['normal'] = {}
    local average = {}
    average['normal'] = {}
    local n_pixels = {}

    -- Load the model
    prev_model_file = cmd_params.prev_model_file
    model = torch.load(prev_model_file)
    model:evaluate()
    print("Model file:", prev_model_file)


    
    _batch_input_cpu = torch.Tensor(1,3,network_input_height,network_input_width)



    n_thresh = 250;
    thresh = torch.Tensor(n_thresh);
    for i = 1, n_thresh do
        thresh[i] = 0.1 + i * 0.01;
    end


    local WKDR = torch.Tensor(n_iter, n_thresh):fill(0)
    local WKDR_eq = torch.Tensor(n_iter, n_thresh):fill(0)
    local WKDR_neq = torch.Tensor(n_iter, n_thresh):fill(0)
    local fmse = torch.Tensor(n_iter):fill(0)
    local fmse_si_obj = torch.Tensor(n_iter):fill(0)
    local fmse_si_img = torch.Tensor(n_iter):fill(0)
    local fmselog = torch.Tensor(n_iter):fill(0) 
    local flsi = torch.Tensor(n_iter):fill(0) 
    local fabsrel = torch.Tensor(n_iter):fill(0) 
    local fsqrrel = torch.Tensor(n_iter):fill(0)
    -- for normal
    local total_angle_difference = 0;
    local n_total_normal_point = 0

    -- main loop
    for i = 1, n_iter do   

        -- read image, scale it to the input size
        local img = image.load(data_handle[i].img_filename)
        local img_orig_h = img:size(2)
        local img_orig_w = img:size(3)
        _batch_input_cpu[{1,{}}]:copy( image.scale(img,network_input_width ,network_input_height))
        
        -- test data        
        local _single_data = {};
        _single_data[1] = data_handle[i]
        
        -- forward
        local network_output = model:forward(_batch_input_cpu:cuda());  
        cutorch.synchronize()
        local temp = network_output:clone()
        if torch.type(network_output) == 'table' then
            network_output = network_output[1]
        end


        -- if i == 1 then
        --     local max_depth = torch.max(network_output)
        --     local min_depth = torch.min(network_output)
        --     max_depth = max_depth * 1.5
        --     min_depth = min_depth / 1.5

        --     local step = (max_depth - min_depth) / n_thresh
        --     for i = 1, n_thresh do
        --         thresh[i] = step + i * step;
        --     end
        --     print(thresh)
        -- end

        -- variables that are going to be used throughout the remaining session
        local orig_size_pred_z = torch.Tensor(img_orig_h, img_orig_w)
        local orig_size_normal = torch.Tensor()
        local world_coord = torch.Tensor()
        local gtz = torch.Tensor()
        
        ---------------------------------------------------------------------------------------------------
                            -- Obtain Surface Normals  
                            -- ********************* 
                            --  PAY ATTENTION HERE!
                            -- *********************
        ---------------------------------------------------------------------------------------------------
        if cmd_params.model == 'chen' then
            ------------- normalize depth before computing the normals (for the models that have both positive and negative depth values)
            temp = normalize_output_depth_with_NYU_mean_std( temp )     -- optional (for the models that trains on relative depth only)
        end
        -------------Option 2: apply gaussian smoothing on depth before computing the normals.
        -- visualize_depth(temp[{1}], 'before smoothing.png')
        -- temp[{1,1}] = smooth_depth_map( temp[{1,1}] )     -- optional (for the models that trains on relative depth only)        
        -- visualize_depth(temp[{1}], 'after smoothing.png')

        world_coord = depth_to_world_coord_network:forward(temp):clone()
        local normal_image = torch.Tensor(3, network_input_height, network_input_width)
        normal_image:copy(world_coord_to_normal_network:forward(world_coord))
                
        ---------------------------------------------------------------------------------------------------
                            -- Relative depth error
        ---------------------------------------------------------------------------------------------------
        --image.scale(src, width, height, [mode])    Scale it to the original size!
        orig_size_pred_z:copy( image.scale(network_output[{1,1,{}}]:double(), img_orig_w, img_orig_h) ) 
        _evaluate_WKDR(orig_size_pred_z, _single_data[1], WKDR[{i,{}}], WKDR_eq[{i,{}}], WKDR_neq[{i,{}}]);


        ---------------------------------------------------------------------------------------------------
                            -- Load the ground truth depth values
        ---------------------------------------------------------------------------------------------------
        if true then

            local gtz = load_hdf5_z(string.gsub(data_handle[i].img_filename, '.png', '_gt_depth.h5'), 'gt_depth')
            assert(gtz:size(1) == 480)
            assert(gtz:size(2) == 640)
            

            if img_orig_h ~= 480 then
                orig_size_pred_z = image.scale( network_output[{1,1,{}}]:double(), 640, 480  )  
            end
            
            -- print(i)
            
            ---------------------------------------------------------------------------------------------------
            -- metric error
            ---------------------------------------------------------------------------------------------------
            -- evaluate the data at the cropped area        
            local _crop = 16
            local gtz_cropped = gtz:sub(45,471,41,601)
            local orig_size_pred_z_cropped = orig_size_pred_z:sub(45,471,41,601)        
            -- evaluate
            local norm_orig_size_pred_z_crop = normalize_output_depth_with_NYU_mean_std( orig_size_pred_z_cropped )
            fmse[i], fmselog[i], flsi[i], fabsrel[i], fsqrrel[i] = metric_error(gtz_cropped, norm_orig_size_pred_z_crop)
            fmse_si_img[i] = img_scale_invariant_error(gtz_cropped, orig_size_pred_z_cropped)
            if cmd_params.si_obj_err then
                if cmd_params.model == 'chen' then
                    local _normed = normalize_output_depth_with_NYU_mean_std( orig_size_pred_z )
                    fmse_si_obj[i] = object_scale_invariant_error(gtz, _normed, data_handle[i].img_filename)
                elseif cmd_params.model == 'ours' then
                    fmse_si_obj[i] = object_scale_invariant_error(gtz, orig_size_pred_z, data_handle[i].img_filename)
                end
            else
               fmse_si_obj[i] = 0 
            end

            -- print(string.format('rmse_si_obj:%f',math.sqrt(fmse_si_obj[i])))

            ---------------------------------------------------------------------------------------------------
            -- Normal error
            ---------------------------------------------------------------------------------------------------
            local normal_target = read_normal(data_handle[i].img_filename)
            local sum_angle_diff = angle_diff(normal_image, normal_target)
 	        total_angle_difference = total_angle_difference + sum_angle_diff
            -- update the number of point pair
            n_total_normal_point = n_total_normal_point + normal_target.n_point

        end
    


        collectgarbage()
        collectgarbage()
        collectgarbage()
        collectgarbage()
        collectgarbage()


    end


    -- get averaged WKDR and so on 
    WKDR = torch.mean(WKDR,1)
    WKDR_eq = torch.mean(WKDR_eq,1)
    WKDR_neq = torch.mean(WKDR_neq,1)
    overall_summary = torch.Tensor(n_thresh, 4)


    -- find the best threshold according to our criteria
    min_max = 100;
    min_max_i = 1;
    for i = 1 , n_thresh do
        overall_summary[{i,1}] = thresh[i]
        overall_summary[{i,2}] = WKDR[{1,i}]
        overall_summary[{i,3}] = WKDR_eq[{1,i}]
        overall_summary[{i,4}] = WKDR_neq[{1,i}]
        if math.max(WKDR_eq[{1,i}], WKDR_neq[{1,i}]) < min_max then
            min_max = math.max(WKDR_eq[{1,i}], WKDR_neq[{1,i}])
            min_max_i = i;
        end
    end

    
    -- -- print the final output
    -- if min_max_i > 1 then
    --     if min_max_i < n_thresh then
    --         print(overall_summary[{{min_max_i,min_max_i},{}}])
    --     end
    -- end
    print(string.format('WKDR:\n%f',overall_summary[{min_max_i,2}]))
    print(string.format('rmse:\n%f',math.sqrt(torch.mean(fmse))))
    print(string.format('rmse_si_img:\n%f',math.sqrt(torch.mean(fmse_si_img))))
    -- print(string.format('rmse_si_obj:%f',math.sqrt(torch.mean(fmse_si_obj))))
    print(string.format('rmselog:\n%f',math.sqrt(torch.mean(fmselog))))
    print(string.format('lsi:\n%f',math.sqrt(torch.mean(flsi))))
    print(string.format('absrel:\n%f',torch.mean(fabsrel)))
    print(string.format('sqrrel:\n%f',torch.mean(fsqrrel)))
    print(string.format('angular diff:\n%f', total_angle_difference / n_total_normal_point))
    
end

