require 'nn'
require 'image'
require 'csvigo'
require 'hdf5'
require 'measure'

local network_input_height = 128
local network_input_width = 416

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



            _line_idx = _line_idx + 1 
        end                 

    end


    return _sample_idx, _data;
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

local function load_normal_bin(normal_bin_name, n_point, width, height)
    local file = torch.DiskFile(normal_bin_name, 'r'):binary();    
    local normal = torch.DoubleTensor(file:readDouble(5 * n_point))
    file:close();                
    normal = torch.view(normal, n_point, 5)                 -- tested
    normal = normal:t()

    -- pay atteniton to the order!
    local x = torch.Tensor()
    x:resize(n_point):copy(normal[{1,{}}]:int())
    local y = torch.Tensor()
    y:resize(n_point):copy(normal[{2,{}}]:int())
    normal = normal[{{3,5},{}}]:clone()


    local gt_normal = torch.Tensor(1, 3, height, width)
    local gt_mask = torch.Tensor(height, width):fill(0.0)
    for i = 1 , n_point do        
        if torch.sum(normal[{{1,3},i}]:ne(normal[{{1,3},i}])) == 0 then
            gt_normal[{ 1, {1,3}, y[i], x[i] }]:copy( normal[{{1,3},i}] )
            gt_mask[{y[i], x[i]}] = 1;
        

        end
    end
    gt_mask = torch.gt(gt_mask, 0.0)



    return gt_normal, gt_mask
end


function normalize_with_mean_std( input, mean, std, b_remove_neg )
    
    local normed_input = input:clone()
    normed_input = normed_input - torch.mean(normed_input);
    normed_input = normed_input / torch.std(normed_input);
    normed_input = normed_input * std;
    normed_input = normed_input + mean;
    
    if b_remove_neg then
        -- remove and replace the depth value that are negative
        if torch.sum(normed_input:lt(0)) > 0 then
            -- fill it with the minimum of the non-negative plus a eps so that it won't be 0
            normed_input[normed_input:lt(0)] = torch.min(normed_input[normed_input:gt(0)]) + 0.00001
        end
    end

    return normed_input
end


local function metric_error_monodepth(gt_z, pred_z)
    -- the input gt_z and pred_z should be 2-dimensional tensor
    local valid_mask = torch.gt(gt_z, 0.0)
    local masked_gtz = gt_z:maskedSelect(valid_mask) 
    local masked_resize_pred_z = pred_z:maskedSelect(valid_mask)

    local mask_gt80 = torch.gt(masked_resize_pred_z, 80.0)
    masked_resize_pred_z:maskedFill(mask_gt80, 80.0)

    -- get error after normalization    
    local fmse = torch.mean(torch.pow(masked_gtz - masked_resize_pred_z, 2))    
    local fabsrel = torch.mean( torch.cdiv(torch.abs( masked_resize_pred_z - masked_gtz ), masked_gtz ))
    local fsqrrel = torch.mean( torch.cdiv(torch.pow( masked_resize_pred_z - masked_gtz ,2), masked_gtz ))
    local fmselog = torch.mean(torch.pow(torch.log(masked_gtz) - torch.log(masked_resize_pred_z), 2))
    local flsi = torch.mean(  torch.pow(torch.log(masked_resize_pred_z)  - torch.log(masked_gtz) + torch.mean(torch.log(masked_gtz) - torch.log(masked_resize_pred_z)) , 2 )  )
    
    -- scale-shift inv mse
    local transformed_masked_resize_pred_z = find_least_square_scale_shift(masked_resize_pred_z, masked_gtz)    
    local fmse_si = torch.mean(torch.pow(transformed_masked_resize_pred_z - masked_gtz, 2))


    return fmse, fmselog, flsi, fabsrel, fsqrrel, fmse_si
end


local function metric_error(gt_z, pred_z)
    local std_of_KITTI_training = 11.308058
    local mean_of_KITTI_training = 14.854678
    -- the input gt_z and pred_z should be 2-dimensional tensor
    local valid_mask = torch.gt(gt_z, 0.0)
    local masked_gtz = gt_z:maskedSelect(valid_mask) 
    local masked_resize_pred_z = pred_z:maskedSelect(valid_mask)
    -- print(masked_gtz:size())



    -- get error after normalization
    local normed_pred_z_neg = normalize_with_mean_std( masked_resize_pred_z, mean_of_KITTI_training, std_of_KITTI_training, false )
    local fmse = torch.mean(torch.pow(masked_gtz - normed_pred_z_neg, 2))    
    local fabsrel = torch.mean( torch.cdiv(torch.abs( normed_pred_z_neg - masked_gtz ), masked_gtz ))
    local fsqrrel = torch.mean( torch.cdiv(torch.pow( normed_pred_z_neg - masked_gtz ,2), masked_gtz ))

    local normed_pred_z_pos = normalize_with_mean_std( masked_resize_pred_z, mean_of_KITTI_training, std_of_KITTI_training, true )
    local fmselog = torch.mean(torch.pow(torch.log(masked_gtz) - torch.log(normed_pred_z_pos), 2))
    local flsi = torch.mean(  torch.pow(torch.log(normed_pred_z_pos)  - torch.log(masked_gtz) + torch.mean(torch.log(masked_gtz) - torch.log(normed_pred_z_pos)) , 2 )  )
    


    -- scale-shift inv mse
    local transformed_masked_resize_pred_z = find_least_square_scale_shift(masked_resize_pred_z, masked_gtz)    
    local fmse_si = torch.mean(torch.pow(transformed_masked_resize_pred_z - masked_gtz, 2))


    return fmse, fmselog, flsi, fabsrel, fsqrrel, fmse_si
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
cmd:option('-valid_set', '', 'validation file name');
cmd:option('-test_set','', 'test file name');
cmd:option('-model','ours', 'ours/monodepth');

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






local function read_NYU_gt_normal_and_mask(i, mode)
    local normal_field_name 
    local mask_field_name 
    if mode == 'test' then
        normal_field_name = 'test_normal'
        mask_field_name = 'test_mask'
    else
        normal_field_name = 'train_normal'
        mask_field_name = 'train_mask'
    end
    -- read the groundtruth normal map            
    local gt_normal = normal_hdf:read(normal_field_name):partial( {i, i}, {1, 3}, {1, 480}, {1, 640} )
    local n_y = gt_normal[{1,3,{}}]:clone()
    local n_z = gt_normal[{1,2,{}}]:clone()
    local n_x = gt_normal[{1,1,{}}]:clone()
    gt_normal[{1,1,{}}]:copy(-n_x)
    gt_normal[{1,2,{}}]:copy(-n_y)
    gt_normal[{1,3,{}}]:copy(-n_z)


    local gt_mask = mask_hdf:read(mask_field_name):partial( {i, i}, {1, 1}, {1, 480}, {1, 640} )
    gt_mask = gt_mask[{1,1,{}}]
    gt_mask = gt_mask:byte()

    return gt_normal, gt_mask
end


require 'cudnn'
require 'cunn'
require 'cutorch'
-- depth to normal
require './models/world_coord_to_normal'
require './models/img_coord_to_world_coord'
require 'mattorch'


local depth_to_world_coord_network = nn.img_coord_to_world_coord():cuda()
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


local eigen_res = {fmse = torch.Tensor(n_iter):fill(0),
                   fmse_si_img = torch.Tensor(n_iter):fill(0),
                   fmselog = torch.Tensor(n_iter):fill(0),
                   flsi = torch.Tensor(n_iter):fill(0),
                   fabsrel = torch.Tensor(n_iter):fill(0),
                   fsqrrel = torch.Tensor(n_iter):fill(0)
                    }

local garg_res = {fmse = torch.Tensor(n_iter):fill(0),
                   fmse_si_img = torch.Tensor(n_iter):fill(0),
                   fmselog = torch.Tensor(n_iter):fill(0),
                   flsi = torch.Tensor(n_iter):fill(0),
                   fabsrel = torch.Tensor(n_iter):fill(0),
                   fsqrrel = torch.Tensor(n_iter):fill(0)
                    }                    
local WKDR = torch.Tensor(n_iter, n_thresh):fill(0)
local WKDR_eq = torch.Tensor(n_iter, n_thresh):fill(0)
local WKDR_neq = torch.Tensor(n_iter, n_thresh):fill(0)


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
    
    
   
    -- variables that are going to be used throughout the remaining session
    local orig_size_pred_z = torch.Tensor(img_orig_h, img_orig_w)
    local temp 
    if cmd_params.model == 'monodepth' then
        local monodepth_filename = string.format('./monodepth_result/KITTI_mono_depth%d.mat', i)
        local monodepth_result = mattorch.load(monodepth_filename)['mono_depth_result']:double()
        orig_size_pred_z:copy(monodepth_result:t())
        temp = torch.Tensor(1,1,network_input_height, network_input_width)
        temp[{1,1,{}}]:copy(image.scale(orig_size_pred_z,network_input_width ,network_input_height))
        temp = temp:cuda()

    elseif cmd_params.model == 'ours' then 
        -- forward
        local network_output = model:forward(_batch_input_cpu:cuda());  
        cutorch.synchronize()
        temp = network_output:clone()
        if torch.type(network_output) == 'table' then
            network_output = network_output[1]
        end       
        orig_size_pred_z:copy( image.scale(network_output[{1,1,{}}]:double(), img_orig_w, img_orig_h) )         
    end

    
    local orig_size_normal = torch.Tensor()
    local world_coord = torch.Tensor()
    
    -- ---------------------------------------------------------------------------------------------------
    --                     -- Obtain Surface Normals  
    -- ---------------------------------------------------------------------------------------------------
    world_coord = depth_to_world_coord_network:forward(temp):clone()
    local normal_image = torch.Tensor(3, network_input_height, network_input_width)
    normal_image:copy(world_coord_to_normal_network:forward(world_coord))
    orig_size_normal = image.scale(normal_image:double(), img_orig_w, img_orig_h) 
            
    -- ---------------------------------------------------------------------------------------------------
    --                     -- Relative depth error
    -- ---------------------------------------------------------------------------------------------------
    -- --image.scale(src, width, height, [mode])    Scale it to the original size!
    

    _evaluate_WKDR(orig_size_pred_z, _single_data[1], WKDR[{i,{}}], WKDR_eq[{i,{}}], WKDR_neq[{i,{}}]);


    -- ---------------------------------------------------------------------------------------------------
    --                     -- Load the ground truth depth values
    -- ---------------------------------------------------------------------------------------------------
    if cmd_params.mode == 'test' then

        local gtz = load_hdf5_z(string.gsub(data_handle[i].img_filename, '.png', '_gt_depth.h5'), 'depth')
        assert(gtz:size(1) == img_orig_h)
        assert(gtz:size(2) == img_orig_w)
        
        ---------------------------------------------------------------------------------------------------
        -- metric error
        ---------------------------------------------------------------------------------------------------
        -- crop
        local eigen_crop = torch.IntTensor({0.3324324 * img_orig_h + 1,  0.91351351 * img_orig_h,  
                                 0.0359477 * img_orig_w + 1,   0.96405229 * img_orig_w})            -- mind the +1. This is to make sure that the crop region is exactly the same as in the monodepth result.
        local garg_crop = torch.IntTensor({0.40810811 * img_orig_h + 1,  0.99189189 * img_orig_h,   
                                 0.03594771 * img_orig_w + 1,   0.96405229 * img_orig_w})
        local gtz_eigen_crop = gtz:sub(eigen_crop[1],eigen_crop[2],eigen_crop[3],eigen_crop[4])
        local gtz_garg_crop = gtz:sub(garg_crop[1],garg_crop[2],garg_crop[3],garg_crop[4])
        local pred_z_eigen_crop = orig_size_pred_z:sub(eigen_crop[1],eigen_crop[2],eigen_crop[3],eigen_crop[4])        
        local pred_z_garg_crop = orig_size_pred_z:sub(garg_crop[1],garg_crop[2],garg_crop[3],garg_crop[4])

        if cmd_params.model == 'monodepth' then
            metric_func = metric_error_monodepth
        else
            metric_func = metric_error
        end

        -- evaluate eigen
        eigen_res['fmse'][i], eigen_res['fmselog'][i], 
        eigen_res['flsi'][i], eigen_res['fabsrel'][i], 
        eigen_res['fsqrrel'][i], eigen_res['fmse_si_img'][i] = metric_func(gtz_eigen_crop, pred_z_eigen_crop)

        garg_res['fmse'][i], garg_res['fmselog'][i], 
        garg_res['flsi'][i], garg_res['fabsrel'][i], 
        garg_res['fsqrrel'][i], garg_res['fmse_si_img'][i] = metric_func(gtz_garg_crop, pred_z_garg_crop)
        
   
    end



    if cmd_params.vis then
        local _pred_z_img = torch.Tensor(1,img_orig_h,img_orig_w)
        local _normal_rgb = torch.Tensor(3,img_orig_h,img_orig_w)
        local _output_img = torch.Tensor(3, img_orig_h,img_orig_w * 5)

        -- predicted depth
        _pred_z_img:copy(orig_size_pred_z:double())
        _pred_z_img = _pred_z_img:add( - torch.min(_pred_z_img) )
        _pred_z_img = _pred_z_img:div( torch.max(_pred_z_img:sub(1,-1, 20, img_orig_h - 20, 20, img_orig_w - 20)) )
        
        _output_img[{1,{1,img_orig_h},{img_orig_w + 1,img_orig_w * 2}}]:copy(_pred_z_img)
        _output_img[{2,{1,img_orig_h},{img_orig_w + 1,img_orig_w * 2}}]:copy(_pred_z_img)
        _output_img[{3,{1,img_orig_h},{img_orig_w + 1,img_orig_w * 2}}]:copy(_pred_z_img)

        -- -- rgb            
        _output_img[{{1},{1,img_orig_h},{1,img_orig_w}}]:copy(img[{1,{}}])
        _output_img[{{2},{1,img_orig_h},{1,img_orig_w}}]:copy(img[{2,{}}])
        _output_img[{{3},{1,img_orig_h},{1,img_orig_w}}]:copy(img[{3,{}}])

                    
        -- the normal map  
        _normal_rgb:copy(orig_size_normal)          
        _normal_rgb:add(1)
        _normal_rgb:div(2)
        _output_img:sub(1,-1, 1,-1, 2 * img_orig_w + 1, 3 * img_orig_w):copy(_normal_rgb)

        

        -- the groundtruth depth
        if gtz ~= nil then
            local _gtz_img = gtz:clone()
            _gtz_img = _gtz_img:add( - torch.min(_gtz_img) )
            _gtz_img = _gtz_img:div( torch.max(_gtz_img) )
            _output_img[{1,{1,img_orig_h},{3 * img_orig_w + 1,img_orig_w * 4}}]:copy(_gtz_img)
            _output_img[{2,{1,img_orig_h},{3 * img_orig_w + 1,img_orig_w * 4}}]:copy(_gtz_img)
            _output_img[{3,{1,img_orig_h},{3 * img_orig_w + 1,img_orig_w * 4}}]:copy(_gtz_img)
        end

        
        image.save(cmd_params.output_folder.. '/' .. i .. '.png', _output_img)  
        image.save(cmd_params.output_folder.. '/' .. i .. '_normal.png', _normal_rgb)  
        image.save(cmd_params.output_folder.. '/' .. i .. '_pred_z_img.png', _pred_z_img) 
        image.save(cmd_params.output_folder.. '/' .. i .. '_img.png', img)   

        if cmd_params.mesh then

            save_mesh(cmd_params.output_folder.. '/' .. i .. '.obj', world_coord)

        end
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


-- print the final output
print(overall_summary)
print("====================================================================")
if min_max_i > 1 then
    if min_max_i < n_thresh then
        print(overall_summary[{{min_max_i-1,min_max_i+1},{}}])
    end
end


if cmd_params.mode == 'test' then
    print("====================================================================")
    print("Eigen Crop Result:")
    print(string.format('rmse:\t%f',math.sqrt(torch.mean(eigen_res['fmse']))))
    print(string.format('rmse_si_img:%f',math.sqrt(torch.mean(eigen_res['fmse_si_img']))))
    print(string.format('rmselog:%f',math.sqrt(torch.mean(eigen_res['fmselog']))))
    print(string.format('lsi:\t%f',math.sqrt(torch.mean(eigen_res['flsi']))))
    print(string.format('absrel:\t%f',torch.mean(eigen_res['fabsrel'])))
    print(string.format('sqrrel:\t%f',torch.mean(eigen_res['fsqrrel'])))

    print("====================================================================")
    print("Garg Crop Result:")
    print(string.format('rmse:\t%f',math.sqrt(torch.mean(garg_res['fmse']))))
    print(string.format('rmse_si_img:%f',math.sqrt(torch.mean(garg_res['fmse_si_img']))))
    print(string.format('rmselog:%f',math.sqrt(torch.mean(garg_res['fmselog']))))
    print(string.format('lsi:\t%f',math.sqrt(torch.mean(garg_res['flsi']))))
    print(string.format('absrel:\t%f',torch.mean(garg_res['fabsrel'])))
    print(string.format('sqrrel:\t%f',torch.mean(garg_res['fsqrrel'])))
end



