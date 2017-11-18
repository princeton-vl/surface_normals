require 'nn'
require 'image'
require 'csvigo'
require 'hdf5'
require 'measure'


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


local function parse_normal_line(line)    
    local splits = line:split(',')
    local sample = {};
    sample.img_filename = splits[ 1 ]    
    sample.n_point = tonumber(splits[ 2 ])
    
    -- there may also be focal length on the 3rd term   -- to test
    if #splits > 2 then
        sample.focal_length = tonumber(splits[3])
    end

    -- the x and y coordinate is on the 4th and 5th term
    sample.x = splits[4]
    sample.y = splits[5]

    return sample
end

local function parse_normal_csv(filename)
    local _handle = {}

    -- take care of the case where filename is a nil, i.e., no training data
    if filename == nil then
        return _handle
    end

    -- read the number of lines
    local _n_lines = 0
    for _ in io.lines(filename) do
      _n_lines = _n_lines + 1
    end

    -- read the hdf5 file that stores the groundtruth normal
    -- _this_normal_vector[1] is x , [2] is y, [3] is z. 
    -- It's a right hand coordinate system. Its x axis points right, y axis points up, z axis points toward us
    -- But in the current way we generate normal, we are using a left hand coordinate system, i.e, x points left, y points down, z points toward us.
    -- Therefore, we need to convert it.
    local hdf5_handle = hdf5.open(string.gsub(filename, ".csv", ".h5"),'r')

    -- read in the image name address
    local csv_file_handle = io.open(filename, 'r');    
    local _sample_idx = 0
    while _sample_idx < _n_lines do

        local this_line = csv_file_handle:read()
        _sample_idx = _sample_idx + 1
        
        _handle[_sample_idx] = parse_normal_line(this_line)    
        
        local _this_normal_vector  = hdf5_handle:read('/normal'):partial({1, 3}, {_sample_idx, _sample_idx})
        _this_normal_vector[{{1,2}}]:mul(-1)
        _handle[_sample_idx].gt_normal = _this_normal_vector:clone()
        
    end    
    csv_file_handle:close();

    return _handle
end

local function read_gt_normal()

end


local function resize_n_channel_tensor(three_dim_tensor, dst_width, dst_height)
    assert(three_dim_tensor:nDimension() == 3)
    local n_channel = three_dim_tensor:size(1)

    local resize_output = torch.Tensor(n_channel, dst_height, dst_width)
    for i = 1, n_channel do
        resize_output[{i,{}}]:copy( image.scale( three_dim_tensor[{i,{}}], dst_width, dst_height))
    end
    
    return resize_output
end


local function inpaint_pad_output_FCRN(output)
    assert(output:size(2) == 128)
    assert(output:size(3) == 160)
    
    -- input height = 240
    -- input width = 320


    local resize_height = 228
    local resize_width = 304
    local resize_output = resize_n_channel_tensor(output:double(), resize_width, resize_height)

    
    local dst_out_height = 240
    local dst_out_width = 320
    local n_channel = resize_output:size(1)

    local padded_output1 = torch.Tensor(n_channel, resize_height, dst_out_width);
    padded_output1[{{},{}, {9, 312}}]:copy(resize_output)
    -- pad left 
    for i = 1 , 8 do
        padded_output1[{{},{}, i}]:copy(padded_output1[{{},{},9}])                
    end
    -- pad right
    for i = 313 , 320 do
        padded_output1[{{},{}, i}]:copy(padded_output1[{{},{},312}])                
    end


    local padded_output2 = torch.Tensor(n_channel, dst_out_height, dst_out_width);
    padded_output2[{{},{7, 234}, {}}]:copy(padded_output1)
    -- pad top and down    
    for i = 1 , 6 do
        padded_output2[{{}, i, {}}]:copy(padded_output1[{{}, 1,{}}])        
    end
    for i = 235, 240 do
        padded_output2[{{}, i, {}}]:copy(padded_output1[{{}, resize_height,{}}])
    end

    return padded_output2
end



local function inpaint_pad_output_eigen(output)
    assert(output:size(2) == 109)
    assert(output:size(3) == 147)
    
    -- input height = 240
    -- input width = 320
    -- [12 .. 227] height
    -- [14 .. 305] width

    local resize_height = 227 - 12 + 1
    local resize_width = 305 - 14 + 1
    local resize_output = resize_n_channel_tensor(output:double(), resize_width, resize_height)

    
    local dst_out_height = 240
    local dst_out_width = 320
    local n_channel = resize_output:size(1)

    local padded_output1 = torch.Tensor(n_channel, resize_height, dst_out_width);
    padded_output1[{{},{}, {14, 305}}]:copy(resize_output)
    -- pad left 
    for i = 1 , 13 do
        padded_output1[{{},{}, i}]:copy(padded_output1[{{},{},14}])                
    end
    -- pad right
    for i = 306 , 320 do
        padded_output1[{{},{}, i}]:copy(padded_output1[{{},{},305}])                
    end


    local padded_output2 = torch.Tensor(n_channel, dst_out_height, dst_out_width);
    padded_output2[{{},{12, 227}, {}}]:copy(padded_output1)
    -- pad top and down    
    for i = 1 , 11 do
        padded_output2[{{}, i, {}}]:copy(padded_output1[{{}, 1,{}}])        
    end
    for i = 228, 240 do
        padded_output2[{{}, i, {}}]:copy(padded_output1[{{}, resize_height,{}}])
    end

    return padded_output2
end

local function visualize_normal(normal, filename)
    local _normal_img = normal:clone()
    _normal_img:add(1)
    _normal_img:mul(0.5)
    image.save(filename, _normal_img) 

    print("Done saving to ", filename)
end

local function visualize_depth(z, filename)
    local _z_img = z:clone()
    _z_img = _z_img:add( - torch.min(_z_img) )
    _z_img = _z_img:div( torch.max(_z_img) )
    image.save(filename, _z_img) 
end

local function normalize_to_unit_vector(vector)
    local sqrt_sum = math.sqrt(torch.sum(torch.pow(vector,2)))
    local normalized_vector = vector:clone()
    normalized_vector:div(sqrt_sum)

    return normalized_vector
end

local function f_angular_difference_in_degree(n1, n2)
    local norm_n1 = normalize_to_unit_vector(n1)
    local norm_n2 = normalize_to_unit_vector(n2)

    local cos = torch.sum(torch.cmul(norm_n1, norm_n2))      -- dot product of normals , seems quite expensive move
    local acos = math.acos(cos)
    return acos / math.pi * 180
end

local function get_normal_from_depth_NYU(nyu_depth, depth_to_world_coord_network, world_coord_to_normal_network)
    local gpu_depth = nyu_depth:cuda()
    gpu_depth:resize(1,1,240,320)
    -- normal image
    local world_coord = depth_to_world_coord_network:forward(gpu_depth):clone()

    local normal_image = torch.Tensor(3, 240, 320)

    normal_image:copy(world_coord_to_normal_network:forward(world_coord))
                      
    return normal_image
end


local function read_eigen15_result(data_handle, i)
        local img = image.load(data_handle[i].img_filename)
        local img_orig_h = img:size(2)
        local img_orig_w = img:size(3)

        local image_basename = paths.basename(data_handle[i].img_filename)
        image_basename = string.gsub(image_basename, '.thumb', '_eigen15_result.h5')
        local folder_name = paths.dirname(data_handle[i].img_filename)

        local h5_filename = paths.concat(folder_name, 'result' , image_basename)
        local myFile = hdf5.open(h5_filename, 'r')
        local eigen_normal_result = myFile:read('normal'):all()
        local eigen_depth_result = myFile:read('depth'):all()
        myFile:close()
        eigen_normal_result = eigen_normal_result[{1,{}}]

        -- ######## The way the normal is organized can be illustrated by this segment of Matlab code:
        -- normal = h5read('debug.h5', '/normal');
        -- a(:,:,1) = normal(:,:,1)';
        -- a(:,:,2) = normal(:,:,2)';
        -- a(:,:,3) = normal(:,:,3)';
        -- x = a(:,:,1);
        -- z = a(:,:,2);
        -- y = a(:,:,3);
        -- a(:,:,2) = -y;
        -- a(:,:,1) = -x;
        -- a(:,:,3) = -z;
        -- figure(1); imshow( (a+1)/2);


        -- we are using a left hand coordinate system, i.e, x points left, y points down, z points toward us.
        -- flip the coordinates system                  --Not checked    Have you checked the xyz order??????
        local x = eigen_normal_result[{1,{}}]:clone()
        local z = eigen_normal_result[{2,{}}]:clone()
        local y = eigen_normal_result[{3,{}}]:clone()
        eigen_normal_result[{1,{}}]:copy(x:mul(-1))
        eigen_normal_result[{2,{}}]:copy(y:mul(-1))
        eigen_normal_result[{3,{}}]:copy(z:mul(-1))


        -- pad and resize back to 240 x 320
        local padded_320_240_eigen_normal = inpaint_pad_output_eigen(eigen_normal_result)
        local padded_320_240_eigen_depth = inpaint_pad_output_eigen(eigen_depth_result)

        -- get normal from estimated depth
        local normal_from_depth = get_normal_from_depth_NYU(padded_320_240_eigen_depth, depth_to_world_coord_network, world_coord_to_normal_network)

        -- resize to the original Scale                 -- checked
        local orig_size_recov_n = resize_n_channel_tensor(padded_320_240_eigen_normal, img_orig_w, img_orig_h)       --image.scale(src, width, height, [mode])
        local orig_size_n_from_d = resize_n_channel_tensor(normal_from_depth, img_orig_w, img_orig_h)       --image.scale(src, width, height, [mode])


        local estimated_normal_at_xy = orig_size_recov_n[{{}, data_handle[i].y, data_handle[i].x}]:clone()          -- this clone() is a must, otherwise, it won't work with resize()!
        estimated_normal_at_xy = normalize_to_unit_vector(estimated_normal_at_xy)
        estimated_normal_at_xy:resize(3,1)


        local normal_from_depth_at_xy = orig_size_n_from_d[{{}, data_handle[i].y, data_handle[i].x}]:clone()          -- this clone() is a must, otherwise, it won't work with resize()!
        if normal_from_depth_at_xy[3] < 0 then
            normal_from_depth_at_xy[3] = 0
        end
        normal_from_depth_at_xy = normalize_to_unit_vector(normal_from_depth_at_xy)
        normal_from_depth_at_xy:resize(3,1)


        
        -- visualize_depth(padded_320_240_eigen_depth, 'depth.png')
        -- visualize_normal(padded_320_240_eigen_normal, 'eigen_normal.png')
        -- visualize_normal(normal_from_depth, 'normal_from_depth.png')
        -- image.save("orig_img.png", img) 
        -- print("done")
        -- io.read()

        return normal_from_depth_at_xy
end






local function read_FCRN_result(data_handle, i)
    local img = image.load(data_handle[i].img_filename)
    local img_orig_h = img:size(2)
    local img_orig_w = img:size(3)
    local image_basename = paths.basename(data_handle[i].img_filename)
    image_basename = string.gsub(image_basename, '.thumb', '_FCRN_result.h5')
    local folder_name = paths.dirname(data_handle[i].img_filename)

    local h5_filename = paths.concat(folder_name, 'result' , image_basename)
    local myFile = hdf5.open(h5_filename, 'r')
    local FCRN_depth_result = myFile:read('depth'):all()
    myFile:close()
    FCRN_depth_result:resize(1, FCRN_depth_result:size(1), FCRN_depth_result:size(2))


    -- we are using a left hand coordinate system, i.e, x points left, y points down, z points toward us.

    -- pad and resize back to 240 x 320
    local padded_320_240_FCRN_depth = inpaint_pad_output_FCRN(FCRN_depth_result)

    -- get normal from estimated depth
    local normal_from_depth = get_normal_from_depth_NYU(padded_320_240_FCRN_depth, depth_to_world_coord_network, world_coord_to_normal_network)

    -- resize to the original Scale                 -- checked
    local orig_size_n_from_d = resize_n_channel_tensor(normal_from_depth, img_orig_w, img_orig_h)       --image.scale(src, width, height, [mode])

    local normal_from_depth_at_xy = orig_size_n_from_d[{{}, data_handle[i].y, data_handle[i].x}]:clone()          -- this clone() is a must, otherwise, it won't work with resize()!
    if normal_from_depth_at_xy[3] < 0 then
        normal_from_depth_at_xy[3] = 0
    end
    normal_from_depth_at_xy = normalize_to_unit_vector(normal_from_depth_at_xy)
    normal_from_depth_at_xy:resize(3,1)
    
    -- visualize_depth(padded_320_240_FCRN_depth, 'depth.png')
    -- visualize_normal(normal_from_depth, 'normal_from_depth.png')
    -- image.save("orig_img.png", img) 
    -- print("done")
    -- io.read()
    
    return normal_from_depth_at_xy
end


local function read_Marr_result(data_handle, i)
        local img = image.load(data_handle[i].img_filename)
        local img_orig_h = img:size(2)
        local img_orig_w = img:size(3)

        local image_basename = paths.basename(data_handle[i].img_filename)
        image_basename = string.gsub(image_basename, '.thumb', '_Marr_result.h5')
        local folder_name = paths.dirname(data_handle[i].img_filename)

        local h5_filename = paths.concat(folder_name, 'result' , image_basename)
        local myFile = hdf5.open(h5_filename, 'r')
        local Marr_result = myFile:read('normal'):all()
        myFile:close()
        Marr_result = Marr_result:transpose(2,3);
        

        assert(Marr_result:size(2) == img_orig_h)
        assert(Marr_result:size(3) == img_orig_w)

        -- -- Matlab code to verify the correctness of the surface normal read
        -- load('_save.mat');   % _save.mat comes from demo_code in the Marr source folder
        -- x = predns(:,:,1);
        -- y = predns(:,:,2);
        -- z = predns(:,:,3);
        -- a(:,:,1) = x;
        -- a(:,:,2) = -y;
        -- a(:,:,3) = z;
        -- imshow( (a+1)/2);
        -- h5create('debug.h5','/normal',size(predns)); 
        -- h5write('debug.h5', '/normal', predns);
        -- predns(20,10,:)
        -- -- The code shows that the x points left, y points up, z points toward us

        -- -- read hdf5 debug code, used with the above Matlab code:
        -- myFile = hdf5.open('debug.h5', 'r')
        -- Marr_result = myFile:read('normal'):all()
        -- Marr_result = Marr_result:transpose(2,3)
        -- Marr_result[{{},20,10}]


        -- we are using a left hand coordinate system, i.e, x points left, y points down, z points toward us.
        -- flip the coordinates system                  -- xyz Order: checked ! 
        -- Marr_result[{1,{}}] is x, Marr_result[{2,{}}] is y, Marr_result[{3,{}}] is z
        local y = Marr_result[{2,{}}]:clone()
        Marr_result[{2,{}}]:copy(y:mul(-1))


        local estimated_normal_at_xy = Marr_result[{{}, data_handle[i].y, data_handle[i].x}]:clone()          -- this clone() is a must, otherwise, it won't work with resize()!
        normalize_to_unit_vector(estimated_normal_at_xy)
        estimated_normal_at_xy:resize(3,1)



        -- visualize_normal(Marr_result, string.format("%d_marr.png", i))
        return estimated_normal_at_xy
end



function evaluate_one_sample(gt_normal, estimated_normal_at_xy, i, record)
    -- check the angular difference
    local _ang_diff = f_angular_difference_in_degree(gt_normal, estimated_normal_at_xy)
    
    -- print(gt_normal)
    -- print(estimated_normal_at_xy)
    -- print(_ang_diff)
    -- io.read()

    if _ang_diff < 30 then
        record.n_less_than_30 = record.n_less_than_30 + 1
    end
    if _ang_diff < 22.5 then
        record.n_less_than_22_5 = record.n_less_than_22_5 + 1
    end
    if _ang_diff < 11.25 then
        record.n_less_than_11_25 = record.n_less_than_11_25 + 1
    end

    record.tensor_angular_diffs[i] = _ang_diff

    return record
end

function print_evaluation_result(record)
    local mean = torch.mean(record.tensor_angular_diffs)
    local median = torch.median(record.tensor_angular_diffs)[1]

    local p_11_25 = record.n_less_than_11_25 / n_iter * 100
    local p_22_5 = record.n_less_than_22_5 / n_iter * 100
    local p_30 = record.n_less_than_30 / n_iter * 100
    print(string.format("Angular Diff:\nMethod\t\tMean\t\tMedians\t\t11.25\t\t22.5\t\t30\n%s&\t%f&\t%f&\t%.2f&\t%.2f&\t%.2f\t \\\\", cmd_params.model, mean, median, p_11_25, p_22_5, p_30 ))
end

function get_network_result(b_direct_normal, data_handle, i, model, world_coord_to_normal_network, depth_to_world_coord_network, depth_to_world_coord_network_focal, b_vis)
    -- read image, scale it to the input size
    local img = image.load('../../data/SNOW_Toolkit/' .. data_handle[i].img_filename)
    local img_orig_h = img:size(2)
    local img_orig_w = img:size(3)
    local network_input_height = 240
    local network_input_width = 320
    local _batch_input_cpu = torch.Tensor(1,3,network_input_height,network_input_width)

    -- pay attention to the grayscale images

    if img:size(1) == 1 then
        print(data_handle[i].img_filename, ' is gray')
        local temp = image.scale(img,network_input_width ,network_input_height)
        _batch_input_cpu[{1,1,{}}]:copy(temp);    -- Note that the image read is in the range of 0~1
        _batch_input_cpu[{1,2,{}}]:copy(temp);    
        _batch_input_cpu[{1,3,{}}]:copy(temp);    
    else
        _batch_input_cpu[{1,{}}]:copy( image.scale(img,network_input_width ,network_input_height)) 
    end


    
                    
    -- forward to obtain the data
    local network_output = model:forward(_batch_input_cpu:cuda());  
    cutorch.synchronize()


    -- normal image
    local normal_image = torch.Tensor(3, network_input_height, network_input_width)
    if not b_direct_normal then
        if torch.type(network_output) == 'table' then
            world_coord = depth_to_world_coord_network_focal:forward(network_output):clone()
        else
            world_coord = depth_to_world_coord_network:forward(network_output):clone()
        end        
        normal_image:copy(world_coord_to_normal_network:forward(world_coord))
    else
        normal_image:copy(network_output)
    end
    
    local orig_size_normal = image.scale(normal_image:double(), img_orig_w, img_orig_h) 
                  
    local estimated_normal_at_xy = orig_size_normal[{{}, data_handle[i].y, data_handle[i].x}]:clone()          -- this clone() is must, otherwise, it won't work with resize()!
    normalize_to_unit_vector(estimated_normal_at_xy)
    estimated_normal_at_xy:resize(3,1)

    collectgarbage()


    if b_vis then
        local _pred_z_img = torch.Tensor(1,img_orig_h,img_orig_w)
        local _normal_rgb = torch.Tensor(3,img_orig_h,img_orig_w)
        local _output_img = torch.Tensor(3, img_orig_h,img_orig_w * 5)

        -- rgb
        _output_img[{{},{1,img_orig_h},{1,img_orig_w}}]:copy(img)

        -- predicted depth
        if not b_direct_normal then
            local temp = torch.Tensor()
            if torch.type(network_output) == 'table' then
                temp = network_output[1]:double()
            else
                temp = network_output            
            end
            _pred_z_img:copy( image.scale(temp[{1,{}}], img_orig_w, img_orig_h)  )
            _pred_z_img = _pred_z_img:add( - torch.min(_pred_z_img) )
            _pred_z_img = _pred_z_img:div( torch.max(_pred_z_img ) )       

            _output_img[{1,{1,img_orig_h},{img_orig_w + 1,img_orig_w * 2}}]:copy(_pred_z_img)
            _output_img[{2,{1,img_orig_h},{img_orig_w + 1,img_orig_w * 2}}]:copy(_pred_z_img)
            _output_img[{3,{1,img_orig_h},{img_orig_w + 1,img_orig_w * 2}}]:copy(_pred_z_img)
        end
        

        -- the normal map  
        _normal_rgb:copy(orig_size_normal)          
        _normal_rgb:add(1)
        _normal_rgb:div(2)
        _output_img:sub(1,-1, 1,-1, 2 * img_orig_w + 1, 3 * img_orig_w):copy(_normal_rgb)

        image.save(cmd_params.output_folder.. '/' .. i .. '.png', _output_img)  
    end


    -- visualize_normal(orig_size_normal, string.format("%d_ours.png", i))
    return estimated_normal_at_xy
end


----------------------------------------------[[
--[[

Main Entry

]]--
------------------------------------------------



cmd = torch.CmdLine()
cmd:text('Options')

cmd:option('-num_iter',10,'number of training iteration')
cmd:option('-prev_model_file','','Absolute / relative path to the previous model file. Resume training from this file')
cmd:option('-vis', false, 'visualize output')
cmd:option('-mesh', false, 'visualize output')
cmd:option('-output_folder','./SNOW_output_images','image output folder')
cmd:option('-mode','test','mode: test or validate')
cmd:option('-valid_set', 'SNOW_val_release.csv', 'validation file name')
cmd:option('-test_set','SNOW_test_release.csv', 'test file name')
cmd:option('-model','ours', 'eigen15, marr, FCRN, or ours')
cmd:option('-direct_normal', false, 'the network directly predicts surface normals')
cmd_params = cmd:parse(arg)


if cmd_params.mode == 'test' then
    csv_file_name = '../../data/SNOW_Toolkit/' .. cmd_params.test_set       -- test set
elseif cmd_params.mode == 'validate' then
    csv_file_name = '../../data/SNOW_Toolkit/' .. cmd_params.valid_set        -- validation set 1001 images       
end




preload_t7_filename = string.gsub(csv_file_name, "csv", "t7")
if not paths.filep(preload_t7_filename) then
    print('loading csv file...', csv_file_name)    
    data_handle = parse_normal_csv(csv_file_name)
    n_sample = #data_handle

    torch.save(preload_t7_filename, data_handle)
else
    print('loading pre load t7 file...', preload_t7_filename)
    data_handle = torch.load(preload_t7_filename)
    n_sample = #data_handle
end


print("Hyper params: ")
print("csv_file_name:", csv_file_name);
print("N test samples:", n_sample);
n_iter = math.min( n_sample, cmd_params.num_iter )
print(string.format('Number of sample we are going to examine(n_iter) = %d',n_iter))

record = {}
record.n_less_than_30 = 0
record.n_less_than_22_5 = 0
record.n_less_than_11_25 = 0
record.tensor_angular_diffs = torch.Tensor(n_iter)


-- Load the trained model
prev_model_file = cmd_params.prev_model_file
model = torch.load(prev_model_file)
model:evaluate()
print("Evaluating on our model")
print("Model file:", prev_model_file)

-- main loop
for i = 1, n_iter do   

    local estimated_normal_at_xy = get_network_result(cmd_params.direct_normal, data_handle, i, model, world_coord_to_normal_network, depth_to_world_coord_network, depth_to_world_coord_network_focal, false)        
    record = evaluate_one_sample(data_handle[i].gt_normal:double(), estimated_normal_at_xy, i, record)

    
    local gt_normal = data_handle[i].gt_normal:double()
    -- print_json(record, estimated_normal_at_xy, data_handle, gt_normal, i )


    collectgarbage()
    collectgarbage()
    collectgarbage()
    collectgarbage()    
end
print_evaluation_result(record)



