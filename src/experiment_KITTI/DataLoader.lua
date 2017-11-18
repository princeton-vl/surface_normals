require('../common/KITTI_params')
require('./DataPointer')
require 'image'
require 'xlua'
require 'hdf5'





local DataLoader = torch.class('DataLoader')

function DataLoader:__init(relative_depth_filename, normal_filename, n_max_depth, n_max_normal)
    print(">>>>>>>>>>>>>>>>> Using DataLoader")       

    self.n_max_depth = nil or n_max_depth
    self.n_max_normal = nil or n_max_normal

    print(self.n_max_depth, self.n_max_normal)      -- if the n_max_depth = nil or n_max_normal = nil , by default it will use the maximum number of point(pair).

    self:parse_depth_and_normal(relative_depth_filename, normal_filename)    
    
    -- setup data pointer to the two sources of data
    self.data_ptr_relative_depth = DataPointer(self.n_relative_depth_sample)
    self.data_ptr_normal = DataPointer(self.n_normal_sample)
    



    print(string.format('DataLoader init: \n \t%d relative depth samples \n \t%d normal samples', self.n_relative_depth_sample, self.n_normal_sample))
end

local function parse_relative_depth_line(line, n_max_depth)    
    local splits = line:split(',')
    local sample = {};
    sample.img_filename = splits[ 1 ]    
    sample.n_point = tonumber(splits[ 3 ])

    -- restrict the number of point if specified
    if n_max_depth ~= nil then
        sample.n_point = math.min(n_max_depth, sample.n_point)
    end

    return sample
end

local function parse_normal_line(line, n_max_normal)    
    local splits = line:split(',')
    local sample = {};
    sample.img_filename = splits[ 1 ]    
    sample.n_point = tonumber(splits[ 2 ])
    
    -- restrict the number of point if specified 
    if n_max_normal ~= nil then
        sample.n_point = math.min(n_max_normal, sample.n_point)
    end

    -- there may also be focal length on the 3rd term   -- to test
    if #splits > 2 then
        sample.focal_length = tonumber(splits[3])
    end

    return sample
end

local function parse_csv(filename, parsing_func, n_max_point)
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

    -- read in the image name address
    local csv_file_handle = io.open(filename, 'r');    
    local _sample_idx = 0
    while _sample_idx < _n_lines do

        local this_line = csv_file_handle:read()
        _sample_idx = _sample_idx + 1
        
        _handle[_sample_idx] = parsing_func(this_line, n_max_point)    
        
    end    
    csv_file_handle:close();

    return _handle
end

function DataLoader:parse_depth_and_normal(relative_depth_filename, normal_filename)
    if relative_depth_filename ~= nil then
        -- the file is a csv file    
        local _simplified_relative_depth_filename = string.gsub(relative_depth_filename, ".csv", "_name.csv");

        -- simplify the csv file into just address lines
        if not paths.filep(_simplified_relative_depth_filename) then
            local command = 'grep \'.png\' ' .. relative_depth_filename .. ' > ' .. _simplified_relative_depth_filename
            print(string.format("executing: %s", command))
            -- io.read()
            os.execute(command)
        else
            print(_simplified_relative_depth_filename , " already exists.")
        end

        -- parse both csv file
        self.relative_depth_handle = parse_csv(_simplified_relative_depth_filename, parse_relative_depth_line, self.n_max_depth)    

        -- the handle for the relative depth point pairs                to do 
        local hdf5_filename = string.gsub(relative_depth_filename, ".csv", ".h5");
        self.relative_depth_handle.hdf5_handle = hdf5.open(hdf5_filename,'r')  
    else
        self.relative_depth_handle = {}        
    end
    
    if normal_filename ~= nil then
        -- read in the normal data    
        self.normal_handle = parse_csv(normal_filename, parse_normal_line, self.n_max_normal)
    else
        self.normal_handle = {}
    end
 

    -- update the number of samples
    self.n_normal_sample = #self.normal_handle
    self.n_relative_depth_sample = #self.relative_depth_handle
end


function DataLoader:close()
end

local function mixed_sample_strategy1(batch_size)       -- to do
    local n_depth =  torch.random(0,batch_size)     -- also consider the case where there is no depth sample at all

    return n_depth, batch_size - n_depth
end

local function mixed_sample_strategy2(batch_size)
    local n_depth =  math.ceil(batch_size / 2)

    return n_depth, batch_size - n_depth
end


_batch_target_relative_depth_gpu = {};
_batch_target_normal_gpu = {};
for i = 1 , g_args.bs  do                                 -- to test
    _batch_target_relative_depth_gpu[i] = {}
    _batch_target_relative_depth_gpu[i].y_A = torch.CudaTensor()
    _batch_target_relative_depth_gpu[i].x_A = torch.CudaTensor()
    _batch_target_relative_depth_gpu[i].y_B = torch.CudaTensor()
    _batch_target_relative_depth_gpu[i].x_B = torch.CudaTensor()
    _batch_target_relative_depth_gpu[i].ordianl_relation = torch.CudaTensor()

    _batch_target_normal_gpu[i] = {}
    _batch_target_normal_gpu[i].x = torch.CudaTensor()
    _batch_target_normal_gpu[i].y = torch.CudaTensor()
    _batch_target_normal_gpu[i].normal = torch.CudaTensor()
end
_batch_target_normal_gpu.focal_length = torch.CudaTensor()      -- a 2D (N x 1) tensor that stores focal length    -- to test
_batch_target_relative_depth_gpu.gt_depth = torch.Tensor();
_batch_target_relative_depth_gpu.gt_depth_size = {};
_batch_target_normal_gpu.gt_depth = torch.Tensor();
_batch_target_normal_gpu.gt_depth_size = torch.Tensor();


function DataLoader:load_indices( depth_indices, normal_indices, b_load_gtz )
    b_load_gtz = b_load_gtz or false

    local n_depth, n_normal
    if depth_indices ~= nil and self.n_relative_depth_sample > 0 then
        n_depth = depth_indices:size(1)
    else
        n_depth = 0
    end

    if normal_indices ~= nil and self.n_normal_sample > 0 then
        n_normal = normal_indices:size(1)
    else
        n_normal = 0
    end


    if n_depth == 0 and n_normal == 0 then
        assert(false, "---->>>> Warning: Both n_depth and n_normal equal 0 in DataLoader:load_indices().")        
    end


    local batch_size = n_depth + n_normal

    local color = torch.Tensor();    
    color:resize(batch_size, 3, g_input_height, g_input_width); 
    if n_depth > 0 then
        _batch_target_relative_depth_gpu.gt_depth:resize(batch_size, 400, 1300)
        _batch_target_relative_depth_gpu.gt_depth_size = {};
    end
    if n_normal > 0 then
        _batch_target_normal_gpu.gt_depth:resize(batch_size, 400, 1300)
        _batch_target_normal_gpu.gt_depth_size = {};
    end


    _batch_target_relative_depth_gpu.n_sample = n_depth
    _batch_target_normal_gpu.n_sample = n_normal

    -- Read the relative depth data
    for i = 1, n_depth do    
        
        local idx = depth_indices[i]
        local img_name = self.relative_depth_handle[idx].img_filename
        local n_point = self.relative_depth_handle[idx].n_point

        -- print(string.format("Loading %s, n_depth_pair = %d", img_name, n_point))
        
        -- read the input image
        color[{i,{}}]:copy(image.load(img_name));    -- Note that the image read is in the range of 0~1
        -- read the groundtruth depth
        if b_load_gtz then
            local gt_z_filename = string.gsub(img_name, '.png', '_gt_depth.h5')
            if path.exists(gt_z_filename) then                
                local gtz_h5_handle = hdf5.open(gt_z_filename, 'r')
                -- print(gt_z_filename)
                local in_gt = gtz_h5_handle:read('/depth'):all()
                _batch_target_relative_depth_gpu.gt_depth_size[i] = in_gt:size();
                _batch_target_relative_depth_gpu.gt_depth[{i,{1,in_gt:size(1)}, {1,in_gt:size(2)}}]:copy(in_gt)
                gtz_h5_handle:close()
            else
                print("File not found:", gt_z_filename)
                _batch_target_relative_depth_gpu.gt_depth[{i,{}}]:copy(torch.rand(400,1300))
            end
            
        end


        -- relative depth
        local _hdf5_offset = 5 * (idx - 1) + 1        
        local _this_sample_hdf5  = self.relative_depth_handle.hdf5_handle:read('/data'):partial({_hdf5_offset, _hdf5_offset + 4}, {1, n_point})

        assert(_this_sample_hdf5:size(1) == 5)
        assert(_this_sample_hdf5:size(2) == n_point)

        -- Pay attention to the order!!!!        
        _batch_target_relative_depth_gpu[i].y_A:resize(n_point):copy(_this_sample_hdf5[{1,{}}])         -- to check if is correct
        _batch_target_relative_depth_gpu[i].x_A:resize(n_point):copy(_this_sample_hdf5[{2,{}}])
        _batch_target_relative_depth_gpu[i].y_B:resize(n_point):copy(_this_sample_hdf5[{3,{}}])
        _batch_target_relative_depth_gpu[i].x_B:resize(n_point):copy(_this_sample_hdf5[{4,{}}])
        _batch_target_relative_depth_gpu[i].ordianl_relation:resize(n_point):copy(_this_sample_hdf5[{5,{}}])
        _batch_target_relative_depth_gpu[i].n_point = n_point

        -- -- for debug        
        -- print(img_name)
        -- for k = 1, n_point do    
        --     print(string.format("%d,%d,%d,%d,%d", _batch_target_relative_depth_gpu[i].y_A[k], _batch_target_relative_depth_gpu[i].x_A[k], _batch_target_relative_depth_gpu[i].y_B[k], _batch_target_relative_depth_gpu[i].x_B[k], _batch_target_relative_depth_gpu[i].ordianl_relation[k]))
        -- end
        -- io.read()
    end       


    -- Read the normal data
    _batch_target_normal_gpu.focal_length:resize(n_normal,1)        -- ATTENTION: the length of focal_length is n_normal!!!!
    for i = n_depth + 1 , batch_size do

        local idx = normal_indices[i - n_depth]        
        local img_name = self.normal_handle[idx].img_filename
        local n_point = self.normal_handle[idx].n_point

        -- print(string.format("Loading %s, n_normal_point = %d", img_name, n_point))

        -- the focal length   -- to test
        _batch_target_normal_gpu.focal_length[{i - n_depth,1}] = self.normal_handle[idx].focal_length
        

        -- read the input image, pay attention to the image index
        color[{i,{}}]:copy(image.load(img_name));    -- Note that the image read is in the range of 0~1
        -- read the groundtruth depth
        if b_load_gtz then
            local gt_z_filename = string.gsub(img_name, '.png', '_gt_depth.h5')
            if path.exists(gt_z_filename) then
                local gtz_h5_handle = hdf5.open(gt_z_filename, 'r')
                -- print(gt_z_filename)
                local in_gt = gtz_h5_handle:read('/depth'):all()
                _batch_target_normal_gpu.gt_depth[{i - n_depth,{1,in_gt:size(1)}, {1,in_gt:size(2)}}]:copy(in_gt)
                _batch_target_normal_gpu.gt_depth_size[i - n_depth] = in_gt:size();
                gtz_h5_handle:close()
            else
                print("File not found:", gt_z_filename)
                _batch_target_normal_gpu.gt_depth[{i - n_depth,{}}]:copy(torch.rand(480,640))
            end         
        end

        -- normal
        -- normal[{{3,3},{}}] is x , normal[{{4,4},{}}] is y, normal[{{5,5},{}}] is z. 
        -- It's a left hand coordinate system, i.e, x points left, y points down, z points toward us.
        local normal_name = string.gsub(img_name, ".png", "_normal.bin")
        local file = torch.DiskFile(normal_name, 'r'):binary();    
        local normal = torch.DoubleTensor(file:readDouble(5 * n_point))
        file:close();                
        normal = torch.view(normal, n_point, 5)                 -- tested
        normal = normal:t()
        normal = normal:cuda()

        -- pay atteniton to the order!
        _batch_target_normal_gpu[i - n_depth].x:resize(n_point):copy(normal[{1,{}}]:int())
        _batch_target_normal_gpu[i - n_depth].y:resize(n_point):copy(normal[{2,{}}]:int())
        _batch_target_normal_gpu[i - n_depth].normal = normal[{{3,5},{}}]:clone()       -- To check.  Do we need to clone()? 
        _batch_target_normal_gpu[i - n_depth].n_point = n_point
        

        -- -- debug
        -- print(normal:size())
        -- print(torch.type(normal))
        -- print(n_point)
        -- print(normal:sub(1,-1, 1, 4))
        -- print(_batch_target_normal_gpu[i - n_depth].x:sub(1,4))
        -- print(_batch_target_normal_gpu[i - n_depth].y:sub(1,4))
        -- print(_batch_target_normal_gpu[i - n_depth].normal:sub(1,-1, 1, 4))
        -- print(img_name)
        -- print(_batch_target_normal_gpu.focal_length)
        -- io.read()
    end

    return color:cuda(), {_batch_target_relative_depth_gpu, _batch_target_normal_gpu}
end


function DataLoader:load_next_batch(batch_size)

    -- determine the number of relative depth samples and normal sample used in this iteration      -- to test
    local n_depth, n_normal;
    if self.n_normal_sample > 0 and self.n_relative_depth_sample > 0 then
        n_depth, n_normal = mixed_sample_strategy1(batch_size)
    elseif self.n_normal_sample > 0 then
        n_normal = batch_size
        n_depth = 0
    elseif self.n_relative_depth_sample > 0 then
        n_normal = 0
        n_depth = batch_size
    else
        n_normal = 0
        n_depth = 0
        assert(false, ">>>>>>>>>    Error: No depth sample or normal sample!")
    end
    -- print(string.format("n_depth = %d, n_normal = %d", n_depth, n_normal))


    -- Obtain the indices for each group of data
    local depth_indices = self.data_ptr_relative_depth:load_next_batch(n_depth)
    local normal_indices = self.data_ptr_normal:load_next_batch(n_normal)

    return self:load_indices( depth_indices, normal_indices )
end



function DataLoader:reset()
    self.current_pos = 1
end
