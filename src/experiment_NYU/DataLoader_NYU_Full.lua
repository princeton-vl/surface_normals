require('../common/NYU_params')
require('./DataPointer')
require 'image'
require 'xlua'
require 'hdf5'





local DataLoader = torch.class('DataLoader')

function DataLoader:__init(absolute_depth_filename, normal_filename, n_max_depth, n_max_normal)
    print(">>>>>>>>>>>>>>>>> Using DataLoader")       

    self.n_max_normal = nil or n_max_normal

    print(self.n_max_normal)      -- if the n_max_depth = nil or n_max_normal = nil , by default it will use the maximum number of point(pair).

    self:parse_depth_and_normal(absolute_depth_filename, normal_filename)    
    
    -- setup data pointer to the two sources of data
    self.data_ptr_relative_depth = DataPointer(self.n_relative_depth_sample)
    self.data_ptr_normal = DataPointer(self.n_normal_sample)
    



    print(string.format('DataLoader init: \n \t%d relative depth samples \n \t%d normal samples', self.n_relative_depth_sample, self.n_normal_sample))
end

local function _get_depth_data(res, filename)

    local myFile = hdf5.open(filename, 'r')
    local depth = myFile:read('/metric_depth'):all()
    myFile:close()

    res:copy(depth)
end

local function _get_normal_data(res, filename)
    local myFile = hdf5.open(filename, 'r')
    local normal = myFile:read('/normal'):all()
    normal = normal:transpose(2,3)
    local nan_mask = normal:ne(normal)
    normal[nan_mask] = 0
    myFile:close()

    res:copy(normal)
end

function DataLoader:parse_full_depth_input_file(_filename)
    print("Parsing full depth input file", _filename)
    local number = 0;
    number = 0;
    for _ in io.lines(_filename) do
      number = number + 1
    end

    print("number of line = ", number)

    local file_handle = io.open(_filename, 'r');
    self.absolute_depth_handle ={}

    for i = 1 , number do
        self.absolute_depth_handle[i] = {};
        self.absolute_depth_handle[i].img_filename = file_handle:read();
        self.absolute_depth_handle[i].depth_filename = string.gsub(self.absolute_depth_handle[i].img_filename, ".png", '_depth.h5')
    end
    file_handle:close();

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

function DataLoader:parse_depth_and_normal(absolute_depth_filename, normal_filename)
    if absolute_depth_filename ~= nil then
       self:parse_full_depth_input_file(absolute_depth_filename) 
    else
        self.absolute_depth_handle = {}        
    end
    
    if normal_filename ~= nil then
        -- read in the normal data    
        self.normal_handle = parse_csv(normal_filename, parse_normal_line, self.n_max_normal)
    else
        self.normal_handle = {}
    end
 

    -- update the number of samples
    self.n_normal_sample = #self.normal_handle
    self.n_relative_depth_sample = #self.absolute_depth_handle
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


local function visualize_normal(normal, filename)
    local _normal_img = normal:clone()
    _normal_img:add(1)
    _normal_img:mul(0.5)
    image.save(filename, _normal_img) 

    print("Done saving to ", filename)
end


_batch_target_depth_gpu = {};
_batch_target_normal_gpu = torch.CudaTensor()
_batch_target_depth_gpu.full_metric_depth = torch.CudaTensor()

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
        _batch_target_depth_gpu.full_metric_depth:resize(n_depth, 1, g_input_height, g_input_width);
        _batch_target_depth_gpu.full_metric_depth:zero()
    end

    if n_normal > 0 then
        _batch_target_normal_gpu:resize(n_normal, 3, g_input_height, g_input_width);
    end

    _batch_target_depth_gpu.n_sample = n_depth

    -- Read the absolute full depth map
    for i = 1, n_depth do    
        
        local idx = depth_indices[i]
        local img_name = self.absolute_depth_handle[idx].img_filename
        local depth_filename = self.absolute_depth_handle[idx].depth_filename;

        -- print(string.format("Loading %s", img_name))
        
        -- read the input image
        color[{i,{}}]:copy(image.load(img_name));    -- Note that the image read is in the range of 0~1
        -- read the full depthmap
        _get_depth_data( _batch_target_depth_gpu.full_metric_depth[{i,{}}], depth_filename);

    end       


    -- Read the normal data
    for i = n_depth + 1 , batch_size do

        local idx = normal_indices[i - n_depth]        
        local img_name = self.normal_handle[idx].img_filename
        local n_point = self.normal_handle[idx].n_point

        -- print(string.format("Loading %s, n_normal_point = %d", img_name, n_point))

        

        -- read the input image, pay attention to the image index
        color[{i,{}}]:copy(image.load(img_name));    -- Note that the image read is in the range of 0~1
        -- read the groundtruth depth

        -- normal
        -- normal[{{3,3},{}}] is x , normal[{{4,4},{}}] is y, normal[{{5,5},{}}] is z. 
        -- It's a left hand coordinate system, i.e, x points left, y points down, z points toward us.
        local normal_name = string.gsub(img_name, "_NC.png", "_normal_qual_depth.h5")
        
        _get_normal_data( _batch_target_normal_gpu[{i,{}}], normal_name);

    end

    return color:cuda(), {_batch_target_depth_gpu, _batch_target_normal_gpu}
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
