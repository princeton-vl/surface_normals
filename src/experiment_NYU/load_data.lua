local train_depth_path = nil 
local train_normal_path = nil 

local valid_depth_path = nil
local valid_normal_path = nil 

local base_data_path = '../../data/'
if g_args.t_depth_file ~= '' then
	train_depth_path = base_data_path .. g_args.t_depth_file
end

if g_args.t_normal_file ~= '' then
	train_normal_path = base_data_path .. g_args.t_normal_file
end

if g_args.v_depth_file ~= '' then
	valid_depth_path = base_data_path .. g_args.v_depth_file
end

if g_args.v_normal_file ~= '' then
	valid_normal_path = base_data_path .. g_args.v_normal_file
end



if train_depth_path == nil then
	print("Error: Missing training file for depth!")
	os.exit()
end

if valid_depth_path == nil then
	print("Error: Missing validation file for depth!")
	os.exit()
end

if train_normal_path == nil and train_depth_path == nil then
	print("Error: No training files at all.")
	os.exit()
end

if (train_normal_path == nil and valid_normal_path ~= nil) or (train_normal_path ~= nil and valid_normal_path == nil)  then
	print("Error: Either train_normal_path or valid_normal_path is not valid")
	os.exit()
end

------------------------------------------------------------------------------------------------------------------


function TrainDataLoader()
	local _train_depth_path = train_depth_path
	local _train_normal_path = train_normal_path
	if g_args.n_max_depth == 0 then
		_train_depth_path = nil
		print("\t\t>>>>>>>>>>>>Warning: No depth training data specified!")
	end	

	if g_args.n_max_normal == 0 then
		_train_normal_path = nil
		print("\t\t>>>>>>>>>>>>Warning: No normal training data specified!")
	end

	if train_depth_path == nil and train_normal_path == nil then
		assert(false, ">>>>>>>>>	Error: Both normal data and depth data are nil!")
	end

	return DataLoader(_train_depth_path, _train_normal_path, g_args.n_max_depth, g_args.n_max_normal)   		
end

function ValidDataLoader()
    return DataLoader(valid_depth_path, valid_normal_path)
end

function Train_During_Valid_DataLoader()
	local _n_max_depth = g_args.n_max_depth
	local _n_max_normal = g_args.n_max_normal
	if g_args.n_max_depth == 0 then
		_n_max_depth = 800
	end	
	if g_args.n_max_normal == 0 then
		_n_max_normal = 5000
	end

    return DataLoader(train_depth_path, train_normal_path, _n_max_depth, _n_max_normal)
end