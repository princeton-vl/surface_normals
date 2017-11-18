require 'hdf5'
require('optim')
require('os')
require('cunn')
require('paths')


-- paths.dofile('../evaluation/evaluate.lua')



cmd = torch.CmdLine()
cmd:option('-m', 'hourglass3', 'model file definition')
cmd:option('-bs', 4, 'batch size')
cmd:option('-it', 0, 'Iterations')
cmd:option('-lt', 1000, 'Loss file saving refresh interval (seconds)')
cmd:option('-mt', 10000, 'Model saving interval (iterations)')
cmd:option('-et', 3000, 'Model evaluation interval (iterations)')
cmd:option('-lr', 1e-2, 'Learning rate')
cmd:option('-w_n', 1, 'Weight for the normal loss')
cmd:option('-t_depth_file','','Training file for relative depth')
cmd:option('-v_depth_file','','Validation file for relative depth')
cmd:option('-t_normal_file','','Training file for normal')
cmd:option('-v_normal_file','','Validation file for normal')
cmd:option('-rundir', '', 'Running directory')
cmd:option('-ep', 10, 'Epochs')
cmd:option('-start_from','', 'Start from previous model')
cmd:option('-diw',false,'Is training on the DIW dataset')
cmd:option('-snow',false,'Is training on the SNOW dataset')
cmd:option('-nyu',false,'Is training on the nyu full metric depth')
cmd:option('-direct_normal',false,'Is directly predicting normal')
cmd:option('-margin', 1, 'margin for the margin loss')
cmd:option('-var_thresh', 1, 'threshold for variance loss')
cmd:option('-n_max_depth', 800, 'maximum number of depth pair loaded per sample during training(validation is excluded)')
cmd:option('-n_max_normal', 50000, 'maximum number of normal point loaded per sample during training(validation is excluded)')
cmd:option('-n_scale', 1, 'number of scale we want to compare to')

g_scales = {1, 4, 8}

g_args = cmd:parse(arg)

-- Data Loader
if g_args.direct_normal and g_args.snow then
    paths.dofile('./DataLoader_SNOW.lua')
    require('./validation_crit/validate_crit_direct_normal_SNOW')
    -- here the model should be hourglass3_softplus_direct_normal_l2_SNOW.lua
elseif g_args.direct_normal then
    paths.dofile('./DataLoader_NYU_Full.lua')
    require('./validation_crit/validate_crit_direct_normal')
    -- here the model should be hourglass3_softplus_direct_normal_neg_cos.lua or hourglass3_softplus_direct_normal_l2.lua
elseif g_args.nyu then
    paths.dofile('./DataLoader_NYU_Full.lua')
    require('./validation_crit/validate_crit_NULL')
elseif g_args.snow and g_args.diw then
    paths.dofile('./DataLoader_SNOW_DIW.lua')
    require('./validation_crit/validate_crit_SNOW_DIW')
elseif g_args.diw then
    paths.dofile('./DataLoader_DIW.lua')
    require('./validation_crit/validate_crit_DIW')
elseif g_args.snow then
    paths.dofile('./DataLoader_SNOW.lua')
    require('./validation_crit/validate_crit_SNOW')
else
    if g_args.n_scale == 1 then
        paths.dofile('./DataLoader.lua')
    else
        paths.dofile('./DataLoader_multi_res.lua')
    end    
    require('./validation_crit/validate_crit1')
end
paths.dofile('load_data.lua')
g_train_loader = TrainDataLoader()
g_valid_loader = ValidDataLoader()
g_train_during_valid_loader = Train_During_Valid_DataLoader()



----------to modify
if g_args.it == 0 then
    g_args.it = g_args.ep * (g_train_loader.n_normal_sample + g_train_loader.n_relative_depth_sample) / g_args.bs
end

-- Run path
g_args.rundir = '../results/' .. g_args.m .. '/' .. g_args.rundir ..'/'
paths.mkdir(g_args.rundir)
torch.save(g_args.rundir .. '/g_args.t7', g_args)


-- Model
local config = {}
require('./models/' .. g_args.m)
if g_args.start_from ~= '' then
    require 'cudnn'
    print(g_args.rundir .. g_args.start_from)
    g_model = torch.load(g_args.rundir .. g_args.start_from);
    if g_model.period == nil then
        g_model.period = 1
    end
    g_model.period = g_model.period + 1
    config = g_model.config
else
    g_model = get_model()
    g_model.period = 1
end
g_model:training()
config.learningRate = g_args.lr



-- Criterion. get_criterion is a function, which is specified in the network model file
if get_criterion == nil then
    print("Error: no criterion specified!!!!!!!")
    os.exit()
end


-- Validation Criteria



-- Function that obtain depth from the model output, used in validation
get_depth_from_model_output = f_depth_from_model_output()
if get_depth_from_model_output == nil then
    print("Error: get_depth_from_model_output is undefined!!!!!!!")
    os.exit()    
end


-- Variables that used globally
g_criterion = get_criterion()
g_model = g_model:cuda()
g_criterion = g_criterion:cuda()
g_params, g_grad_params = g_model:getParameters()




local function default_feval(current_params)
    -- timer = torch.Timer()
    local batch_input, batch_target = g_train_loader:load_next_batch(g_args.bs)
    -- print('Load Data: ' .. timer:time().real .. ' seconds')
    

    -- reset grad_params
    g_grad_params:zero()    


    
    --forward & backward
    -- timer = torch.Timer()
    local batch_output = g_model:forward(batch_input)    
    -- print('g_model:forward: ' .. timer:time().real .. ' seconds')
    -- timer = torch.Timer()
    local batch_loss = g_criterion:forward(batch_output, batch_target)
    -- print('g_criterion:forward: ' .. timer:time().real .. ' seconds')
    -- timer = torch.Timer()
    local dloss_dx = g_criterion:backward(batch_output, batch_target)
    -- print('g_criterion:backward: ' .. timer:time().real .. ' seconds')
    -- timer = torch.Timer()
    g_model:backward(batch_input, dloss_dx)    
    -- print('g_model:backward: ' .. timer:time().real .. ' seconds')


    collectgarbage()

    return batch_loss, g_grad_params
end

local function save_loss_accuracy(t_loss, t_WKDR, v_loss, v_WKDR)       -- to check
    -- first convert to tensor    
    local _v_loss_tensor = torch.Tensor(v_loss)    
    local _t_loss_tensor = torch.Tensor(t_loss)
    local _v_WKDR_tensor = torch.Tensor(v_WKDR)    
    local _t_WKDR_tensor = torch.Tensor(t_WKDR)

    -- print(_v_loss_tensor)
    -- print(_t_loss_tensor)
    -- print(_v_WKDR_tensor)
    -- print(_t_WKDR_tensor)

    -- first remove the existing file
    local _full_filename = g_args.rundir .. 'loss_accuracy_record_period' .. g_model.period .. '.h5'
    os.execute("rm " .. _full_filename)
    -- print("rm " .. _full_filename)
    
    local myFile = hdf5.open(_full_filename, 'w')
    myFile:write('/t_loss', _t_loss_tensor)        
    myFile:write('/v_loss', _v_loss_tensor)        
    myFile:write('/t_WKDR', _t_WKDR_tensor)        
    myFile:write('/v_WKDR', _v_WKDR_tensor)  
    myFile:close()    
end

local function save_model(model, dir, current_iter, config)
    model:clearState()          -- to do: save the state of the optimizer
    model.config = config
    torch.save(dir .. '/model_period'.. model.period .. '_' .. current_iter  .. '.t7' , model)
end

local function save_best_model(model, dir, config, iter)
    model:clearState()          -- to do: save the state of the optimizer
    model.config = config
    model.iter = iter
    torch.save(dir .. '/Best_model_period' .. model.period .. '.t7' , model)
end









-----------------------------------------------------------------------------------------------------

if feval == nil then
    feval = default_feval
end





local best_valid_set_error_rate = 1
local train_loss = {};
local train_WKDR = {};
local valid_depth_loss = {};
local valid_WKDR = {};

local lfile = torch.DiskFile(g_args.rundir .. '/training_loss_period' .. g_model.period .. '.txt', 'w')

for iter = 1, g_args.it do
    local params, current_loss = optim.rmsprop(feval, g_params, config)
    print(current_loss[1])
    lfile:writeString(current_loss[1] .. '\n')

    


    train_loss[#train_loss + 1] = current_loss[1]

    if iter % g_args.mt == 0 then        
        print(string.format('Saving model at iteration %d...', iter))
        save_model(g_model, g_args.rundir, iter, config)        
    end
    if iter % g_args.lt == 0 then
        print(string.format('Flusing training loss file at iteration %d...', iter))
        lfile:synchronize()        
        __loss_normal_file:synchronize()
        __loss_depth_file:synchronize()
    end
    -- if false then
    if iter % g_args.et == 0 or iter == 1 then        -- to do
        print(string.format('Evaluating at iteration %d...', iter))
        
        local _train_depth_loss, _train_eval_WKDR, _train_normal_loss, _train_angle_diff, _train_rmse, _train_rmse_si, _train_lsi = evaluate( g_train_during_valid_loader, g_model, g_criterion, 100 )  
        local _valid_depth_loss, _valid_eval_WKDR, _valid_normal_loss, _valid_angle_diff, _valid_rmse, _valid_rmse_si, _valid_lsi = evaluate( g_valid_loader, g_model, g_criterion, 100 )  


        
        -- Pay attention that the _train_depth_loss is not used here
        valid_depth_loss[#valid_depth_loss + 1] = _valid_depth_loss
        valid_WKDR[#valid_WKDR + 1] = _valid_eval_WKDR
        train_WKDR[#train_WKDR + 1] = _train_eval_WKDR
        save_loss_accuracy(train_loss, train_WKDR, valid_depth_loss, valid_WKDR)


        if best_valid_set_error_rate > _valid_eval_WKDR then
            best_valid_set_error_rate = _valid_eval_WKDR
            save_best_model(g_model, g_args.rundir, config, iter)
        end
    end
end

-- evaluate(g_model, g_args.bs, g_valid_loader)
lfile:close()
__loss_normal_file:close()
__loss_depth_file:close()
__valid_angular_diff_file:close()
__valid_normal_loss_file:close()
__training_angular_diff_file:close()
__training_rmse_file:close()
__training_rmse_si_file:close()
__training_lsi_file:close()
__valid_rmse_file:close()
__valid_rmse_si_file:close()
__valid_lsi_file:close()


g_train_loader:close()
g_valid_loader:close()
g_train_during_valid_loader:close()
save_model(g_model, g_args.rundir, g_args.it, config)
