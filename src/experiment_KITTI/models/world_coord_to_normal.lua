require 'nngraph'
require('../../common/KITTI_params')

-- -- for debug only
-- local g_input_width = 640
-- local g_input_height = 480

-- local g_fx_rgb = 5.1885790117450188e+02;
-- local g_fy_rgb = -5.1946961112127485e+02;
-- local g_cx_rgb = 3.2558244941119034e+02;
-- local g_cy_rgb = 2.5373616633400465e+02;


local cross_prod, Parent = torch.class('nn.cross_prod', 'nn.Module')
local elementwise_division, Parent = torch.class('nn.elementwise_division', 'nn.Module')
local spatial_normalization, Parent = torch.class('nn.spatial_normalization', 'nn.Module')
local vector_path_layer, parent = torch.class('nn.vector_path_layer', 'nn.Module')


-------------------------------------------------------------------------------

function vector_path_layer:__init(kernel)
   Parent.__init(self)   
   self.model = nn.SpatialConvolution(1, 1, kernel:size(2), kernel:size(1), 1, 1, 1, 1)
   self.model.weight:copy(kernel) 
   self.model.bias:zero()
end

function vector_path_layer:updateOutput(input)
   -- the input should be a 3 channel tensor X,Y,Z
   if self.output then     
      if self.output:type() ~= input:type() then
         self.output = self.output:typeAs(input);
      end
      self.output:resizeAs(input)
   end

   self.output:sub(1,-1, 1,1):copy(self.model:forward(input:sub(1,-1, 1,1)))
   self.output:sub(1,-1, 2,2):copy(self.model:forward(input:sub(1,-1, 2,2)))
   self.output:sub(1,-1, 3,3):copy(self.model:forward(input:sub(1,-1, 3,3)))
   return self.output
end

function vector_path_layer:updateGradInput(input, gradOutput)
   -- the input should be a 3 channel tensor X,Y,Z
   -- the gradOutput should also be a 3 channel tensor X,Y,Z
   if self.gradInput then     
      if self.gradInput:type() ~= input:type() then
         self.gradInput = self.gradInput:typeAs(input);
      end
      self.gradInput:resizeAs(input)      
      self.gradInput:zero()
   end

   -- self.model.gradInput:zero()      -- no need to zero it, it won't accumulate
   self.gradInput:sub(1,-1, 1,1):copy(self.model:backward(input:sub(1,-1, 1,1), gradOutput:sub(1,-1, 1,1)))
   -- self.model.gradInput:zero()      -- no need to zero it, it won't accumulate
   self.gradInput:sub(1,-1, 2,2):copy(self.model:backward(input:sub(1,-1, 2,2), gradOutput:sub(1,-1, 2,2)))
   -- self.model.gradInput:zero()      -- no need to zero it, it won't accumulate
   self.gradInput:sub(1,-1, 3,3):copy(self.model:backward(input:sub(1,-1, 3,3), gradOutput:sub(1,-1, 3,3)))

   return self.gradInput
end

---------------------------------------------------------------------------------------
function elementwise_division:__init()
   Parent.__init(self) 
   self.gradInput = {}
   self.gradInput[1] = torch.Tensor()     
   self.gradInput[2] = torch.Tensor()     
end

function elementwise_division:updateOutput(input)
   if self.output then     
      if self.output:type() ~= input[1]:type() then
         self.output = self.output:typeAs(input[1]);
      end
      self.output:resizeAs(input[1])
   end
   -- the input is a table
   self.output:cdiv(input[1], input[2])
   return self.output
end

function elementwise_division:updateGradInput(input, gradOutput)
   if self.gradInput then     
      for i = 1 , 2 do
         if self.gradInput[i]:type() ~= input[i]:type() then
            self.gradInput[i] = self.gradInput[i]:typeAs(input[i]);
         end
         self.gradInput[i]:resizeAs(input[i])
      end      
   end
   

   self.gradInput[1]:cdiv(gradOutput, input[2])
   
   self.gradInput[2]:cmul(gradOutput, input[1])
   self.gradInput[2]:cdiv(input[2])
   self.gradInput[2]:cdiv(input[2])
   self.gradInput[2]:mul(-1)   

   return self.gradInput
end

--------------------------------------------------------------------------------------
function spatial_normalization:__init()
   Parent.__init(self)   
   self.square_buffer = torch.Tensor()
   self.square_sum_buffer = torch.Tensor()
   self.input_size_buffer = torch.Tensor()
end

function spatial_normalization:updateOutput(input)
   if self.output then     
      if self.output:type() ~= input:type() then
         self.output = self.output:typeAs(input);
         self.square_buffer = self.square_buffer:typeAs(input);
         self.square_sum_buffer = self.square_sum_buffer:typeAs(input);
      end
      self.output:resizeAs(input)
      self.square_buffer:resizeAs(input)
      self.square_sum_buffer:resize(input:size(1),1,input:size(3),input:size(4))
   end

   -- the input is a N x 3 x H x W tensor
   self.square_buffer:copy(input)
   self.square_buffer:pow(2)   
   
   
   self.square_sum_buffer:add(self.square_buffer[{{},1,{}}], self.square_buffer[{{},2,{}}])
   self.square_sum_buffer:add(self.square_buffer[{{},3,{}}])
   self.square_sum_buffer:sqrt()
   self.square_sum_buffer:add(0.000000001)                        -- in case of divide by zero  
   

   self.output:copy(input)   
   self.output[{{},1,{}}]:cdiv(self.square_sum_buffer)            -- x
   self.output[{{},2,{}}]:cdiv(self.square_sum_buffer)            -- y
   self.output[{{},3,{}}]:cdiv(self.square_sum_buffer)            -- z

   return self.output
end

function spatial_normalization:updateGradInput(input, gradOutput)
   if self.gradInput then     
      for i = 1 , 2 do
         if self.gradInput:type() ~= input:type() then
            self.gradInput = self.gradInput:typeAs(input);
         end
         self.gradInput:resizeAs(input)
         self.input_size_buffer:resizeAs(input)
      end      
   end

------------------------This part may be unnecessary since already done in updateOutput---------------------------
   self.square_buffer:copy(input)
   self.square_buffer:pow(2)   
   self.square_sum_buffer:add(self.square_buffer[{{},1,{}}], self.square_buffer[{{},2,{}}])
   self.square_sum_buffer:add(self.square_buffer[{{},3,{}}])
   self.square_sum_buffer:sqrt()
   self.square_sum_buffer:add(0.000000001)                        -- in case of divide by zero  


------------------------------------------------------------------------------------------------------------------
   self.square_sum_buffer:pow(3)    
   self.input_size_buffer:zero()
   self.gradInput:zero()

   self.input_size_buffer[{{},1}]:add(self.square_buffer[{{},2,{}}], self.square_buffer[{{},3,{}}])       -- y^2 + z^2    
   self.input_size_buffer[{{},2}]:addcmul(-1, input[{{},1,{}}], input[{{},2,{}}])                         -- -xy      
   self.input_size_buffer[{{},3}]:addcmul(-1, input[{{},1,{}}], input[{{},3,{}}])                         -- -xz   
   self.input_size_buffer:cmul(gradOutput)
   self.gradInput[{{},1}]:add(self.input_size_buffer[{{},1}], self.input_size_buffer[{{},2}])
   self.gradInput[{{},1}]:add(self.input_size_buffer[{{},3}])
   self.gradInput[{{},1}]:cdiv(self.square_sum_buffer)
   self.input_size_buffer:zero()


   self.input_size_buffer[{{},1}]:addcmul(-1, input[{{},1,{}}], input[{{},2,{}}])                         -- -xy      
   self.input_size_buffer[{{},2}]:add(self.square_buffer[{{},1,{}}], self.square_buffer[{{},3,{}}])       -- x^2 + z^2    
   self.input_size_buffer[{{},3}]:addcmul(-1, input[{{},2,{}}], input[{{},3,{}}])                         -- -yz       
   self.input_size_buffer:cmul(gradOutput)
   self.gradInput[{{},2}]:add(self.input_size_buffer[{{},1}], self.input_size_buffer[{{},2}])
   self.gradInput[{{},2}]:add(self.input_size_buffer[{{},3}])
   self.gradInput[{{},2}]:cdiv(self.square_sum_buffer)
   self.input_size_buffer:zero()



   self.input_size_buffer[{{},1}]:addcmul(-1, input[{{},1,{}}], input[{{},3,{}}])                         -- -xz      
   self.input_size_buffer[{{},2}]:addcmul(-1, input[{{},2,{}}], input[{{},3,{}}])                         -- -yz       
   self.input_size_buffer[{{},3}]:add(self.square_buffer[{{},1,{}}], self.square_buffer[{{},2,{}}])       -- x^2 + y^2   
   self.input_size_buffer:cmul(gradOutput)
   self.gradInput[{{},3}]:add(self.input_size_buffer[{{},1}], self.input_size_buffer[{{},2}])
   self.gradInput[{{},3}]:add(self.input_size_buffer[{{},3}])
   self.gradInput[{{},3}]:cdiv(self.square_sum_buffer)

   return self.gradInput
end



--------------------------------------------------------------------------------------
function cross_prod:__init()
   Parent.__init(self)   
   self.gradInput = {}   
   self.gradInput[1] = torch.Tensor()
   self.gradInput[2] = torch.Tensor()
end

function cross_prod:updateOutput(input)
   -- the input is a table with two tensor, each with 3 channels
   -- the output is a tensor
   if self.output then     
      if self.output:type() ~= input[1]:type() then
         self.output = self.output:typeAs(input[1]);
      end
      self.output:resizeAs(input[1])
   end

   self.output:zero()      -- do this first

  
   
   -- remember that it's shallow copy
   self.output:copy(torch.cross(input[1], input[2], 2))
   
   return self.output
end

function cross_prod:updateGradInput(input, gradOutput)
   -- the input is a table with two tensor
   -- the output is a tensor   
   if self.gradInput then     
      for i = 1 , 2 do
         if self.gradInput[i]:type() ~= input[i]:type() then
            self.gradInput[i] = self.gradInput[i]:typeAs(input[i]);
         end
         self.gradInput[i]:resizeAs(input[i])
      end      
   end

   self.gradInput[1]:zero()
   self.gradInput[2]:zero()
   
   
   -- remember that they are shallow copy
   local u = input[1]
   local v = input[2]
   local dE_ds = gradOutput
   local dE_du = self.gradInput[1]
   local dE_dv = self.gradInput[2]


   local u1 = u[{{},1,{}}]    
   local v1 = v[{{},1,{}}]    
   local dE_ds1 = dE_ds[{{},1,{}}]    
   local dE_du1 = dE_du[{{},1,{}}]    
   local dE_dv1 = dE_dv[{{},1,{}}]    

   local u2 = u[{{},2,{}}]    
   local v2 = v[{{},2,{}}]    
   local dE_ds2 = dE_ds[{{},2,{}}]    
   local dE_du2 = dE_du[{{},2,{}}]    
   local dE_dv2 = dE_dv[{{},2,{}}]    


   local u3 = u[{{},3,{}}]    
   local v3 = v[{{},3,{}}]    
   local dE_ds3 = dE_ds[{{},3,{}}]    
   local dE_du3 = dE_du[{{},3,{}}]    
   local dE_dv3 = dE_dv[{{},3,{}}]    

   dE_du1:addcmul(-1, dE_ds2, v3)
   dE_du1:addcmul(dE_ds3, v2)

   dE_du2:addcmul(dE_ds1, v3)      
   dE_du2:addcmul(-1, dE_ds3, v1)

   dE_du3:addcmul(-1, dE_ds1,v2)
   dE_du3:addcmul(dE_ds2, v1)

   dE_dv1:addcmul(dE_ds2, u3)
   dE_dv1:addcmul(-1, dE_ds3, u2)

   dE_dv2:addcmul(-1, dE_ds1, u3)
   dE_dv2:addcmul(dE_ds3, u1)

   dE_dv3:addcmul(dE_ds1, u2)
   dE_dv3:addcmul(-1, dE_ds2, u1)


   return self.gradInput
end

--------------------------------------------------------------------------------------

local function normal_path( index1, index2 )   
   local concat_path = nn.ConcatTable()
   concat_path:add(nn.SelectTable(index1))
   concat_path:add(nn.SelectTable(index2))   
   -- the output of concat_path is a table with two tensor selected from the input table
   
   local path = nn.Sequential()
   path:add(concat_path)
   path:add(nn.cross_prod())
   path:add(nn.spatial_normalization())

   return path
end

function normalization_path()
   local root_square_sum_path = nn.Sequential()
   root_square_sum_path:add(nn.Square())
   root_square_sum_path:add(nn.Sum(2))
   root_square_sum_path:add(nn.Sqrt())
   root_square_sum_path:add(nn.AddConstant(0.0000000001))
   root_square_sum_path:add(nn.Replicate(3, 2))

   local concat_path = nn.ConcatTable()
   concat_path:add(nn.Identity())
   concat_path:add(root_square_sum_path)

   local main_path = nn.Sequential()
   main_path:add(concat_path)
   main_path:add(nn.elementwise_division())

   return main_path
end


function world_coord_to_normal()

   local kernel_up = torch.Tensor( { {0,-1,0}, {0,0,0},{0,1,0} } )
   local kernel_left = torch.Tensor( { {0,0,0}, {-1,0,1}, {0,0,0} } )

   local v_up = nn.vector_path_layer(kernel_up)
   local v_left = nn.vector_path_layer(kernel_left)

   local v_path = nn.ConcatTable()
   v_path:add(v_up)
   v_path:add(v_left)


   local model = nn.Sequential()
   model:add(v_path)
   -- up to this point, the output is a table, with 2 tensors: v_up, v_left


   local n_path = normal_path(1,2)

   model:add(n_path)
   -- up to this point, the output is a tensors
   
   return model
end
