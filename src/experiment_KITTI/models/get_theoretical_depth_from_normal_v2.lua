require 'nn'
require('../../common/KITTI_params')

local ABC_sum, parent = torch.class('nn.ABC_sum', 'nn.Module')
local elementwise_div, Parent = torch.class('nn.elementwise_div', 'nn.Module')
local elementwise_mul, Parent = torch.class('nn.elementwise_mul', 'nn.Module')
local elementwise_shift, Parent = torch.class('nn.elementwise_shift', 'nn.Module')
local elementwise_concat, Parent = torch.class('nn.elementwise_concat', 'nn.Module')

function ABC_sum:__init(mode)
	parent.__init(self)      
   self.constant_X_Y_1 = torch.Tensor(3,g_input_height, g_input_width)

   local _offset_x = 0
   local _offset_y = 0

   if mode == 'center' then
      _offset_x = 0
      _offset_y = 0
   elseif mode == 'left' then
      _offset_x = -1
      _offset_y = 0
   elseif mode == 'right' then
      _offset_x = 1
      _offset_y = 0
   elseif mode == 'up' then
      _offset_x = 0
      _offset_y = -1
   elseif mode == 'down' then
      _offset_x = 0
      _offset_y = 1
   end

	for y = 1 , g_input_height do
	   for x = 1 , g_input_width do
	      self.constant_X_Y_1[{1,y,x}] = (x + _offset_x - g_cx_rgb) / g_fx_rgb
	      self.constant_X_Y_1[{2,y,x}] = (y + _offset_y - g_cy_rgb) / g_fy_rgb
	   end
	end
   self.constant_X_Y_1[{3,{}}]:fill(1)
end

function ABC_sum:updateOutput(input) 
   -- the input is a 3 channel volume - the groundtruth normal map: 3 dimension, assume normal[{{},1,{}}] is the x, then y , then z
   -- the output is the Ax/f + By/f + C
   if self.output then     
      if self.output:type() ~= input:type() then
         self.output = self.output:typeAs(input);
      end
      self.output:resize(input:size(1), 1, input:size(3), input:size(4))      

      if self.constant_X_Y_1:type() ~= input:type() then
         self.constant_X_Y_1 = self.constant_X_Y_1:typeAs(input);
      end   
   end

   for batch_idx = 1 , input:size(1) do      -- this might not be the fastest way to do it
      self.output[{batch_idx,{}}]:copy(torch.sum(torch.cmul(self.constant_X_Y_1, input[{batch_idx, {}}]), 1))
   end

   return self.output
end

function ABC_sum:updateGradInput(input, gradOutput) 
   if self.gradInput then     
      if self.gradInput:type() ~= input:type() then
         self.gradInput = self.gradInput:typeAs(input);
      end
      self.gradInput:resizeAs(input)      
      self.gradInput:zero()
   end

   for batch_idx = 1 , input:size(1) do
      self.gradInput[{batch_idx, {}}]:copy(self.constant_X_Y_1)
   end
   self.gradInput[{{}, 1, {}}]:cmul(gradOutput[{batch_idx, {}}])
   self.gradInput[{{}, 2, {}}]:cmul(gradOutput[{batch_idx, {}}])
   self.gradInput[{{}, 3, {}}]:cmul(gradOutput[{batch_idx, {}}])
   
   return self.gradInput
end





------------------------
function elementwise_div:__init()
   Parent.__init(self) 
   self.gradInput = {}
   self.gradInput[1] = torch.Tensor()     
   self.gradInput[2] = torch.Tensor()     
end

function elementwise_div:updateOutput(input)
   -- the input is a table, input[1] / input[2]
   if self.output then     
      if self.output:type() ~= input[1]:type() then
         self.output = self.output:typeAs(input[1]);
      end
      self.output:resizeAs(input[1])
   end
   
   self.output:cdiv(input[1], input[2])

   -- ignore the element with 0 divisor
   local zero_mask = input[2]:eq(0)
   self.output[zero_mask] = 0

   return self.output
end

function elementwise_div:updateGradInput(input, gradOutput)
   -- the input is a table, input[1] / input[2]
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

   -- ignore the element with 0 divisor
   local zero_mask = input[2]:eq(0)
   self.gradInput[1][zero_mask] = 0
   self.gradInput[2][zero_mask] = 0

   return self.gradInput
end

----------------------------
function elementwise_mul:__init()
   Parent.__init(self) 
   self.gradInput = {}
   self.gradInput[1] = torch.Tensor()     
   self.gradInput[2] = torch.Tensor()     
end

function elementwise_mul:updateOutput(input)
   if self.output then     
      if self.output:type() ~= input[1]:type() then
         self.output = self.output:typeAs(input[1]);
      end
      self.output:resizeAs(input[1])
   end
   -- the input is a table
   self.output:cmul(input[1], input[2])
   return self.output
end

function elementwise_mul:updateGradInput(input, gradOutput)
   if self.gradInput then     
      for i = 1 , 2 do
         if self.gradInput[i]:type() ~= input[i]:type() then
            self.gradInput[i] = self.gradInput[i]:typeAs(input[i]);
         end
         self.gradInput[i]:resizeAs(input[i])
         self.gradInput[i]:zero()
      end      
   end

   self.gradInput[1]:cmul(gradOutput, input[2])
   self.gradInput[2]:cmul(gradOutput, input[1])

   return self.gradInput
end
-----------------------------------

function elementwise_shift:__init(mode)
   Parent.__init(self) 
   self.mode = mode
end

function elementwise_shift:updateOutput(input)
   -- the input is 4 dim, shift the 3th and 4th dim
   if self.output then     
      if self.output:type() ~= input:type() then
         self.output = self.output:typeAs(input);
      end
      self.output:resizeAs(input)
      self.width = input:size(4)
      self.height = input:size(3)
      self.output:zero()
   end
   
   -- the input is a table
   if self.mode == 'left' then
      self.output[{{},{},{},{1,self.width-1}}]:copy(input[{{},{},{},{2, self.width}}])
   elseif self.mode == 'right' then
      self.output[{{},{},{},{2, self.width}}]:copy(input[{{},{},{},{1,self.width-1}}])
   elseif self.mode == 'up' then
      self.output[{{},{},{1,self.height - 1},{}}]:copy(input[{{},{},{2, self.height},{}}])
   elseif self.mode == 'down' then
      self.output[{{},{},{2, self.height},{}}]:copy(input[{{},{},{1,self.height - 1},{}}])
   end

   return self.output
end

function elementwise_shift:updateGradInput(input, gradOutput)
   if self.gradInput then     
      if self.gradInput:type() ~= input:type() then
         self.gradInput = self.gradInput:typeAs(input);
      end
      self.gradInput:resizeAs(input)
      self.gradInput:zero()
   end

   if self.mode == 'left' then
      self.gradInput[{{},{},{},{2, self.width}}]:copy(gradOutput[{{},{},{},{1,self.width-1}}])
   elseif self.mode == 'right' then
      self.gradInput[{{},{},{},{1,self.width-1}}]:copy(gradOutput[{{},{},{},{2, self.width}}])
   elseif self.mode == 'up' then
      self.gradInput[{{},{},{2, self.height},{}}]:copy(gradOutput[{{},{},{1,self.height - 1},{}}])
   elseif self.mode == 'down' then
      self.gradInput[{{},{},{1,self.height - 1},{}}]:copy(gradOutput[{{},{},{2, self.height},{}}])
   end

   return self.gradInput
end
-----------------------------------

function elementwise_concat:__init()
   Parent.__init(self) 
   self.gradInput = {}
end

function elementwise_concat:updateOutput(input)
   -- input is an array. Each element is a 4 dimesional tensor. 
   -- we concatenate these tensors along the second dimension
   -- we also assume that each element in the input has the same number of channels
   if self.output then     
      if self.output:type() ~= input[1]:type() then
         self.output = self.output:typeAs(input[1]);
      end

      self.output:resize(input[1]:size(1), #input * input[1]:size(2), input[1]:size(3), input[1]:size(4))
      self.output:zero()
   end
   -- the input is a table
   for i = 1, #input do
      self.output[{{}, {input[1]:size(2) * (i-1) + 1, input[1]:size(2) * i }, {}}]:copy(input[i])
   end
   
   return self.output
end

function elementwise_concat:updateGradInput(input, gradOutput)
   if self.gradInput then     
      for i = 1 ,#input do
         if self.gradInput[i] == nil then
            self.gradInput[i] = torch.Tensor()
         end

         if self.gradInput[i]:type() ~= input[i]:type() then
            self.gradInput[i] = self.gradInput[i]:typeAs(input[i]);
         end
         self.gradInput[i]:resizeAs(input[i])
         self.gradInput[i]:zero()
      end      
   end

   for i = 1 ,#input do
      self.gradInput[i]:copy(gradOutput[{{}, {input[1]:size(2) * (i-1) + 1, input[1]:size(2) * i }, {}}])
   end      
   return self.gradInput
end



-------------------------------------

require 'nngraph'

function get_theoretical_depth()
   print("get_theoretical_depth_v2")
   local depth_input = nn.Identity()():annotate{name = 'depth_input', graphAttributes = {color = 'blue'}}
   local normal_input = nn.Identity()():annotate{name = 'normal_input', graphAttributes = {color = 'green'}}
   local ABC_right = nn.ABC_sum('right')(normal_input):annotate{name = 'ABC_right'}
   local ABC_left = nn.ABC_sum('left')(normal_input):annotate{name = 'ABC_left'}
   local ABC_down = nn.ABC_sum('down')(normal_input):annotate{name = 'ABC_down'} 
   local ABC_up = nn.ABC_sum('up')(normal_input):annotate{name = 'ABC_up'}
   
   local z_i_left = nn.elementwise_shift('right')(depth_input)
   local z_i_right = nn.elementwise_shift('left')(depth_input)
   local z_i_up = nn.elementwise_shift('down')(depth_input)
   local z_i_down = nn.elementwise_shift('up')(depth_input)

   local left_div_right = nn.elementwise_div()({ABC_left, ABC_right}):annotate{name = 'left_div_right'}
   local D_right = nn.elementwise_mul()({left_div_right, z_i_left}):annotate{name = 'D_right'}
   local z_o_right = nn.elementwise_shift('right')(D_right):annotate{name = 'z_o_right'}

   local right_div_left = nn.elementwise_div()({ABC_right, ABC_left}):annotate{name = 'right_div_left'}
   local D_left = nn.elementwise_mul()({right_div_left, z_i_right}):annotate{name = 'D_left'}
   local z_o_left = nn.elementwise_shift('left')(D_left):annotate{name = 'z_o_left'}

   local up_div_down = nn.elementwise_div()({ABC_up, ABC_down}):annotate{name = 'up_div_down'}
   local D_down = nn.elementwise_mul()({up_div_down, z_i_up}):annotate{name = 'D_down'}
   local z_o_down = nn.elementwise_shift('down')(D_down):annotate{name = 'z_o_down'}

   local down_div_up = nn.elementwise_div()({ABC_down, ABC_up}):annotate{name = 'down_div_up'}
   local D_up = nn.elementwise_mul()({down_div_up, z_i_down}):annotate{name = 'D_up'}
   local z_o_up = nn.elementwise_shift('up')(D_up):annotate{name = 'z_o_up'}

   local z_concat = nn.elementwise_concat()({z_o_up, z_o_down, z_o_left, z_o_right}):annotate{name = 'concat'}

   local model = nn.gModule({normal_input, depth_input}, {z_concat})

   return model
end

function get_shifted_mask()
   print("get_shifted_mask_v2")

   local mask_input = nn.Identity()():annotate{name = 'mask_input', graphAttributes = {color = 'green'}}

   local mask_o_left = nn.elementwise_shift('left')(mask_input)
   local mask_o_right = nn.elementwise_shift('right')(mask_input)
   local mask_o_up = nn.elementwise_shift('up')(mask_input)
   local mask_o_down = nn.elementwise_shift('down')(mask_input)

   local mask_concat = nn.elementwise_concat()({mask_o_up, mask_o_down, mask_o_left, mask_o_right}):annotate{name = 'concat'}

   local model = nn.gModule({mask_input}, {mask_concat})

   return model
end
