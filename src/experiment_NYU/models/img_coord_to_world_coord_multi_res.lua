require 'nn'
require('../../common/NYU_params')

-- -- for debug only
-- local g_input_width = 640
-- local g_input_height = 480

-- local g_fx_rgb = 5.1885790117450188e+02;
-- local g_fy_rgb = -5.1946961112127485e+02;
-- local g_cx_rgb = 3.2558244941119034e+02;
-- local g_cy_rgb = 2.5373616633400465e+02;

local img_coord_to_world_coord_multi_res, parent = torch.class('nn.img_coord_to_world_coord_multi_res', 'nn.Module')


function img_coord_to_world_coord_multi_res:__init(scale)
   local width = g_input_width / scale
   local height = g_input_height / scale
   local _cx_rgb = g_cx_rgb / scale
   local _cy_rgb = g_cy_rgb / scale
   local _fx_rgb = g_fx_rgb / scale
   local _fy_rgb = g_fy_rgb / scale
   
	parent.__init(self)      
	self.constant_x = torch.Tensor(height, width)    -- this should be cuda tensor, maybe
	self.constant_y = torch.Tensor(height, width)
	for y = 1 , height do				-- to test
	   for x = 1 , width do
	      self.constant_x[{y,x}] = (x - _cx_rgb) / _fx_rgb
	      self.constant_y[{y,x}] = (y - _cy_rgb) / _fy_rgb
	   end
	end
end

function img_coord_to_world_coord_multi_res:updateOutput(input) -- the input is depth map, haven't checked the ouput though
   if self.output then     
      if self.output:type() ~= input:type() then
         self.output = self.output:typeAs(input);
      end
      self.output:resize(input:size(1), 3, input:size(3), input:size(4))      

      if self.constant_x:type() ~= input:type() then
         self.constant_x = self.constant_x:typeAs(input);
         self.constant_y = self.constant_y:typeAs(input);
      end   
   end
   
   self.output[{{}, 1, {}}]:copy(input)
   self.output[{{}, 2, {}}]:copy(input)
   self.output[{{}, 3, {}}]:copy(input)


   for batch_idx = 1 , input:size(1) do      -- this might not be the fastest way to do it
      self.output[{batch_idx, 1, {}}]:cmul(self.constant_x)
      self.output[{batch_idx, 2, {}}]:cmul(self.constant_y)
   end
   
   return self.output
end

function img_coord_to_world_coord_multi_res:updateGradInput(input, gradOutput) 
   if self.gradInput then     
      if self.gradInput:type() ~= input:type() then
         self.gradInput = self.gradInput:typeAs(input);
      end
      self.gradInput:resizeAs(input)      
      self.gradInput:zero()
   end

   for batch_idx = 1 , input:size(1) do      -- this might not be the fastest way to do it
      self.gradInput[{batch_idx, {}}]:addcmul(gradOutput[{batch_idx, 1, {}}], self.constant_x)
      self.gradInput[{batch_idx, {}}]:addcmul(gradOutput[{batch_idx, 2, {}}], self.constant_y)
      self.gradInput[{batch_idx, {}}]:add(gradOutput[{batch_idx, 3, {}}])
   end

   return self.gradInput
end