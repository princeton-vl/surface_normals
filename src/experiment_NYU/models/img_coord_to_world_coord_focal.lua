require 'nn'
require('../../common/NYU_params')

-- -- for debug only
-- local g_input_width = 10
-- local g_input_height = 20






local img_coord_to_world_coord_focal, parent = torch.class('nn.img_coord_to_world_coord_focal', 'nn.Module')


function img_coord_to_world_coord_focal:__init()
   parent.__init(self)      
   self.gradInput = {torch.Tensor(), torch.Tensor()}

   local _cx_rgb = g_input_width / 2
   local _cy_rgb = g_input_height / 2

   self.constant_x = torch.Tensor(g_input_height, g_input_width)    -- this should be cuda tensor, maybe
   self.constant_y = torch.Tensor(g_input_height, g_input_width)
   for y = 1 , g_input_height do
      for x = 1 , g_input_width do
         self.constant_x[{y,x}] = (x - _cx_rgb)
         self.constant_y[{y,x}] = -(y - _cy_rgb)                     -- VERY IMPORTANT!  The negative sign!!!!!!!!! to test
      end
   end
end

function img_coord_to_world_coord_focal:updateOutput(input) 
-- The input is an array. 
--    1. The first component is the depth map. A 4D tensor.
--    2. The second component is the predicted focal length. A 2D tensor. The first dimension is the number of batch.  -- to do.
-- The ouput is a N x 3 x H x W tensor

   if self.output then     
      if self.output:type() ~= input[1]:type() then
         self.output = self.output:typeAs(input[1]);
      end
      self.output:resize(input[1]:size(1), 3, input[1]:size(3), input[1]:size(4))      

      if self.constant_x:type() ~= input[1]:type() then
         self.constant_x = self.constant_x:typeAs(input[1]);
         self.constant_y = self.constant_y:typeAs(input[1]);
      end   
   end
   assert(input[1]:size(1) == input[2]:size(1))

   self.output[{{}, 1, {}}]:copy(input[1])
   self.output[{{}, 2, {}}]:copy(input[1])
   self.output[{{}, 3, {}}]:copy(input[1])
   
   for batch_idx = 1 , input[1]:size(1) do      -- this might not be the fastest way to do it
      self.output[{batch_idx, 1, {}}]:cmul(self.constant_x)
      self.output[{batch_idx, 2, {}}]:cmul(self.constant_y)
      
      self.output[{batch_idx, {1,2}, {}}]:div(input[2][{batch_idx,1}])   -- divided by the predicted focal length      
   end
   
   return self.output
end

function img_coord_to_world_coord_focal:updateGradInput(input, gradOutput) 
   if self.gradInput then     
      if self.gradInput[1]:type() ~= input[1]:type() then
         self.gradInput[1] = self.gradInput[1]:typeAs(input[1]);
      end

      if self.gradInput[2]:type() ~= input[2]:type() then
         self.gradInput[2] = self.gradInput[2]:typeAs(input[2]);
      end

      self.gradInput[1]:resizeAs(input[1])
      self.gradInput[2]:resizeAs(input[2])
      self.gradInput[1]:zero()
      self.gradInput[2]:zero()
   end


   local buffer = self.output:clone()
   buffer:cmul(gradOutput)
   for batch_idx = 1 , input[1]:size(1) do      -- this might not be the fastest way to do it   -- to do
      -- depth
      self.gradInput[1][{batch_idx, {}}]:addcmul(gradOutput[{batch_idx, 1, {}}], self.constant_x)
      self.gradInput[1][{batch_idx, {}}]:addcmul(gradOutput[{batch_idx, 2, {}}], self.constant_y)
      self.gradInput[1][{batch_idx, {}}]:div(input[2][{batch_idx,1}])

      self.gradInput[1][{batch_idx, {}}]:add(gradOutput[{batch_idx, 3, {}}])

      -- focal length
      self.gradInput[2][{batch_idx,1}] = torch.sum(buffer[{batch_idx,{1,2}}]:div(-input[2][{batch_idx,1}]))
   end

   return self.gradInput
end


