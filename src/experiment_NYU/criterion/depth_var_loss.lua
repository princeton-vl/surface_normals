require 'nn'
require 'cunn'
local remove_mean, Parent = torch.class('nn.remove_mean', 'nn.Module')
local get_variance, Parent = torch.class('nn.get_variance', 'nn.Module')


-------------------------------------------------------------------------------

function remove_mean:__init(kernel)
   Parent.__init(self)   
end

function remove_mean:updateOutput(input)
   -- the input should be a 4 channel tensor
   if self.output then     
      if self.output:type() ~= input:type() then
         self.output = self.output:typeAs(input);
      end
      self.output:resizeAs(input)
   end

   self.output:copy(input)
   for batch_idx = 1, input:size(1) do
      self.output[{batch_idx, {}}]:add( -torch.mean(input[{batch_idx, {}}]) )
   end
   
   return self.output
end

function remove_mean:updateGradInput(input, gradOutput)
   if self.gradInput then     
      if self.gradInput:type() ~= input:type() then
         self.gradInput = self.gradInput:typeAs(input);
      end
      self.gradInput:resizeAs(input)      
      self.gradInput:zero()
   end

   local n_pixel = input:size(3) * input:size(4)
   local r_n = 1 / n_pixel

   -- use self.gradInput as a temporary buffer
   self.gradInput:copy(gradOutput)
   self.gradInput:mul(-r_n)
   for batch_idx = 1, input:size(1) do
      local temp_sum = torch.sum(self.gradInput[{batch_idx,{}}])
      self.gradInput[{batch_idx, {}}]:copy(gradOutput[{batch_idx, {}}])
      self.gradInput[{batch_idx, {}}]:add(temp_sum)
   end

   return self.gradInput
end



--------------------------------------------------------------------------------------



function get_variance:__init(kernel)
   Parent.__init(self)   
end

function get_variance:updateOutput(input)
   -- the input should be a 4 channel tensor
   if self.output then     
      if self.output:type() ~= input:type() then
         self.output = self.output:typeAs(input);
      end
      
      local n_batch = input:size(1)
      self.output:resize(n_batch,1)
   end

   local n_pixel = input:size(3) * input:size(4)
   for batch_idx = 1, input:size(1) do
      self.output[{batch_idx, 1}] = torch.sum(torch.pow(input[{batch_idx, 1}],2)) / n_pixel
   end
   
   return self.output
end

function get_variance:updateGradInput(input, gradOutput)
   if self.gradInput then     
      if self.gradInput:type() ~= input:type() then
         self.gradInput = self.gradInput:typeAs(input);
      end
      self.gradInput:resizeAs(input)      
   end

   local n_pixel = input:size(3) * input:size(4)
   local coef = 2 / n_pixel
   self.gradInput:copy(input)
   for batch_idx = 1, input:size(1) do
      self.gradInput[{batch_idx, 1}]:mul(coef)
      self.gradInput[{batch_idx, 1}]:mul(gradOutput[{batch_idx,1}])
   end

   return self.gradInput
end








----------------------------------------------------------------------------------------
require 'cunn'

local depth_var_loss, parent = torch.class('nn.depth_var_loss', 'nn.Criterion')


function depth_var_loss:__init(thresh)
    parent.__init(self)
    self.buffer = torch.Tensor()

    print(string.format("\n>>>>>>>>>>>>>>>>>>>>>>Criterion: depth_var_loss: thresh = %f", thresh))

    -- get variance
    self.get_var = nn.Sequential()
    self.get_var:add(nn.remove_mean())
    self.get_var:add(nn.get_variance())

    -- self.thresh
    self.thresh = thresh

end



function depth_var_loss:updateOutput(input, target)
   -- The input is the 4D depth channel
   local var = self.get_var:forward(input)


   -- v1
   var:mul(-1)
   var:add(self.thresh)
   self.output = 0
   for batch_idx = 1, input:size(1) do
      self.output = self.output + math.max(0, var[{batch_idx,1}])
   end

   -- -- v2
   -- var:add(-self.thresh)
   -- self.output = torch.sum(var:pow(2))


   -- print("depth var loss = ", self.output)
   return self.output
end



function depth_var_loss:updateGradInput(input, target)    
    -- pre-allocate memory and reset gradient to 0
    if self.gradInput then
        local nElement = self.gradInput:nElement()        
        if self.gradInput:type() ~= input:type() then
            self.gradInput = self.gradInput:typeAs(input);
        end
        self.gradInput:resizeAs(input)
    end

    -- v1    
    -- get the intermediate gradient    
    local temp = self.get_var:forward(input)
        
    for batch_idx = 1, input:size(1) do
        if temp[{batch_idx, 1}] < self.thresh then
            temp[{batch_idx, 1}] = -1
        else
            temp[{batch_idx, 1}] = 0
        end
    end

    self.gradInput:copy(self.get_var:backward(input, temp))



    -- -- v2
    -- -- get the intermediate gradient    
    -- local temp = self.get_var:forward(input)
    
    -- for batch_idx = 1, input:size(1) do
    --     temp[{batch_idx, 1}] = 2 * (temp[{batch_idx, 1}] - self.thresh)
    -- end

    -- self.gradInput:copy(self.get_var:backward(input, temp))



    return self.gradInput
end
