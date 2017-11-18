require 'nn'
require 'cunn'

g_input_width = 416
g_input_height = 128

g_fx_rgb = 246.2849;
g_fy_rgb = -241.6745;
g_cx_rgb = 208.0629;
g_cy_rgb = 57.8963;



-- K = [
--   246.2849         0  208.0629
--          0  241.6745   57.8963
--          0         0    1.0000
--     ]