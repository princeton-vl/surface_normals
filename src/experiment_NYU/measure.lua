function rmse(a, b)
	return torch.sqrt((a - b):pow(2):sum() / a:numel())
end

function depth_rmse_linear(y1, y2)
	return rmse(y1, y2)
end

function depth_rmse_log(y1, y2)
	return rmse(torch.log(y1), torch.log(y2))
end

function depth_scale_invariant_rmse_log(y1, y2)
	local ly1 = torch.log(y1)
	local ly2 = torch.log(y2)
	local d = ly1 - ly2
	local n = d:numel()
	local alpha = torch.pow(d, 2):sum() / n - (d:sum() / n) ^ 2
	return torch.sqrt((ly1 - ly2 + alpha):pow(2):sum() / (2 * n))
end

function threshold_delta(y1, y2, thres)
	local to_count = torch.cmax(torch.cdiv(y1, y2), torch.cdiv(y2, y1))
	return torch.le(to_count, thres):sum() / to_count:numel()
end

function threshold_1(y1, y2)
	return threshold_delta(y1, y2, 1.25)
end

function threshold_2(y1 ,y2)
	return threshold_delta(y1, y2, 1.25^2)
end

function threshold_3(y1, y2)
	return threshold_delta(y1, y2, 1.25^3)
end

-- y2 is the groundtruth
function abs_rel_diff(y1, y2)
	return torch.cdiv(torch.abs(y1 - y2), y2):sum() / y1:numel()
end

function sqr_rel_diff(y1, y2)
	return torch.cdiv((y1 - y2):pow(2), y2):sum() / y1:numel()
end

local depth_measures = {
    ['delta<1.25'      ] = threshold_1,
    ['delta<1.25^2'    ] = threshold_2,
    ['delta<1.25^3'    ] = threshold_3,
    ['abs_rel_diff'    ] = abs_rel_diff,
    ['sqr_rel_diff'    ] = sqr_rel_diff,
    ['RMSE_linear'     ] = depth_rmse_linear,
    ['RMSE_log'        ] = depth_rmse_log,
    ['RMSE_log(sc.inv)'] = depth_scale_invariant_rmse_log
}

function measure_depth(y1, y2, mask)
	local measures = {}
	for name, func in pairs(depth_measures) do
		measures[name] = func(y1:maskedSelect(mask), y2:maskedSelect(mask))
	end
	return measures
end


function angle(n1, n2)
	assert(n1:size()[1] == 3)
	local n11 = torch.pow(n1, 2):sum(1)
	local n22 = torch.pow(n2, 2):sum(1)
	local n12 = torch.cmul(n1, n2):sum(1)

	return torch.acos(torch.cdiv(n12, torch.sqrt(torch.cmul(n11, n22))):clamp(-1, 1))
end


function mean_angle_error(n1, n2)
	return angle(n1, n2):mean() / math.pi * 180
end

function median_angle_error(n1, n2)
	return torch.median(angle(n1, n2):view(-1):double())[1] / math.pi * 180
end

function rmse_error(n1, n2)
	return torch.sqrt(angle(n1, n2):pow(2):sum() / (n1:numel() / 3))
end

function good_pixels(n1, n2, thres)
	return torch.le(angle(n1, n2), thres):sum() / (n1:numel() / 3)
end

function good_pixels_11_25(n1, n2)
	return good_pixels(n1, n2, 11.25 * math.pi / 180)
end

function good_pixels_22_5(n1, n2)
	return good_pixels(n1, n2, 22.5 * math.pi / 180)
end

function good_pixels_30(n1, n2)
	return good_pixels(n1, n2, 30 * math.pi / 180)
end

local normal_measures = {
	['mean_angle_error'  ] = mean_angle_error,
	['median_angle_error'] = median_angle_error,
	['within_11.25'      ] = good_pixels_11_25,
	['within_22.5'       ] = good_pixels_22_5,
	['within_30'         ] = good_pixels_30
}

function measure_normal(n1, n2, mask)
	assert(n1:size()[1] == 1)

	local masked_n1 = torch.zeros(3, mask:sum())
	local masked_n2 = torch.zeros(3, mask:sum())
	for i =1,3 do
		masked_n1[i] = n1[{1, i}]:maskedSelect(mask)
		masked_n2[i] = n2[{1, i}]:maskedSelect(mask)
	end

	local measures = {}
	for name, func in pairs(normal_measures) do
		measures[name] = func(masked_n1, masked_n2)
	end
	
	return measures
end
