# Surface Normals in the Wild

Code for reproducing the results in the following paper:


	Surface Normals in the Wild,
	Weifeng Chen, Donglai Xiang, Jia Deng
	International Conference on Computer Vision (ICCV), 2017.


Please check out the [project site](http://www-personal.umich.edu/~wfchen/surface-normals-in-the-wild/)  for more details.


# Setup

1. Install the Torch 7 framework as described in http://torch.ch/docs/getting-started.html#_. Please make sure that you have the `cudnn`, `hdf5`, 'mattorch' and `csvigo` modules installed.

2. Clone this repo.

		https://github.com/wfchen-umich/surface_normals.git


# Evaluating on pre-trained models 

## Setup

Please first download the [data files](https://drive.google.com/open?id=0B02I7-1fYj-ceXI4cGlSNDBPcm8) and [pre-trained models](https://drive.google.com/open?id=0B02I7-1fYj-cdmsza01XQ2pIV28) into the `surface_normals` folder. Download the SNOW dataset from the [project site](http://www-personal.umich.edu/~wfchen/surface-normals-in-the-wild/).  

Untar `data.tar.gz` into `surface_normals`. Untar `results.tar.gz` into `surface_normals/src`. Untar `SNOW_Toolkit.tar.gz` into `surface_normals/data`. Untar `SNOW_images.tar.gz` into `surface_normals/data/SNOW_Toolkit`.
	
	

## NYU Experiments

Change directory into `/surface_normals/src/experiment_NYU`.


### NYU Subset

To evaluate the pre-trained models ( trained on the NYU labeled training subset), run the following commands:

* d_n_al:
		
		th test_model_on_NYU_NO_CROP.lua -num_iter 1000 -prev_model_file ../results/hourglass3_softplus_margin_log/wn1_n5000_d800/model_period2_100000.t7 -test_set 654_NYU_MITpaper_test_imgs_orig_size_points.csv -mode test

* d_n_dl:

		th test_model_on_NYU_NO_CROP.lua -num_iter 1000 -prev_model_file ../results/hourglass3_softplus_margin_log_depth_from_normal/wn100_n5000_d800/model_period2_100000.t7 -test_set 654_NYU_MITpaper_test_imgs_orig_size_points.csv -mode test


### NYU Full

To evaluate the pre-trained models ( trained on the full NYU labeled training subset), run the following commands:

* d_n_al_F:
	
		th test_model_on_NYU_NO_CROP.lua -num_iter 1000 -prev_model_file ../results/hourglass3_softplus_margin_log/wn1_n5000_d10000_fullNYU/model_period3_100000.t7 -test_set 654_NYU_MITpaper_test_imgs_orig_size_points.csv -mode test

* d_n_dl_F:

		th test_model_on_NYU_NO_CROP.lua -num_iter 1000 -prev_model_file ../results/hourglass3_softplus_margin_log_depth_from_normal/wn100_n5000_d10000_fullNYU/model_period3_90000.t7 -test_set 654_NYU_MITpaper_test_imgs_orig_size_points.csv -mode test


## SNOW Experiments

Normals from Predicted Depth:

* d_n_al_F_SNOW
	
		th test_model_on_SNOW.lua -num_iter 100000 -prev_model_file ../results/hourglass3_softplus_margin_log/SNOW12_from_n5000_d10000_1e-4/model_period3_100000.t7 -mode test


## KITTI Experiments

Change directory into `/surface_normals/src/experiment_KITTI`. Run the following commands:

* d:
	
		th test_model_on_KITTI.lua -num_iter 1000 -prev_model_file ../results/hourglass3_softplus_margin_log_depth_from_normal/KITTI_1e-4_n0_run2_1e-5/model_period10_200000.t7 -test_set eigen_test_files_combine.csv -mode test

* d_n_al:
	
		th test_model_on_KITTI.lua -num_iter 1000 -prev_model_file ../results/hourglass3_softplus_margin_log/KITTI_1e-4_d5000_n5000_run2_1e-5/model_period7_150000.t7 -test_set eigen_test_files_combine.csv -mode test

* d_n_dl:

		th test_model_on_KITTI.lua -num_iter 1000 -prev_model_file ../results/hourglass3_softplus_margin_log_depth_from_normal/KITTI_1e-4_n5000_run2_1e-5/model_period7_160000.t7 -test_set eigen_test_files_combine.csv -mode test
