import os
import os.path
import glob
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

home = os.path.expanduser('~')

parser = argparse.ArgumentParser()

parser.add_argument('group', type=str)
parser.add_argument('file_num', type=str)

prefix = home + '/MIAL/KV_paired_ER_tubules/STED'

climp_pref = '3_23_2021 CLIMP COS7 Paired STED Decon_Series0'
ctrl_pref = '3_23_2021 Control COS7 Paired STED Decon_Series0'
rtn_pref = '3_23_2021 RTN4ACOS7 Paired STED Decon_Series0'

def pipeline_creator(group, file_num):
	if group == 'Climp':
		grp_pref = climp_pref
	elif group == 'Control':
		grp_pref = ctrl_pref
	else:
		grp_pref = rtn_pref

	sample = prefix + '/' + group + '/samples/' + grp_pref + str(file_num) + '_decon_ch02.tif' 
	grp_sample = mpimg.imread(sample)
	plt.imshow(grp_sample)
	plt.show()
	
	enhanced_sample = prefix + '/'+ group + '/enh/' + grp_pref + str(file_num) + '_decon_ch02_std_enhance.png'
	grp_enh = mpimg.imread(enhanced_sample)
	plt.imshow(grp_enh)
	plt.show()
	
	tubules = prefix + '/' + group + '/tub_dil/' + grp_pref + str(file_num) + '_decon_ch02_std_enhance_tubules.png'
	grp_tub = mpimg.imread(tubules)
	plt.imshow(grp_tub)
	plt.show()
	
	selected_tubules = prefix + '/' + group + '/selection_dil/' + grp_pref + str(file_num) + '_decon_ch02_std_enhance_tubules_labeled_selected.png'
	grp_sel = mpimg.imread(selected_tubules)
	plt.imshow(grp_sel)
	plt.show()


opt = parser.parse_args()

group = opt.group
file_num = opt.file_num

pipeline_creator(group, file_num)