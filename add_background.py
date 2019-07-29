#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Adding a background on pictures.
"""

import os
import sys
import random
import numpy as np
from PIL import Image
import time

# paths
src_dir = '/mnt/lin2/ineru/detector_dataset/test_input'
dst_dir = '/mnt/lin2/ineru/detector_dataset/test_output'
bg_dir  = '/mnt/lin2/ineru/detector_dataset/background/'

PARTS_COPY_JPG = ['train', 'valid']  # selecting datasets for a background adding 
PARTS_ADD_BG = ['train', 'valid'] # selection datasets for a background adding 
MAX_NUM_BG_ON_PNG = 10
BG_SCALE_ALPHA = 230


def merge_with_background(im, bg, alpha_value=220):
	""" Merge the picture "im" with the background "bg".
	Inputs
	------
	alpha_value : smaller value implies to stronger background
	"""

	bg = bg.transpose(Image.FLIP_TOP_BOTTOM).resize(im.size)
	bg.putalpha(200)

	a = np.array(im)
	b = np.array(bg)

	im_alpha = a[:,:,3]
	bg_alpha = b[:,:,3]

	bg_alpha = np.where(im_alpha == 255, bg_alpha, 0)
	im_alpha = np.where(im_alpha == 255, alpha_value, 0)

	a[:,:,3] = im_alpha
	b[:,:,3] = bg_alpha

	a_im = Image.fromarray(a)
	b_im = Image.fromarray(b)

	b_im.paste(a_im, (0, 0), a_im) # or Image.paste(im, box=None, mask=None)
	b = np.array(b_im)
	bg_alpha = np.where(im_alpha == alpha_value, 255, 0)
	b[:,:,3] = bg_alpha
	b_img = Image.fromarray(b)

	return b_img


def add_background_to_image(foreground, background):
	""" Adding the background on the picture.
	"""
	if foreground.size != background.size:
		background = background.resize(foreground.size)
	background.paste(foreground, (0, 0), foreground)
	return background


def copy_files_with_background(src_dir, dst_dir, bg_dir, parts):
	""" Copying files with a background adding.
	"""

	count = {'valid': 0, 'train': 0}
	src_dir = src_dir.rstrip('/')
	dst_dir = dst_dir.rstrip('/')
	os.system('mkdir -p {}'.format(dst_dir))

	for part in parts:

		bg_subdir  = os.path.join(bg_dir, part)
		src_subdir = os.path.join(src_dir, part)
		dst_subdir = os.path.join(dst_dir, part)

		os.system('mkdir -p {}'.format(dst_subdir))
		filenames = os.listdir(src_subdir)
		if len(filenames) == 0: 
			print('{0} is an empty subdir'.format(src_subdir))
			continue

		all_bg_filenames = os.listdir(bg_subdir)
		if len(all_bg_filenames) == 0: 
			print('{0} is an empty subdir'.format(bg_subdir))
			raise Exception()

		for filename in filenames:

			basename = os.path.splitext(filename)[0]
			ext = os.path.splitext(filename)[1]

			if ext == '.jpg':			
				
				if part in PARTS_COPY_JPG: 	
				
					cmd = 'cp {} {}'.format(src_subdir + '/' + basename + '.txt', 
											dst_subdir + '/' + basename + '.txt')
					print(cmd)
					os.system(cmd)

					# It just copies jpg files. It works fast but size of files is also stay large
					cmd = 'cp {} {}'.format(src_subdir + '/' + filename, 
											dst_subdir + '/' + filename)
					print(cmd)
					os.system(cmd)
					count[part] += 1

			if ext == '.png':
				
				if part in PARTS_ADD_BG:

					if len(all_bg_filenames) > MAX_NUM_BG_ON_PNG:
						bg_filenames = random.sample(all_bg_filenames, MAX_NUM_BG_ON_PNG)
					else:
						bg_filenames = all_bg_filenames

					img_fg_path = src_subdir + '/' + filename
					img_fg = Image.open(img_fg_path)
					json_file_path = src_subdir + '/' + basename + '.jpg.json'
					print('json_file_path:', json_file_path)

					for i, bg_filename in enumerate(bg_filenames):
						
						bg_path = bg_subdir + '/' + bg_filename
						img_bg = Image.open(bg_path)

						try:
							if not os.path.isfile(json_file_path):
								# If the scale is empty
								img_fg_new = merge_with_background(img_fg, img_bg, alpha_value=BG_SCALE_ALPHA)
								print('merge_with_background')
							else:
								img_fg_new = img_fg
							img = add_background_to_image(img_fg_new, img_bg)
						
						except Exception as ex:
							print('filename:', filename)
							print('bg_path:', bg_path)
							print('Error in add_background_to_image')
							os.system('echo "add_background_to_image: {} {}" | cat >> _ERRORS.log'.\
								format(filename, bg_path))
							print(ex)
							time.sleep(10)
							break

						dst_path_base = os.path.join(dst_subdir, basename)

						dst_path_jpg = dst_path_base + '_bg' + str(i) + '.jpg'
						print(i, ':', dst_path_jpg)
						img.save(dst_path_jpg, quality=85, optimize=True, progressive=True)
						count[part] += 1

						dst_path_txt = dst_path_base + '_bg' + str(i) + '.txt'
						cmd = 'cp {} {}'.format(src_subdir + '/' + basename + '.txt', 
												dst_path_txt)
						os.system(cmd)


if __name__ == '__main__':

	parts = ['train', 'valid']
	copy_files_with_background(src_dir, dst_dir, bg_dir, parts)
	
