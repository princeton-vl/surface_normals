# File utils
# ydawei@umich.edu

import os
import pickle


def exec_print(cmd):
	print(cmd)
	os.system(cmd)


def list_all_files_w_ext(folder, ext, recursive=False, cache=False):
	FILENAME_CACHE_DIR = 'cache_filename/'
	FILENAME_CACHE = FILENAME_CACHE_DIR + folder.replace('/', '_')
	if recursive:
		FILENAME_CACHE += '-R'
	FILENAME_CACHE += ext + '.bin'

	if cache:
		try:
			with open(FILENAME_CACHE, 'rb') as f:
				return pickle.load(f)
		except:
			pass

	filenames = []
	if recursive:
		for root, ignored, fnames in os.walk(folder):
			for fname in fnames:
				if fname.endswith(ext):
					filenames.append(os.path.join(root, fname))
	else:
		for fname in os.listdir(folder):
			if fname.endswith(ext):
				filenames.append(os.path.join(folder,fname))

	if cache:
		if not os.path.exists(FILENAME_CACHE_DIR):
			os.makedirs(FILENAME_CACHE_DIR)
		with open(FILENAME_CACHE, 'wb') as f:
			pickle.dump(filenames, f, True)

	return filenames


def load_everything_from(module_name):
    g = globals()
    m = __import__(module_name)
    names = getattr(m, '__all__', None)
    if names is None:
        names = [name for name in dir(m) if not name.startswith('_')]
    for name in names:
        g[name] = getattr(m, name)
