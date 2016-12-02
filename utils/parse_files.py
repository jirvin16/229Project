import os
import scipy.io.wavfile as wav
from scipy import signal
import numpy as np
from pipes import quote
import h5py
from pylab import *

def convert_mp3_to_wav(filename, sample_frequency):
	ext = filename[-4:]
	if(ext != '.mp3'):
		return
	files = filename.split('/')
	orig_filename = files[-1][0:-4]
	orig_path = filename[0:-len(files[-1])]
	new_path = ''
	if(filename[0] == '/'):
		new_path = '/'
	for i in xrange(len(files)-1):
		new_path += files[i]+'/'
	tmp_path = new_path + 'tmp'
	new_path += 'wave'
	if not os.path.exists(new_path):
		os.makedirs(new_path)
	if not os.path.exists(tmp_path):
		os.makedirs(tmp_path)
	filename_tmp = tmp_path + '/' + orig_filename + '.mp3'
	new_name = new_path + '/' + orig_filename + '.wav'
	# sample_freq_str = "{0:.1f}".format(float(sample_frequency)/1000.0)
	cmd = 'lame -a -m m {0} {1}'.format(quote(filename), quote(filename_tmp))
	os.system(cmd)
	cmd = 'lame --decode {0} {1}'.format(quote(filename_tmp), quote(new_name))
	# cmd = 'lame --decode {0} {1} --resample {2}'.format(quote(filename_tmp), quote(new_name), sample_freq_str)
	os.system(cmd)
	return new_name

def convert_folder_to_wav(directory, sample_rate=44100):
	for file in os.listdir(directory):
		fullfilename = directory+file
		if file.endswith('.mp3'):
			convert_mp3_to_wav(filename=fullfilename, sample_frequency=sample_rate)
		if file.endswith('.flac'):
			convert_flac_to_wav(filename=fullfilename, sample_frequency=sample_rate)
	return directory + 'wave/'

def convert_wav_files_to_nptensor(directory, out_dir):
	files = []
	for file in os.listdir(directory+"/wave/"):
		if file.endswith('.wav'):
			files.append(directory+"/wave/"+file)
	num_files = len(files)
	genres = sorted(list(set([file[11:-10] for file in files])))
	X = []
	y = []
	# if num_files > 20:
	# 	num_files = 20
	num_weird = 0
	for file_idx in xrange(num_files):
		file = files[file_idx]
		print 'Processing: ', (file_idx+1),'/',num_files
		print 'Filename: ', file
		genre = genres.index(file[11:-10])
		_, x = wav.read(file)
		num_splits = 6
		split_second_length = x.shape[0] / num_splits
		for i in range(num_splits):
			f, t, Sxx = signal.spectrogram(x[i*split_second_length:i*split_second_length+split_second_length], nperseg=128, fs=4000)
			if Sxx.shape[1] == 988:
				X.append(Sxx)
				y.append(genre)
	x_data = np.array(X)#.astype('float16')
	y_data = np.array(y)
	assert y_data.shape[0] == x_data.shape[0]
	print x_data.shape, y_data.shape
	mean_x = np.mean(x_data) #Mean across num examples and num timesteps
	std_x = np.sqrt(np.mean(np.abs(x_data-mean_x)**2)) # STD across num examples and num timesteps
	std_x = np.maximum(1.0e-8, std_x) #Clamp variance if too tiny
	x_data[:][:] -= mean_x #Mean 0
	x_data[:][:] /= std_x #Variance 1
	with h5py.File(out_dir+'data.h5', 'w') as h5f:
		h5f.create_dataset('X', data=x_data)
		h5f.create_dataset('y', data=y_data)
		h5f.create_dataset('mean', data=mean_x)
		h5f.create_dataset('std', data=std_x)
		h5f.create_dataset('genres', data=genres)
	print 'Done!'






