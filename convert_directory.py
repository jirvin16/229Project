from utils.parse_files import *

# config = nn_config.get_neural_net_configuration()
config = {
	'dataset_directory' : 'data/',
	'output_filename' : 'data_vectors/',
	# 'sampling_frequency' : 44100
	'sampling_frequency' : 44100 / (49)
}
input_directory = config['dataset_directory']
output_filename = config['output_filename'] 
freq = config['sampling_frequency'] #sample frequency in Hz

clip_len = 8 		#length of clips for training. Defined in seconds
block_size = freq / 4 #block sizes used for training - this defines the size of our input state
max_seq_len = int(round((freq * clip_len) / block_size)) #Used later for zero-padding song sequences
#Step 1 - convert MP3s to WAVs
# new_directory = convert_folder_to_wav(input_directory, freq)
#Step 2 - convert WAVs to frequency domain with mean 0 and standard deviation of 1
convert_wav_files_to_nptensor(input_directory, block_size, max_seq_len, output_filename)