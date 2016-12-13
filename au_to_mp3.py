from pydub import AudioSegment
import os

# input_directory = "au_files/genres/"
# output_directory = "data"
# suffix_length = len(".au")

# for directory in os.listdir(input_directory):
# 	for file in os.listdir(os.path.join(input_directory, directory)):
# 		AudioSegment.from_file(os.path.join(input_directory, directory, file)).export(os.path.join(output_directory, file[:-suffix_length]+".mp3"), format="mp3")

input_directory = "/deep/group/dlbootcamp/jirvin16/229/unique/"
output_directory = "/deep/group/dlbootcamp/jirvin16/229/unique_data/"

for directory in os.listdir(input_directory):
	genre_dir = os.path.join(input_directory, directory)
	if os.path.isdir(genre_dir):
		file_index = 0	
		for file in os.listdir(genre_dir):
			infile_name = os.path.join(input_directory, directory, file)
			if len(str(file_index)) == 1:
				number = "00" + str(file_index)
			elif len(str(file_index)) == 2:
				number = "0" + str(file_index)
			else:
				number = str(file_index)
			outfile_name = os.path.join(output_directory, directory + number + ".mp3")
			AudioSegment.from_file(infile_name).export(outfile_name, format="mp3")
			file_index += 1