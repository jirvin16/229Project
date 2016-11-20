from pydub import AudioSegment
import os

input_directory = "au_files/genres/"
output_directory = "data"

for directory in os.listdir(input_directory):
	for au_file in os.listdir(os.path.join(input_directory, directory)):
		AudioSegment.from_file(os.path.join(input_directory, directory, au_file)).export(os.path.join(output_directory, au_file[:-3]+".mp3"), format="mp3")