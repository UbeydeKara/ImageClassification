import os
from pathlib import Path

# name
model_class = '<class_name>'
folder = 'test'

# dir
cur_dir = 'Images/{}/{}'.format(folder, model_class)
file_count = len(os.listdir(cur_dir))

# iterate files in directory
file_index = 0

for filename in os.listdir(cur_dir):
    os.rename(r'{}/{}'.format(cur_dir, filename), r'{}/{}_{:04}.jpg'.format(cur_dir, model_class, file_index))
    print('{}/{}_{:04}.jpg'.format(cur_dir, model_class, file_index))
    file_index = file_index + 1