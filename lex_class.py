import os
import file_reader as fr
import sklearn


path = 'data/'
categories = ['Animals', 'Children', 'Fiction', 'Medicine', 'Religion']
twenty_train = sklearn.datasets.load_files(container_path=path, description=None, categories=categories, load_content=True, shuffle=True, encoding=None, decode_error='ignore', random_state=0)

print(twenty_train.target_names)