def data_iterator(data_path, max_size, batch_size):
	for i in range(10):
		yield np.arange(387000).reshape(batch_size, max_size, 100), np.arange(batch_size)