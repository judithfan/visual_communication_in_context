#### to remove gray background pixels from image in target data directory ####

import os

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', type=str, help='full path to images', default='./object/')
	args = parser.parse_args()

	imgs = [i for i in os.listdir(args.data) if i.split('.')[-1]=='png']
	for im in imgs:
		print('converting {}'.format(im))
		path = os.path.join(args.data,im)
		os.system('convert {} -transparent "rgb(127,127,127)" {}'.format(path,path))
