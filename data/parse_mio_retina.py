import sys
import os

def create_img_path(num):
	_end ='.jpg'
	begining = 'train'
	temp = str(num)+_end
	final = os.path.join(begining,temp)
	return final


def read_csv_file(file_):
	with open('retina_mio_tcd_dataset.csv','w') as out:
		with open(file_) as f:
			for line in f:
				line = line.strip()
				line_split = line.split(',')
				im_ = line_split[0]
				label_ = line_split[1]
				x1 = line_split[2]
				y1 = line_split[3]
				x2 = line_split[4]
				y2 = line_split[5]

				new_im_ = create_img_path(im_)
				copy_line = line_split.copy()
				copy_line[0] = new_im_
				copy_line[1] = x1
				copy_line[2] = y1
				copy_line[3] = x2
				copy_line[4] = y2
				copy_line[5] = label_
				new_line = ','.join(copy_line)
				new_line = new_line + '\n'
				#import pdb
				#pdb.set_trace()
				out.write(new_line)

def main():
	file_ = sys.argv[1]
	read_csv_file(file_)


if __name__ == '__main__':
    main()