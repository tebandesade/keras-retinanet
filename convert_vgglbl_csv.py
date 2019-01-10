import os
import argparse
import json 
import numpy as np

def parse_args():
	parser = argparse.ArgumentParser(description='Test Retina Model with MIO or Bogota images')
	parser.add_argument( '--annotfile', help="annotation json file", default='', type=str)
	parser.add_argument('--output', help="output dir", default='', type=str)
	parser.add_argument('--bogota', help="output dir", default='', type=str)
	return parser.parse_args()

#Based on https://github.com/roytseng-tw/Detectron.pytorch/blob/master/lib/utils/boxes.py
def xywh_to_xyxy(xywh):
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format."""
    if isinstance(xywh, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xywh) == 4
        x1, y1 = xywh[0], xywh[1]
        x2 = x1 + np.maximum(0., xywh[2] - 1.)
        y2 = y1 + np.maximum(0., xywh[3] - 1.)
        return (x1, y1, x2, y2)
    elif isinstance(xywh, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack(
            (xywh[:, 0:2], xywh[:, 0:2] + np.maximum(0, xywh[:, 2:4] - 1))
        )
    else:
        raise TypeError('Argument xywh must be a list, tuple, or numpy array.')

def get_bbox(dict_):
	anotaciones = dict_['annotations']
	anots = []
	for anot in anotaciones:
		ret_temp = []
		cat_id = anot['category_id']
		bbox   = anot['bbox']
		bbox   = xywh_to_xyxy(bbox)
		ret_temp.append(bbox)
		ret_temp.append(cat_id)
		anots.append(ret_temp)
	return anots 

def get_category(dict_):
	categorias = dict_['categories']
	dict_name = {}
	dict_num  = {}
	for cat in categorias:
		dict_name[cat['name']] = cat['id']
		dict_num[cat['id']] = cat['name']
	return dict_num, dict_name 

def get_rta(dict_):
 	imagenes = []
 	for im in dict_['images']:
				rta = im['file_name']
				width = im['width']
				height = im['height']
				temp = [rta,width,height]
				imagenes.append(temp)
	#			for k, v in dict_['_via_img_metadata'].items():				print('paila paso algo')
 	return imagenes

def verifyx1(x1,width):

	if x1>width:
		return width-1
	elif x1> 0 :
#		import pdb
#			pdb.set_trace()
		return x1
	else:
		return 1
def verifyx2(x2, width):
	if x2<width:
		return x2
	elif x2<=0:
		return 1
	else:
		return width-1

def verifyy(y,height):
	if y<height:
		return y
	elif y<=0:
		return 1 
	else:
		return height-1
def verifyXXX(x1,x2):
	if (x1==x2):
		x1= x1-1
		return  x1 , x2
	else:
		return x1, x2
def verifyYYY(y1,y2):
	if(y1==y2):
		y1 = y1-1
		return y1, y2
	else:
		return y1, y2 
def verifyBbox(bbox, width, height):

	x1 = verifyx1(int(bbox[0]),width)
	y1 = verifyy(int(bbox[1]),height)
	x2 = verifyx2(int(bbox[2]),width)
	y2 = verifyy(int(bbox[3]),height)
	x1 ,x2 =verifyXXX(x1,x2)

	temp = [x1,y1,x2,y2]
	return temp
def iter_imgs_bbxs(out, imgs,bboxs,num_dict):
	for io, jo in zip(imgs,bboxs):
		lbl = jo[1]
		jo = verifyBbox(jo[0], io[1],io[2])

		txt = str(io[0]) +', '+ str(jo[0])+ ',' + str(jo[1])+ ','+ str(int(jo[2]))+ ','+ str(int(jo[3]))+ ',' +  str(num_dict[lbl]) + '\n'
		out.write(txt)

def writeDataFile(output, imgs,bboxs,num_dict):

	with open(output,'w') as out:
		iter_imgs_bbxs(out,imgs,bboxs,num_dict)

def writeIdFile(out_file, nam_dict):
	output = 'ID_'+str(out_file)
	with open(output,'w') as out:
		for ele, olo  in nam_dict.items():
			ele = ele.strip()

			txt = str(ele) + ',' + str(olo) + '\n'
			out.write(txt)
#imagen / x1, y1, x2, y2 , cat 
def parse_vgg(file_, out_file):
	with open(file_, 'r') as f:
		dictionary = json.load(f)
		images = get_rta(dictionary)
		bboxes = get_bbox(dictionary)
		num_dict, nam_dict = get_category(dictionary)

	writeDataFile(out_file, images,bboxes, num_dict)
	writeIdFile(out_file,nam_dict)

def iter_boxes_v2(regions,base):
		rtas = []
		anots = []
		for el in regions:
			x1 = el['shape_attributes']['x']
			y1 = el['shape_attributes']['y']
			width = el['shape_attributes']['width']
			height = el['shape_attributes']['height']
			x2     = xywh_to_xyxy([x1,y1,width,height])[2]
			y2     = xywh_to_xyxy([x1,y1,width,height])[3]
			x1     = verifyx1(int(x1),width)
			y1     = verifyy(int(y1),height)
			x2     = verifyx2(int(x2),width)
			y2     = verifyy(int(y2),height)
			x1 , x2  = verifyXXX(int(x1),x2)
			y1, y2 = verifyYYY(y1,y2)
			temp = []	
			try:
					lbl  = el['region_attributes']['type']
			except:
				import pdb
				pdb.set_trace()
				print('oka')
			if lbl=='':
				import pdb
				pdb.set_trace()
			else:
				temp.append([x1,y1,x2,y2])
				temp.append(lbl)
			rtas.append(base)
			anots.append(temp)
		return rtas, anots

def create_file(output,rtas,anots):
		with open(output,'w') as out:	
				for u,v in zip(rtas,anots):
					u = u.strip()
					txt = u +','+str(v[0][0])+','+str(v[0][1])+','+str(v[0][2])+','+str(v[0][3])+','+str(v[1])+'\n'
					print(txt)
					out.write(txt)

def iter_images_v2(images):
	ret_rutas = []
	ret_anots = []
	for k,v in images.items():
		base = 'data_bogota/1_7_8_15'
		rta = v['filename'].strip()
		rta = os.path.join(base,rta)
		rtas, anotas = iter_boxes_v2(v['regions'],rta)
		ret_rutas = ret_rutas + rtas
		ret_anots = ret_anots + anotas
	return ret_rutas, ret_anots

def parse_vgg_v2(file_, out_file):
	with open(file_, 'r') as f:
		dictionary = json.load(f)
		images = dictionary['_via_img_metadata']
		rtas , anots = iter_images_v2(images)
		create_file(out_file,rtas,anots)


def main(args):
	annot_file = args.annotfile
	out_file   = args.output
	bogota     = args.bogota
	print(bogota=='True')
	if (bogota=='True'):
		parse_vgg_v2(annot_file,out_file)
	else:
		parse_vgg(annot_file, out_file)


if __name__=='__main__':
	args = parse_args()
	print(args)
	main(args)