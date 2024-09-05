from PIL import Image
import os
import glob
from PIL import Image
from imagecorruptions import corrupt
import numpy as np
item_list = ['candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2','pcb1','pcb2',
                    'pcb3','pcb4','pipe_fryum']
for type_cor in ['brightness','contrast','defocus_blur','gaussian_noise']:
    for _class_ in item_list:
        path_orginal = './visa/1cls/' + _class_ + '/' + 'test' #path to the test set of original mvtec 
        path = './visa_'+type_cor+'/' + _class_ + '/' + 'test' #path to the corrupted mvtec 
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
            print("The new directory is created!")
        type_sets = glob.glob(path_orginal+'/*/')
        for type in type_sets:
            path_type = type
            path_type_new = path_type.replace('visa/1cls', 'visa_'+type_cor)
            print(path_type_new)
            isExist = os.path.exists(path_type_new)
            if not isExist:
                os.makedirs(path_type_new)
                print("The new directory is created!")
            image_names = glob.glob(path_type + '/*.jpg')
            for image_name in image_names:
                path_to_image = image_name
                print(path_to_image)
                image = Image.open(path_to_image)
                image = np.array(image)
                corrupted = corrupt(image, corruption_name=type_cor, severity=3)
                im = Image.fromarray(corrupted)
                im.save(path_to_image.replace('visa/1cls', 'visa_'+type_cor))

