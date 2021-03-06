import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

#### image file reading
addr=os.getcwd()+'/'
os.chdir(addr)

file_list=os.listdir()
len_list=len(file_list)
black_criteria=4096       # at 256*256 resized image


addr='skeleton_original_size' # target folder name
resize_folder = 'resized' # save folder name

try:
    os.mkdir(resize_folder)
except:
    print('The name of file already exists')
# imag extensiosns opencv can support  
extension_CV=['bmp','dib','jpeg','jpg','jpe','jp2','png','webp','pbm','pgm','ppm','ras','tiff','tif','peg'] 

#### Define Functions ####
## Figure Resizing
def resize_imag(image,dim_tuple):
    resized_imag=cv2.resize(image,dim_tuple,interpolation=cv2.INTER_AREA)
    if np.sum(resized_imag)<=black_criteria:
        resized_imag=contrast_imag(resized_imag)
    return resized_imag

def crop_img(A):
        A = np.array(A)
        check_pnt_1 = np.where(A <= [100, 100, 100])
        if not check_pnt_1[0].size < A.size // 2:
            A[:, :] = [0, 0, 0]
            A[check_pnt_1[0:2]] = [255, 255, 255]

        std_len = min(len(A), len(A[0]))//2
        B = np.where(A <= [100, 100, 100])
        try:
            pos_x, pos_y = sum(B[0])//len(B[0]), sum(B[1])//len(B[1])
            C = A[:, pos_y-std_len:pos_y+std_len, :]
            C = cv2.resize(C, (256,256), interpolation =cv2.INTER_CUBIC)
            return C
        except:
            pass

## Contrast
def contrast_imag(image): # edge was drawn in dark line
    edge_mark=image>=200
    image[:]=255 # make all black
    image[edge_mark]=0 # make edge white
    return image


## Saving files
def save_imag(image,address,name_with_ext,extension_str):
    if name_with_ext[-4:]=='jpeg':
        onlyName=name_with_ext[:-4]
    else:
        onlyName=name_with_ext[:-3]        
    cv2.imwrite(address+onlyName+extension_str,image) ## onlyName includes .(dot)
    

def search(dir):
    files = os.listdir(os.getcwd() +"/"+ dir)
    try:
        os.mkdir(os.getcwd() +"/"+ resize_folder + '/' + dir)
    except:
        print('The name of file already exists')

    for file in files:
        fullFilename = os.path.join(os.getcwd()+"/"+ dir, file)
        if os.path.isdir(fullFilename):
            search(dir + "/" +file)
        else:
            print(file)
            img = cv2.imread(fullFilename)
            resized = crop_img(img)
            save_imag(resized, os.getcwd() + '/resized/' + dir + "/", file, 'png')

search(addr)
