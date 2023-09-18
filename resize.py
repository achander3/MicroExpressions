from PIL import Image
import os, sys

dir_path = "D:/NCSU/CS-B/project/Data/CASME2/"

def resize_im(path):
    if os.path.isfile(path):
        im = Image.open(path).resize((224,224), Image.Resampling.LANCZOS)
        parent_dir = os.path.dirname(path)
        img_name = os.path.basename(path).split('.')[0]
        im.save(os.path.join(parent_dir, img_name + '.jpg'), 'JPEG', quality=90)

def resize_all(mydir):
    for subdir , _ , fileList in os.walk(mydir):
        for f in fileList:
            try:
                full_path = os.path.join(subdir,f)
                resize_im(full_path)
            except Exception as e:
                print(e)
                break

if __name__ == '__main__':
    resize_all(dir_path)