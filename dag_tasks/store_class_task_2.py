# Bunch of import statements
import os
import shutil
"""
Task for node that stores classified images belonding to it's assigned class.
"""
def task(file_, pathin, pathout):
    out_list = []
    source = os.path.join(pathin, file_) 
    destination = os.path.join(pathout, file_)
    try: 
        out_list.append(shutil.copyfile(source, destination))
    except: 
        print("ERROR while copying file in store_class_task.py")
    return out_list 
	
if __name__ == "__main__":
    filelist = ['n04146614_10015.JPEG']
    class_num = 2
    for f in filelist: 
        task(f, "./classified_images/" + str(class_num) + "/", "./store_class_"+ str(class_num) + "/")
