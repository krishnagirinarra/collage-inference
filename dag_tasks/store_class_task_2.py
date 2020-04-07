# Bunch of import statements
import os
import shutil
"""
Task for node that stores classified images belonding to it's assigned class.
"""
def task(filelist, pathin, pathout):
    out_list = []
    for f in filelist:
        source = os.path.join(pathin, f) 
        destination = os.path.join(pathout, f)
        try: 
            out_list.append(shutil.copyfile(source, destination))
        except: 
            print("ERROR while copying file in store_class_task.py")
    return out_list 
	
if __name__ == "__main__":
    filelist = ['n04146614_10015.JPEG']
    class_num = 2 
    task(filelist, "./classified_images/" + str(class_num) + "/", "./store_class_"+ str(class_num) + "/")
