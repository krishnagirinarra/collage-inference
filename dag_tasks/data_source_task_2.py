# Bunch of import statements
import os
import shutil
"""
Task for data input node
"""
def task(filelist, pathin, pathout):
    out_list = []
    for f in filelist:
        source = os.path.join(pathin, f) 
        destination = os.path.join(pathout, f)
        try: 
            out_list.append(shutil.copyfile(source, destination))
        except: 
            print("ERROR while copying file in data_source_task.py")
    return out_list 
	
if __name__ == "__main__":
    filelist = ['n04146614_10015.JPEG']
    class_name = "schoolbus"
    task(filelist, "./datasources/" + class_name + "/", "./to_master/")
