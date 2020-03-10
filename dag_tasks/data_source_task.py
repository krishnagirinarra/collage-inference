# Bunch of import statements

"""
Task for data input node
"""
def task(filelist, pathin, pathout):
	# pathout should be the same as pathin
	return [os.path.join(pathout,f) for f in filelist] # List of image files from this node
	
	
if __name__ == "__main__":
	filelist = ['cat1.jpg', 'car1.jpg', 'cat2.jpg', 'dog1.jpg']
	task(filelist, "/home/keshavba/DARPA/imgdir1", "/home/keshavba/DARPA/imgdir1")
