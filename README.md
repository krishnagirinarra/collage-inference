## Installation pre-requisites
python3, pip3, torch, torchvision, flask, scikit-image, pandas, grequests  

### Folder organization
master_service contains the codes for master service in the collage inference.  
'worker_ips' file inside this folder contains the ips of the worker nodes.  
resnet_worker contains codes for the worker that does inference using the ResNet-32 model.  
collage_worker contains codes for the worker that does inference using the collage-cnn model
