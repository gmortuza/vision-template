# vision-template
This is a starter template for computer vision related projects. This template uses tensorflow
and keras. `config.yaml` file is used to adapt this template with any project. 
## Build with docker
**Build project**  
In the project root folder run the following command to build docker container
```docker
docker build -t vision .
```
**Run project**
```docker
docker run -it vision --gpus all -v $(pwd)/inputs:/home/project/inputs -v $(pwd)/outputs:/home/projects/outputs vision python train.py/search_hparam.py
```
The inputs folder can be pretty big as it will contains the dataset and pretrained model. So we don't copy this
folder's content into docker container instead we mount this folder in a docker container.  
Once docker is finished running all it's content go deleted. But we need to the output of the trained model.
mounting the outputs folder enabled us to save data from docker container to our local machine.
