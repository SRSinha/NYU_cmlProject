## Multi-Modal Multi Environment Performance Comparison on Google Cloud Platform | Project - NYU Cloud and Machine Learning

In recent years, due to the enormous amount of data, and the advancements in Big Data and Machine Learning Applications, Cloud Computing has become ubiquitous. Most of the applications in today’s world are deployed to the cloud so that they can easily be scaled and accessed by the end-users. Along with deploying the applications in the bare Virtual Machines, the advent of containerized Software has given the developers a hard time to choose the environment where they should deploy their applications. There are different kinds of applications and Machine Learning workloads and it becomes very important to choose the best environment where the application should be deployed for maximum resource utilization.

We have evaluated 3 environments on `GCP` for 2 different workloads: _CNN_ and _RNN_ based models
1. VM
2. Docker
3. Singularity

The files above have been organized in the following manner:
`cnn`
- Code and profiling for MNIST digit recognition.

`cnn.codeScripts`
- Contains the main python file for training
- contains two scripts for nvprof and nsys profiling

`cnn.outputs`
- Contains two folders `bare` and `docker` containing output metrics for the VM and docker profiling respectively.
- Contains the file _Dockerfile_ for building image to be used to spin up Docker container.
- folder `time` contains the _real_, _user_ and _sys_ time for the different batch-sizes.
- file named _kernels_vm_docker.txt_ contains the different kernels observed for the same code for batch-size 64.

`rnn`
- Code and profiling for sentiment analysis.

`rnn.codeScripts`
- Contains the main python file for training
- contains two scripts for nvprof and nsys profiling

`rnn.inputData`
- Contains the custom subset input for the profiling.

`rnn.outputs`
- Contains two folders `bare` and `docker` containing output metrics for the VM and docker profiling respectively.
- file named _time_command_outputs.txt_ contains the _real_, _user_ and _sys_ time for the different batch-sizes.
- file named _kernels_vm_docker.txt_ contains the different kernels observed for the same code for batch-size 64.


Steps below can be followed for _CNN_ metrics profiling for comparison on the 3 environments, assuming `docker` and `singularity` has been installed in the VM.

#### VM
1. Copy the 3 files from the folder `codeScripts` in the VM
2. Default input data-size has been set 512, can be changed in line-number:120
3. To run nvprof profiling for batch-size 64 run
```
sh nvprof.sh 64
```
4. To run nsys profiling for batch-size 64 runtime run
```
sh nsys.sh 64
```
5. Outputs would be generated in the folder name $BATCH_SIZE, here 64


#### Docker
1. The Docker container image can be generated from the given Dockerfile in the folder `codeScripts`
2. Spin up the docker container using the command
```
docker run -it --gpus all --privileged -v /usr/local/cuda:/usr/local/cuda mnist bash
```
3. Default input data-size has been set 512, can be changed in line-number:120
4. To run nvprof profiling for batch-size 64 run
```
sh nvprof.sh 64
```
5. To run nsys profiling for batch-size 64 runtime run
```
sh nsys.sh 64
```
6. Outputs would be generated in the folder name $BATCH_SIZE, here 64

#### Singularity
1. The Singularity image can be generated using:
```
sudo singularity pull mnist.sif docker://pytorch/pytorch 
```
2. Next run the singularity container with the command
```
sudo singularity shell --bind /usr/local/cuda --nv mnist.sif bash
```
3. Default input data-size has been set 512, can be changed in line-number:120
4. To run nvprof profiling for batch-size 64 run
```
sh nvprof.sh 64
```
5. To run nsys profiling for batch-size 64 runtime run
```
sh nsys.sh 64
```
6. Outputs would be generated in the folder name $BATCH_SIZE, here 64


# Running the Application

## Setup & Installtion

Make sure you have the latest version of Python installed.

```bash
git clone <repo-url>
```

```bash
pip install -r requirements.txt
```

## Running The App

```bash
python main.py
```

## Viewing The App

Go to `http://127.0.0.1:8000`
