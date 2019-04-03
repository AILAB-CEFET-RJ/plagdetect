# Plagdetect

The objective of this project is to create a tool for Intrinsic Plagiarism Detection. It is still under
construction.

The Deep LSTM network for text similarity was found at [this repository](https://github.com/dhwajraj/deep-siamese-text-similarity).

## Environment setup

Create conda environtment:

```conda create -n siamese python=2.7 numpy=1.13.3 tensorflow-gpu=1.12.0 gensim=1.0.1 nltk=3.2.2 memory_profiler=0.54.0 h5py=2.8.0```

If you want to train using your processor or your machine has no GPU, install `tensorflow=1.12.0` 
instead to proceed with the installation.

**Note:** Remember to activate the environment once it is created using `source activate siamese`.

### Installing NVIDIA driver

The NVIDIA driver used in this project is 396.64. You can install it by running the following commands:

```sudo add-apt-repository ppa:graphics-drivers/ppa```

```sudo apt-get update```

```sudo apt install nvidia-kernel-source-396```

```sudo apt-get install nvidia-driver-396 nvidia-modprobe```

You may use a different version of the driver as long as TensorFlow can recognize the GPUs in your
system. To make sure TensorFlow can make use of your GPU(s), run the following command under the
`scripts` directory:

```python check_devices.py```

If you can see your GPU(s) listed in the output, it means TensorFlow can make use of them.

---

### Downloading NLTK Punkt

The Punkt package of NLTK library is used in this project to separate a text documents by sentences. 
In order to download this library, go to the `scripts` directory and run the following command:

```python download_punkt.py``` 


---

### Adding this repository to PYTHONPATH variable

In order to make run scripts properly, make sure to add the root directory of this project 
to PYTHONPATH variable. On Linux, it can be done by the following commands:

```echo 'export PYTHONPATH=$PYTHONPATH:/path/to/this/repository' >> ^/.bashrc```
```source ~/.bashrc```

Note that you have to replace */path/to/this/repository/* to the actual path in your system.
In case you are not sure about where it is, just open the command prompt in the root directory and
execute the command below:

```pwd```

### Download dataset

You can download the dataset (PAN CORPUS 11) used in this project in the following link:

https://drive.google.com/open?id=1zyJ6FOleogiS-Zqs1e3ZjOceP0MuOAbe

Once downloaded, extract files to a directory named `dataset` under the root of this project.


### Create database:

After downloading the dataset, run this command under the `scripts` directory of the project. 

```python gen_db.py```

By default, the script will look up for the documents in `../dataset` directory and create the
database named `plag.db`, both of them under the root directory of this project.

---

### Generate train/val/test datasets:

Once the database is created, go to the `lstm` directory and run this script generate the dataset:

```python gen_ds.py```


---

### Download word embeddings

The word embeddings used on this project can be downloaded in the following link:

https://drive.google.com/open?id=1u79f3d2PkmePzyKgubkbxOjeaZCJgCrt

Once downloaded, make sure to unzip the file and place it under `lstm` directory.
The expected name of the file is `wiki.simple.vec`.

---

### Train neural network

After completing all the steps above, you may train the model running the following command under
the `lstm` directory:

```python train.py```

