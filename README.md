# Plagdetect

The objective of this project is to create a tool for Intrinsic Plagiarism Detection. It is still under
construction.

The Deep LSTM network for text similarity was found at [this repository](https://github.com/dhwajraj/deep-siamese-text-similarity).

## Useful commands

Create conda environtment:

```conda install -n siamese python=2.7 numpy=1.12.0 tensorflow-gpu=1.12.0 gensim=1.0.1 nltk=3.2.2 memory_profiler=0.54.0 h5py=2.8.0```

If you want to train using your processor or your machine has no GPU, install `tensorflow=1.12.0` 
instead. Remember to activate the environment once it is created using `source activate siamese`.

### Installing NVIDIA driver

The NVIDIA driver used in this project is 396.64. You can install it by running the following commands:

```sudo add-apt-repository ppa:graphics-drivers/ppa```

```sudo apt-get update```

```sudo apt install nvidia-kernel-source-396```

```sudo apt-get install nvidia-driver-396 nvidia-modprobe```



---

Create database:

```python gen_db.py --data <data> --db <db>```

Run this command under the root directory of the project. You may use `-h` option to get more info 
about the command.

---

Create dataset:

Once the database is created, go to the **lstm** folder and generate the dataset:

```python gen_ds.py```

More information about the command is provided when running with `-h`.

---

Train neural network: 

```python train.py --database= <database>```

Make sure to run this command under **lstm** folder. The ``--database`` argument must contain the path 
to the database.

**Note:** In order to make run scripts properly, make sure to add the rood folder of this project 
to PYTHONPATH variable. On Linux, it can be done by the following command:

```export PYTHONPATH=$PYTHONPATH:/path/to/this/repository```

Note that you have to replace */path/to/this/repository/* to the actual path in your system.
In case you are not sure about where it is, just open the command prompt in the root folder and
execute the command below:

```pwd```
