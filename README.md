# Plagdetect

The objective of this project is to create a tool for Intrinsic Plagiarism Detection. It is still under
construction.

The Deep LSTM network for text similarity was found at [this repository](https://github.com/dhwajraj/deep-siamese-text-similarity).

## Useful commands

Create conda environtment:

```conda install -n siamese python=2.7 numpy=1.11.0 tensorflow-gpu=1.2.1 gensim=1.0.1 nltk=3.2.2 h5py```

If you want to train using your processor or your machine has no GPU, install `tensorflow=1.2.1` instead.
Remember to activate the environment once it is created using `source activate siamese`.

Create database:

```python generate_db.py```

Run this command under the root directory of the project.

Run: 

```python train.py is_char_based=False --training_files=../plag.db```

Make sure to run this command under **deep_siamese_text_similarities** folder.
