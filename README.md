# MLOPS_project of group 60 in MLOPS Jan 2024
MLops project for January course 

Using this dataset 
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge

## Overall goal of the project
Write a classifier for toxic comments from the Kaggl toxic comment classification competition. The classifier should take one comment and output the probability of the comment belonging in each of the 7 classes.

## Framework
FastAi and PyTorch-Transformers (formerly known as pytorch_pretrained_bert) from HuggingFace. The second framework will be used to load the pretrained model and tokenizer; while the first one will be used (instead of PyTorch-Lightning to improve the code, this because FastAi was used in the example we based our model on).

## Data
We will use the Kaggle Toxic comment classification dataset. The dataset consists of comments from Wikipedia, with a label for each comment. Labels consist of one of 6 classes: toxic, severe_toxic, obscene, threat, insult, identity_hate. Link: [Toxic comment classification challenge data](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

## Models
We will use a pretrained BERT model and then train the model on the dataset. Because the BERT base model was trained on a large amount of data, training it on this specific task should yield good results.
