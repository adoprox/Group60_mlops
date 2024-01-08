# MLOPS_project of group 60 in MLOPS Jan 2024
MLops project for January course 

Using this dataset 
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge

## Overall goal of the project
The project aims to develop a classifier for identifying toxic comments, as part of the Kaggle Toxic Comment Classification Challenge. This classifier's primary function is to analyze individual comments and estimate the likelihood of each comment falling into one of seven categories. The categories include six specific classes: toxic, severe toxic, obscene, threat, insult, and identity hate, along with a seventh for general classification.

## Framework
To achieve this, we are utilizing a combination of FastAi and PyTorch-Transformers (now known as pytorch_pretrained_bert) frameworks. PyTorch-Transformers, a product of HuggingFace, is instrumental in loading the pretrained model and tokenizer. We chose to use pytorch-lightning as a high-level framework to reduce boilerplate code that we would have to write.

## Data
Our data source is the Kaggle Toxic Comment Classification dataset, which comprises various comments sourced from Wikipedia. Each comment in this dataset is tagged with one or more labels corresponding to the six toxic categories. The dataset's structure and labels allow for a comprehensive training regime, catering to our classifier's need for diverse and complex examples. Interested parties can access the dataset through the provided Kaggle link: [Toxic comment classification challenge data](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

## Models
For the modeling aspect, we are leveraging a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model. The choice of BERT is strategic; given its extensive training on a large corpus, it is highly capable of understanding nuanced language patterns. Our approach involves further training this base model on the specific dataset to fine-tune it for our classification task. This method is expected to harness BERT's advanced language processing capabilities, making it adept at recognizing and classifying varying degrees of toxicity in comments.
