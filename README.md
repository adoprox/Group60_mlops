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

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── toxic_comments  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Using DVC and working with files
git clone https://github.com/adoprox/Group60_mlops.git
cd Group60_mlops.git 
dvc pull
This creates a new directory with all the files needed for the model to work. 

## Gcloud setup
The following section contains documentation and rules for how to interact with the cloud setup.

### Region

All operations should be done in region eu-west-4 and zone eu-west-4a (if fine-grained zones are needed)

### Storage buckets

Any traing, testing, validation, prediction data should be added to the bucket group_60_data.
Any trained models should be added to the bucket group_60_models.