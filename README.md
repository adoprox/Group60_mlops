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
dvc pull
This creates a new directory with all the files needed for the model to work. 

#### Known issues

#### DVC fails to pull

If you get this error:

    ```
    ERROR: failed to pull data from the cloud - Checkout failed for following targets: models
    I your cache up to date?
    ```

The please dvc pull from the google cloud remote: `dvc pull -r gcloud-storage`

## Wandb sweeps

1. To set custom logging directory, set this environment variable before running the sweep: 
`export WANDB_DIR=./outputs/wandb_logs/`

2. Create a new sweep using: 
`wandb sweep ./toxic_comments/models/config/sweep.yaml`

3. Run the sweep with the command that gets output after you create the sweep

> Note: when you update the sweep.yaml you will need to create a new sweep to use the updated configuration file. If you reuse the old sweep, it will also reuse the old configuration file!

## Docker containers

### Commands to build docker containers
1. Training container: `docker build -f dockerfiles/train_model.dockerfile . -t trainer:latest`
2. Prediction container: `docker build -f dockerfiles/predict_model.dockerfile . -t predict:latest`
3. Inference container: `docker build -f dockerfiles/inference_streamlit.dockerfile . -t inference:latest`
Predict is still under work

### Commands to run docker containers
The docker containers are set up without an entrypoint. The data root folder is in the configuration, the default is set to ./data/processed.

#### Running the training container:
`docker run -v ./data:/data -v ./models:/models -e WANDB_API_KEY='<your-api-key>' group60_trainer:latest python3 ./toxic_comments/train_model.py`

IMPORTANT: to add GPU support to a container, add the flag `--gpus all` to above command, like so:

`docker run -v ./data:/data -v ./models:/models -e WANDB_API_KEY='<your-api-key>' --gpus all group60_trainer:latest python3 ./toxic_comments/train_model.py`

##### Command for running in cloud:

`docker run -v ./data:/data -v ./models:/models -e WANDB_API_KEY='<your-api-key>' --gpus all gcr.io/propane-facet-410709/bert-toxic-trainer:latest python3 ./toxic_comments/train_model.py`

## Gcloud setup
The following section contains documentation and rules for how to interact with the cloud setup.

### Region

All operations should be done in region eu-west-4 and zone eu-west-4a (if fine-grained zones are needed)

### Storage buckets

Any traing, testing, validation, prediction data should be added to the bucket group_60_data.
Any trained models should be added to the bucket group_60_models.

### Creating inference instance

The following command can be used to create a new inference service based on the latest version of the streamlit inference container:
`gcloud run deploy inference-streamlit --image gcr.io/propane-facet-410709/inference-streamlit:latest --platform managed --region europe-west4 --allow-unauthenticated --port 8501`

Additionally, a new instance will be deployed via a trigger whenever a push to main happens.

## Training the model on a compute instance

1. Create an instance with GPU, choose one of the Deep learning images. When starting the instance, make sure the nvidia drivers and cuda are installed correctly. Make sure the VM has access to all APIs.
2. Clone the repository
3. Run dvc pull, supply credentials
4. Train model
5. Run dvc add models/
6. Run dvc push -r gcloud-drive

Alternatively, the model can also be trained within a container. For that:
1. Create an instance with GPU, choose one of the Deep learning images. When starting the instance, make sure the nvidia drivers and cuda are installed correctly. Make sure the
2. Pull the container with: `docker pull gcr.io/propane-facet-410709/bert-toxic-trainer:latest`
3. Install gcloud, gsutil, etc. 
4. Copy training data from cloud storage to container: `gsutil rsync -r gs://group_60_data/data ./data`. This command will copy the data stored in the bucket to the local data directory (assuming current directoy is the project root)
5. Run the container with above command.
6. wandb should automatically upload the model checkpoints. But they can also be uploaded using: `gsutil rsync -r ./local/path/to/models gs://group_60_models`

## Predict
The prediction script can classify a comment or a list of comments given as input:
- **List of string:** python toxic_comments/predict_model.py +file=<file_name>.csv
- **One string:** python toxic_comments/predict_model.py +text="comment to classify"
  
You can also specify the model to use by adding the parameter:
"++predict.checkpoint_path=path_model"

_n.b. The '=' is a special character, if it is present in the path, it needs to be preceded by the special character '\'_
