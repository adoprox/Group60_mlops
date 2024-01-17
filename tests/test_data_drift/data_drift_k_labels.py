import pandas as pd
import numpy as np

import os

from evidently import ColumnMapping
from evidently.report import Report
from evidently.test_suite import TestSuite

from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import DataQualityPreset
from evidently.metric_preset import TextOverviewPreset

from evidently.tests import(    
    TestNumberOfRows,
    TestNumberOfColumns,
    TestNumberOfMissingValues,
    TestShareOfMissingValues,
    TestNumberOfColumnsWithMissingValues,
    TestShareOfColumnsWithMissingValues,
    TestNumberOfRowsWithMissingValues,
    TestShareOfRowsWithMissingValues,
    TestNumberOfDifferentMissingValues,
    TestNumberOfConstantColumns,
    TestNumberOfEmptyRows,
    TestNumberOfEmptyColumns,
    TestNumberOfDuplicatedRows,
    TestNumberOfDuplicatedColumns,
    TestColumnsType,
    TestConflictTarget,
    TestConflictPrediction,
    TestHighlyCorrelatedColumns,
    TestNumberOfDriftedColumns,
    TestShareOfDriftedColumns,
)


# Used to process text-related metrics
import nltk
nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')

DATA_PATH_RAW = ""
DATA_PATH_PREDICTION = ""

seed = 42
threshold = 0.5

# Data pre-processing ------------------------------------------------

train_data = pd.read_csv(DATA_PATH_RAW + "train.csv")
data_curr= pd.read_csv(DATA_PATH_PREDICTION + "prediction.csv")

# These are used to make the data set (instead of drop and for sample)
column_labels = train_data.columns.tolist()[2:]
column_no_id = train_data.columns.tolist()[1:]

# Create subsets based on toxic and clean comments
train_toxic = train_data[train_data[column_labels].sum(axis=1) > 0]
# Clean = no all labels are 0, toxic at least one label is 1
train_clean = train_data[train_data[column_labels].sum(axis=1) == 0]

n_toxic_train = len(train_toxic)  # 16225

# Randomly sample 16225 clean comments
train_clean_sampled = train_clean.sample(n=n_toxic_train, random_state=seed)

# Combine the toxic and sampled clean comments
df = pd.concat([train_toxic, train_clean_sampled], axis=0)
data_review = df[column_no_id]

# Change prediction scores to 0/1
for column in column_labels:
    data_curr[column] = data_curr[column].apply(lambda x: 0 if x < threshold else 1)

# Add comment-length column

data_curr["comment_length"] = data_curr["comment_text"].apply(
    lambda x: len(x)
)
data_review["comment_length"] = data_review["comment_text"].apply(
    lambda x: len(x)
)

# Setting up the tests ---------------------------------

column_mapping = ColumnMapping(
    numerical_features=['comment_length'],
    categorical_features=column_labels,
    text_features=['comment_text'] 
)

# Running the tests ----------------------------------------

data_drift_report = Report(metrics=[
    DataDriftPreset(num_stattest='ks', num_stattest_threshold=0.2),
    DataQualityPreset(),
    TextOverviewPreset(column_name = "comment_text")
])

data_drift_report.run(reference_data=data_review, current_data=data_curr, column_mapping=column_mapping)

# Make folder if it does not exists

if not os.path.isdir("tests"):
    os.mkdir("tests")
if not os.path.isdir("reports"):
    os.mkdir("reports")

data_drift_report.save_html('reports/data_drift_k_labels.html')



# Running tests -----------------------------------------------
dataset_tests = TestSuite(tests=[
    TestNumberOfRows(),
    TestNumberOfColumns(),
    TestNumberOfMissingValues(),
    TestShareOfMissingValues(),
    TestNumberOfColumnsWithMissingValues(),
    TestShareOfColumnsWithMissingValues(),
    TestNumberOfRowsWithMissingValues(),
    TestShareOfRowsWithMissingValues(),
    TestNumberOfDifferentMissingValues(),
    TestNumberOfConstantColumns(),
    TestNumberOfEmptyRows(),
    TestNumberOfEmptyColumns(),
    TestNumberOfDuplicatedRows(),
    TestNumberOfDuplicatedColumns(),
    TestColumnsType(),
    TestConflictTarget(),
    TestConflictPrediction(),
    TestHighlyCorrelatedColumns(),
    TestNumberOfDriftedColumns(),
    TestShareOfDriftedColumns(),
])
dataset_tests.run(reference_data=data_review, current_data=data_curr, column_mapping=column_mapping)


dataset_tests.save_html("tests/data_k_labels.html")


