import pandas as pd
import numpy as np

from itertools import product

from evidently import ColumnMapping
from evidently.report import Report
from evidently.test_suite import TestSuite

from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import DataQualityPreset
from evidently.metric_preset import TargetDriftPreset
from evidently.metric_preset import TextOverviewPreset

from evidently.tests import (
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
    TestTargetFeaturesCorrelations,
    TestPredictionFeaturesCorrelations,
    TestCorrelationChanges,
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

# create label dictionary
dict_k_to_single = {}
dict_single_to_k = {}

combs = list(product([0,1], repeat=6))
for i,c in enumerate(combs):
    dict_k_to_single[c] = i
    dict_single_to_k[i] = c


# Merge colums into 1
data_curr['ColumnMerged'] = data_curr[data_curr.columns[1:]].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1
)
data_review['ColumnMerged'] = data_review[data_review.columns[1:]].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1
)

def convert_string_to_label(x:str) -> int:
    # Split the sting to a list
    x = x.split(",")
    # Convert the individual elements to integers
    x = [int(val) for val in x]
    # Transform to a tuple
    x = tuple(x)
    # Change to label using the dictionary
    x = dict_k_to_single[x]
    return x

# Make the label column
data_curr['labels'] = data_curr['ColumnMerged'].apply(
    lambda x: convert_string_to_label(x)
)
data_review['labels'] = data_review['ColumnMerged'].apply(
    lambda x: convert_string_to_label(x)
)

# Drop the old columns
data_curr = data_curr[[column_no_id[0],"labels"]]
data_review = data_review[[column_no_id[0],"labels"]]

# Add comment-length column

data_curr["comment_length"] = data_curr["comment_text"].apply(
    lambda x: len(x)
)
data_review["comment_length"] = data_review["comment_text"].apply(
    lambda x: len(x)
)


# Setting up the tests ---------------------------------

column_mapping = ColumnMapping(
    target = "labels",
    numerical_features=['comment_length'],
    text_features=['comment_text'],
    task = "classification"
)

# Running the reports ----------------------------------------

data_drift_report = Report(metrics=[
    DataDriftPreset(num_stattest='ks', num_stattest_threshold=0.2),
    DataQualityPreset(),
    TargetDriftPreset(),
    TextOverviewPreset(column_name="comment_text")
])

data_drift_report.run(reference_data=data_review, current_data=data_curr, column_mapping=column_mapping)

data_drift_report.save_html('data_drift_report.html')


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
    TestTargetFeaturesCorrelations(),
    TestPredictionFeaturesCorrelations(),
    TestCorrelationChanges(),
    TestNumberOfDriftedColumns(),
    TestShareOfDriftedColumns(),
])

dataset_tests.run(reference_data=data_review, current_data=data_curr, column_mapping=column_mapping)
dataset_tests.save_html("data_tests.html")



