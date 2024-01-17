import pandas as pd
import numpy as np

import os

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

import  hydra


DATA_PATH_RAW = "data/raw/"
DATA_PATH_PREDICTION = "outputs/"
PATH = "tests/test_data_drift/" 



# Create label dictionary
dict_k_to_single = {}
dict_single_to_k = {}
combs = list(product([0,1], repeat=6))
for i,c in enumerate(combs):
    dict_k_to_single[c] = i
    dict_single_to_k[i] = c
    
def convert_string_to_label(x:str) -> int:
    """
    Helper function to transform the targets into labels
    x : tuple of targests (ex: (0,1,1,0,1,0))
    return label 0-63
    """
    # Split the sting to a list
    x = x.split(",")
    # Convert the individual elements to integers
    x = [int(val) for val in x]
    # Transform to a tuple
    x = tuple(x)
    # Change to label using the dictionary
    x = dict_k_to_single[x]
    return x


@hydra.main(version_base="1.3", config_name="config.yaml", config_path="")
def data_drift(config):
    # Data pre-processing ------------------------------------------
    train_data = pd.read_csv(DATA_PATH_RAW + "train.csv")
    data_curr= pd.read_csv(DATA_PATH_PREDICTION + "predictions.csv")

    assert len(data_curr) > 1, "Cannot do analysis on a single row"
    assert len(train_data) > 1, "Cannot do analysis on a single row"

    # These are used to make the data set (instead of drop and for sample)
    column_labels = train_data.columns.tolist()[2:]
    column_no_id = train_data.columns.tolist()[1:]

    # Create subsets based on toxic and clean comments
    train_toxic = train_data[train_data[column_labels].sum(axis=1) > 0]
    # Clean = no all labels are 0, toxic at least one label is 1
    train_clean = train_data[train_data[column_labels].sum(axis=1) == 0]

    n_toxic_train = len(train_toxic)  # 16225

    # Randomly sample 16225 clean comments
    train_clean_sampled = train_clean.sample(n=n_toxic_train, random_state=config.train.seed)

    # Combine the toxic and sampled clean comments
    df = pd.concat([train_toxic, train_clean_sampled], axis=0)
    data_review = df[column_no_id]

    # Change prediction scores to 0/1
    for column in column_labels:
        data_curr[column] = data_curr[column].apply(lambda x: 0 if x < config.predict.threshold else 1)

    # Merge colums into 1
    data_curr['ColumnMerged'] = data_curr[data_curr.columns[1:]].apply(
        lambda x: ','.join(x.dropna().astype(str)),
        axis=1
    )
    data_review['ColumnMerged'] = data_review[data_review.columns[1:]].apply(
        lambda x: ','.join(x.dropna().astype(str)),
        axis=1
    )

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

    # Run reports
    data_drift_report.run(reference_data=data_review, current_data=data_curr, column_mapping=column_mapping)

    # Make folder if it does not exists
    if not os.path.isdir(PATH + "tests"):
        os.mkdir(PATH + "tests")
    if not os.path.isdir(PATH + "reports"):
        os.mkdir(PATH + "reports")

    # Save report as html
    data_drift_report.save_html(PATH + 'reports/data_drift_single.html')


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

    # Run the tests
    dataset_tests.run(reference_data=data_review, current_data=data_curr, column_mapping=column_mapping)
    # Save test results as html
    dataset_tests.save_html(PATH + "tests/data_single.html")


if __name__ == "__main__":
    data_drift()