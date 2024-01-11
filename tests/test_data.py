import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader
from tests import _PATH_DATA


train_file = f"{Path(_PATH_DATA)}/processed/train.pt"
val_file = f"{Path(_PATH_DATA)}/processed/val.pt"
test_file = f"{Path(_PATH_DATA)}/processed/test.pt"

max_seq_length = 128 #from hydra config
@pytest.mark.skipif(
    not os.path.exists(train_file) or not os.path.exists(val_file) or not os.path.exists(test_file), reason="Data files not found"
)
def test_data():
    
    # Load datasets
    train_dataset = torch.load(train_file)
    val_dataset = torch.load(val_file)
    test_dataset = torch.load(test_file)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    # Define the expected sizes of your datasets
    # Replace with the actual expected sizes
    N_train = len(train_dataset)
    N_val = len(val_dataset)
    N_test = len(test_dataset)

    # Run tests
    run_data_length_test(train_loader, N_train)
    run_data_length_test(val_loader, N_val)
    run_data_length_test(test_loader, N_test)


    run_data_shape_test(train_loader, max_seq_length)
    run_data_shape_test(val_loader, max_seq_length)
    run_data_shape_test(test_loader, max_seq_length)
    

    run_data_labels_test(train_loader)
    run_data_labels_test(val_loader)
    run_data_labels_test(test_loader)


def run_data_length_test(loader: torch.utils.data.DataLoader, N_datapoints):
    """
    This method asserts that the total number of datapoints in the DataLoader is equal to N_datapoints.
    """
    total_count = 0
    for _, _, _ in loader:
        total_count += 1

    # Since the total count gives the number of batches, multiply by batch size to get the total number of data points
    total_datapoints = total_count * loader.batch_size

    # Adjust for the last batch which might be smaller
    last_batch_size = len(loader.dataset) % loader.batch_size
    if last_batch_size != 0:
        total_datapoints = total_datapoints - loader.batch_size + last_batch_size

    assert (
        total_datapoints == N_datapoints
    ), f"DataLoader contains {total_datapoints} datapoints, expected {N_datapoints}"


def run_data_shape_test(loader: torch.utils.data.DataLoader, max_length):
    """
    Assert that each datapoint has shapes for input_ids and attention_masks
    as [max_length] since we are using BERT tokenization.
    """
    for batch in loader:
        input_ids, attention_masks, _ = batch
        for i in range(len(input_ids)):
            assert input_ids[i].shape == torch.Size([max_length]), f"Input ID shape is {input_ids[i].shape}, expected {torch.Size([max_length])}"
            assert attention_masks[i].shape == torch.Size([max_length]), f"Attention mask shape is {attention_masks[i].shape}, expected {torch.Size([max_length])}"


def run_data_labels_test(loader: torch.utils.data.DataLoader):
    """
    This method asserts that all individual labels are present in the dataset.
    """
    all_labels = set()
    for _, _, labels in loader:
        # Assuming labels are in a binary format for each class, e.g., [0, 1, 1]
        labels = labels.bool().numpy()
        # Iterate through each label in the batch
        for label in labels:
            # Update the set of all labels encountered across all batches
            all_labels.update(set(label.nonzero()[0]))

    # Check that all expected labels are present
    expected_labels = set(range(loader.dataset.tensors[-1].shape[1]))
    assert all_labels == expected_labels, f"Not all labels are present in the dataset. Found labels: {all_labels}"
