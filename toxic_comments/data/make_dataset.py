import pandas as pd

import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

# to configs
import hydra


# Token and Encode Function
def tokenize_and_encode(tokenizer, comments, labels, max_length):
    """Return  tokenized inputs, attention masks, and labels as PyTorch tensors."""

    # Initialize empty lists to store tokenized inputs and attention masks
    input_ids = []
    attention_masks = []

    # Iterate through each comment in the 'comments' list
    for comment in comments:
        # Tokenize and encode the comment using the BERT tokenizer
        encoded_dict = tokenizer.encode_plus(
            comment,
            # Add special tokens like [CLS] and [SEP]
            add_special_tokens=True,
            # Truncate or pad the comment to 'max_length'
            max_length=max_length,
            # Pad the comment to 'max_length' with zeros if needed
            # Depricated but other does not seem to work..
            pad_to_max_length=True,
            # Return attention mask to mask padded tokens
            return_attention_mask=True,
            # Return PyTorch tensors
            return_tensors="pt",
        )

        # Append the tokenized input and attention mask to their respective lists
        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])

    # Concatenate the tokenized inputs and attention masks into tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # Convert the labels to a PyTorch tensor with the data type float32
    labels = torch.tensor(labels, dtype=torch.float32)

    return input_ids, attention_masks, labels


@hydra.main(version_base=None, config_name="config_data.yaml", config_path="")
def make_data(config):
    """Load data, tokenize them and save in data/processed"""
    ## Hyperparameters
    # seed (to control randomness)
    seed = config.settings.seed
    # proption of non-traing data
    test_val_size = config.settings.test_val_size
    # proportion of validation data
    val_size = config.settings.val_size
    # max_seq_length (for tokenizing since we use transformers)
    max_seq_length = config.settings.max_seq_length
    # spefic bert model name
    bert_model_name = config.settings.bert_model_name

    # Load data
    train_data = pd.read_csv("./data/raw/train.csv")

    column_labels = train_data.columns.tolist()[2:]
    # From visualization (add visual)

    # Create subsets based on toxic and clean comments
    train_toxic = train_data[train_data[column_labels].sum(axis=1) > 0]
    # Clean = no all labels are 0, toxic at least one label is 1
    train_clean = train_data[train_data[column_labels].sum(axis=1) == 0]

    n_toxic_train = len(train_toxic)  # 16225

    # Randomly sample 16225 clean comments
    train_clean_sampled = train_clean.sample(n=n_toxic_train, random_state=seed)

    # Combine the toxic and sampled clean comments
    df = pd.concat([train_toxic, train_clean_sampled], axis=0)

    # Shuffle the data to avoid any order bias during training
    dataframe = df.sample(frac=1, random_state=42)

    # Split data into training and  validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        dataframe["comment_text"], dataframe.iloc[:, 2:], test_size=val_size, random_state=seed
    )

    # Split data into training, testing sets & validation sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        dataframe["comment_text"], dataframe.iloc[:, 2:], test_size=test_val_size, random_state=seed
    )

    # validation set
    test_texts, val_texts, test_labels, val_labels = train_test_split(
        test_texts, test_labels, test_size=val_size, random_state=seed
    )

    # Token Initialization
    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)

    # Tokenize and Encode the comments and labels for the training set
    input_ids, attention_masks, train_labels = tokenize_and_encode(
        tokenizer, train_texts, train_labels.values, max_seq_length
    )

    # Tokenize and Encode the comments and labels for the test set
    test_input_ids, test_attention_masks, test_labels = tokenize_and_encode(
        tokenizer, test_texts, test_labels.values, max_seq_length
    )

    # Tokenize and Encode the comments and labels for the validation set
    val_input_ids, val_attention_masks, val_labels = tokenize_and_encode(
        tokenizer, val_texts, val_labels.values, max_seq_length
    )

    # Create final data_set object using TensorDatasets
    train_dataset = TensorDataset(input_ids, attention_masks, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

    # Print information
    print("Saving processed data in ./data/processed/")

    # Save datasets
    PATH = "./data/processed/"
    torch.save(train_dataset, PATH + "train.pt")
    torch.save(val_dataset, PATH + "val.pt")
    torch.save(test_dataset, PATH + "test.pt")


if __name__ == "__main__":
    make_data()
