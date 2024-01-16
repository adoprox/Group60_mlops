import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from models.model import ToxicCommentClassifier
from transformers import BertTokenizer
import hydra
import sys
import pandas as pd
import numpy as np


# Token and Encode Function
def tokenize_and_encode(tokenizer, comments):
    """Return  tokenized inputs, attention masks as PyTorch tensors."""

    # Initialize empty lists to store tokenized inputs and attention masks
    input_ids = []
    attention_masks = []

    # Iterate through each comment in the 'comments' list
    for comment in comments:
        # check the validity of data format
        if not isinstance(comment, list):
            comment = [comment]

        # Tokenize and encode the comment using the BERT tokenizer
        encoded_dict = tokenizer.encode_plus(
            comment,
            # Add special tokens like [CLS] and [SEP]
            add_special_tokens=True,
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

    return input_ids, attention_masks


def predict(inputs, config):
    # define the device to use
    device = config.train.device
    checkpoint_path = config.predict.checkpoint_path

    # load the model
    model = ToxicCommentClassifier.load_from_checkpoint(checkpoint_path)  #using in-built function to recall checkpoint 

    # compute the ids and attention_mask for the model
    bert_model_name = config.model.bert_model_name
    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)

    input_ids = []
    attention_masks = []

    input_ids, attention_masks = tokenize_and_encode(tokenizer, inputs)

    user_dataset = TensorDataset(input_ids, attention_masks)
    user_loader = DataLoader(user_dataset, batch_size=1, shuffle=False)

    # Predict each comment given
    predicted = []
    model.eval()
    with torch.no_grad():
        for batch in user_loader:
            input_ids, attention_mask = [t.to(device) for t in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.sigmoid(logits)
            predicted.append(predictions.cpu().numpy())

    return predicted


@hydra.main(version_base="1.3", config_name="default.yaml", config_path="models/config")
def predict_user_input(config):
    # Compute prediction
    user_input = [config.text]
    result = predict(user_input, config)

    # Save results
    labels_list = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    print(user_input)
    print(result[0])
    r_df = pd.DataFrame(result[0], columns=labels_list)
    r_df.to_csv("outputs/predictions.csv")


@hydra.main(version_base="1.3", config_name="default.yaml", config_path="models/config")
def predict_file_input(config):
    # Load data
    file_input = pd.read_csv(config.file)

    # Compute predictions
    results = predict(list(file_input["Comment"]), config)

    labels_list = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    # save the result
    r_df = pd.DataFrame(np.concatenate(results), columns=labels_list)
    r_df.to_csv("outputs/predictions.csv")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise ValueError(
            "Wrong number of Arguments! use one of the format \n - python ../predict_model.py '+file=path_file' \n -python ../predict_model.py '+file=path_file' "
        )

    if sys.argv[1].startswith("+text"):
        predict_user_input()
    elif sys.argv[1].startswith("+file"):
        predict_file_input()
    else:
        raise ValueError(sys.argv[1] + ": Invalid command")
