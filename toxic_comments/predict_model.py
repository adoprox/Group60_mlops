import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from models.model import ToxicCommentClassifier
from transformers import BertTokenizer
import hydra
import sys

# Token and Encode Function
def tokenize_and_encode(tokenizer, comments):
    """Return  tokenized inputs, attention masks, and labels as PyTorch tensors."""

    # Initialize empty lists to store tokenized inputs and attention masks
    input_ids = []
    attention_masks = []
 
    # Iterate through each comment in the 'comments' list
    for comment in comments:
 
        # Tokenize and encode the comment using the BERT tokenizer
        encoded_dict = tokenizer.encode_plus(comment,
            # Add special tokens like [CLS] and [SEP]
            add_special_tokens=True,
 
            # Pad the comment to 'max_length' with zeros if needed
            # Depricated but other does not seem to work..
            pad_to_max_length=True, 
            # Return attention mask to mask padded tokens
            return_attention_mask=True,
 
            # Return PyTorch tensors
            return_tensors='pt'
        )
 
        # Append the tokenized input and attention mask to their respective lists
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
 
    # Concatenate the tokenized inputs and attention masks into tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

@hydra.main(version_base= "1.3", config_name="config_predict.yaml", config_path = "")
def predict_user_input(config):

    # define the device to use
    device = config.hyperparameters.device
    
    # Check if all the arguments are present and load the input
    if len(sys.argv) != 2:
        raise ValueError('Wrong number of Arguments! use the format python ../predict_model.py \'+text=text_to_classify\'')
    user_input = [config.text]
    
    #load the model
    model = ToxicCommentClassifier.load_from_checkpoint("models/bert_toxic_classifier_logs/version_16/checkpoints/epoch=0-step=125.ckpt")

    # compute the ids and attention_mask for the model
    bert_model_name = config.hyperparameters.bert_model_name
    tokenizer = BertTokenizer.from_pretrained(bert_model_name,do_lower_case=True)
    input_ids, attention_masks = tokenize_and_encode(tokenizer,user_input)

    user_dataset = TensorDataset(input_ids, attention_masks)
    user_loader = DataLoader(user_dataset, batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
	    for batch in user_loader:
              input_ids, attention_mask = [t.to(device) for t in batch]
              outputs = model(input_ids, attention_mask=attention_mask)
              logits = outputs.logits
              predictions = torch.sigmoid(logits)
        
    # Keep just the meaningful label
    predicted_labels = (predictions.cpu().numpy() > 0.8).astype(int)

    labels_list = ['toxic', 'severe_toxic', 'obscene',
				'threat', 'insult', 'identity_hate']
	

    result = [labels_list[i] for i in range(len(predicted_labels[0]))  if predicted_labels[0][i]>0]

    # Print the type of toxicity, if present
    if len(result)==0:
        print('Not Toxic')
    else:
        print(result)

if __name__ == '__main__':
    
    predict_user_input()