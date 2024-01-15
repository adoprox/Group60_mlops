import torch
import numpy as np
import pytorch_lightning as pl
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score
import wandb


# Define a pytorch ligtning module for Toxic Comment Classification
class ToxicCommentClassifier(pl.LightningModule):
    def __init__(
        self,
        batch_size=32,
        lr=2e-5,
        bert_model_name="bert-base-uncased",
        use_short_data=None,
        num_workers=0,
        data_root=None,
    ):
        """
        Initialize the ToxicCommentClassifier.

        Args:
            batch_size (int): Batch size
            lr (float): Learning rate
            bert_model_name (str): Name of the BERT model to use.
            use_short_data (int): Number of instances to use for training, validation, and testing.
            num_workers (int): Number of CPU workers for data loading.
            data_root (str): Root path where preprocessed data is stored.
        """

        super().__init__()

        self.save_hyperparameters()
        self.batch_size = batch_size
        self.lr = lr
        self.use_short_data = use_short_data
        self.num_workers = num_workers
        self.data_root = data_root

        # Model Initialization
        self.model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=6)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)
        self.test_step_outputs = []

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for input.
            labels (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Model predictions.
        """
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)
    
    def training_step(self, batch, batch_idx):
        """
        Training step for the Lightning module.

        Args:
            batch: Batch of training data.
            batch_idx: Index of the batch.

        Returns:
            torch.Tensor: Training loss.
        """
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        self.log("train_loss", loss)
        wandb.log({"loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the Lightning module.

        Args:
            batch: Batch of validation data.
            batch_idx: Index of the batch.
        """
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        """
        Test step for the Lightning module.

        Args:
            batch: Batch of test data.
            batch_idx: Index of the batch.

        Returns:
            dict: Dictionary containing predicted probabilities, predicted labels, and true labels.
        """
        input_ids, attention_mask, labels = [t.to(self.device) for t in batch]
        outputs = self.model(input_ids, attention_mask=attention_mask)

        predicted_probs_batch = torch.sigmoid(outputs.logits)
        predicted_probs = predicted_probs_batch.cpu().numpy()
        predicted_labels = (predicted_probs > 0.5).astype(int)

        true_labels = labels.cpu().numpy()

        self.test_step_outputs.append(
            {"predicted_probs": predicted_probs, "predicted_labels": predicted_labels, "true_labels": true_labels}
        )

        return {"predicted_probs": predicted_probs, "predicted_labels": predicted_labels, "true_labels": true_labels}

    def on_test_epoch_end(self):
        """
        Operations to perform at the end of the test epoch.
        Computes and logs accuracy, precision, and recall.
        """
        outputs = self.test_step_outputs
        predicted_labels = np.concatenate([out["predicted_labels"] for out in outputs], axis=0)
        true_labels = np.concatenate([out["true_labels"] for out in outputs], axis=0)

        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average="micro")
        recall = recall_score(true_labels, predicted_labels, average="micro")

        self.log("test_accuracy", accuracy, on_epoch=True, prog_bar=True)
        self.log("test_precision", precision, on_epoch=True, prog_bar=True)
        self.log("test_recall", recall, on_epoch=True, prog_bar=True)

        return accuracy, precision, recall

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        """
        Train DataLoader for the Lightning module.
        Loads the training dataset and applies optional subsampling.
        """
        train_dataset = torch.load(self.data_root + "train.pt")
        if self.use_short_data is not None:
            train_indx = list(range(self.use_short_data))
            train_dataset = Subset(train_dataset, train_indx)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        """
        Validation DataLoader for the Lightning module.
        Loads the validation dataset and applies optional subsampling.
        """
        val_dataset = torch.load(self.data_root + "val.pt")
        if self.use_short_data is not None:
            val_indx = list(range(int(self.use_short_data / 10)))
            val_dataset = Subset(val_dataset, val_indx)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        """
        Test DataLoader for the Lightning module.
        Loads the test dataset and applies optional subsampling.
        """
        val_dataset = torch.load(self.data_root + "test.pt")
        if self.use_short_data is not None:
            val_indx = list(range(int(self.use_short_data / 10)))
            val_dataset = Subset(val_dataset, val_indx)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
