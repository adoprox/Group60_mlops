import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from transformers import BertForSequenceClassification, BertTokenizer
import wandb
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

# for managin hyperparameters
import hydra

PATH_TO_DATA = "./data/processed/"

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Define a pytorch ligtning module
class ToxicCommentClassifier(pl.LightningModule):
    def __init__(self, batch_size=32, lr=2e-5, bert_model_name = 'bert-base-uncased', use_short_data = None, num_workers= 0, data_root=None):
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
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        self.log('train_loss', loss)
        wandb.log({"loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        self.log('val_loss', loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
    
    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = [t.to(self.device) for t in batch]
        outputs = self.model(input_ids, attention_mask=attention_mask)

        predicted_probs_batch = torch.sigmoid(outputs.logits)
        predicted_probs = predicted_probs_batch.cpu().numpy()
        predicted_labels = (predicted_probs > 0.5).astype(int)

        true_labels = labels.cpu().numpy()

        self.test_step_outputs.append({'predicted_probs': predicted_probs, 'predicted_labels': predicted_labels, 'true_labels': true_labels})

        return {'predicted_probs': predicted_probs, 'predicted_labels': predicted_labels, 'true_labels': true_labels}
    
    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        predicted_probs = np.concatenate([out['predicted_probs'] for out in outputs], axis=0)
        predicted_labels = np.concatenate([out['predicted_labels'] for out in outputs], axis=0)
        true_labels = np.concatenate([out['true_labels'] for out in outputs], axis=0)

        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='micro')
        recall = recall_score(true_labels, predicted_labels, average='micro')

        self.log('test_accuracy', accuracy, on_epoch=True, prog_bar=True)
        self.log('test_precision', precision, on_epoch=True, prog_bar=True)
        self.log('test_recall', recall, on_epoch=True, prog_bar=True)

        return accuracy,precision,recall

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        train_dataset = torch.load(self.data_root + "train.pt")
        if self.use_short_data is not None:
            train_indx = list(range(self.use_short_data))
            train_dataset = Subset(train_dataset, train_indx)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        val_dataset = torch.load(self.data_root + "val.pt")
        if self.use_short_data is not None:
            val_indx = list(range(int(self.use_short_data / 10)))
            val_dataset = Subset(val_dataset, val_indx)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        val_dataset = torch.load(self.data_root + "test.pt")
        if self.use_short_data is not None:
            val_indx = list(range(int(self.use_short_data / 10)))
            val_dataset = Subset(val_dataset, val_indx)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)



@hydra.main(version_base= "1.3", config_name="config_train.yaml", config_path = "")
def train(config):
    # Initialize TensorBoard and wandb logger
    logger = pl.loggers.TensorBoardLogger("./models", name="bert_toxic_classifier_logs")
    wandb_logger = WandbLogger(log_model="all", project="bert_toxic_classifier")

    # log training device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
    wandb.log({"device": str(device)})

    wandb.log({"configuration": OmegaConf.to_yaml(config)})

    # Set seed
    torch.manual_seed(config.hyperparameters.seed)

    # Define Lightning Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator=config.hyperparameters.device,
        max_epochs=config.hyperparameters.num_epochs,
        log_every_n_steps= config.hyperparameters.print_every
    )

    # Create instance of your LightningModule
    model = ToxicCommentClassifier(batch_size=config.hyperparameters.batch_size,
                                   lr=config.hyperparameters.lr,
                                   bert_model_name=config.hyperparameters.bert_model_name,
                                   use_short_data=config.hyperparameters.use_short_data,
                                   num_workers=config.hyperparameters.num_workers,
                                   data_root = config.hyperparameters.data_root) # Use small dataset to speed up training
    
    wandb.watch(model, log_freq=100)

    # Train the model
    trainer.fit(model)
    # save logged data
    logger.save()
    # Test the model
    trainer.test(model)

if __name__ == '__main__':
    wandb.init(project="bert_toxic_classifier")
    train()