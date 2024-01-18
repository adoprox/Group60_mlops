import torch
from torch.utils.data import DataLoader, random_split
from toxic_comments.models.model import ToxicCommentClassifier
import hydra
import torch.nn.utils.prune as prune
from pathlib import Path

@hydra.main(version_base="1.3", config_name="default.yaml", config_path="models/config")
def prune_and_quantize(config):
    checkpoint_path = config.predict.checkpoint_path

    # load the model, use strict = False to work even if some parameters are missing
    model = ToxicCommentClassifier.load_from_checkpoint(checkpoint_path, strict=False)
    model.eval()
    device_quantize = "cpu"
    model.to(device_quantize)

    # Apply pruning to the classifier layer
    parameters_to_prune = (
        (model.model.classifier, "weight"),
        (model.model.classifier, "bias"),
    )
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.2,
    )

    # To make the pruning permanent
    prune.remove(model.model.classifier, "weight")
    prune.remove(model.model.classifier, "bias")

    # Construct the relative path to the 'train.pt' file
    base_path = Path(__file__).resolve().parent.parent
    processed_data_path = base_path / 'data' / 'processed' / 'train.pt'
    # Load the tensor data from the file
    train_dataset = torch.load(processed_data_path)
    #qconfig = get_default_qconfig('fbgemm')
    float_qparams_weight_only_qconfig = torch.quantization.float_qparams_weight_only_qconfig

    def set_embedding_qconfig(model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                module.qconfig = float_qparams_weight_only_qconfig

    # Prepare and calibrate the model with the calibration dataset
    torch.quantization.prepare(model, inplace=True)
    
    calibration_dataloader = create_calibration_dataloader(
        dataset=train_dataset, # Replace with your dataset variable
        calibration_split=0.001,
        batch_size=32,
        num_workers=0
    )
    calibrate_model(model, calibration_dataloader, device_quantize)
    set_embedding_qconfig(model)
    model_quantized = torch.quantization.convert(model, inplace=False)

    # Create the checkpoint dictionary
    checkpoint = {
        'state_dict': model_quantized.state_dict(),
        'hyper_parameters': model_quantized.hparams,  # Assuming hyperparameters are stored in model.hparams
        # Include any other keys that PyTorch Lightning expects
    }
    checkpoint['pytorch-lightning_version'] = '2.1.2'

    # Save the checkpoint
    torch.save(checkpoint, 'models/production/production_quantized.ckpt')


def calibrate_model(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            # Pass both input_ids and attention_mask to the model
            model(input_ids, attention_mask= attention_mask)


def create_calibration_dataloader(dataset, calibration_split=0.1, batch_size=32, num_workers=0):
    num_calibration_samples = int(len(dataset) * calibration_split)
    num_training_samples = len(dataset) - num_calibration_samples
    calibration_dataset, _ = random_split(dataset, [num_calibration_samples, num_training_samples])

    def collate_fn(batch):
        
        input_ids = torch.stack([torch.tensor(item[0]) for item in batch])
        attention_masks = torch.stack([torch.tensor(item[1]) for item in batch])
        return input_ids, attention_masks

    calibration_dataloader = DataLoader(
        calibration_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return calibration_dataloader


if __name__ == "__main__":
    
    prune_and_quantize()
    
    