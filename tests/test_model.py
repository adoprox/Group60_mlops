import os
import pytest
from omegaconf import open_dict
import torch
from hydra import initialize, compose
from toxic_comments.models.model import ToxicCommentClassifier
from torch.utils.data import DataLoader
from tests import _PROJECT_ROOT

# Assuming the path to the saved processed data
PATH_TO_DATA = "./data/processed/"

# Check if CUDA is available for GPU tests
cuda_available = torch.cuda.is_available()


# Calculate the relative config path
@pytest.mark.skipif(not os.path.exists(_PROJECT_ROOT + "/toxic_comments"), reason="Config files not found")
@pytest.fixture(scope="module")
def config():
    """Fixture for loading the configuration."""
    with initialize("../toxic_comments/models/config"):
        cfg = compose(config_name="default.yaml")

    return cfg


@pytest.fixture(scope="module")
def model(config):
    """Fixture for creating an instance of the model."""
    # Temporarily open the config for modification
    with open_dict(config):
        # Remove 'data_root' from the config
        if "data_root" in config.model:
            del config.model["data_root"]

    # Initialize model with modified hyperparameters
    return ToxicCommentClassifier(**config.model)


@pytest.fixture(scope="module")
def test_loader():
    """Fixture for loading the test data."""
    # Load test data
    test_dataset = torch.load(PATH_TO_DATA + "test.pt")
    return DataLoader(test_dataset, batch_size=32, shuffle=False)


def test_model_output_shape(model, test_loader):
    """Test if the model produces the expected output shape."""
    input_ids, attention_masks, labels = next(iter(test_loader))
    outputs = model(input_ids, attention_masks)
    assert outputs.logits.shape == (
        input_ids.size(0),
        model.model.num_labels,
    ), f"Logits shape is {outputs.logits.shape}, expected {(input_ids.size(0), model.model.num_labels)}"


def test_model_label_prediction(model, test_loader):
    """Test if the predicted labels are binary (0 or 1) for each class."""
    input_ids, attention_masks, labels = next(iter(test_loader))
    outputs = model(input_ids, attention_masks)
    predicted_probs = torch.sigmoid(outputs.logits)
    predicted_labels = (predicted_probs > 0.5).int()
    assert set(predicted_labels.flatten().tolist()) <= {
        0,
        1,
    }, "Predicted labels should be binary (0 or 1) for each class"


def test_loss_calculation(model, test_loader):
    """Test if the model calculates the loss."""
    input_ids, attention_masks, labels = next(iter(test_loader))
    outputs = model(input_ids, attention_masks, labels)
    assert outputs.loss is not None, "Model did not calculate loss"


def test_model_save_load(model, tmp_path):
    """Test if the model can be successfully saved and loaded."""
    save_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), save_path)
    model_loaded = ToxicCommentClassifier()
    model_loaded.load_state_dict(torch.load(save_path))
    assert model_loaded is not None, "Failed to load the saved model"


# test impossible to run due to memory consumption of having model on
# both CPU and GPU.

# @pytest.mark.skipif(not cuda_available, reason="CUDA is not available")
# def test_predictions_cuda_consistency(model, test_loader):
#     input_ids, attention_masks, labels = next(iter(test_loader))
#     outputs_cpu = model.to("cpu")(input_ids, attention_masks)
#     model_gpu = model.to("cuda")
#     outputs_gpu = model_gpu(input_ids.to("cuda"), attention_masks.to("cuda"))
#     assert torch.equal(
#         outputs_cpu.logits, outputs_gpu.logits.to("cpu")
#     ), "CPU and GPU model predictions are not consistent"


def test_gradient_flow(model, test_loader):
    """Test if gradients are being computed, not vanishing, and not exploding."""
    input_ids, attention_masks, labels = next(iter(test_loader))
    model = model.to("cpu")
    outputs = model(input_ids, attention_masks, labels)
    outputs.loss.backward()
    for param in model.parameters():
        assert param.grad is not None, "Gradients are not being computed"
        assert not torch.all(param.grad == 0), "Gradients are vanishing"
        assert torch.isfinite(param.grad).all(), "Gradients are exploding"
