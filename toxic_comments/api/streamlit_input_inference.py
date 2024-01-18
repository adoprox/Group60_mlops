from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import streamlit as st
from toxic_comments.predict_model import predict, load_model
from hydra import initialize, compose


@st.cache_resource
def load_config():
    """Decorator to cache loaded config. Avoids errors of hydra complaining about reinitialization."""
    with initialize("../models/config"):
        return compose(
            config_name="default.yaml",
            overrides=["predict.checkpoint_path=models/production/production_quantized.ckpt"],
        )


@st.cache_resource(hash_funcs={DictConfig: OmegaConf.to_container})
def load_model_decorator(config: DictConfig):
    """Decorator for caching of loaded model"""
    return load_model(config)


cfg = load_config()
tokenizer, model, device = load_model_decorator(cfg)
input = st.text_area(label="input")
# Convert the input characters to a list of words
# input = input.split()
input = [input]

# def run_input_prediction():
#     output =
if st.button(label="Run inference"):
    output = predict(input, tokenizer, model, device)
    output
