from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import streamlit as st
from toxic_comments.predict_model import predict, load_model
from hydra import initialize, compose
from datetime import datetime

prediction_labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

st.title("Check if a comment is toxic")

st.write(f'This little application uses a BERT model trained on the toxic comment dataset to check if the text in the text field is toxic or not.')

st.write(f'It will predict the probability of the text being in each of the following classes: {prediction_labels}')

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
    st.header("Results: ")
    prediction = predict(input, tokenizer, model, device)
    output_with_labels = zip(prediction_labels, prediction[0][0])
    results_formatted = [f'* {label}: {p*100:.1f}%' for label, p in output_with_labels]

    for line in results_formatted:
        st.markdown(line)
    st.write(f'Computed on {datetime.now().strftime("%H:%M:%S")}')