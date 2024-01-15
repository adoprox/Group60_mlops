import streamlit as st
from toxic_comments.predict_model import predict
from hydra import initialize, compose

with initialize("../models/config"):
    cfg = compose(config_name="default.yaml", overrides=["predict.checkpoint_path=models/production/deploy.ckpt"])

input = st.text_area(label="input")

# def run_input_prediction():
#     output = 
if st.button(label="Run inference"):
    output = predict(input, cfg)
    output