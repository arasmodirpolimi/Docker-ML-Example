import gradio as gr
import pickle
import numpy as np
import os
import sys

# Load trained model and encoders
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
except FileNotFoundError as e:
    sys.stderr.write(f"Missing file: {e}\n")
    raise

def recommend_stack(project_type, team_size, perf_need, experience):
    try:
        pt = encoders["project_type"].transform([project_type])[0]
        pn = encoders["perf_need"].transform([perf_need])[0]
        ex = encoders["experience"].transform([experience])[0]
        input_data = np.array([[pt, int(team_size), pn, ex]])
        pred_encoded = model.predict(input_data)[0]
        stack = encoders["stack"].inverse_transform([pred_encoded])[0]
        return f"🔧 Recommended Tech Stack: {stack}"
    except Exception as err:
        return f"⚠️ Error making recommendation: {err}"

demo = gr.Interface(
    fn=recommend_stack,
    inputs=[
        gr.Radio(["Web App", "API", "ML App", "Real-time App"], label="Project Type"),
        gr.Slider(1, 10, step=1, label="Team Size", value=3),
        gr.Radio(["Low", "Medium", "High"], label="Performance Need"),
        gr.Radio(["Beginner", "Intermediate", "Expert"], label="Experience Level"),
    ],
    outputs="text",
    title="Tech Stack Advisor",
    description="Get a recommended tech stack based on your project and team!",
    allow_flagging="never",  # ✅ disable flagging to avoid PermissionError
)

demo.launch(server_name="0.0.0.0", server_port=7860)
