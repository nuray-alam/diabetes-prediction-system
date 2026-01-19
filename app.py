## Gradio App

import gradio as gr
import pandas as pd
import pickle

# 1. Load the Model
with open("diabetes_prediction_model.pkl", "rb") as file:
    model = pickle.load(file)

NUMERIC_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

CATEGORICAL_FEATURES = ["BMI_Category", "Glucose_Category"]


def create_bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def create_glucose_category(glucose: float) -> str:
    if glucose < 100:
        return "Normal"
    if glucose < 126:
        return "Prediabetic"
    return "Diabetic"


def predict_diabetes(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree_function,
    age,
):
    bmi_category = create_bmi_category(bmi)
    glucose_category = create_glucose_category(glucose)

    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree_function,
        "Age": age,
        "BMI_Category": bmi_category,
        "Glucose_Category": glucose_category,
    }

    input_df = pd.DataFrame([data], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0][1]
        label = "Diabetic" if proba >= 0.5 else "Non-Diabetic"
        return label, float(proba)

    prediction = model.predict(input_df)[0]
    label = "Diabetic" if prediction == 1 else "Non-Diabetic"
    return label, None


with gr.Blocks(title="Diabetes Prediction") as demo:
    gr.Markdown(
        "# Diabetes Prediction\n"
        "Enter patient details to predict diabetes likelihood. "
        "BMI and Glucose categories are derived automatically."
    )

    with gr.Row():
        pregnancies = gr.Number(label="Pregnancies (e.g., 2)", precision=0)
        glucose = gr.Number(label="Glucose (e.g., 120)")
        blood_pressure = gr.Number(label="BloodPressure (e.g., 70)")
        skin_thickness = gr.Number(label="SkinThickness (e.g., 20)")

    with gr.Row():
        insulin = gr.Number(label="Insulin (e.g., 85)")
        bmi = gr.Number(label="BMI (e.g., 28.0)")
        diabetes_pedigree_function = gr.Number(
            label="DiabetesPedigreeFunction (e.g., 0.5)",
        )
        age = gr.Number(label="Age (e.g., 33)", precision=0)

    predict_button = gr.Button("Predict")

    with gr.Row():
        label_output = gr.Textbox(label="Prediction")
        proba_output = gr.Number(label="Diabetes Probability", precision=4)

    predict_button.click(
        predict_diabetes,
        inputs=[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree_function,
            age,
        ],
        outputs=[label_output, proba_output],
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                "### Quick Notes\n"
                "- Higher glucose levels generally increase diabetes risk.\n"
                "- Higher BMI is often linked with higher risk.\n"
                "- This tool is informational and not medical advice."
            )
        with gr.Column(scale=3):
            gr.Markdown("")


if __name__ == "__main__":
    demo.launch()
