import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Smartphone Usage Prediction",
    page_icon="📱",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load("best_model.joblib")

    with open("metadata.json", "r") as f:
        metadata = json.load(f)

    return model, metadata

try:
    model, metadata = load_model()
except Exception as e:
    st.error(f"Model loading failed: {type(e).__name__}: {e}")
    st.exception(e)
    st.stop()

# ---------------- PAGE HEADER ----------------

st.title("📱 Smartphone Usage Analysis")
st.subheader("Prediction Among Students Using Machine Learning")

st.write(
    "Explore smartphone usage patterns and obtain an "
    "estimated category based on a machine learning model."
)

st.info(
    "This is an educational project, not a medical diagnostic "
    "tool. Results depend on the dataset and model limitations."
)

# ---------------- INPUT FORM ----------------

st.header("Enter Student Information")

st.write(
    "Complete the fields below. The inputs are generated "
    "from the features used during model training."
)

input_values = {}

with st.form("prediction_form"):

    for feature in metadata["features"]:

        name = feature["name"]

        if feature["type"] == "number":

            min_value = float(feature["min"])
            max_value = float(feature["max"])
            default_value = float(feature["default"])

            if min_value > max_value:
                min_value, max_value = max_value, min_value

            default_value = min(
                max(default_value, min_value),
                max_value
            )

            if min_value == max_value:
                input_values[name] = st.number_input(
                    name.replace("_", " "),
                    value=default_value,
                    disabled=True,
                    key=name
                )
            else:
                input_values[name] = st.number_input(
                    name.replace("_", " "),
                    min_value=min_value,
                    max_value=max_value,
                    value=default_value,
                    key=name
                )

        else:

            options = feature["options"]
            default_value = feature["default"]

            if not options:
                st.warning(f"No options available for {name}.")
                st.stop()

            default_index = (
                options.index(default_value)
                if default_value in options
                else 0
            )

            input_values[name] = st.selectbox(
                name.replace("_", " "),
                options=options,
                index=default_index,
                key=name
            )

    submitted = st.form_submit_button(
        "Predict Usage Category",
        use_container_width=True
    )

# ---------------- PREDICTION ----------------

if submitted:

    input_df = pd.DataFrame(
        [input_values],
        columns=[f["name"] for f in metadata["features"]]
    )

    try:
        prediction = model.predict(input_df)[0]

        st.divider()
        st.header("Prediction Result")

        if prediction == "Low":
            st.success("Predicted category: Low")

        elif prediction == "Moderate":
            st.warning("Predicted category: Moderate")

        else:
            st.error("Predicted category: High")

        st.write(
            f"The model estimates the category as **{prediction}**."
        )

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_df)[0]
            class_names = model.classes_

            st.subheader("Model probability estimates")

            probability_df = pd.DataFrame({
                "Category": class_names,
                "Probability (%)": probabilities * 100
            })

            st.bar_chart(
                probability_df.set_index("Category")
            )

            st.dataframe(
                probability_df.round(2),
                hide_index=True,
                use_container_width=True
            )

        st.caption(
            "These probabilities describe the model's estimates, "
            "not a clinical assessment or a guarantee."
        )

        st.subheader("Responsible smartphone use")

        st.write(
            "Consider balancing recreational phone use with "
            "study time, sleep, physical activity and offline "
            "social interaction. Seek support if phone use "
            "interferes with daily life."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------- PROJECT INFORMATION ----------------

st.divider()

st.header("About This Project")

st.write(
    "This project uses a Kaggle dataset, data preprocessing, "
    "classification algorithms and model evaluation to explore "
    "smartphone usage patterns among teenagers."
)

st.write(
    f"Selected model: {metadata['model_name']}"
)

st.write(
    "Dataset source: "
    "https://www.kaggle.com/datasets/sumedh1507/teen-phone-addiction"
)

st.caption(
    "Developed as a student data science and machine learning project."
)
