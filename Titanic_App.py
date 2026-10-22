import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

# Step 1 : Page Configuration

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

st.title("Titanic Survival Predictor")
st.write("Machine Learning Classification Dashboard")
st.divider()


# Step 2 : Load Saved Models and Results

@st.cache_resource
def LoadModels():
    decision_tree_model = joblib.load("Titanic_DecisionTree.pkl")
    knn_model = joblib.load("Titanic_KNN.pkl")

    return decision_tree_model, knn_model


try:
    decision_tree_model, knn_model = LoadModels()
    results_df = pd.read_csv("Titanic_Model_Results.csv")
except FileNotFoundError:
    st.error(
        "Required files are missing. First run Titanic_CaseStudy.py "
        "to generate both models and Titanic_Model_Results.csv."
    )
    st.stop()


# Step 3 : Display Model Comparison

st.header("1. Model Performance Comparison")

st.dataframe(results_df, use_container_width=True)

fig, ax = plt.subplots()

ax.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

ax.set_title("Titanic Model Accuracy Comparison")
ax.set_xlabel("Machine Learning Model")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)

st.pyplot(fig)
plt.close(fig)


# Step 4 : Select Model

st.header("2. Select Machine Learning Model")

selected_model = st.selectbox(
    "Choose a model for prediction:",
    ["Decision Tree", "KNN"]
)

if selected_model == "Decision Tree":
    model = decision_tree_model
else:
    model = knn_model


# Step 5 : Enter Passenger Details

st.header("3. Enter Passenger Details")

st.write("Enter passenger information to predict survival.")

with st.form("passenger_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=0.0,
            max_value=100.0,
            value=30.0
        )

        fare = st.number_input(
            "Fare",
            min_value=0.0,
            value=25.0
        )

        sex = st.selectbox(
            "Sex (dataset code)",
            [0, 1]
        )

        pclass = st.selectbox(
            "Passenger Class",
            [1, 2, 3]
        )

    with col2:
        sibsp = st.number_input(
            "Number of Siblings/Spouses Aboard",
            min_value=0,
            max_value=10,
            value=0
        )

        parch = st.number_input(
            "Number of Parents/Children Aboard",
            min_value=0,
            max_value=10,
            value=0
        )

        embarked = st.selectbox(
            "Embarked (dataset code)",
            [0, 1, 2]
        )

    predict_button = st.form_submit_button("Predict Survival")


# Step 6 : Make Prediction

if predict_button:

    # Create input using the original dataset feature names
    input_data = pd.DataFrame([{
        "Age": age,
        "Fare": fare,
        "Sex": sex,
        "sibsp": sibsp,
        "Parch": parch,
        "Pclass": pclass,
        "Embarked": float(embarked)
    }])

    # Apply the same preprocessing as the training script
    input_data = pd.get_dummies(
        input_data,
        columns=["Embarked"],
        drop_first=True,
        dtype=int
    )

    # Match the exact feature columns used during training
    expected_columns = list(model.feature_names_in_)

    input_data = input_data.reindex(
        columns=expected_columns,
        fill_value=0
    )

    prediction = model.predict(input_data)[0]

    st.divider()
    st.header("4. Prediction Result")

    if prediction == 1:
        st.success("Prediction: Passenger is predicted to survive.")
    else:
        st.error("Prediction: Passenger is predicted not to survive.")

    st.write("Selected Model:", selected_model)

    # Show probability when supported by the model
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_data)[0]
        classes = model.classes_

        probability_df = pd.DataFrame({
            "Class": classes,
            "Estimated Probability": probabilities
        })

        st.subheader("Estimated Class Probabilities")
        st.dataframe(probability_df, use_container_width=True)

        survival_probability = probabilities[
            list(classes).index(1)
        ]

        st.metric(
            "Estimated Survival Probability",
            f"{survival_probability * 100:.2f}%"
        )

    st.caption(
        "This is an educational machine learning project. "
        "Predictions are estimates, not guarantees."
    )
