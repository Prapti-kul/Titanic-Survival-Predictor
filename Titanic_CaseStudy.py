import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Step 1 : Load Data

#-----------------------------------------------------
#   Function Name : LoadData
#   Description : Load the data from CSV
#   Input : Name of csv file
#   Output : Data frame
#-----------------------------------------------------
def LoadData(filename):
    df = pd.read_csv(filename)

    print("Dataset loaded successfully")
    print(df.head())

    return df


# Step 2 : Data Preprocessing

#-----------------------------------------------------
#   Function Name : PreprocessData
#   Description : It performs data preprocessing
#   Input : Data Frame
#   Output : Updated Data frame
#-----------------------------------------------------
def PreprocessData(df):

    df = df.drop(
        columns=[
            "Passengerid",
            "zero",
            "name"
        ],
        errors="ignore"
    )

    # Handle missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    df["Embarked"] = df["Embarked"].fillna(
        df["Embarked"].mode()[0]
    )

    # Convert categorical data into numeric columns
    df = pd.get_dummies(
        df,
        columns=["Embarked"],
        drop_first=True,
        dtype=int
    )

    print(df.head())
    print("Data preprocessing completed")

    return df


# Step 3 : Split Data

#-----------------------------------------------------
#   Function Name : SplitData
#   Description : It performs splitting activity
#   Input : Data Frame
#   Output : Four subsets for training and testing
#-----------------------------------------------------
def SplitData(df):

    X = df.drop("Survived", axis=1)
    Y = df["Survived"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    print("Dataset splitting completed successfully")

    return X_train, X_test, Y_train, Y_test


# Step 4 : Train the models

#-----------------------------------------------------
#   Function Name : TrainModels
#   Description : It performs model training
#   Input : Training features and labels
#   Output : Trained models
#-----------------------------------------------------
def TrainModels(X_train, Y_train):

    models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            max_depth=5
        ),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ])
    }

    for name, model in models.items():
        model.fit(X_train, Y_train)
        print(name, "model trained successfully")

    return models


# Step 5 : Evaluate models

#-----------------------------------------------------
#   Function Name : EvaluateModels
#   Description : It evaluates trained models
#   Input : Models, testing data and labels
#   Output : Model performance
#-----------------------------------------------------
def EvaluateModels(models, X_test, Y_test):

    results = []

    for name, model in models.items():

        Y_pred = model.predict(X_test)

        accuracy = accuracy_score(Y_test, Y_pred)

        print("\nModel:", name)
        print("Accuracy:", accuracy)
        print("Confusion Matrix:")
        print(confusion_matrix(Y_test, Y_pred))
        print("Classification Report:")
        print(classification_report(Y_test, Y_pred))

        results.append({
            "Model": name,
            "Accuracy": accuracy
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("Titanic_Model_Results.csv", index=False)

    print("\nModel comparison:")
    print(results_df)

    return results_df


# Step 6 : Preserve Models

#-----------------------------------------------------
#   Function Name : PreserveModels
#   Description : It saves trained models into .pkl files
#   Input : Trained models
#   Output : Saved model files
#-----------------------------------------------------
def PreserveModels(models):

    joblib.dump(
        models["Decision Tree"],
        "Titanic_DecisionTree.pkl"
    )

    joblib.dump(
        models["KNN"],
        "Titanic_KNN.pkl"
    )

    print("Decision Tree model saved successfully")
    print("KNN model saved successfully")


#-----------------------------------------------------
#   Function Name : main
#   Description : Entry point function
#   Input : None
#   Output : None
#-----------------------------------------------------
def main():

    # Step 1
    df = LoadData("MarvellousTitanicDataset.csv")

    # Step 2
    df = PreprocessData(df)

    # Step 3
    X_train, X_test, Y_train, Y_test = SplitData(df)

    # Step 4
    models = TrainModels(X_train, Y_train)

    # Step 5
    EvaluateModels(models, X_test, Y_test)

    # Step 6
    PreserveModels(models)


if __name__ == "__main__":
    main()
