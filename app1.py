import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, roc_curve, auc, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import json
from streamlit_lottie import st_lottie

# Load Lottie animation
def load_lottie_animation(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

# Sidebar options
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose a section", ["Home", "Visualizations", "Models", "Model Comparisons", "Assessment"])

# File uploader widget
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

@st.cache_data
def preprocess_data(df):
    label_encoders = {}
    for column in ['Gender', 'Drug_Type', 'Frequency_of_Use', 'Mental_Health_Status', 'Employment_Status', 'Social_Support']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    X = df.drop(columns=['Verified'])
    y = df['Verified']

    # Add noise to numerical features
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    X[numerical_cols] += np.random.normal(0, 0.1, X[numerical_cols].shape)

    # Create random outliers
    num_outliers = int(0.05 * X.shape[0])
    outliers = np.random.uniform(low=-1, high=1, size=(num_outliers, X.shape[1]))
    outlier_labels = np.random.randint(0, 2, size=num_outliers)

    X_with_outliers = np.vstack([X, outliers])
    y_with_outliers = np.concatenate([y, outlier_labels])

    # Shuffle the dataset
    df_with_outliers = pd.DataFrame(X_with_outliers, columns=X.columns)
    df_with_outliers['Verified'] = y_with_outliers
    df_shuffled = df_with_outliers.sample(frac=1, random_state=42).reset_index(drop=True)

    X_shuffled = df_shuffled.drop(columns=['Verified'])
    y_shuffled = df_shuffled['Verified']
    
    return X_shuffled, y_shuffled, X_shuffled.columns

@st.cache_data
def train_models(X_train, y_train):
    # Train models with predefined parameters
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)

    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)

    # Simplified neural network
    nn_model = Sequential()
    nn_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    nn_model.add(Dropout(0.3))
    nn_model.add(Dense(32, activation='relu'))
    nn_model.add(Dense(1, activation='sigmoid'))

    nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Reduced epochs for faster training
    nn_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, callbacks=[early_stopping], verbose=1)

    return rf_model, xgb_model, nn_model
def plot_metrics(metrics_df):
    metrics_df.set_index('Model', inplace=True)
    plt.figure(figsize=(12, 8))
    metrics_df[['Accuracy', 'Precision', 'F1 Score']].plot(kind='bar', figsize=(12, 8))
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.grid(True)
    st.pyplot(plt)


def evaluate_models(X_test, y_test, rf_model, xgb_model, nn_model):
    # Make predictions
    rf_pred = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    nn_pred = (nn_model.predict(X_test) > 0.5).astype(int).flatten()

    # Calculate metrics
    def calculate_metrics(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return accuracy, precision, recall, f1, rmse

    models = {
        'Random Forest': rf_pred,
        'XGBoost': xgb_pred,
        'Neural Network': nn_pred,
    }

    metrics = {}
    for model_name, predictions in models.items():
        metrics[model_name] = calculate_metrics(y_test, predictions)

    metrics_df = pd.DataFrame(metrics).T
    metrics_df.reset_index(inplace=True)
    metrics_df.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'RMSE']
    
    return metrics_df

def plot_roc_curves(X_test, y_test, rf_model, xgb_model, nn_model):
    plt.figure(figsize=(12, 8))
    
    # Random Forest
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
    rf_auc = auc(rf_fpr, rf_tpr)
    plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
    
    # XGBoost
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs)
    xgb_auc = auc(xgb_fpr, xgb_tpr)
    plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {xgb_auc:.2f})')
    
    # Neural Network
    nn_probs = nn_model.predict(X_test).flatten()
    nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_probs)
    nn_auc = auc(nn_fpr, nn_tpr)
    plt.plot(nn_fpr, nn_tpr, label=f'Neural Network (AUC = {nn_auc:.2f})')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    st.pyplot(plt)

def plot_confusion_matrices(X_test, y_test, rf_model, xgb_model, nn_model):
    models = {
        'Random Forest': rf_model,
        'XGBoost': xgb_model,
        'Neural Network': nn_model,
    }

    for model_name, model in models.items():
        if model_name == 'Neural Network':
            predictions = (model.predict(X_test) > 0.5).astype(int).flatten()
        else:
            predictions = model.predict(X_test)
        
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=['Not Verified', 'Verified'], 
                    yticklabels=['Not Verified', 'Verified'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.grid(True)
        st.pyplot(plt)

def plot_feature_importances(rf_model, xgb_model, feature_names):
    # Plot feature importances for Random Forest
    plt.figure(figsize=(12, 8))
    rf_importances = rf_model.feature_importances_
    indices = np.argsort(rf_importances)[::-1]
    plt.title('Random Forest Feature Importances')
    plt.bar(range(len(indices)), rf_importances[indices], align='center')
    plt.xticks(range(len(indices)), np.array(feature_names)[indices], rotation=90)
    plt.xlim([-1, len(indices)])
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.grid(True)
    st.pyplot(plt)

    # Plot feature importances for XGBoost
    plt.figure(figsize=(12, 8))
    xgb_importances = xgb_model.feature_importances_
    indices = np.argsort(xgb_importances)[::-1]
    plt.title('XGBoost Feature Importances')
    plt.bar(range(len(indices)), xgb_importances[indices], align='center')
    plt.xticks(range(len(indices)), np.array(feature_names)[indices], rotation=90)
    plt.xlim([-1, len(indices)])
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.grid(True)
    st.pyplot(plt)
def plot_histograms(numeric_df):
    st.subheader('Feature Histograms')
    for column in numeric_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(numeric_df[column], kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True)
        st.pyplot(plt)
def plot_correlation_heatmap(numeric_df):
    st.subheader('Correlation Heatmap')
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.grid(True)
    st.pyplot(plt)
def plot_pairplot(numeric_df):
    st.subheader('Pairplot of Features')
    if len(numeric_df.columns) > 1:
        sns.pairplot(numeric_df)
        st.pyplot(plt)
def plot_boxplots(df, numeric_df):
    st.subheader('Boxplots for Numerical Features by Categorical Variables')
    categorical_columns = ['Gender', 'Drug_Type', 'Frequency_of_Use', 'Mental_Health_Status', 'Employment_Status', 'Social_Support']
    for column in categorical_columns:
        if column in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[column], y=numeric_df['Assessment_Score'])
            plt.title(f'Boxplot of Assessment Score by {column}')
            plt.xlabel(column)
            plt.ylabel('Assessment Score')
            plt.grid(True)
            st.pyplot(plt)

# New plots
def plot_drug_distribution_pie(df):
    genders = ['Male', 'Female', 'Other']
    drug_labels = {0: 'Cocaine', 1: 'Heroin', 2: 'Methamphetamine', 3: 'Prescription Drugs', 4: 'Alcohol'}
    gender_labels = {0: 'Male', 1: 'Female', 2: 'Other'}

    for gender in genders:
        plt.figure(figsize=(10, 6))
        gender_df = df[df['Gender'] == gender]
        drug_distribution = gender_df['Drug_Type'].value_counts(normalize=True)
        drug_distribution.rename(index=drug_labels, inplace=True)

        plt.pie(drug_distribution, labels=drug_distribution.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
        plt.title(f'Drug Type Distribution for {gender}')
        plt.legend(title='Drug Type', loc='upper left', labels=drug_distribution.index)
        plt.axis('equal')
        st.pyplot(plt)


def plot_rehab_distribution(df):
    plt.figure(figsize=(8, 8))
    mental_health_distribution = df['Mental_Health_Status'].value_counts(normalize=True)
    mental_health_labels = {0: 'Depressed', 1: 'Anxiety', 2: 'Angry', 3: 'Happy', 4: 'Sad', 5: 'Normal'}

    mental_health_distribution.rename(index=mental_health_labels, inplace=True)

    plt.pie(mental_health_distribution, labels=mental_health_distribution.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"), wedgeprops=dict(width=0.4))
    plt.title('Mental Health Status Distribution')
    plt.legend(title='Mental Health Status', loc='upper left', labels=mental_health_distribution.index)
    plt.axis('equal')
    st.pyplot(plt)
    


# Load and display Lottie animation based on selected option
if option == "Home":
    st.title("Welcome to the Drug Dependency and Rehab Assessment Tool")
    st.write("This tool helps in assessing drug dependency and provides insights into whether rehab is needed.")
    
    # Load and display Lottie animation
    lottie_animation = load_lottie_animation('Animation - 1723391518711.json')
    st_lottie(lottie_animation)

elif option == "Visualizations":
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Overview")
        st.dataframe(df.head())

        # Separate numeric and categorical columns
        numeric_df = df.select_dtypes(include=[np.number])

        st.write("### Drug Type Distribution by Gender (Pie Charts)")
        plot_drug_distribution_pie(df)

        st.write("### Mental Health Status Distribution")
        plot_rehab_distribution(df)

        
        plot_histograms(numeric_df)

        
        plot_correlation_heatmap(numeric_df)

        
        plot_pairplot(numeric_df)

        
        plot_boxplots(df, numeric_df)
        
    else:
        st.warning("Please upload a CSV file to view visualizations.")

elif option == "Models":
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        X, y, feature_names = preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf_model, xgb_model, nn_model = train_models(X_train, y_train)

        st.write("### Models have been trained successfully!")

        st.write("### Model Evaluation")
        metrics_df = evaluate_models(X_test, y_test, rf_model, xgb_model, nn_model)
        st.dataframe(metrics_df)

        # Plot metrics
        st.write("### Performance Metrics")
        plot_metrics(metrics_df)
        
        st.write("### ROC Curves")
        plot_roc_curves(X_test, y_test, rf_model, xgb_model, nn_model)
        
        st.write("### Confusion Matrices")
        plot_confusion_matrices(X_test, y_test, rf_model, xgb_model, nn_model)
        
    else:
        st.warning("Please upload a CSV file to train models.")

elif option == "Model Comparisons":
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        X, y, feature_names = preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf_model, xgb_model, nn_model = train_models(X_train, y_train)
        st.write("### Models Comparison Through Confusion Matrices ")

        st.write("### Confusion Matrices")
        plot_confusion_matrices(X_test, y_test, rf_model, xgb_model, nn_model)
        
    else:
        st.warning("Please upload a CSV file to compare models.")


# Rehab Assessment
elif option == "Assessment":
    st.title("Drug Rehab Assessment")
    st.write("Please fill out the following assessment questions:")

    age = st.slider("What is your age?", 10, 100, 30)
    gender = st.selectbox("What is your gender?", ["Male", "Female", "Other"])
    drug_type = st.selectbox("What type of drug are you using?", ["Alcohol", "Marijuana", "Opioids", "Cocaine", "Methamphetamine", "Other"])
    duration_of_use = st.slider("How long have you been using this drug? (in years)", 0, 50, 5)
    frequency_of_use = st.selectbox("How often do you use this drug?", ["Daily", "Weekly", "Monthly", "Occasionally"])
    mental_health_status = st.selectbox("What is your current mental health status?", ['Depressed', 'Anxiety', 'Angry', 'Happy', 'Sad', 'Normal'])
    previous_rehab_attempts = st.selectbox("Have you previously attempted rehab?", ["Yes", "No"])
    employment_status = st.selectbox("What is your current employment status?", ["Employed", "Unemployed", "Student", "Retired"])
    social_support = st.selectbox("Do you have social support from friends or family?", ["Yes", "No"])

    if st.button("Submit"):
        st.write("Thank you for completing the assessment.")

        # Define points for each question
        points = {
            'drug_type': {
                "Alcohol": 20,
                "Marijuana": 10,
                "Opioids": 30,
                "Cocaine": 40,
                "Methamphetamine": 50,
                "Other": -10
            },
            'frequency_of_use': {
                "Daily": -20,
                "Weekly": -10,
                "Monthly": 0,
                "Occasionally": -5
            },
            'mental_health_status': {
                "Depressed": -30,
                "Anxiety": -20,
                "Angry": -10,
                "Happy": 10,
                "Sad": -20,
                "Normal": -10
            },
            'previous_rehab_attempts': {
                "Yes": 10,
                "No": -10
            },
            'employment_status': {
                "Employed": 0,
                "Unemployed": -10,
                "Student": -5,
                "Retired": -10
            },
            'social_support': {
                "Yes": 10,
                "No": -10
            }
        }

        # Calculate the assessment score
        score = (
            points['drug_type'].get(drug_type, 0) +
            points['frequency_of_use'].get(frequency_of_use, 0) +
            points['mental_health_status'].get(mental_health_status, 0) +
            points['previous_rehab_attempts'].get(previous_rehab_attempts, 0) +
            points['employment_status'].get(employment_status, 0) +
            points['social_support'].get(social_support, 0)
        )

        st.write(f"Your assessment score is: {score}")

        # Provide feedback based on the score
        if score < 0:
            st.write("Based on your responses, it is recommended that you seek rehab.")
        elif score > 0:
            st.write("Based on your responses, rehab may be beneficial but not immediately necessary.")
        else:
            st.write("Based on your responses, rehab may not be necessary at this time.")
