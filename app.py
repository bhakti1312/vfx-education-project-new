import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------------
# PROJECT TITLE
# ----------------------------

st.title("Role of VFX in Education using Machine Learning")
st.subheader("Developed by: Bhakti Shinde")

# ----------------------------
# FILE UPLOAD
# ----------------------------

human_file = st.file_uploader("Upload Human Dataset CSV")

ai_file = st.file_uploader("Upload AI Dataset CSV")

# ----------------------------
# FUNCTION
# ----------------------------

def evaluate(df):

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted') * 100
    recall = recall_score(y_test, y_pred, average='weighted') * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100

    return accuracy, precision, recall, f1


# ----------------------------
# PROCESS
# ----------------------------

if human_file and ai_file:

    human_df = pd.read_csv(human_file)
    ai_df = pd.read_csv(ai_file)

    h_acc, h_pre, h_rec, h_f1 = evaluate(human_df)
    ai_acc, ai_pre, ai_rec, ai_f1 = evaluate(ai_df)

    results = pd.DataFrame({

        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Human Response": [h_acc, h_pre, h_rec, h_f1],
        "AI Generated": [ai_acc, ai_pre, ai_rec, ai_f1]

    })

    st.write("Performance Comparison")
    st.dataframe(results)

# ----------------------------
# GRAPH
# ----------------------------

    fig, ax = plt.subplots()

    x = range(len(results["Metric"]))

    ax.bar(x, results["Human Response"], width=0.4)

    ax.bar([i + 0.4 for i in x], results["AI Generated"], width=0.4)

    ax.set_xticks([i + 0.2 for i in x])

    ax.set_xticklabels(results["Metric"])

    ax.set_ylabel("Score (%)")

    ax.set_title("Human vs AI Comparison")

    st.pyplot(fig)

# ----------------------------
# BEST RESULT MESSAGE
# ----------------------------

    if h_acc > ai_acc:

        st.success("Human Response is Higher than AI Generated")

    else:

        st.warning("AI Generated is Higher")

# ----------------------------
# DOWNLOAD REPORT
# ----------------------------

    csv = results.to_csv(index=False)

    st.download_button(

        label="Download Report",

        data=csv,

        file_name="VFX_Report.csv",

        mime="text/csv"

    )
