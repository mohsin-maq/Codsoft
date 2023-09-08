import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Load the saved Logistic Regression model
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the CountVectorizer and TF-IDF Transformer fitted on the training data
with open('count_vectorizer.pkl', 'rb') as cv_file, open('tfidf_transformer.pkl', 'rb') as tfidf_file:
    count_vectorizer = pickle.load(cv_file)
    tfidf_transformer = pickle.load(tfidf_file)

# Set page title and icon
st.set_page_config(page_title="Spam Detector", page_icon=":shield:")

# Add a title with custom style
st.title("Spam Message Detector")
st.markdown("<p style='text-align: center; color: #007BFF;'>Detect SPAM in Text Messages</p>", unsafe_allow_html=True)

# Add an app logo
from PIL import Image
image = Image.open('insurance.png')
st.sidebar.image(image, caption='Be Safe!', use_column_width=True)
st.sidebar.header("How to Use:")
st.sidebar.markdown("""
1. Enter a text message in the text box on the main panel.
2. Click the "Check SPAM" button to analyze the message.
3. The app will classify the message as either "SPAM" or "NOT SPAM."
4. You will also see the SPAM probability percentage.
""")

# Add any additional information or links to resources if needed
st.sidebar.markdown("""
If you encounter any issues or have feedback, please contact us at [support@example.com](mailto:support@example.com).
""")

# Add explanation
st.markdown("This app uses a trained Logistic Regression model to detect whether a text message is SPAM or NOT SPAM.")
st.markdown("Use the slider to adjust the SPAM threshold and click 'Check SPAM' to get the prediction.")

# Add a slider for threshold
threshold = st.slider("Threshold for Predicting SPAM", 0.0, 1.0, 0.5)

# Add a text input box for user input
user_input = st.text_area("Enter a text message:")

# Check SPAM button
if st.button("Check SPAM"):
    # Transform the user input using the loaded transformers
    user_input_counts = count_vectorizer.transform([user_input])
    user_input_tfidf = tfidf_transformer.transform(user_input_counts)

    # Predict using the loaded Logistic Regression model
    prediction = model.predict(user_input_tfidf)
    probability = model.predict_proba(user_input_tfidf)

    # Display the prediction result with colored text and appropriate sign
    if prediction[0] == 0:
        result_text = "NOT SPAM"
        result_color = "#00FF00" 
        sign = "✔️"
    else:
        result_text = "SPAM"
        result_color = "red"
        sign = "❗"

    # Display the result with colored text and sign
    st.markdown(f"<h1 style='color: {result_color};'>{sign} This message is {result_text}</h1>", unsafe_allow_html=True)

    # Display SPAM Probability with increased size
    st.write(f"<h2 style='font-size: 24px;'>SPAM Probability: {probability[0][1]*100:.2f}%</h2>", unsafe_allow_html=True)





