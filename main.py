import os                                                             # Import the os module for interacting with the operating system.
import nltk                                                           # Import the nltk library for natural language processing.
import ssl                                                            # Import the ssl module to work with secure socket layers (SSL).
import streamlit as st                                                # Import the streamlit library for building interactive web applications.
import random                                                         # Import the random module for generating random numbers.
from sklearn.feature_extraction.text import TfidfVectorizer           # Import TfidfVectorizer from scikit-learn for text vectorization.
from sklearn.linear_model import LogisticRegression                   # Import LogisticRegression from scikit-learn for logistic regression classification.
ssl._create_default_https_context = ssl._create_unverified_context    # Set up SSL context to create an unverified context for HTTPS connections
nltk.data.path.append(os.path.abspath("nltk_data"))                   # Append the absolute path to the nltk_data directory to the nltk data path
nltk.download('punkt')                                                # Download the 'punkt' dataset from nltk

#Define some intents of the chatbot.
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": ["To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.", "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.", "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": ["A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.", "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."]
    }
]
#Prepare the intents and train a Machine Learning model for the chatbot.
# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()                            # Initialize TfidfVectorizer for converting patterns into numerical features
clf = LogisticRegression(random_state=0, max_iter=10000)  # Initialize Logistic Regression classifier with specified parameters

# Preprocess the data                              
tags = []                                                 #  Initialize empty lists to store tags and patterns
patterns = []
for intent in intents:                                    # Iterate through each intent in the intents list
        for pattern in intent['patterns']:                # Iterate through each pattern in the 'patterns' key of the intent
                tags.append(intent['tag'])                # Append the tag (intent category) and pattern to the respective lists
                patterns.append(pattern)
# Training the model
x = vectorizer.fit_transform(patterns)                    # Convert the patterns into a numerical feature matrix using TfidfVectorizer
y = tags                                                  # Assign the tags to the target variable
clf.fit(x, y)                                             # Train the Logistic Regression model using the feature matrix (x) and target variable (y)

#Write a Python function to chat with the chatbot:


def chatbot(input_text):
    # Convert the input text into a numerical feature using the vectorizer
    input_text = vectorizer.transform([input_text])
    
    # Predict the tag using the trained classifier
    tag = clf.predict(input_text)[0]
    
    # Iterate through intents to find the matching tag
    for intent in intents:
        if intent['tag'] == tag:
            # Select a random response from the matched intent
            response = random.choice(intent['responses'])
            return response
        
# To turn this chatbot into an end-to-end chatbot,
# I need to deploy it to interact with the chatbot using a user interface. 
# To deploy the chatbot, I will use the streamlit library in Python,       
        
#So, here’s how I can deploy the chatbot using Python:
# Initialize a global variable to keep track of the number of interactions
# ... (previous code remains unchanged)

# Initialize a global variable to keep track of the number of interactions
counter = 0

# Define the main function to create the Streamlit application
def main():
    global counter  # Declare counter as a global variable
    
    # Set the title of the Streamlit app to "Chatbot"
    st.title("Chatbot")

    # Display a welcome message in the Streamlit app
    st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

    # Increment the counter for each interaction
    counter += 1

    # Take user input using a text input box with a unique key for each interaction
    user_input = st.text_input("You:", key=f"user_input_{counter}")

    # Check if the user has entered any input
    if user_input:
        # Call the chatbot function to generate a response based on the user input
        response = chatbot(user_input)

        # Display the chatbot's response in a text area with a unique key for each interaction
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")

        # Check if the chatbot's response indicates a farewell message
        if response.lower() in ['goodbye', 'bye']:
            # Display a thank-you message and end the Streamlit app
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

# Ensure that the main function is executed when the script is run
if __name__ == '__main__':
    main()


# To run this chatbot, I need tpo use the command mentioned below in my terminal:
#             streamlit run filename.py

