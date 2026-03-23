import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Step 1: Dataset
data = {
    "text": [
        "Payment not processed",
        "Unable to login",
        "App is crashing",
        "Need help with account",
        "Refund not received",
        "Website is slow",
        "Password reset issue",
        "Order not delivered",
        "General inquiry about product",
        "Technical error in app"
    ],
    "category": [
        "Billing",
        "Technical",
        "Technical",
        "General",
        "Billing",
        "Technical",
        "Technical",
        "Billing",
        "General",
        "Technical"
    ]
}

df = pd.DataFrame(data)

# Step 2: Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['category']

# Step 3: Train model
model = MultinomialNB()
model.fit(X, y)

# Step 4: Test prediction
test = ["App not working"]
test_vec = vectorizer.transform(test)

prediction = model.predict(test_vec)
print("Predicted Category:", prediction[0])

# Step 5: Priority logic
if prediction[0] == "Technical":
    print("Priority: High")
elif prediction[0] == "Billing":
    print("Priority: Medium")
else:
    print("Priority: Low")
