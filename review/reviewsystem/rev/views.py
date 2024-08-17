from django.shortcuts import render
from django.http import HttpResponseRedirect
import re
import torch
import torch.nn as nn
import joblib
import numpy as np
import nltk
import nltk.sentiment
import nltk.sentiment.util

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text):
        lstm_out, (hidden, cell) = self.lstm(text)
        return torch.sigmoid(self.fc(self.dropout(hidden[-1])))


# Instantiate the LSTM model
input_dim = 5000  # Update this if your TF-IDF vectorizer uses a different number of features
hidden_dim = 256
output_dim = 1

model = LSTMModel(input_dim, hidden_dim, output_dim)

# Load the saved model state dictionary
model.load_state_dict(
    torch.load(r'C:\Users\babin\OneDrive\Desktop\review\reviewsystem\static\LSTM_model_state.pth',
               map_location=torch.device('cpu')))

# Load the TF-IDF vectorizer
vectorization = joblib.load(r'C:\Users\babin\OneDrive\Desktop\review\reviewsystem\static\TFIDF_vectorization_v2.pkl')


def clean_text(text):
    cleaned_text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    cleaned_text = cleaned_text.strip()  # Trim leading and trailing spaces
    cleaned_text = cleaned_text.lower()  # Convert to lowercase
    return cleaned_text


def review(request):
    if request.method == 'POST':
        obb = request.POST.get('review')
        text = clean_text(obb)
        print('-------------------------------')
        print('Input Review = ', text)
        print('-------------------------------')

        # Transform text using TF-IDF vectorizer
        text_vectorized = vectorization.transform([text])
        text_tensor = torch.tensor(text_vectorized.toarray(), dtype=torch.float32)

        # Add sequence dimension for LSTM input
        text_tensor = text_tensor.unsqueeze(1)  # Shape: (1, 1, input_dim)

        # Make a prediction using the LSTM model
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            prediction = model(text_tensor).squeeze(1).item()

        print('-------------------------------')
        print('Prediction = ', prediction)
        print('-------------------------------')

        prediction1 = round(prediction * 100, 2)

        # Sentiment analysis using VADER
        sid = nltk.sentiment.vader.SentimentIntensityAnalyzer()
        compound = sid.polarity_scores(obb)["compound"]
        print("result of sentiment", compound)

        if compound > 0:
            senti = "positive"
        elif compound < 0:
            senti = "Negative"
        else:
            senti = "neutral"

        if prediction < 0.5:
            aa = f'This review is {100 - prediction1}% chance to be fake '
            confi = 100 - prediction1
        else:
            aa = f'This review is {prediction1}% chance to be genuine '
            confi = prediction1

        context = {
            'kk': aa,
            'per': confi,
            'sr': senti
        }
        return render(request, 'rev/review.html', context)

    return render(request, 'rev/review.html')
