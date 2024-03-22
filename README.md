# SMS Spam Classification Challenge

This project demonstrates a machine learning pipeline for classifying SMS messages as "spam" or "ham" (not spam) using Python, Scikit-learn, and Gradio. The goal is to accurately predict whether a text message is spam, providing a user-friendly interface for real-time predictions.

## Features

- **Data Preprocessing:** Utilizes `TfidfVectorizer` to transform text messages into a format suitable for machine learning models.
- **Model Training:** Employs a `LinearSVC` model within a pipeline to classify messages, trained on a dataset of labeled SMS messages.
- **Interactive Prediction Interface:** A Gradio application enables users to input SMS messages and receive instant spam/ham classification.

## Installation

Before running this project, ensure you have Python installed on your system. Then, install the required dependencies by running:

```bash
pip install pandas scikit-learn gradio
```

## Dataset

The project uses the `SMSSpamCollection.csv` dataset, which contains labeled SMS messages as "spam" or "ham". The dataset is split into training and testing sets to evaluate the model's performance.

## Usage

To use this project:

1. Load the `SMSSpamCollection.csv` dataset.
2. Run the `sms_classification` function to train the model.
3. Launch the Gradio interface to input new SMS messages and receive spam/ham classifications.

### Example Code

```python
# Load the dataset and train the model
df = pd.read_csv('SMSSpamCollection.csv', delimiter=',', names=['label', 'text_message'], skiprows=1)
text_clf = sms_classification(df)

# Create and launch the Gradio app
sms_app = create_app()
sms_app.launch()
```

## Functions

- `sms_classification(sms_text_df)`: Trains the spam classification model using the provided dataset.
- `sms_prediction(text)`: Predicts whether a given SMS message is spam or ham.
- `create_app()`: Sets up the Gradio interface for interactive spam/ham classification.

## Testing

Test the application with various SMS messages to evaluate the model's performance. Example test messages include promotional texts, normal conversations, and phishing attempts.

## Contributing

Feel free to fork this project and submit pull requests with improvements or additional features. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT License](https://opensource.org/licenses/MIT)


