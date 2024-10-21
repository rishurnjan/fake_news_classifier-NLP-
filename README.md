### **Fake News Classifier using LSTM and Word Embedding**

This project builds a classifier that detects fake news using an LSTM (Long Short-Term Memory) model with word embeddings. The model is trained using the [Fake News Dataset](https://www.kaggle.com/c/fake-news/data#) from Kaggle.

#### **Dataset Overview**
- The dataset contains 4 columns: 
  - `id`: Unique identifier for the news article
  - `title`: Title of the news article
  - `author`: Author of the news article
  - `text`: Full text of the news article
  - `label`: Binary label, where 1 indicates fake news and 0 indicates real news.

#### **Steps Involved:**

1. **Data Preparation**
   - The dataset is loaded and missing values are dropped.
   - Text preprocessing is done using the NLTK library, which includes:
     - Lowercasing the text
     - Removing special characters
     - Tokenizing and stemming the words
     - Removing stop words
   - The processed text is then converted into word embeddings using one-hot encoding.

2. **Model Architecture**
   - A Sequential LSTM model is built using Keras:
     - **Embedding Layer**: This converts the input data into dense vector embeddings.
     - **LSTM Layer**: LSTM processes the sequence data and captures temporal dependencies.
     - **Dense Layer**: Outputs a single value with a sigmoid activation function for binary classification.

3. **Training and Evaluation**
   - The model is compiled using `binary_crossentropy` as the loss function, `adam` as the optimizer, and accuracy as the evaluation metric.
   - The data is split into training and testing sets (67% training and 33% testing).
   - The model is trained for 10 epochs with a batch size of 64.
   - The evaluation is done using a confusion matrix, accuracy score, and classification report.

#### **Results:**
- **Accuracy**: 91%
- **Confusion Matrix**:
  ```
  [[3088,  331],
   [ 211, 2405]]
  ```
  - Precision, recall, and F1-score are also calculated and reported.

#### **Possible Improvements:**
- **Increase Dataset Size**: Adding more data can help improve model performance.
- **Use Pre-trained Word Embeddings**: Utilizing pre-trained word embeddings like GloVe or Word2Vec can potentially enhance the model's understanding of words.
- **Hyperparameter Tuning**: Adjusting the learning rate, number of LSTM units, and dropout rates could improve the accuracy.
- **Model Variations**: Trying other architectures like GRU or transformers for more complex sequence handling.

#### **Code Snippet for Model Creation:**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

embedding_vector_feature = 40
model = Sequential()
model.add(Embedding(5000, embedding_vector_feature, input_length=sent_len))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

#### **Conclusion:**
This project demonstrates the application of an LSTM network for detecting fake news based on article titles. While achieving 91% accuracy, the model could benefit from further improvements with larger datasets, better embeddings, and more sophisticated models.
