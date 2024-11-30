# **NLP-Driven Rasa-Assiste ChatBot360: Seq2Seq Chatbot for Customer Support**

---

## **1. Overview**

The **NLP-Driven Rasa-Assiste ChatBot360** is a Seq2Seq-based chatbot designed to handle customer inquiries in a customer support environment. The chatbot aims to simulate interactions typically encountered in customer service, such as product information requests, status inquiries, and troubleshooting. This project involves the implementation of Sequence-to-Sequence (Seq2Seq) models using PyTorch and incorporates data pre-processing, model training, evaluation, and deployment.

**Dataset Downloaded:**
- The dataset used in this project is the [Customer Support on Twitter Dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).

**Project Components:**
- **Model Implementation**: Encoder-Decoder architecture with attention mechanisms.
- **Deployment**: Both as a web-based application and a console-based chatbot.
- **Evaluation**: Using metrics such as BLEU and METEOR for performance assessment.
- **Challenges and Future Directions**: Details on limitations, improvements, and enhancements.

---

## **2. Dataset Preparation**

### **2.1 Dataset Overview**

The **Customer Support on Twitter** dataset is sourced from Kaggle. This dataset contains interactions between customers and support representatives on Twitter, making it suitable for training a customer support chatbot. The dataset includes various types of questions and corresponding responses that can be used to train the model to handle typical customer service conversations.

**Link to Dataset:**
- [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)

**Types of Data**:
- Customer questions
- Support responses
- Product inquiries
- Status updates and troubleshooting

### **2.2 Data Cleaning and Preprocessing**

Data preparation is critical for training a chatbot model effectively. Here is how the dataset was processed:

- **Text Normalization**: All text was converted to lowercase to maintain uniformity, and unnecessary HTML tags, special characters, and punctuation were removed.
- **Tokenization**: Sentences were split into individual words or tokens to allow the model to learn from the components of each sentence.
- **Lemmatization and Stemming**: Words were converted to their root forms (e.g., "running" to "run") to ensure consistency.
- **Entity Replacement**: Specific entities like names, order numbers, and dates were replaced with generic placeholders such as `<name>`, `<order_id>`, and `<date>`. This helps the model focus on the structure of the interaction rather than memorizing specific details.
  
---

## **3. Seq2Seq Model Implementation**

### **3.1 Model Choice**

The model used in this project is based on **Sequence-to-Sequence (Seq2Seq)** architecture, which is well-suited for mapping one sequence to another. This is ideal for customer queries and responses. The Seq2Seq model comprises two parts:

- **Encoder**: Reads the input sequence and converts it to a context vector.
- **Decoder**: Takes the context vector and generates an output sequence.

The model is implemented using **PyTorch**, which provides the necessary modules and tools for defining, training, and optimizing the Seq2Seq architecture.

### **3.2 Model Design**

#### **Encoder**
The **Encoder** processes input sentences and outputs a fixed-size context vector. In this project, an LSTM or GRU was used for the encoder:

- **Input Embedding**: The input words are transformed into numerical vectors using **pre-trained word embeddings** (GloVe vectors are used in this project).
- **RNN/LSTM/GRU Layers**: To learn the temporal dependencies between tokens, the encoder comprises one or more layers of recurrent units.

#### **Decoder**
The **Decoder** is responsible for generating the response sequence based on the context vector from the encoder:

- The **context vector** from the encoder is used as input to the decoder.
- The decoder uses **RNN/LSTM/GRU layers** similar to the encoder, with an additional **Attention mechanism** that allows it to focus on relevant parts of the input during generation.
- **Attention Mechanism**: Enhances response quality by enabling the decoder to focus on specific elements of the input sequence during each step.

### **3.3 Training the Model**

#### **Input Representation**
- **Word Embeddings**: Pre-trained embeddings, such as **GloVe**, were used to represent input tokens numerically. In this project, embeddings from `glove.6B.*.txt` were used, providing meaningful relationships between words.

#### **Hyperparameters**
- **Learning Rate**: Initial learning rate set to `0.001`.
- **Batch Size**: Batch size of `16` used for training.
- **Epochs**: Trained for an epoch to evaluate initial model performance (since computationally expensive).

#### **Loss Function**
- **Cross-Entropy Loss** was used to measure the error between the predicted responses and the actual target responses during training.

#### **Optimizer**
- **Adam** optimizer was chosen due to its efficiency in dealing with sparse gradients.

### **3.4 Evaluation Metrics**

- **BLEU Score**: Measures how similar the generated responses are to reference responses.
- **METEOR Score**: Evaluates the generated responses based on synonyms, stemming, and exact matches.

### **3.5 Model Checkpoints**

- During training, **model checkpoints** were saved after each epoch. This allows for resuming training if interrupted and provides versions to compare during performance analysis.

---

## **4. Deployment**

### **4.1 Web Application**

A **Flask-based web application** was developed to interact with the trained chatbot. The web app allows users to enter queries and receive responses in a conversational manner.

**Directory Structure for Web Application**:
```
webapp/
├── app.py                       # Flask app for chatbot interactions
├── templates/
│   └── index.html               # HTML file for chatbot UI
```

#### **Flask Application (`app.py`)**
- The Flask app serves as a backend for handling user queries, processing inputs, generating responses, and displaying them.
- The chatbot retains the conversation history, allowing users to see a complete interaction flow.

#### **Web Interface (`index.html`)**
- A simple, Bootstrap-enhanced interface is used to create an interactive chatbot experience.
- Users can input their questions and see a response generated by the trained model.

### **4.2 Console-based Interface**

In addition to the web application, a **console-based interface** was implemented. Users can interact with the chatbot directly through the terminal by typing queries and receiving responses.
![web-app-screenshot.png](screenshots%2Fweb-app-screenshot.png)
**Console Interaction Script**:
- **console_chat.py**: Implements a console-based interface for direct interaction.

---

## **5. Challenges and Future Directions**

### **5.1 Challenges Faced**

#### **1. Computational Limitations**
- **Training on 10% of Dataset**: Due to limited computational resources, only 10% of the dataset was used for training. Training on the full dataset requires significant resources, which were not available at the time.
- **Solution**: The next iteration will involve scaling up to cloud-based GPUs for more extensive training on the entire dataset.

#### **2. Parameter Optimization**
- **Continuous Training**: The model needs to be retrained continuously with updated data to improve its accuracy. Optimizing the parameter set (e.g., learning rate decay, gradient clipping) would allow for better performance.

#### **3. Hyperparameter Tuning**
- **Automated Hyperparameter Tuning**: Although some basic hyperparameters were set manually, hyperparameter tuning using **grid search** or **Bayesian optimization** could improve performance by systematically exploring the parameter space.

#### **4. Attention Mechanisms**
- Implementing more sophisticated **attention mechanisms**, such as **multi-head attention**, can help the model to learn complex relationships between tokens and provide more contextually accurate responses.

### **5.2 Future Directions**

#### **1. Transformer-Based Models**
- Transitioning to **Transformer** models, which excel at parallelization, could help improve both the accuracy and the efficiency of response generation. Transformers have been proven to outperform RNNs in many sequence-to-sequence tasks.

#### **2. Intent Classification**
- Integrating an **intent classification** layer into the chatbot would allow it to understand the purpose behind each query. This will help to route the user's request to the appropriate response model or logic.

#### **3. Continuous Model Improvement**
- **Online Learning**: Implement an online learning pipeline to continuously improve the chatbot based on new data gathered from user interactions.
- **Feedback Loop**: Users could provide feedback on responses, which would then be used for retraining the model to make it more robust and contextually aware.

#### **4. Dataset Augmentation**
- The training dataset can be augmented with synthetic data or by including interactions collected from other customer service platforms. This augmentation would allow the model to become more robust to different types of customer inquiries.

---

## **6. Project Structure**

Here is an overview of the project structure:

```
NLP-Driven-Rasa-Assiste-ChatBot360/
│
├── .venv/                            # Python virtual environment
├── data/                             # Raw and processed datasets
│   └── customer_support.csv          # Dataset file
├── logs/                             # Logs for training runs and evaluations
├── models/                           # Model checkpoints and saved models
│   ├── decoder_with_attention.pth
│   ├── embedding_matrix.pth
│   ├── encoder.pth
│   ├── saved_classes.pkl
│   ├── seq2seq_attention_best_model.pth
│   └── word2idx.pkl
├── notebooks/                        # Jupyter notebooks for data exploration and training
│   ├── data_preprocessing.ipynb
│   ├── model_evaluation.ipynb
│   ├── model_fine_tuning.ipynb
│   ├── model_loading_and_testing.ipynb
│   └── model_training.ipynb
├── screenshots/                      # Screenshots for documentation and visualization
│   └── web-app-screenshot.png
├── utils/                            # Utility modules for models
│   ├── decoder.py
│   ├── encoder.py
│   └── seq2seq.py
├── webapp/                           # Web application files
│   ├── app.py                        # Flask app for chatbot
│   └── templates/
│       └── index.html                # HTML for chatbot UI
├── glove.6B.50d.txt                  # Pre-trained GloVe word vectors
├── glove.6B.100d.txt
├── glove.6B.200d.txt
├── glove.6B.300d.txt
├── LICENSE                           # Project license
├── Notes.txt                         # Miscellaneous notes and information
├── README.md                         # Project description and instructions
└── requirements.txt                  # List of dependencies
```

---

## **7. Getting Started**

### **7.1 Installation**

To set up and run the chatbot locally, follow these steps:

**Step 1: Clone the Repository**
```sh
git clone https://github.com/your-username/NLP-Driven-Rasa-Assiste-ChatBot360.git
cd NLP-Driven-Rasa-Assiste-ChatBot360
```

**Step 2: Create and Activate Virtual Environment**
```sh
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

**Step 3: Install Dependencies**
Install the required packages listed in the `requirements.txt` file:
```sh
pip install -r requirements.txt
```

**Step 4: Download Dataset**
Download the **Customer Support on Twitter** dataset from Kaggle and place it in the `data/` folder.

### **7.2 Running the Application**

**1. Train the Model**
Run the training script to train the Seq2Seq model:
```sh
python train.py
```

**2. Launch the Web Application**
To interact with the chatbot through a web interface, launch the Flask app:
```sh
cd webapp
python app.py
```
- Open your browser and go to `http://127.0.0.1:5000/` to start chatting with the bot.

**3. Use the Console Application**
To test the chatbot via a console-based interface:
```sh
python console_chat.py
```

---

## **8. Conclusion**

The **NLP-Driven Rasa-Assiste ChatBot360** is a versatile Seq2Seq-based chatbot designed for customer support. Through its Encoder-Decoder architecture, pre-trained word embeddings, and attention mechanisms, it aims to provide coherent and contextually relevant responses to common customer queries.

The project faces challenges such as computational limitations and the need for continuous improvement, but it has significant potential for further development. Future iterations will focus on advanced Transformer-based models, hyperparameter tuning, and integrating intent classification to make the chatbot more robust and capable.

---

## **9. Future Enhancements**

- **Intent Recognition**: Add an intent recognition mechanism to better classify and respond to various types of user queries.
- **Sentiment Analysis**: Integrate sentiment analysis to adjust responses according to the user's emotional state.
- **More Powerful Models**: Upgrade to Transformer-based models like BERT or GPT to enhance performance.
- **Robust Evaluation Metrics**: Implement more diverse evaluation metrics beyond BLEU and METEOR, such as **ROUGE** or human evaluations.

Feel free to **fork** the project, contribute enhancements, and use it for building more advanced customer support solutions.
