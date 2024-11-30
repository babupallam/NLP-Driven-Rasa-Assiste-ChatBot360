# %% Section 1: Import Libraries and Load Utilities from Files

import os
import torch
import nltk
import pickle
from flask import Flask, render_template, request, session, redirect, url_for
import importlib.util

# Ensure nltk resources are available for tokenization
nltk.download('punkt')

# Ensure reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1: Set the root directory correctly
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

# Debug: Print current working directory to confirm it's set correctly
print(f"Current working directory set to: {project_root}")

# Define the utils path relative to the current directory
utils_path = os.path.join(project_root, "..", "utils")

# Debug: Print the utils path
print(f"Looking for 'utils' folder at: {utils_path}")

# Function to load a module by its file path
def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load encoder.py
encoder_path = os.path.join(utils_path, "encoder.py")
print(f"Looking for 'encoder.py' at: {encoder_path}")
try:
    encoder_module = load_module("encoder", encoder_path)
    Encoder = encoder_module.Encoder
    print(f"Encoder loaded successfully from {encoder_path}")
except FileNotFoundError as e:
    print(f"Error: {e}. The encoder file path is incorrect. Please verify the file exists.")
    raise

# Load decoder.py
decoder_path = os.path.join(utils_path, "decoder.py")
try:
    decoder_module = load_module("decoder", decoder_path)
    DecoderWithAttention = decoder_module.DecoderWithAttention
    print(f"Decoder loaded successfully from {decoder_path}")
except FileNotFoundError as e:
    print(f"Error: {e}. The decoder file path is incorrect.")
    raise

# Load seq2seq.py
seq2seq_path = os.path.join(utils_path, "seq2seq.py")
try:
    seq2seq_module = load_module("seq2seq", seq2seq_path)
    Seq2SeqWithAttention = seq2seq_module.Seq2SeqWithAttention
    print(f"Seq2Seq model loaded successfully from {seq2seq_path}")
except FileNotFoundError as e:
    print(f"Error: {e}. The Seq2Seq file path is incorrect.")
    raise

# %% Section 2: Load Saved Components

# Paths to saved models and data
models_path = os.path.join(project_root, "..", "models")
encoder_weights_path = os.path.join(models_path, "encoder.pth")
decoder_weights_path = os.path.join(models_path, "decoder_with_attention.pth")
seq2seq_weights_path = os.path.join(models_path, "seq2seq_attention_best_model.pth")
embedding_matrix_path = os.path.join(models_path, "embedding_matrix.pth")
word2idx_path = os.path.join(models_path, "word2idx.pkl")

# Load word2idx from word2idx.pkl
try:
    with open(word2idx_path, 'rb') as f:
        word2idx = pickle.load(f)
    print("word2idx loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading word2idx: {e}. Make sure the file exists at {word2idx_path}.")
    raise

# Load embedding matrix
try:
    embedding_matrix = torch.load(embedding_matrix_path, map_location=device)
    print("Embedding matrix loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading embedding matrix: {e}. Make sure the file exists at {embedding_matrix_path}.")
    raise

# Create idx2word dictionary from word2idx for converting token IDs back to words
idx2word = {idx: word for word, idx in word2idx.items()}

# Hyperparameters
input_size = len(word2idx)
output_size = len(word2idx)
hidden_size = 256  # Adjusted according to training

# Initialize the models
encoder = Encoder(input_size, embedding_matrix, hidden_size).to(device)
decoder = DecoderWithAttention(output_size, embedding_matrix, hidden_size).to(device)
model = Seq2SeqWithAttention(encoder, decoder, device).to(device)

# Load pre-trained weights
try:
    encoder.load_state_dict(torch.load(encoder_weights_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_weights_path, map_location=device))
    model.load_state_dict(torch.load(seq2seq_weights_path, map_location=device))
    print("Model weights loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model weights: {e}. Make sure the model files exist at {models_path}.")
    raise

encoder.eval()
decoder.eval()
model.eval()
print("Models loaded successfully.")

# %% Section 3: Define Flask App for Chatbot

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed to use sessions

# Route for the homepage
@app.route('/')
def home():
    # Initialize chat history in session
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('index.html', chat_history=session['chat_history'])

# Route to handle user input and generate chatbot response
@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    if 'chat_history' not in session:
        session['chat_history'] = []

    # Prepare input and generate response
    source_input = prepare_input_text(user_input, word2idx)
    response = generate_response(model, source_input, word2idx, idx2word)

    # Append user input and bot response to chat history
    session['chat_history'].append({'user': user_input, 'bot': response})
    session.modified = True  # Notify Flask that session has been updated

    return redirect(url_for('home'))

# Function to clean and tokenize input text for use by the model
def prepare_input_text(input_text, word2idx, max_len=20):
    tokens = input_text.lower().split()  # Simple tokenization (splitting by space)
    token_ids = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]
    token_ids = token_ids[:max_len]  # Truncate if longer than max_len
    token_ids += [word2idx['<PAD>']] * (max_len - len(token_ids))
    return torch.tensor([token_ids], dtype=torch.long).to(device)

# Function to generate response from the model
def generate_response(model, source_input, word2idx, idx2word):
    model.eval()
    encoder_outputs, hidden, cell = model.encoder(source_input)
    input_token = torch.tensor([word2idx['<SOS>']], dtype=torch.long).to(device)
    output_tokens = []

    max_output_length = 20
    for _ in range(max_output_length):
        output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
        top1 = output.argmax(1)
        output_tokens.append(top1.item())

        if top1.item() == word2idx['<EOS>']:
            break

        input_token = top1

    response_sentence = [idx2word[token] for token in output_tokens if token not in [word2idx['<PAD>'], word2idx['<EOS>']]]
    return ' '.join(response_sentence)

# Run the app
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
