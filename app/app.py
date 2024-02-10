
import torch
from model import Encoder, Decoder, Seq2SeqTransformer, translate_sentence
from model import EncoderLayer , DecoderLayer , AttentionLayer, PositionwiseFeedforwardLayer
from flask import Flask, render_template, request

import json
import os

# Configuration for not using CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = ""

app = Flask(__name__)

# Initialize device
device = torch.device('cpu')  # Adjust if using GPU

# Load vocab and model configuration
vocab_transform = torch.load('./Models/vocab.pt')
with open('./Models/model_config.json', 'r') as config_file:
    loaded_config = json.load(config_file)

params, state = torch.load('./Models/Additive_Attn_Seq2SeqTransformer.pt', map_location=device)
model = Seq2SeqTransformer(**params, device=device)
model.load_state_dict(state)
model.eval()  # Set the model to evaluation mode

@app.route('/')
@app.route('/translate', methods=['GET', 'POST'])
def translate():
    src_sentence = ''
    translation = ''
    if request.method == 'POST':
        src_sentence = request.form.get('src_sentence', '')
        if src_sentence:
            translation = translate_sentence(src_sentence, vocab_transform['en'], vocab_transform['bn'], model, device)
    
    return render_template('index.html', src_sentence=src_sentence, translation=translation)

if __name__ == '__main__':
    app.run(debug=True)
