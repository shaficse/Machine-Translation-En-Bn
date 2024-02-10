# A3: Machine Translation

## Table of Contents
- [Overview](#overview)
- [Task 1: Preparation and Training](#task-1-preparation-and-training)
  - [Foundational Research Papers](#11-foundational-research-papers)
  - [Data Description](#12-data-description)
  - [Technical Components](#13-technical-components)
  - [Getting Started](#14-getting-started)
- [Task 2: Model Comparison and Analysis](#task-2-model-comparison-and-analysis)
- [Task 3: Text Generation - Web Application Development](#task-3-text-generation---web-application-development)
- [Contributing & Support](#contributing--support)
- [License](#license)
- [Acknowledgments](#acknowledgments)


## Overview
In this project, we delved into the domain of neural machine translation, with a specific focus on translating between English and Bangla languages. Our objective was to explore various attention mechanisms within the Transformer architecture to enhance translation quality and efficiency.

## Task 1: Preparation and Training
This task involves setting up the project, preparing the data, and training the LSTM language model.

### 1.1 Foundational Research Papers
Our project is inspired by the seminal paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762), which introduced the Transformer architecture. Here's a concise summary:
  <img src="figures/transformer1.png">
- **Concept**: Transformer replaces recurrent layers with self-attention mechanisms for capturing global dependencies in sequences.
- **Key Features**: 
  - Self-Attention: Enables capturing long-range dependencies efficiently.
  - Positional Encoding: Incorporates token order information without recurrent connections.
  - Encoder-Decoder Architecture: Consists of encoder and decoder layers for input processing and output generation.
- **Advantages**: 
  - Parallelization: Allows for efficient parallel processing of sequences.
  - Long-Range Dependencies: Better handling of long-range dependencies compared to recurrent architectures.
  - Scalability: Highly scalable and effective on large datasets with minimal modifications.
- **Impact**: 
  - Transformed NLP: Became the standard architecture for various NLP tasks, including machine translation.
  - Legacy: Paved the way for subsequent advancements in NLP research and applications.

Our project builds upon these foundational concepts to explore English-to-Bangla machine translation.


### 1.2 Data Description

This dataset specification outlines the use of a machine translation dataset from the Hugging Face datasets library.

#### Language Selection
- **Source Language (`SRC_LANGUAGE`)**: English (`en`) - the language we translate from.
- **Target Language (`TRG_LANGUAGE`)**: Bengali (`bn`) - the language we translate into.

#### Dataset Selection
- **Dataset Used**: OPUS-100 Corpus, focusing on the "bn-en" language pair, available on Hugging Face's dataset repository.
    - Chosen for its high-quality, parallel sentences, essential for effective machine translation model training.

For more information and to access the dataset, visit the [Hugging Face Datasets Library](https://huggingface.co/datasets/opus100).

---


### 1.3 Technical Components

#### Dataset Sampling and Preprocessing Overview

- Initialize a random number generator with a fixed seed for reproducibility.

- Randomly select 3000 examples from the `train` split for a balanced approach to computational efficiency and dataset diversity.

    - Splits: Train (2100 samples), Validation (450 samples), Test (450 samples)

#### Tokenizing with spaCy Models

- Use spaCy to set up tokenizers for both source and target languages, enhancing preprocessing with its advanced tokenization capabilities.

- **Setup Details**:

    - The target language (Bangla) tokenizer uses `'xx_ent_wiki_sm'` for effective across various languages, including Bangla.
    - The source language (English) tokenizer relies on `'en_core_web_sm'`, optimized for web text with efficient tokenization and tagging.

To download spaCy models, use the following commands in our terminal:

```bash
# For the multi-language entity recognition model
python -m spacy download xx_ent_wiki_sm

# For the small English model optimized for web content
python -m spacy download en_core_web_sm
```




#### Technical Components of the Transformer Model

This section provides an overview of the key components within the Transformer model architecture, highlighting the Encoder, Decoder, and their supporting layers. The Transformer model is renowned for its effectiveness in handling sequence-to-sequence tasks, such as machine translation and text summarization.

**EncoderLayer Class**
- A single layer in the Transformer model encoder, crucial for transforming input sequences into higher-level representations.
- **Key Components** include the Self-Attention Mechanism, Positionwise Feedforward Network, and Normalization and Dropout, each contributing to the model's ability to capture context and dependencies within the input.

**Encoder Class**
- Serves as the core of the Transformer's encoder, converting input sequences into rich representations that encapsulate semantic and contextual relationships.
- Incorporates Token and Positional Embedding, multiple Encoder Layers, and Dropout, sequentially refining the input representation.

**AttentionLayer Class**
- Facilitates selective focus on different parts of the input sequence, employing General, Multiplicative, and Additive attention mechanisms.
- **Key Features**:
  - **Configurability**: Supports various attention types, enabling flexibility to tackle diverse data modeling challenges.
  - **Equations for Attention Mechanisms**:

    1. **General Attention**: 
    \(e_i = s^T h_i \in \mathbb{R}\) where \(d_1 = d_2\).

    2. **Multiplicative Attention**:
     \(e_i = s^T W h_i \in \mathbb{R}\) where \(W \in \mathbb{R}^{d_2 \times d_1}\).

    3. **Additive Attention**: 
    \(e_i = v^T \tanh(W_1 h_i + W_2 s) \in \mathbb{R}\).

  - These equations delineate how each attention type calculates the energy score \(e_i\), which is pivotal for determining the focus areas of the model's input sequence.

**PositionwiseFeedforwardLayer**
- Processes inputs at each sequence position independently, crucial for learning complex patterns and adding depth to the Transformer's encoding capabilities.

**DecoderLayer**
- A key component of the Transformer's decoder, designed to generate the output sequence informed by the context of the encoded source sequence and the target sequence itself.
- Utilizes Self-Attention, Encoder-Decoder Attention, and Positionwise Feedforward Networks to refine the target sequence representation.

**Decoder Class**
- Central to the decoding process, this class generates the output sequence from the encoded source sequence, applying Token and Positional Embeddings, Decoder Layers, and an Output Linear Layer to produce next-token predictions.

**Seq2SeqTransformer Class**
- Integrates the Encoder and Decoder to form a complete sequence-to-sequence model, suited for various NLP tasks.
- Employs Source and Target Masks to manage attention focus and ensure the integrity of sequence processing.

The Transformer model architecture leverages these components to efficiently process and generate sequences, achieving state-of-the-art performance on numerous NLP tasks. Its parallelizable training and adaptability across different applications make it a cornerstone of modern natural language processing.



### Training Process
- **Dataset Loading**: Utilizes the OPUS-100 Corpus for training, validating, and testing the translation model.
- **Hyperparameter Tuning**: Adjusts learning rate, batch size, and other parameters to optimize model performance.
- **Model Evaluation**: Measures translation accuracy, computational efficiency, and other relevant metrics to compare attention mechanisms.
- **Future Improvement Strategies**: Identifies areas for potential enhancement, such as data augmentation, hyperparameter tuning, and advanced model architectures.

### Implementation Details
- **Framework**: Implemented using PyTorch, a popular deep learning framework.
- **Inspiration**: Draws inspiration from the Transformer model described in the seminal paper "Attention is All You Need" (Vaswani et al., 2017).
- **Credit to Developers**: Acknowledges the contributions of the PyTorch development team and the authors of the Transformer paper for their foundational work in deep learning research.

#### Training
- **Hyperparameter Adjustment**: Configures model hyperparameters such as the number of layers, embedding dimension, hidden dimension, dropout rate, and learning rate.
- **Gradient Clipping**: Implements gradient clipping to prevent exploding gradients during backpropagation, maintaining model stability.
- **Learning Rate Schedulers**: Uses a learning rate scheduler (`ReduceLROnPlateau`) to adjust the learning rate based on validation loss, aiding in model convergence and avoiding overfitting.
- **Batch Processing**: Employs batch processing for efficient model training, reshaping the data into batches of a specified size.
- **Loss Function**: Utilizes Cross-Entropy Loss, suitable for multi-class classification tasks, to compute the loss between predictions and targets.
- **Optimizer**: Adopts the Adam optimizer for model training, updating model parameters based on computed gradients.
- **Evaluation Metrics**: Employs perplexity as the primary metric for model evaluation, providing insight into the model's performance in predicting the next word in a sequence.


### 1.4 Getting Started

To get started with the LSTM language model project, follow these steps to set up the environment, prepare the data, and initiate the training process:

1. **Environment Setup**:
    - Ensure you have Python 3.6+ installed.
    - Create a virtual environment to manage dependencies:
      ```bash
      python -m venv venv
      ```
    - Activate the virtual environment:
      - On Windows: `venv\Scripts\activate`
      - On macOS/Linux: `source venv/bin/activate`
    - Install the required dependencies:
      ```bash
      pip install -r requirements.txt
      ```


## Task 2: Model Comparison and Analysis

### Model Performance Report

This report provides an analysis of the LSTM language model's training performance over 50 epochs, focusing primarily on minimizing Train and Valid Perplexity, and offering an evaluation based on Test Perplexity.

#### Results
- **Total Training Time:** 48 minutes and 12 seconds(On T4 GPU Linux Machine with 16GB memory ).
- **Optimal Epoch:** 30
  - **Train Perplexity:** 45.028
  - **Valid Perplexity:** 88.476
  - **Test Perplexity:** 106.331

#### Analysis
The model demonstrated consistent improvement throughout the training epochs, with Epoch 30 yielding the most balanced performance. The Test Perplexity, while higher than the Validation Perplexity, remains within a reasonable range, suggesting good generalization of the model to unseen data. `However, the noticeable gap between Test and Validation Perplexity indicates potential areas for further model refinement.`

#### Recommendations for Further Improvement
1. **Early Stopping:** Implement an early stopping mechanism to halt training when the Validation Perplexity no longer shows significant improvement. This approach can prevent overfitting and reduce unnecessary computational overhead.
2. **Hyperparameter Tuning:** Conduct thorough experimentation with various sets of hyperparameters to optimize the model's performance further. Consider exploring different learning rates, batch sizes, and LSTM configurations.
3. **Extended Dataset:** Enrich the training dataset with more diverse or comprehensive data sources. This expansion can enhance the model's ability to learn and generalize across a wider array of textual contexts.
4. **Advanced Architectures:** Investigate the integration of more sophisticated model architectures, such as Transformer-based models or attention mechanisms, to potentially capture the dataset's complexities more effectively and improve predictive performance.

This analysis and the subsequent recommendations aim to guide further development and optimization of the LSTM language model, promoting advancements in performance and applicability.


***If anyone wants to use our trained model, please download from here [https://drive.google.com/file/d/1DhDYEpJT4EpHx7bfmuezCMCGKV--kgmx/view?usp=sharing](https://drive.google.com/file/d/1DhDYEpJT4EpHx7bfmuezCMCGKV--kgmx/view?usp=sharing)
& rest of the configuration files are inside the app/models directory***

## Task 3: Text Generation - Web Application Development

This task focuses on deploying the trained LSTM language model as an interactive web application using Flask. The application allows users to input custom prompts and generate text based on those prompts, showcasing the model's text generation capabilities in a user-friendly format.

### Web Application Features
- **Interactive Interface**: Users can input a prompt and adjust parameters such as **maximum sequence length** and **temperature** for text generation.
- **Real-time Text Generation**: The model generates text in real-time, providing immediate feedback based on the user's input.
- **Parameter Tuning**: Users can experiment with different temperatures to influence the creativity and coherence of the generated text.

### Getting Started with the Web Application
1. **Set Up the Flask Environment**:
   - Navigate to the Flask application directory.
   - Ensure all dependencies are installed: `pip install -r requirements.txt`.

2. **Start the Flask Server**:
   - Run the Flask server with  
        ```sh
        python app.py
        ```
   - The server typically starts at [http://127.0.0.1:5000](http://127.0.0.1:5000).


   <img src="figures/a2-app-1.png">

   <img src="figures/a2-app-2.png">

   ***For Live Demo from Huggingface Space [https://huggingface.co/spaces/shaficse/a2-text-gen](https://huggingface.co/spaces/shaficse/a2-text-gen)*** 
3. **Interact with the Application**:
   - Access the web application through the provided URL.
   - Input your prompt and adjust generation parameters as desired.
   - Click 'Generate' to view the model's text output.

### Application Architecture
- **Flask Backend**: Handles requests, interacts with the LSTM model, and serves the generated text.
- **Frontend Interface**: Provides an intuitive UI for users to interact with the model, built using HTML, CSS, and JavaScript.
- **Model Integration**: Seamlessly integrates the trained LSTM model to perform text generation based on user input.

This development phase brings the LSTM language model to a wider audience, allowing for interactive engagement and demonstration of the model's text generation prowess through a web-based platform.


## Contributing & Support
Contributions are welcome. For issues or questions, please open an issue in the repository.

## License
This project is licensed under the MIT License.[LICENSE](LICENSE)

## Acknowledgments

- **Research Inspiration**: Sincere thanks to the authors of the foundational research paper, "Regularizing and Optimizing LSTM Language Models," which provided crucial insights and methodologies for this project.
- **Resource Contributions**: Special appreciation to [Chaklam Silpasuwanchai](https://github.com/chaklam-silpasuwanchai) for his invaluable contributions. The codebase for this project drew inspiration and guidance from his [Python for Natural Language Processing](https://github.com/chaklam-silpasuwanchai/Python-for-Natural-Language-Processing) repository, serving as a vital resource.
- **Dataset Providers**: Heartfelt gratitude to the curators and maintainers of the Harry Potter dataset, whose efforts in compiling and structuring the data have been instrumental for the training and evaluation of the LSTM language models in this project.

These acknowledgments reflect the collaborative spirit and valuable contributions that have significantly enriched this project, and we extend our sincere gratitude to everyone involved.
