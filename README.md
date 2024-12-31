# Emotion-Classification-Thesis 
## Introduction 🎭
Emotions in text, particularly in tweets related to the 2024 elections, reflect the public's feelings about specific issues or candidates. This project focuses on classifying five primary emotions: 
- anger 😡
- sadness 😢
- joy ☺️
- love 🥰
- fear 😱
  
Analyzing these emotions is crucial for understanding collective public sentiment, identifying sensitive issues, and providing valuable insights to support data-driven decision-making.

## Objectives 🎯
This project aims to develop the best emotion classification model for tweets related to the Indonesia 2024 elections by leveraging deep learning models such as IndoBERT, RoBERTa, and DeBERTa, implemented using the Torch framework. 
To achieve the best model, the project compares three data preprocessing approaches:
1. Traditional Method 
2. GPT-based Method
3. Combination of GPT and Traditional Methods
   
Additionally, manual data labeling is conducted using a voting method. This approach was chosen because initial experiments revealed inaccuracies between some labels and their actual emotions. These inaccurate labels were further analyzed and corrected based on human judgment to better align with their emotional context.

The model performance is evaluated using accuracy and weighted average F1-score metrics.

## Data Preprocessing Methods ⚙️⏳
### Traditional Method 🛠️

In this method, preprocessing is performed through the following steps:

1. **Text Normalization**:
   - Convert all text to lowercase.
   - Remove irrelevant elements such as numbers, symbols, URLs, usernames, and tags.

2. **Slang Normalization**:
   - Normalize slang words using five slang dictionaries collected from previous research and created by the authors:
     - **Kamus Singkatan Meisaputri** [1](#reference-1)
     - **Kamus Colloquial Indonesian** [2](#reference-2)
     - **Kamus Alay** [3](#reference-3)
     - **Kamus Gabungan Louis Owen** [4](#reference-4)
     - **Custom Slang Dictionary**: Created by the authors using slang words from the training dataset, refined with proper word suggestions from the Indonesian Dictionary (*Kamus Besar Bahasa Indonesia*, KBBI).

These steps ensure that the text is cleaned and standardized, providing a solid foundation for emotion classification.

### GPT-Based Method 🤖
This method uses the GPT-4.0 API for text cleaning, ensuring consistent and clean text while preserving context for emotion classification, with the following parameters:

- Model: GPT-4.0
- Token Limit: 8192
- Max Tokens: 2500
  
The preprocessing prompt used was:
"Tolong bersihkan typo dan singkatan dalam tweet ini tanpa mengubah konteks kalimat! Sesuaikan pemilihan kata sesuai label, dan gabungkan kata terpisah agar lebih terbaca."

Example:

Input: "gmna kabar? smga sehat, btw km adlh sahabat terbaik <3"
Output: "Bagaimana kabar? Semoga sehat, by the way, kamu adalah sahabat terbaik ❤️"
This method 

### Combination of GPT and Traditional Method 🤖 + 📑 
This method combines the **GPT-based ** approach and **the traditional** preprocessing method. The process is carried out in two steps:
1. GPT-Based Preprocessing (Method 2)
The dataset is initially processed using the steps outlined in the GPT-4 method. This step ensures consistent and clean text while preserving context for emotion classification.
2. Traditional Preprocessing (Method 1)
The output from GPT-4 is then further refined using the following traditional preprocessing steps:
    - Converting all text to lowercase: Standardizes the text to improve consistency.
    - Removing irrelevant elements: Numbers, symbols, URLs, usernames, and tags are eliminated to avoid noise in the data.
    - Normalizing slang words: Slang terms are mapped to their formal equivalents using five slang dictionaries collected from previous research and created by the authors.

## Deep Learning Models 🧠💻
Each dataset that has undergone the preprocessing stage is then used to train several deep learning models, including:
- IndoBERT for dataset in Indonesian
- RoBERTa for dataset in English
- DeBERTa for dataset in English
  
For RoBERTa and DeBERTa, the dataset is translated 🌐🔠 into English since the pre-trained models are based on English corpora. During the translation process, experiments are conducted using three different translation tools:
- Deep Translator
- Google Translator
- TextBlob
The goal of these experiments is to determine which translation tool produces the best quality translations. 

### Model Experimentaion ⚙️📊 
Several parameters were tested to optimize model performance. However, the number of neurons remained the same with the architecture of the pre-trained model used. The parameters used including:
- Pre-trained model (base/large)
- Number of epochs
- Batch size
- Learning rate
- Optimizer

After conducting a series of experiments, the best model was obtained using IndoBERT Large Phase 2 and the most effective preprocessing method, which was the combination of GPT & Traditional approaches.
#### 🏆📈 Best model configuration details :
- Data split: 70% training, 15% validation, 15% testing
- Number of epochs: 5
- Batch size: 32
- Learning rate: 5e-6
- Optimizer: Adam
This model achieved an **accuracy** and** weighted average F1-score** of **86%**.



## Best Model Implementation 🏆 🖥️ 
After obtaining the best model, **IndoBERT Large Phase 2** with the **Combination of GPT & Traditional** for preprocessing, which achieved an** accuracy and weighted average F1-score** of **86%**, the model was implemented on tweets related to the Indonesian 2024 elections.
The 2024 election🗳️  tweets were collected through a scraping process using tweet-harvest. This process generated a sample set of tweets relevant to the 2024 election topic, which was then used as input for the emotion classification model.
Subsequently, the model was integrated into a **Streamlit-based application**, allowing users to directly experiment with the emotion classification model. 

The application features a simple interface that enables users to:
1. View data visualizations on the Indonesian 2024 election tweet samples on the dashboard page, such as:
    - Bar chart 📊 and donut chart 🍩 showing the emotion distribution for each presidential candidate
    - Word Clouds ☁️  for visual representation of keywords  🔑 
    - Emotion trend lines 📉 over time 📅 
    - Last 10 tweets 🐤 and its emotion 🎭
2. Perform real-time ⚡⏱️ emotion classification for a sentence on the emotion classification page, allowing users to test the model's ability to classify emotions directly from the interface.


# Folder Structure
## Application
This folder contains the application code.

## Dataset
This folder contains the datasets used for training and testing the model. It includes data in both Indonesian and English. The data is already split into three sets with the following proportions:
- 70% for training
- 15% for validation
- 15% for testing

## Dictionary
This folder contains the slang word dictionaries created by the authors and a combined file from five slang dictionaries used in the preprocessing steps.

## Model
This folder stores the trained best-performing model file with a .pth extension. 

## Contributors
Belinda Mutiara
Florencia
Gabrielle Felicia Ariyanto

References
<a id="reference-1">[1] Saputri, M. S., Mahendra, R., & Adriani, M. (2018). Emotion Classification on Indonesian Twitter Dataset. Proceedings of the 2018 International Conference on Asian Language Processing, IALP 2018, 90–95. Institute of Electrical and Electronics Engineers Inc. https://doi.org/10.1109/IALP.2018.8629262
<a id="reference-2">[2] Salsabila, N. A., Winatmoko, Y. A., Septiandri, A. A., & Jamal, A. (2018). Colloquial Indonesian Lexicon. 327.
<a id="reference-3">[3] Ibrohim, M. O., & Budi, I. (2018). A Dataset and Preliminaries Study for Abusive Language Detection in Indonesian Social Media. Procedia Computer Science, 135, 222–229. Elsevier B.V. https://doi.org/10.1016/j.procs.2018.08.169
<a id="reference-4">[4] Purwitasari, D., Putra, C. B. P., & Raharjo, A. B. (2023). A stance dataset with aspect-based sentiment information from Indonesian COVID-19 vaccination-related tweets. Data in Brief, 47. https://doi.org/10.1016/j.dib.2023.108951





