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
     - **Kamus Singkatan Meisaputri** [1]
     - **Kamus Colloquial Indonesian** [2]
     - **Kamus Alay** [3]
     - **Kamus Gabungan Louis Owen** [4]
     - **Custom Slang Dictionary**: Created by the authors using slang words from the training dataset, refined with proper word suggestions from the Indonesian Dictionary (*Kamus Besar Bahasa Indonesia*, KBBI).

These steps ensure that the text is cleaned and standardized, providing a solid foundation for emotion classification.



