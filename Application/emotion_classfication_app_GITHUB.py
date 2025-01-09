import pandas as pd
import streamlit as st
import numpy as np
import re
import altair as alt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
import nltk
from nltk.corpus import stopwords
import requests
from io import StringIO
from io import BytesIO
import gdown
import openai
import tiktoken
from transformers import BertTokenizer, BertConfig
from torch.utils.data import DataLoader, Dataset
from st_flexible_callout_elements import flexible_callout

nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(layout="wide") 

file_id = "1s-PDTh55J3bDCEOmTr7CkW-LyG4fPz4X"
download_url = f"https://drive.google.com/uc?id={file_id}"
response = requests.get(download_url)
response.raise_for_status() 

csv_content = StringIO(response.text)
df = pd.read_csv(csv_content)

file_id = "1JmEpu4SluOgfmkGo5QXdv5npLN8CxfQl"
download_url = f"https://drive.google.com/uc?id={file_id}"
response = requests.get(download_url)
response.raise_for_status()  
image = BytesIO(response.content) 
st.image(image, use_column_width=True)

tab1, tab2 = st.tabs(['📊 Dasbor', '😶‍🌫 Klasifikasi Emosi'])

with tab1:
    options = st.multiselect(
            "Filter Kandidat",
            df['candidates'].unique(),
            st.session_state['candidates'] if 'candidates' in st.session_state else [], 
            key="candidates",
            placeholder="Pilih pasangan (dapat lebih dari 1)"
        )
    row1 = st.columns(3)
    row2 = st.columns(2)
    row3 = st.columns(1)

    def filterCandidate(source):
        if 'candidates' in st.session_state and len(st.session_state['candidates']) > 0:
            source = source[source['candidates'].isin(st.session_state['candidates'])]
        return source

    def count_freq(source):
        source['freq'] = source['label'].map(source['label'].value_counts())
        return source

    def getDonutChart(source, tile):
        source = filterCandidate(source)
        source = count_freq(source)

        unique_candidates = source["candidates"].unique()
        if len(unique_candidates) == 1:
            candidate_name = unique_candidates[0]  
            title = f"Presentase Label ({candidate_name})"
            background_color = 'skyblue' 
        else:
            title = "Presentase Label (Seluruh Pasangan)"
            background_color = '#dcdcdc'
        tile.write(f"""
            <div style="background-color: #f0f8ff; color: black; padding: 10px; border-radius: 0px; font-size: 16px; text-align: center;">
                {title}
            </div>
            <br>
        """, unsafe_allow_html=True)

        chart = alt.Chart(source).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="freq", type="quantitative"),
            color=alt.Color(field="label", type="nominal", legend=None),  
        ).properties(
            width=250, 
            height=250 
        )

        tile.altair_chart(chart, theme="streamlit", use_container_width=True)


    def bar_chart(source, tile):
        source = filterCandidate(source)
        source = count_freq(source)
        # tile.write("Label Distribution")
        unique_candidates = source["candidates"].unique()
        if len(unique_candidates) == 1:
            candidate_name = unique_candidates[0]  
            title = f"Distribusi Label ({candidate_name}))"
            background_color = 'skyblue' 
        else:
            title = "Distribusi Label (Seluruh Pasangan)"
            background_color = '#dcdcdc'  

        tile.write(f"""
            <div style="background-color: #f0f8ff; color: black; padding: 10px; border-radius: 0px; font-size: 16px; width: 205%; text-align: center;">
                {title}
            </div>
            <br>
        """, unsafe_allow_html=True)
        return tile.bar_chart(source, x="label", y="freq", color="label", stack=False, use_container_width=False, width = 700)
    
    # Function to download and load stopwords
    def load_stopwords():
        # Download the stopwords if not already done
        nltk.download('stopwords')
        
        # Load the stopwords for Indonesian
        stopwords_indonesian = set(stopwords.words('indonesian'))
        
        # Add custom stopwords if needed
        custom_stopwords = {'http', 'https', 't.co', 'RT', 'amp', 'like'}
        stopwords_indonesian.update(custom_stopwords)
        
        stopwords_indonesian = {word.lower() for word in stopwords_indonesian}
        return stopwords_indonesian

    # Load stopwords once, making them accessible globally
    stopwords_indonesian = load_stopwords()

    def normalize_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Normalize Unicode (if needed)
        return text

    # Function to remove stopwords from text
    def remove_stopwords(text):
        text = normalize_text(text)  # Normalize text
    # Remove stopwords
        filtered_words = [word for word in text.split() if word not in stopwords_indonesian]
        return " ".join(filtered_words)

    # Function to generate wordcloud
    def word_cloud(source, title):
        # Filter based on candidates (if needed)
        source = filterCandidate(source)
        source = source.drop_duplicates(subset=["tweet"])
    
        # Combine all tweets into a single string
        source_text = " ".join(source["tweet"].astype(str))
        
        # Remove stopwords
        cleaned_text = remove_stopwords(source_text)
        
        wordcloud = WordCloud(
            stopwords=stopwords_indonesian,
            background_color='white',
            colormap='coolwarm',
            width=1000,  
            height=600,  
            # max_words=200,  
            contour_color='black',  
            contour_width=1 
        ).generate(cleaned_text)

        unique_candidates = source["candidates"].unique()

        if len(unique_candidates) == 1:
            candidate_name = unique_candidates[0]  
            title = f"Wordcloud ({candidate_name})"
            background_color = 'skyblue'  
        else:
            title = "Wordcloud (Seluruh Pasangan)"
            background_color = '#dcdcdc' 

        
        st.write(f"""
            <div style="background-color: #f0f8ff; color: black; padding: 10px; border-radius: 0px; font-size: 16px; width: 100%; text-align: center;">
                {title}
            </div>
            <br>
        """, unsafe_allow_html=True)

        # Create WordCloud visualization
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")  # Hide axes
        # ax.set_title(title)  
        st.pyplot(fig)


    def timeline_chart(source, tile):
        source = filterCandidate(source)
        source = source.set_index((pd.to_datetime(source['date'])))
        source = source.groupby(pd.Grouper(freq='D')).value_counts(subset=['label'])
        source = source.to_frame()
        source = source.reset_index(level=['label', 'date'])

        # tile.write("Label Distribution Over Time")
        unique_candidates = st.session_state.get('candidates', [])

        if len(unique_candidates) == 1:
            candidate_name = unique_candidates[0]  # Take the only candidate
            title = f"Distribusi Label Berdasarkan Waktu ({candidate_name})"
            background_color = '#dcdcdc'  # Set background color for a single candidate
        elif len(unique_candidates) > 1:
            title = "Distribusi Label Berdasarkan Waktu (Seluruh Pasangan)"
            background_color = '#dcdcdc'  # Set background color for multiple candidates
        else:
            title = "Distribusi Label Berdasarkan Waktu (Seluruh Pasangan)"
            background_color = '#dcdcdc'  # Default background color for no candidates selected

        # Display the title with the background color
        tile.write(f"""
            <div style="background-color: #f0f8ff; color: black; padding: 10px; border-radius: 0px; font-size: 16px; width: 200%; text-align: center;">
                {title}
            </div>
            <br>
        """, unsafe_allow_html=True)

        chart = alt.Chart(source).mark_line(point=True).encode(
            x=alt.X('date:T', title='Date', axis=alt.Axis(format='%b %d')),
            y='count:Q',
            color=alt.Color('label:N', legend=None),  # Hilangkan legend
        ).properties(
            width=1090,  # Width of the chart
            height=500
        )

        return tile.altair_chart(chart, theme="streamlit", use_container_width=False)


    def display_latest_tweets(source, tile):
        source = filterCandidate(source)
        # if candidate:
            # source = source[source['candidates'] == candidate]
        # Set the max column width to None so tweets are not truncated
        pd.set_option("display.max_colwidth", None)

        # Mengonversi kolom 'date' ke format datetime jika belum
        source['date'] = pd.to_datetime(source['date'])

        # Mengurutkan data berdasarkan tanggal secara menurun
        df_sorted = source.sort_values(by='date', ascending=False)

        # Menampilkan 10 tweet terbaru
        latest_10_tweets = df_sorted.head(10)

        # tile.write("Latest Tweets: Top 10")
        unique_candidates = source["candidates"].unique()

# Determine the title based on the number of unique candidates
        if len(unique_candidates) == 1:
            candidate_name = unique_candidates[0]  # Take the only candidate
            title = f"Tweet Terbaru: Top 10 ({candidate_name})"
            background_color = 'skyblue'  # Set background color for single candidate
        else:
            title = "Tweet Terbaru: Top 10 (Seluruh Pasangan)"
            background_color = '#dcdcdc'  # Set background color for multiple candidates

        # Display the title with the background color
        tile.write(f"""
            <div style="background-color: #f0f8ff; color: black; padding: 10px; border-radius: 0px; font-size: 16px; width: 100%; text-align: center;">
                {title}
            </div>
            <br>
        """, unsafe_allow_html=True)
        # tile.dataframe(latest_10_tweets[['tweet', 'label']].reset_index(drop=True), use_container_width=True, hide_index=True)
        tile.write(latest_10_tweets[['tweet', 'label']].to_html(index=False, justify = 'center'), unsafe_allow_html=True)

        # Mengembalikan DataFrame yang sudah diurutkan
        return source

    def area_chart(source, tile):
        # Pastikan untuk memfilter data terlebih dahulu
        source = filterCandidate(source)  # Asumsi: filterCandidate adalah fungsi yang sudah ada

        # Mengubah kolom 'date' menjadi format datetime dan mengatur indeks
        source['date'] = pd.to_datetime(source['date'])
        source = source.set_index('date')

        # Group by berdasarkan tanggal dengan agregasi count (mengabaikan label)
        source_grouped = source.groupby(pd.Grouper(freq='D')).size().reset_index(name='count')

        # Membuat chart area menggunakan Altair
        chart = alt.Chart(source_grouped).mark_area().encode(
            x=alt.X('date:T', title='Date', axis=alt.Axis(format='%b %d')),  # Format tanggal
            y=alt.Y('count:Q', title='Count')  # Sumbu Y untuk jumlah
        ).properties(
            title='Area Chart of Counts'
        )
        return tile.altair_chart(chart, theme="streamlit", use_container_width=True)

    row1[0] = getDonutChart(df, row1[0].container())
    row1[1] = bar_chart(df, row1[1].container())
    

    row2[0] = timeline_chart(df, row2[0].container())
    row2[1] = word_cloud(df, row2[1].container())

    row3[0] = display_latest_tweets(df, row3[0].container())


with tab2:
    url = "https://drive.google.com/uc?id=1P-7TQJKBrFePmamx6-oVesGe-s4JxBfw"
    model_filename = "IndoBERT_best_model.pth"

    if not os.path.exists(model_filename):
        print(f"{model_filename} tidak ditemukan, mengunduh file...")
        gdown.download(url, model_filename, quiet=False)
    else:
        print(f"{model_filename} sudah ada, tidak perlu mengunduh lagi.")
    os.environ["OPENAI_API_KEY"] = "[# Enter your API Key]"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    label_to_emoji = {
        "anger": "😡",
        "love": "❤️",
        "sadness": "😢",
        "happy": "😊",
        "fear": "😱"
    }
    label_to_color = {
        "anger": "#ff937d",
        "love": "#FFC0CB",
        "sadness": "#A2CAED",
        "happy": "#FFD700",
        "fear": "#B2ABD2"
    }

    def send(
        prompt=None,
        text_data=None,
        chat_model="gpt-4.0",
        model_token_limit=8192,
        max_tokens=2500
    ):

        if not prompt:
            return "Error: Prompt is missing. Please provide a prompt."
        if not text_data:
            return "Error: Text data is missing. Please provide some text data."

        tokenizer = tiktoken.encoding_for_model(chat_model)

        token_integers = tokenizer.encode(text_data)

        chunk_size = max_tokens - len(tokenizer.encode(prompt))
        chunks = [
            token_integers[i : i + chunk_size]
            for i in range(0, len(token_integers), chunk_size)
        ]

        chunks = [tokenizer.decode(chunk) for chunk in chunks]

        responses = []
        messages = [
            {"role": "user", "content": prompt},
            {
                "role": "user",
                "content": "To provide the context for the above prompt, I will send you text in parts. When I am finished, I will tiktoken.get_encoding you 'ALL PARTS SENT'. Do not answer until you have received all the parts.",
            },
        ]

        for chunk in chunks:
            messages.append({"role": "user", "content": chunk})

           
            while (
                sum(len(tokenizer.encode(msg["content"])) for msg in messages)
                > model_token_limit
            ):
                messages.pop(1)  

            response = openai.ChatCompletion.create(model=chat_model, messages=messages)
            chatgpt_response = response.choices[0].message["content"].strip()
            responses.append(chatgpt_response)

        messages.append({"role": "user", "content": "ALL PARTS SENT"})
        response = openai.ChatCompletion.create(model=chat_model, messages=messages)
        final_response = response.choices[0].message["content"].strip()
        responses.append(final_response)

        return responses


    from chatgptmax import send
    # Define the prompt
    prompt_text = """
    Tolong bersihkan typo dan singkatan dalam tweet ini tanpa mengubah konteks kalimat! Tolong kalau ada kata-kata yang terpisah dibuat terbaca (digabungkan).
    """

    
    # Create functions
    def replace_acronyms(tweet, acronym_dict):
        """Replace acronyms with their meaning using acronym dictionary."""
        words = tweet.split()
        new_words = [acronym_dict.get(word, word) for word in words]
        return ' '.join(new_words)

    def replace_slang(text, slang_dict):
        """Replace slang words with their formal counterparts using slang dictionary."""
        words = text.split()
        new_words = [slang_dict.get(word, word) for word in words]
        return ' '.join(new_words)
    
    # Function to remove mentions and URLs
    def remove_mentions_urls(tweet):
        # Remove mentions
        tweet = re.sub(r'@\w+', '', tweet)
        # Remove URLs
        tweet = re.sub(r'http\S+|www\.\S+', '', tweet)
        # Remove extra whitespace
        tweet = re.sub(r'\s+', ' ', tweet).strip()
        return tweet


    # Update the cleantext function to avoid misinterpretation
    def cleantext(text):
        # Remove unwanted non-ASCII characters, but allow diacritics and alphabets
        text = re.sub(r'[^\w\s]', ' ', text)  # Keep alphanumeric characters and spaces
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', ' ', text)
        text = re.sub(r'\www.\S+', ' ', text)
        # Remove mentions
        text = re.sub(r'@[\w]+', ' ', text)
        # Remove excessive punctuation, symbols
        text = re.sub(r'[!$%^&*@#()+|~=`{}\[\]%-:";\'<>?,./]', ' ', text)
        # Remove numbers
        text = re.sub(r'[0-9]+', '', text)
        # Collapse multiple spaces into one
        text = re.sub(r'\s+', ' ', text)
        # Trim leading/trailing spaces
        text = text.strip()
        # Convert to lowercase
        text = text.lower()

        return text
    

    class EmotionDetectionDataset(Dataset):
        # Static constant variable
        LABEL2INDEX = {'sadness': 0, 'anger': 1, 'love': 2, 'fear': 3, 'happy': 4}
        INDEX2LABEL = {0: 'sadness', 1: 'anger', 2: 'love', 3: 'fear', 4: 'happy'}
        NUM_LABELS = 5

        def load_dataset(self, path):
            # Load dataset
            dataset = pd.read_csv(path)
            dataset['label'] = dataset['label'].apply(lambda sen: self.LABEL2INDEX[sen])
            return dataset

        def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
            self.data = self.load_dataset(dataset_path)
            self.tokenizer = tokenizer
            self.no_special_token = no_special_token

        def __getitem__(self, index):
            tweet, label = self.data.loc[index,'tweet'], self.data.loc[index,'label']
            subwords = self.tokenizer.encode(tweet, add_special_tokens=not self.no_special_token)
            return np.array(subwords), np.array(label), tweet

        def __len__(self):
            return len(self.data)

    class EmotionDetectionDataLoader(DataLoader):
        def __init__(self, max_seq_len=512, *args, **kwargs):
            super(EmotionDetectionDataLoader, self).__init__(*args, **kwargs)
            self.collate_fn = self._collate_fn
            self.max_seq_len = max_seq_len

        def _collate_fn(self, batch):
            batch_size = len(batch)
            max_seq_len = max(map(lambda x: len(x[0]), batch))
            max_seq_len = min(self.max_seq_len, max_seq_len)

            subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
            mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
            label_batch = np.full((batch_size, 1), -100, dtype=np.int64)

            seq_list = []
            for i, (subwords, label, raw_seq) in enumerate(batch):
                subwords = subwords[:max_seq_len]
                subword_batch[i,:len(subwords)] = subwords
                mask_batch[i,:len(subwords)] = 1
                label_batch[i] = label

                seq_list.append(raw_seq)

            return subword_batch, mask_batch, label_batch, seq_list
 

    # Input field for the user
    sentence = st.text_input("Masukkan Kalimat (dalam Bahasa Indonesia & minimal terdiri dari 2 kata):")
    sentence = sentence.strip()
    cleaned_tweet = ""

    word_count = len(sentence.split()) if sentence else 0

    if sentence:  # Check if input is not empty
        if word_count < 2:
            st.error("Mohon masukan setidaknya dua kata.")  
    else:
        st.text("Tekan Enter untuk memproses kalimat.")


    if word_count >= 2:
        result = ""
        prediction = ""
        prediction_text = ""
        with st.status("Memprediksi emosi dari kalimat anda...", expanded=True) as status:
            st.write("Membersihkan kalimat...")
            combined_input = f"Tweet: {sentence}\n"
            response = send(prompt=prompt_text, text_data=combined_input)
            cleaned_tweet = response[0].content if response else ""
            cleaned_tweet = cleaned_tweet.removeprefix('Tweet: ')



            # Load slang word dictionary
            df_dict = pd.read_csv("https://raw.githubusercontent.com/meisaputri21/Indonesian-Twitter-Emotion-Dataset/master/kamus_singkatan.csv", names=["acronym"])
            # Preprocess dictionary
            df_dict[["acronym", "meaning"]] = df_dict["acronym"].str.split(";", expand=True)
            # Strip whitespaces
            df_dict["acronym"] = df_dict["acronym"].astype(str).str.strip()
            df_dict["meaning"] = df_dict["meaning"].astype(str).str.strip()

            # Load slang dictionary
            df_slang = pd.read_csv("https://raw.githubusercontent.com/Solveware3/Emotion-Classification-Thesis/refs/heads/main/Dictionary/combined_slang_dict.csv")
            df_slang["Slang"] = df_slang["Slang"].astype(str).str.strip()
            df_slang["Formal"] = df_slang["Formal"].astype(str).str.strip()


            cleaned_tweet = replace_acronyms(cleaned_tweet, df_dict)
            cleaned_tweet = replace_slang(cleaned_tweet, df_slang)

            cleaned_tweet = remove_mentions_urls(cleaned_tweet)

            cleaned_tweet = cleantext(cleaned_tweet)

            cleaned_tweet = re.sub(r'[^a-zA-Z\s]', ' ', cleaned_tweet)

            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
               
                device = torch.device("cpu")
            
            st.write("Memproses kalimat...")
        
            # Load Tokenizer and Config
            tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-large-p2')
            config = BertConfig.from_pretrained('indobenchmark/indobert-large-p2')
            config.num_labels = EmotionDetectionDataset.NUM_LABELS

            w2i, i2w = EmotionDetectionDataset.LABEL2INDEX, EmotionDetectionDataset.INDEX2LABEL


            # Load model
            model = torch.load(model_filename)
            model.eval() 

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)

            # Tokenize input text
            tokens = tokenizer(
                cleaned_tweet,
                padding='max_length',  # Pad to max_length
                max_length=128,        
                truncation=True,
                return_tensors="pt"    # Return PyTorch tensors
            )

            # Extract tokenized data and move to the same device as the model
            subword_batch = tokens['input_ids'].to(device)
            mask_batch = tokens['attention_mask'].to(device)
            token_type_batch = tokens.get('token_type_ids', None)
            if token_type_batch is not None:
                token_type_batch = token_type_batch.to(device)

            # Forward pass to get logits
            with torch.no_grad():  # No need to compute gradients
                logits = model(
                    input_ids=subword_batch,
                    attention_mask=mask_batch,
                    token_type_ids=token_type_batch
                )[0]

            st.write("Memprediksi emosi kalimat...")
           
            probabilities = F.softmax(logits, dim=-1).squeeze()

           
            label_idx = torch.topk(probabilities, k=1, dim=-1)[1].squeeze().item()
            confidence = probabilities[label_idx].item() * 100  # Convert to percentage

            # Get the corresponding label and emoji
            label = i2w[label_idx]
            emoji = label_to_emoji.get(label, "🤔")  

            if cleaned_tweet != "":
                st.write('**Kalimat anda berhasil diproses!**')
            
            prediction_text = f"<b>Prediksi Emosi: {label}{emoji} ({confidence:.3f}%)</b>"
            result = f'<b>Teks</b>: {cleaned_tweet}'#<br>{prediction_text}'
            prediction = label

        st.write(f"""
            <div style="background-color: #f0f8ff; color: black; padding: 10px; border-top-left-radius: 10px; border-top-right-radius: 10px; font-size: 16px; text-align: left;">
                {result}
            </div>
        """, unsafe_allow_html=True)
        
        st.write(f"""
            <div style="background-color: {label_to_color[prediction]}; color: #301934; padding: 10px; border-bottom-left-radius: 10px; border-bottom-right-radius: 10px; font-size: 16px; text-align: left; line-height: 1.8;">
                {prediction_text}
            </div>
        """, unsafe_allow_html=True)

        

    
    st.divider() 
    st.write(':blue-background[Contoh Tweet (Anda bisa memilih salah satu dari kalimat berikut sebagai input di atas):]')
    st.write('1. @ganjarpranowo @mohmahfudmd cukses selalu pak Ganjar Mahfud , semoga program ny brjln dengan lancar')
    st.write('2. Prabowo Subianto, sudah dua kali mencalonkan diri tapi belum pernah berhasil meraih kemenangan @krnnggrn ASAL BUKAN PRABOWO YangKEREN MasBOWOGBRAN RakyatMAU YangMENANG #BolehKokPindah02 #asalbukanprabowo #GabungKe02GakDosa https://t.co/xFg1kXf7FQ https://t.co/KbxIqtCm0Z ')
    st.write('3. @msaid_didu 40 banner ukuran 2 x 1.5 m sy sebarkan di Kuningan..., sy pas main ke Kuningan sedih..kota kelahiran Anies Baswedan..tp sepi baliho , banner dll kalah Ama 02 dan 03 https://t.co/ofpTeLqlqp ')
    st.write('4. @tvOneNews Gak Ngaruh Min.. rakyat Indonesia sudah terlanjur cinta dengan AMIN... Anies Baswedan itu Pemimpin yang benar2 Briliant..👍 kita kangen punya pemimpin yang sprti beliau.., TVone sebagai media yang terpercaya harusnya benar2 mendukung untuk kita mendapatkan pemimpin yang elegant.')
    st.write('5. @griselruiz88 @prabowo @gibran_tweet @budimandjatmiko @aniesbaswedan @cakimiNOW @TiurWahyuni @siyasah_aswaja @KonohaTami28877 Jangan2 alutsistanya untuk…. Gak jadi ah takut diculik')

    st.divider() 
    temp = f'Limitasi Aplikasi<br>1. Disarankan untuk memasukkan input dalam bahasa Indonesia.<br>2. Hanya dapat menerima input dengan minimal kata berjumlah 2.'
    flexible_callout(temp, background_color = '#f2f2f2', font_color="#301934")