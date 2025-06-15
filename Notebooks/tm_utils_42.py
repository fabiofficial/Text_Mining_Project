import re
import string
from tqdm import tqdm

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer   

from gensim.models import Word2Vec                         

from transformers import pipeline                          

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Bidirectional,
    Dropout,
    Dense,
)
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
from keras_preprocessing.sequence import pad_sequences     

import torch

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

lemma = WordNetLemmatizer()
stemmer = SnowballStemmer('english')

def preprocess(text_list, lemma = None, stemmer = None, word2vec=False):
    """
    Return the prepocessed text in a list "updates".

    Parameters:
    text_list : list to be preprocessed
    use_lemmatize : bool, optional
        If True, applies lemmatization to the tokens. Default is True.
    use_stemmer : bool, optional
        If True, applies stemming to the tokens. Default is False.
    """

    stop_words = set(stopwords.words('english'))

    updates = []

    for j in text_list:

        text = j

        # Lower case text
        text = text.lower()


        # Remove emojis
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)

        # Remove unknown character �
        text = text.replace("�", "")

        # Remove Regular Unwanted Expressions
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'\d+', '', text)

        # Remove Punctuation
        text = re.sub(rf"[{re.escape(string.punctuation)}]", '', text)

        # Tokenize the text
        tokens = word_tokenize(text)

        #Remove Stopwords
        tokens = [word for word in tokens if word not in stop_words]

        #Lemmatize
        if lemma:
            tokens = [lemma.lemmatize(word) for word in tokens]

        #Stemming
        if stemmer:
            tokens = [stemmer.stem(word) for word in tokens]

        # Rejoin tokens

        if word2vec:
            cleaned_text=tokens
        else:
            cleaned_text = " ".join(tokens)

        updates.append(cleaned_text)

    return updates

def average_embedding(text, model, dim):
    words = text.split()
    vectors = []
    for word in words:
        if word in model:
            vectors.append(model[word])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(dim)
    
def corpus2vec(corpus, w2v):
    index_set = set(w2v.index_to_key)  # List of all words/vocabulary in the model
    word_vec = w2v.get_vector           # Creates a short cut that retrieves the embedding method for a given word
    return [
        [word_vec(word) for word in doc.split() if word in index_set]
        for doc in tqdm(corpus)
    ]

def generate_embeddings(
    texts,
    embeddings_model,
    for_sequence_model=False,
    desc="Generating Embeddings"
):
    """
    Generates either CLS embeddings (1 per text) or full token embeddings (sequence) for each input text.

    Args:
        texts (list of str): List of input texts.
        embeddings_model (callable): Hugging Face pipeline("feature-extraction", ...).
        use_cls (bool): If True, use the CLS token embedding; if False, use mean pooling.
        for_sequence_model (bool): If True, return full token-level embeddings (for RNNs).
        desc (str): tqdm progress bar description.

    Returns:
        If for_sequence_model:
            List[torch.Tensor]: Each tensor is [seq_len, hidden_dim].
        Else:
            torch.Tensor: Tensor of shape [num_texts, hidden_dim].
    """
    processed = []

    for text in tqdm(texts, desc=desc):
        output = embeddings_model(text)  
        token_embeddings = output[0]  # get token vectors from the first (and only) item in batch
        token_embeddings = torch.tensor(token_embeddings)  

        if for_sequence_model:
            processed.append(token_embeddings)  
        else:
            summary_vector = token_embeddings[0]  
            processed.append(summary_vector)

    if for_sequence_model:
        return processed
    else:
        return torch.stack(processed)  

feateng_list=['BOW', 'Word2Vec', 'Glove', 'RobertaBase', 'XLMRoberta', 'Finbert', 'DistilbertCased']

def apply_feateng(X_tr, X_te, y_tr, y_te, feateng, model_name):
    if feateng=='BOW':
        X_tr_cleaned = preprocess(X_tr['text'], lemma=lemma)
        X_te_cleaned = preprocess(X_te['text'])
        bow = CountVectorizer(binary=True)
        X_train_bow = bow.fit_transform(X_tr_cleaned)
        X_val_bow = bow.transform(X_te_cleaned)
        return X_train_bow, X_val_bow, y_tr, y_te

    elif feateng=='Word2Vec':
        X_tr_word2vec = preprocess(X_tr['text'], word2vec=True)
        X_tr_cleaned = preprocess(X_tr['text'])
        X_te_cleaned = preprocess(X_te['text'])
        w2v_model_100 = Word2Vec(sentences=X_tr_word2vec, vector_size=100, window=5, min_count=1, workers=4, sg=1)
        if model_name in ['KNN', 'LogReg']:
            X_train_avgembeddings = np.array([average_embedding(text, w2v_model_100.wv, dim=100) for text in X_tr_cleaned])
            X_val_avgembeddings = np.array([average_embedding(text, w2v_model_100.wv, dim=100) for text in X_te_cleaned])
            return X_train_avgembeddings, X_val_avgembeddings, y_tr, y_te
        else:
            X_train_embeddings = corpus2vec(X_tr_cleaned, w2v_model_100.wv)
            X_val_embeddings = corpus2vec(X_te_cleaned, w2v_model_100.wv)
            train_len = []
            for i in X_train_embeddings:
                train_len.append(len(i))
            X_train_pad = pad_sequences(maxlen=max(train_len),sequences=X_train_embeddings, padding="post", dtype='float32')
            X_val_pad = pad_sequences(maxlen=max(train_len),sequences=X_val_embeddings, padding="post", dtype='float32')
            y_tr_encoded = tf.one_hot(y_tr, depth=3)
            y_te_encoded = tf.one_hot(y_te, depth=3)
            return X_train_pad, X_val_pad, y_tr_encoded, y_te_encoded

    elif feateng=='Glove':
        X_tr_cleaned=preprocess(X_tr['text'], lemma=lemma)
        X_te_cleaned=preprocess(X_te['text'], lemma=lemma)
        if model_name in ['KNN', 'LogReg']:
            X_train_avgembeddings = np.array([average_embedding(text, glove_model_100, dim=100) for text in X_tr_cleaned])
            X_val_avgembeddings = np.array([average_embedding(text, glove_model_100, dim=100) for text in X_te_cleaned])
            return X_train_avgembeddings, X_val_avgembeddings, y_tr, y_te
        else:
            X_train_embeddings = corpus2vec(X_tr_cleaned, glove_model_100)
            X_val_embeddings = corpus2vec(X_te_cleaned, glove_model_100)
            train_len = []
            for i in X_train_embeddings:
                train_len.append(len(i))
            X_train_pad = pad_sequences(maxlen=max(train_len),sequences=X_train_embeddings, padding="post", dtype='float32')
            X_val_pad = pad_sequences(maxlen=max(train_len),sequences=X_val_embeddings, padding="post", dtype='float32')
            y_tr_encoded = tf.one_hot(y_tr, depth=3)
            y_te_encoded = tf.one_hot(y_te, depth=3)
            return X_train_pad, X_val_pad, y_tr_encoded, y_te_encoded


    elif feateng in ['RobertaBase', 'XLMRoberta', 'Finbert', 'DistilbertCased']:
        X_tr_cleaned=preprocess(X_tr['text'])
        X_te_cleaned=preprocess(X_te['text'])
        if feateng=='RobertaBase':
            MODEL="cardiffnlp/twitter-roberta-base-sentiment"
        elif feateng=='XLMROberta':
            MODEL="cardiffnlp/twitter-xlm-roberta-base-sentiment"
        elif feateng=='Finbert':
            MODEL="ProsusAI/finbert"
        elif feateng=='DistilbertCased':
            MODEL="distilbert-base-cased"
        model = pipeline("feature-extraction", model=MODEL, tokenizer=MODEL, batch_size=16, truncation=True)
        if model_name in ['KNN', 'LogReg']:
            X_train_avgembeddings=generate_embeddings(X_tr_cleaned, model)
            X_val_avgembeddings=generate_embeddings(X_te_cleaned, model)
            return X_train_avgembeddings, X_val_avgembeddings, y_tr, y_te

        else:
            X_train_embeddings=generate_embeddings(X_tr_cleaned, model, for_sequence_model=True)
            X_val_embeddings=generate_embeddings(X_te_cleaned, model, for_sequence_model=True)
            train_len = []
            for i in X_train_embeddings:
                train_len.append(len(i))
            X_train_pad = pad_sequences(maxlen=max(train_len),sequences=X_train_embeddings, padding="post", dtype='float32')
            X_val_pad = pad_sequences(maxlen=max(train_len),sequences=X_val_embeddings, padding="post", dtype='float32')
            y_tr_encoded = tf.one_hot(y_tr, depth=3)
            y_te_encoded = tf.one_hot(y_te, depth=3)
            return X_train_pad, X_val_pad, y_tr_encoded, y_te_encoded
        
def obtain_predictions(X_tr, X_te, y_tr, y_te_encoded, model_name):
    # y_te_encoded only needed for LSTM
    if model_name == 'KNN':
        # Obtain model
        knn_args = {"n_neighbors": 5, "metric": "cosine", "weights": "uniform"} 
        model = KNeighborsClassifier(**knn_args)

        # Obtain preds
        model.fit(X_tr, y_tr)
        y_tr_pred = model.predict(X_tr)
        y_te_pred = model.predict(X_te)

    elif model_name == 'LogReg':
        # Obtain model
        logreg_args = {"max_iter": 1000, "solver": "lbfgs"}
        model = LogisticRegression(**logreg_args)

        # Obtain preds
        model.fit(X_tr, y_tr)
        y_tr_pred = model.predict(X_tr)
        y_te_pred = model.predict(X_te)

    elif model_name == 'LSTM':

        print("X_tr:", X_tr.shape, X_tr.dtype)
        print("X_te:", X_te.shape, X_te.dtype)
        print("y_tr:", y_tr.shape, y_tr.dtype)
        print("y_te_encoded:", y_te_encoded.shape, y_te_encoded.dtype)
        # Obtain model
        train_len=X_tr.shape[1]
        embedding_dim =  X_tr.shape[2]
        input_ = Input(shape=(train_len, embedding_dim))

        lstm_out = Bidirectional(LSTM(units=32, return_sequences=False))(input_)
        drop = Dropout(0.4)(lstm_out)
        act  = Dense(3,
                    activation='softmax',
                    kernel_regularizer=regularizers.l2(1e-4)
                    )(drop)

        model = Model(input_, act)
        model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=[CategoricalAccuracy(name="accuracy")])

        # Obtain preds
        model.fit(
            X_tr, y_tr, 
            validation_data=(X_te, y_te_encoded),
            batch_size=16,
            epochs=40,
            verbose=1
        )
        y_tr_pred_enc = model.predict(X_tr)
        y_tr_pred = []
        for prediction in y_tr_pred_enc:
            y_tr_pred.append(np.argmax(prediction, axis=None, out=None))
        y_te_pred_enc = model.predict(X_te)
        y_te_pred = []
        for prediction in y_te_pred_enc:
            y_te_pred.append(np.argmax(prediction, axis=None, out=None))

    return y_tr_pred, y_te_pred

def feateng_crossval(
    X,                            # original DataFrame
    y,                            # original labels Series
    feateng_list,                 # e.g. ['BOW','Word2Vec',…,'RobertaBase',…]
    model_name,                   # 'KNN'|'LogReg'|'LSTM'
    precomputed_cls,              # dict[name->(N,H) array]
    precomputed_seq,              # dict[name->list of N (Lᵢ,H) arrays]
    cv=9,
    seed=SEED
):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    results = []
    for feateng in feateng_list:
        for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
            # ─── pick the right precomputed embeddings ───
            y_tr_labels = y.iloc[tr_idx]
            y_te_labels = y.iloc[te_idx]
            if feateng in precomputed_cls and model_name in ['KNN','LogReg']:
                X_tr = precomputed_cls[feateng][tr_idx]
                X_te = precomputed_cls[feateng][te_idx]
                y_for_pred_tr, y_for_pred_te = y_tr_labels, y_te_labels

            elif feateng in precomputed_seq and model_name == 'LSTM':
                seq_all = precomputed_seq[feateng]
                seq_tr = [seq_all[i] for i in tr_idx]
                seq_te = [seq_all[i] for i in te_idx]
                # pad to the max length in this fold:
                max_len = max(len(s) for s in seq_tr)
                X_tr = tf.keras.preprocessing.sequence.pad_sequences(
                    seq_tr, maxlen=max_len, padding='post', dtype='float32')
                X_te = tf.keras.preprocessing.sequence.pad_sequences(
                    seq_te, maxlen=max_len, padding='post', dtype='float32')
                y_tr_encoded = tf.one_hot(y.iloc[tr_idx], depth=3)
                y_te_encoded = tf.one_hot(y.iloc[te_idx], depth=3)
                y_for_pred_tr, y_for_pred_te = y_tr_encoded, y_te_encoded


            else:
                # fallback for BOW / Word2Vec / Glove
                X_tr, X_te, y_for_pred_tr, y_for_pred_te = apply_feateng(
                    X.iloc[tr_idx], X.iloc[te_idx],
                    y.iloc[tr_idx], y.iloc[te_idx],
                    feateng, model_name
                )

            print(X_tr.shape)
            print(y_for_pred_tr.shape)
            print(y_for_pred_tr)
            # ─── fit & predict ───
            y_tr_pred, y_te_pred = obtain_predictions(
                X_tr, X_te, y_for_pred_tr, y_for_pred_te, model_name
            )

            # Calculate metrics
            train_f1 = f1_score(y_tr_labels, y_tr_pred, average='macro')
            test_f1 = f1_score(y_te_labels, y_te_pred, average='macro')
            train_acc = accuracy_score(y_tr_labels, y_tr_pred)
            test_acc = accuracy_score(y_te_labels, y_te_pred)
            results.append({
                'model': model_name,
                'feateng': feateng,
                'fold': fold,
                'train_f1': train_f1,
                'test_f1': test_f1,
                'train_acc': train_acc,
                'test_acc': test_acc
            })

    return pd.DataFrame(results)

def plot_f1_by_feateng(df, model_name):
    """
    Creates grouped boxplots for train and test F1 scores by feature engineering method,
    with train in blue and test in red, and y-axis fixed between 0.5 and 0.9.

    Parameters:
    - df: DataFrame containing columns ['model', 'feateng', 'train_f1', 'test_f1']
    - model_name: string specifying which model to plot
    """
    # Filter for the chosen model
    df_model = df[df['model'] == model_name]
    featengs = df_model['feateng'].unique()

    data = []
    positions = []

    # Gather data and positions
    for i, feat in enumerate(featengs):
        train_vals = df_model[df_model['feateng'] == feat]['train_f1'].values
        test_vals = df_model[df_model['feateng'] == feat]['test_f1'].values

        pos_base = i * 2
        positions.extend([pos_base, pos_base + 0.4])
        data.extend([train_vals, test_vals])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.35,
        patch_artist=True,
        medianprops={'color': 'black'}
    )

    # Color boxes
    for idx, patch in enumerate(box['boxes']):
        color = 'blue' if idx % 2 == 0 else 'red'
        patch.set_facecolor(color)
        patch.set_edgecolor('black')

    # Configure axes
    xticks = [i * 2 + 0.2 for i in range(len(featengs))]
    ax.set_xticks(xticks)
    ax.set_xticklabels(featengs)
    ax.set_xlabel('Feature Engineering Method')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title(f'Macro F1 by Feature Engineering for {model_name}')
    ax.set_ylim(0.5, 1)

    # Legend
    import matplotlib.patches as mpatches
    train_patch = mpatches.Patch(color='blue', label='train_f1')
    test_patch = mpatches.Patch(color='red', label='test_f1')
    ax.legend(handles=[train_patch, test_patch])

    plt.tight_layout()
    plt.show()

def plot_test_f1_by_model(df):
    """
    Creates grouped boxplots of test_f1 scores for each model, grouped by feature engineering method.

    Parameters:
    - df: DataFrame containing columns ['model', 'feateng', 'test_f1']

    Returns:
    - fig, ax: Matplotlib Figure and Axes objects.
    """
    # Get unique feature engineering methods and models
    featengs = df['feateng'].unique()
    models = df['model'].unique()
    n_models = len(models)

    offset = 0.4
    group_stride = (n_models + 1) * offset  

    # Prepare data, positions, and colors
    data = []
    positions = []
    colors = []

    cmap = plt.get_cmap('tab10')

    for i, feat in enumerate(featengs):
        for j, model in enumerate(models):
            vals = df[(df['feateng'] == feat) & (df['model'] == model)]['test_f1'].values
            pos = i * (n_models + 1) + j
            data.append(vals)
            positions.append(pos)
            colors.append(cmap(j))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bplots = ax.boxplot(
        data,
        positions=positions,
        widths=0.35,
        patch_artist=True,
        medianprops={'color': 'black'}      
    )

    # Color each box
    for patch, color in zip(bplots['boxes'], colors):
        patch.set_facecolor(color)

    # X-axis ticks
    centers = [i * group_stride + (n_models - 1) * offset / 2 for i in range(len(featengs))]
    ax.set_xticks(centers)
    ax.set_xticklabels(featengs)
    ax.set_xlabel('Feature Engineering Method')
    ax.set_ylabel('Test Macro F1 Score')
    ax.set_title('Test Macro F1 by Model and Feature Engineering')
    ax.set_ylim(0.5, 1)

    # Legend
    legend_patches = [mpatches.Patch(color=cmap(idx), label=model) for idx, model in enumerate(models)]
    ax.legend(handles=legend_patches, title='Model')

    plt.tight_layout()
    plt.show()

def make_metrics_dict(y_true, y_pred):
    rpt = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )
    # Only keep class ’0’, ’1’, ’2’ and the ’macro avg’; drop ’weighted avg’ and support counts
    keys_to_keep = ['0', '1', '2', 'macro avg']
    flat = {}
    for label in keys_to_keep:
        met = rpt[label]
        for m_name in ['precision', 'recall', 'f1-score']:
            key = f"{label.replace(' ', '')}{m_name}"
            flat[key] = met[m_name]
    return flat


def make_metrics_dict_transformers(data_loader, model, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    rpt=classification_report(all_labels, all_preds, target_names=["0", "1", "2"],output_dict=True, zero_division=0, digits=4)

    keys_to_keep = ["0", "1", "2", "macro avg"]
    flat = {}
    for label in keys_to_keep:
        met = rpt[label]
        for m_name in ['precision', 'recall', 'f1-score']:
            key = f"{label.replace(' ', '')}{m_name}"
            flat[key] = met[m_name]
    return flat

def success1():
    print(r"""    |\/\  ,.
    /   `' |,-,
   /         /_
 _/            /
(.-,--.       /
/o/  o \     /
\_\    /   _/
(__`--'    _)
 /         |   GREAT SUCCESS, THANKS FOR LOOKING AT THIS NOTEBOOK!
(___,'    \  LOTS OF LOVE FROM GROUP 42
   \_       _\ PS: you unlocked BART 1 out of 3, keep running success to find the other rare BARTs!
     `._..-'""")


def success2():
    print(r"""            |\|\,'\,'\ ,.
            )        ;' |,'
           /              |,'|.
          /                  ` /__
         ,'                    ,-'
        ,'                    :
       (_                     '
     ,'                      ;
     |---._ ,'     .        '
     :   o Y---.__  ;      ;
     /,""-|     o.|     /   GREAT SUCCESS, THANKS FOR LOOKING AT THIS NOTEBOOK!
    ,  `._  `.    ,'     ;    LOTS OF LOVE FROM GROUP 42
    ;         `""'      ;     PS: you unlocked BART 2 out of 3, keep running success to find the other rare BARTs!
   /                   -'.
   \                   G  )
    `-.___,   `.,'
            (`   `     |)\
           / `.       ,'  \
          /    `-----'     \ """)

def success3():
    print(r"""                /\/\,\,\ ,
                /        ` \'\,
               /               '/|_
              /                   /
             /                   /
            /                   ;
            ;-""-.  __       ,
           /      )'    `.     '
          (    o |        )   ;
           ),'""" + r"""\    o   ;  :
           ;\_  `.___/ ,-:
           ;                 @ )
           /                `;-'
        ,. `-.______,|
     ,(`.||       \\\_)|
    ,.-   \      '.        |
     `._  )         )__,;\_
        \    \_   _,--/       ,   `.       GREAT SUCCESS, THANKS FOR LOOKING AT THIS NOTEBOOK!
         \     `--\   :      /      `.     LOTS OF LOVE FROM GROUP 42
          \        \  ;     |         \    PS: you unlocked BART 3 out of 3, keep running success to find the other rare BARTs!
           `-.___ ;|      |       _,'
                    \/'      `-.----' \
                    /          \      \ """)


import random
def success():
    random.choice([success1, success2, success3])()