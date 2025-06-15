# Library imports
import re
import string
from tqdm import tqdm

import numpy as np
import pandas as pd

import random

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


# Set random seed for reproducibility
SEED=42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if using -multi-GPU
tf.random.set_seed(SEED)


# Make PyTorch deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Function definitions
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
        color = (92/255, 102/255, 108/255) if idx % 2 == 0 else (190/255, 214/255, 47/255)
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
    train_patch = mpatches.Patch(color=(92/255, 102/255, 108/255), label='train_f1')
    test_patch = mpatches.Patch(color=(190/255, 214/255, 47/255), label='test_f1')
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

def plot_metric(
    df: pd.DataFrame,
    metric: str = 'f1-score',
    class_label: str = 'macroavg',
    train_color: str = 'blue',
    test_color: str = 'red'
):
    """
    Plots a line plot of the specified metric for train and test results by model,
    preserving the order of models as they appear in the DataFrame.

    Parameters:
    - df: DataFrame containing columns ['model', 'set', ...metric columns...]
    - metric: one of 'precision', 'recall', or 'f1-score'
    - class_label: '0' (Bullish), '1' (Bearish), '2' (Neutral), or 'macroavg'
    - train_color: color for the train line
    - test_color: color for the test line
    """
    # Map class_label to column prefix and display name
    label_map = {
        '0': ('0', 'Bullish'),
        '1': ('1', 'Bearish'),
        '2': ('2', 'Neutral'),
        'macroavg': ('macroavg', 'Macro Avg')
    }
    if class_label not in label_map:
        raise ValueError(f"Invalid class_label {class_label}. "
                         "Choose from '0', '1', '2', 'macroavg'.")
    
    col_prefix, display_name = label_map[class_label]
    col_name = f"{col_prefix}{metric}"
    
    # Check column exists
    if col_name not in df.columns:
        raise KeyError(f"Column '{col_name}' not found in DataFrame.")
    
    # Split train and test
    df_train = df[df['set'] == 'train']
    df_test  = df[df['set'] == 'test']
    
    # Preserve original model order
    model_order = df['model'].drop_duplicates().tolist()
    
    # Create aligned series
    y_train = [df_train.loc[df_train['model'] == m, col_name].values[0] for m in model_order]
    y_test  = [df_test.loc[df_test['model'] == m, col_name].values[0] for m in model_order]
    
    # Plot
    plt.figure(figsize=(6, 5))
    plt.plot(model_order, y_train, label='Train', color=train_color, marker='o')
    plt.plot(model_order, y_test,  label='Test',  color=test_color,  marker='o')
    
    plt.xlabel('Model')
    plt.ylabel(f"{metric.capitalize()} ({display_name})")
    plt.title(f"{display_name} {metric.capitalize()} by Model")
    plt.xticks(rotation=45)
    plt.ylim(0.3,1)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_test_classes_across_models(
    df: pd.DataFrame,
    metric: str = 'f1-score',
    split: str = 'test',
    class_labels: dict = None,
    class_colors: dict = None
):
    """
    Plots the specified metric for each class (including macroavg) across models on the x-axis.

    Parameters:
    - df: DataFrame containing columns ['model', 'set', ...metric columns...]
    - metric: one of 'precision', 'recall', or 'f1-score'
    - split: 'train' or 'test' (default 'test')
    - class_labels: dict mapping class keys to display names
                    default: {'0':'Bullish','1':'Bearish','2':'Neutral','macroavg':'Macro Avg'}
    - class_colors: dict mapping class keys to colors
                    default: standard matplotlib cycle
    """
    # Default class label mapping
    if class_labels is None:
        class_labels = {
            '0': 'Bullish',
            '1': 'Bearish',
            '2': 'Neutral',
            'macroavg': 'Macro Avg'
        }
    class_keys = list(class_labels.keys())
    
    df_sub = df[df['set'] == split]
    models = df_sub['model'].drop_duplicates().tolist()
    
    plt.figure(figsize=(6, 5))
    
    for key in class_keys:
        col = f"{key}{metric}"
        y = [df_sub.loc[df_sub['model'] == m, col].values[0] for m in models]
        # thicker line for macroavg
        lw = 3 if key == 'macroavg' else 1.5
        plt.plot(models, y, marker='o', label=class_labels[key], 
                 color=class_colors[key], linewidth=lw)
    
    plt.xlabel('Model')
    plt.ylabel(metric.capitalize())
    plt.title(f"{metric.capitalize()} across Classes on {split.capitalize()} Set")
    plt.xticks(rotation=45)
    plt.ylim(0.3, 1)
    plt.legend(title="Class")
    plt.tight_layout()
    plt.show()

def evaluate_and_print(model, X_train, y_train, X_val, y_val, model_name, rep_name, config):
    print(f"\n Model: {model_name}, Embedding: {rep_name}, Config: {config}")
    print("--------------------------------------------------------------------------------------")

    results = {}

    # --- TRAINING METRICS ---
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_report = classification_report(y_train, y_train_pred, target_names=['bearish', 'bullish', 'neutral'], output_dict=True)

    print("\nTRAINING METRICS")
    print(f"\nMacro Avg F1-score: {train_report['macro avg']['f1-score']:.4f}")

    # --- VALIDATION METRICS ---
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_report = classification_report(y_val, y_val_pred, target_names=['bearish', 'bullish', 'neutral'], output_dict=True)

    print("\nVALIDATION METRICS")
    print(f"\nMacro Avg F1-score: {val_report['macro avg']['f1-score']:.4f}")

    # --- Return all scores for storage ---
    results = {
        "model": model_name,
        "embedding": rep_name,
        "config": config,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_f1": train_report["macro avg"]["f1-score"],
        "val_f1": val_report["macro avg"]["f1-score"],
        "f1_gap": train_report["macro avg"]["f1-score"] - val_report["macro avg"]["f1-score"]
    }

    return results

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

def success():
    random.choice([success1, success2, success3])()