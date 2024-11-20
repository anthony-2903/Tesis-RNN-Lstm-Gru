import os
import pandas as pd
import numpy as np
import time
import tracemalloc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Directorio donde se encuentran los archivos CSV por año
data_dir = 'modelo/datos1'

# Lista de archivos CSV
csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('_data.csv')]

# Lista para almacenar los DataFrames de cada año
dataframes = []

# Cargar cada archivo CSV en un DataFrame y agregarlo a la lista
for file in csv_files:
    df = pd.read_csv(file)
    df['submission_time'] = pd.to_datetime(df['submission_time'])
    df['year'] = df['submission_time'].dt.year
    dataframes.append(df)

# Concatenar todos los DataFrames en uno solo
df = pd.concat(dataframes, ignore_index=True)

# Feature engineering
df['length'] = df['url'].apply(len)
df['num_dots'] = df['url'].apply(lambda x: x.count('.'))
df['contains_at'] = df['url'].apply(lambda x: '@' in x).astype(int)

# Crear una columna target
df['phishing'] = 1  # Asume que todos los registros en los CSVs son phishing

# Crear ejemplos falsos de no phishing para cada año
years = df['year'].unique()
non_phishing_data = []

for year in years:
    non_phishing_data.append({
        'url': 'https://example.com',
        'length': len('https://example.com'),
        'num_dots': 'https://example.com'.count('.'),
        'contains_at': 0,
        'phishing': 0,
        'year': year
    })
    non_phishing_data.append({
        'url': 'https://google.com',
        'length': len('https://google.com'),
        'num_dots': 'https://google.com'.count('.'),
        'contains_at': 0,
        'phishing': 0,
        'year': year
    })
    non_phishing_data.append({
        'url': 'https://github.com',
        'length': len('https://github.com'),
        'num_dots': 'https://github.com'.count('.'),
        'contains_at': 0,
        'phishing': 0,
        'year': year
    })

non_phishing_df = pd.DataFrame(non_phishing_data)

# Combinar los datos
df = pd.concat([df, non_phishing_df], ignore_index=True)

# Resultados por año
results = []

# Tokenizer para convertir URLs a secuencias de enteros
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['url'])
vocab_size = len(tokenizer.word_index) + 1

for year in years:
    df_year = df[df['year'] == year]
    
    # Convertir URLs a secuencias de enteros
    X = tokenizer.texts_to_sequences(df_year['url'])
    X = pad_sequences(X, maxlen=100)
    y = df_year['phishing']
    
    if df_year.shape[0] > 1:
        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Construir el modelo GRU
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=100))
        model.add(GRU(128))
        model.add(Dense(1, activation='sigmoid'))
        
        # Compilar el modelo
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Evaluar sostenibilidad (uso de recursos)
        start_time = time.time()
        tracemalloc.start()
        
        # Entrenar el modelo
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=1)
        
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        training_time = end_time - start_time
        max_memory_usage = peak / 10**6  # En MB
        
        # Evaluar robustez (con datos ruidosos)
        noise_factor = 0.1
        X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
        X_test_noisy = np.clip(X_test_noisy, 0., 1.)
        
        y_pred_noisy = (model.predict(X_test_noisy) > 0.5).astype("int32")
        accuracy_noisy = accuracy_score(y_test, y_pred_noisy)
        precision_noisy = precision_score(y_test, y_pred_noisy)
        recall_noisy = recall_score(y_test, y_pred_noisy)
        f1_noisy = f1_score(y_test, y_pred_noisy)
        
        # Evaluar eficiencia (tiempo de inferencia)
        start_inference_time = time.time()
        model.predict(X_test)
        end_inference_time = time.time()
        
        inference_time = end_inference_time - start_inference_time
        
        # Hacer predicciones y evaluar el modelo con datos normales
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'year': year,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'training_time': training_time,
            'max_memory_usage': max_memory_usage,
            'accuracy_noisy': accuracy_noisy,
            'precision_noisy': precision_noisy,
            'recall_noisy': recall_noisy,
            'f1_noisy': f1_noisy,
            'inference_time': inference_time
        })

# Imprimir resultados por año
for result in results:
    print(f"Año: {result['year']}")
    print(f"  Accuracy: {result['accuracy']}")
    print(f"  Precision: {result['precision']}")
    print(f"  Recall: {result['recall']}")
    print(f"  F1 Score: {result['f1']}")
    print(f"  Tiempo de entrenamiento: {result['training_time']} segundos")
    print(f"  Uso máximo de memoria: {result['max_memory_usage']} MB")
    print(f"  Robustez (Accuracy con ruido): {result['accuracy_noisy']}")
    print(f"  Robustez (Precision con ruido): {result['precision_noisy']}")
    print(f"  Robustez (Recall con ruido): {result['recall_noisy']}")
    print(f"  Robustez (F1 con ruido): {result['f1_noisy']}")
    print(f"  Tiempo de inferencia: {result['inference_time']} segundos")
    print()