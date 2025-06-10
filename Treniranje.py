# Uvoz knjižnic:
import tensorflow as tf 
# TensorFlow - glavna knjižnica za globoko učenje, omogoča gradnjo in treniranje nevronskih mrež

from tensorflow.keras.models import Sequential
# Sequential model - linearni zapis plasti, kjer se izhod ene plasti posreduje naslednji

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
# Conv2D - konvolucijske plasti za zaznavanje značilnosti v slikah
# MaxPooling2D - zmanjšuje prostorske dimenzije, ohranja pomembne informacije
# BatchNormalization - normalizira aktivacije, pospeši učenje
# Flatten - preoblikuje 2D matrike v 1D vektorje za polno povezane plasti
# Dense - klasične polno povezane plasti
# Dropout - regularizacija, naključno izklopi nevrone med treningom

from tensorflow.keras.utils import to_categorical
# Pretvori številčne labele v one-hot encoding (npr. 5 → [0,0,0,0,0,1,0,...])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Omogoča augmentacijo slik - generiranje novih učnih primerov z transformacijami

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# EarlyStopping - ustavi treniranje, če se model ne izboljšuje
# ReduceLROnPlateau - zmanjša učno hitrost, če se model ne izboljšuje
# ModelCheckpoint - shrani najboljšo različico modela med treningom

import numpy as np
# NumPy - osnovne operacije z matrikami in vektorji

import matplotlib.pyplot as plt
# Matplotlib - vizualizacija grafov in slik

from scipy.io import loadmat
# SciPy funkcija za nalaganje MATLAB .mat datotek

import os
import zipfile
import math

!



# Povezava z Google Drive:
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
# To omogoča dostop do datotek (EMNIST zbirka) in shranjevanje datotek (končni model) na Google Drive 



DRIVE_BASE_PATH = '/content/drive/MyDrive/Colab_Data/' 
# Osnovna pot do mape na Google Drive, kjer so shranjeni podatki

EMNIST_ZIP_PATH = os.path.join(DRIVE_BASE_PATH, 'matlab.zip')
# Kjer je .zip datoteka z EMNIST podatki


EMNIST_EXTRACT_PATH = os.path.join(DRIVE_BASE_PATH, 'emnist_data')
# Pot, kamor se bodo razpakirali EMNIST podatki

MODEL_SAVE_NAME = 'emnist_model.keras' 
# Ime datoteke za shranjevanje modela v .keras formatu 

BEST_MODEL_SAVE_PATH = os.path.join(DRIVE_BASE_PATH, MODEL_SAVE_NAME)
# Polna pot za shranjevanje najboljšega modela

MAPPING_SAVE_PATH = os.path.join(DRIVE_BASE_PATH, 'emnist_byclass_mapping.npy')
# Pot za shranjevanje mapiranja znakov


# Razpakiranje EMNIST podatkov:
if not os.path.exists(os.path.join(EMNIST_EXTRACT_PATH, 'matlab', 'emnist-byclass.mat')):
   
    
    print("Razpakiram EMNIST podatke...")
    
    if not os.path.exists(EMNIST_ZIP_PATH):
        
        print(f"NAPAKA: Datoteka '{EMNIST_ZIP_PATH}' ne obstaja. Prosim, naložite jo na Google Drive.")
        raise FileNotFoundError(f"Datoteka '{EMNIST_ZIP_PATH}' ne obstaja.")
    
    os.makedirs(EMNIST_EXTRACT_PATH, exist_ok=True)
    
    with zipfile.ZipFile(EMNIST_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EMNIST_EXTRACT_PATH)
    
    print(f"EMNIST podatki razpakirani v: {EMNIST_EXTRACT_PATH}")
else:
    print(f"EMNIST podatki že obstajajo v: {EMNIST_EXTRACT_PATH}")

EMNIST_MAT_PATH = os.path.join(EMNIST_EXTRACT_PATH, 'matlab', 'emnist-byclass.mat')

# Nalaganje EMNIST podatkov:
print("Nalagam EMNIST podatke...")
try:
    mat = loadmat(EMNIST_MAT_PATH)
    # Naloži .mat datoteko (MATLAB format) kot Python dictionary
    
    data_struct = mat['dataset'][0,0]
    # EMNIST podatki so shranjeni v strukturiranem formatu z [0,0] indeksiranjem
    
    train_images = data_struct['train'][0,0]['images']
    # Učne slike - vsaka vrstica predstavlja eno sliko (28x28 pikslov)
    
    train_labels = data_struct['train'][0,0]['labels'].squeeze()
    # Učne oznake - številčni identifikatorji razredov 
    
    test_images = data_struct['test'][0,0]['images']
    # Testne slike za evalvacijo modela
    
    mapping_data = data_struct['mapping']
    # Mapiranje med številčnimi oznakami in ASCII kodami znakov
    
    # Preverjanje oblike mapiranja
    if not (mapping_data.ndim == 2 and mapping_data.shape[1] == 2):
        # Mapiranje mora biti 2D array z dvema stolpcema [label, ASCII_code]
        
        if hasattr(mapping_data, 'shape') and mapping_data.shape == (1,1) and \
           hasattr(mapping_data[0,0], 'ndim') and mapping_data[0,0].ndim == 2 and mapping_data[0,0].shape[1] == 2:
            # Včasih je mapiranje zavito v dodatno strukturo - to ga "odvije"
            mapping_data = mapping_data[0,0]
        else:
            # Če struktura ni pričakovana, prikaže napako
            raise ValueError(f"Nepričakovana oblika za 'mapping': {mapping_data.shape if hasattr(mapping_data, 'shape') else 'N/A'}.")

except FileNotFoundError:
    print(f"NAPAKA: Datoteka EMNIST '{EMNIST_MAT_PATH}' ni najdena. Preverite pot.")
    raise
except Exception as e:
    print(f"NAPAKA pri nalaganju EMNIST podatkov: {e}")
    raise

# Shranjevanje mapiranja
if not os.path.exists(MAPPING_SAVE_PATH) or True: 
    np.save(MAPPING_SAVE_PATH, mapping_data)
    # Shrani mapiranje v NumPy format za hitro nalaganje
    print(f"Mapiranje shranjeno/posodobljeno v: {MAPPING_SAVE_PATH}")

num_classes = mapping_data.shape[0]
# Število razredov = število vrstic v mapiranju (62 za ByClass)
print(f"Število razredov: {num_classes}")

# Predprocesiranje slik:
def preprocess_emnist_images(images):
    # Predprocesira EMNIST slike za uporabo v CNN modelu. Format: (N, 28, 28, 1)
    
    images = images.astype('float32') / 255.0
    # Normalizacija: pretvori iz [0,255] v [0,1], kar stabilizira in pospeši učenje
    
    processed_images = [img.reshape(28, 28).T for img in images]     
    processed_images = np.array(processed_images)
    return np.expand_dims(processed_images, axis=-1) 
     # Vsako sliko (784 pikslov) preoblikuje v 28x28 matriko

# Predprocesiranje učnih in testnih podatkov:
X_train = preprocess_emnist_images(train_images)
X_test = preprocess_emnist_images(test_images)

# Pretvorba oznak v one-hot encoding
y_train_cat = to_categorical(train_labels, num_classes)
# Pretvori [5] v [0,0,0,0,0,1,0,...] - potrebno za categorical_crossentropy loss
y_test_cat = to_categorical(test_labels, num_classes)

print(f"Oblika učnih slik: {X_train.shape}, oznak: {y_train_cat.shape}")
# Prikaže končne dimenzije za preverjanje

# Vizualizacija primerov:
print("\nPrimeri slik iz učne množice (X_train):")
plt.figure(figsize=(10,4))

for i in range(min(10, X_train.shape[0])):
    # Prikaže do 10 primerov
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i].squeeze(), cmap='gray') 
    
    actual_char_label = chr(int(mapping_data[train_labels[i], 1]))
    
    
    plt.title(f"Oznaka: {actual_char_label}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Gradnja CNN modela:
model = Sequential([
    # Sequential omogoča linearno povezovanje plasti
    
    #Prvi konvolucijski blok: 
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    # Conv2D(32, (3,3)): 32 filtrov velikosti 3x3 pikslov
    # activation='relu': ReLU aktivacijska funkcija
    # input_shape=(28,28,1): definira obliko vhodnih podatkov
    
    BatchNormalization(),
    # Normalizira aktivacije za stabilnejše učenje
    # Izračuna povprečje in standardni odklon batch-a, nato normalizira

    MaxPooling2D(pool_size=(2, 2)),
    # Zmanjša prostorske dimenzije za faktor 2 (28x28 → 14x14)
    # Vzame maksimalno vrednost iz vsakega 2x2 okna
    
    
    # Drugi konvolucijski blok:
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    # Podvoji število filtrov (32 → 64) za zaznavanje kompleksnejših vzorcev
    # Manjše prostorske dimenzije omogočajo več filtrov
    
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    # Dodatno zmanjšanje: 14x14 → 7x7


    # Tretji konvolucijski blok:
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    # Ponovno podvojitev filtrov (64 → 128)
    
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    # Končna prostorska dimenzija: 7x7 → 3x3 (z zaokroževanjem)
    
    Dropout(0.4), 
    # Prvi dropout - 40% nevronov naključno "izključi" med treningom
    # Prepreči pretirano prilagajanje (overfitting)

    # Klasifikacijski del:
    Flatten(),
    # Preoblikuje 3D tenzor (3,3,128) v 1D vektor 
    # Potrebno za prehod na polno povezane plasti
    
    Dense(256, activation='relu'),
    # Polno povezana plast z 256 nevroni
    # Kombinira vse zaznane značilnosti za odločanje
    
    BatchNormalization(),
    # Normalizacija tudi za polno povezane plasti
    
    Dropout(0.5), 
    # Močnejši dropout (50%) pred izhodno plastjo
    # Standardna praksa za preprečevanje overfitting-a
    
    Dense(num_classes, activation='softmax')
    # Izhodna plast z 62 nevroni (en za vsak razred)
    # Softmax zagotavlja, da se verjetnosti seštejejo v 1
    # Omogoča interpretacijo kot verjetnosti članstva v razredih
])

model.summary()
# Prikaže arhitekturo modela (plasti, parametre, oblike)

# Kompilacija modela:
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# Optimizator je algoritem, ki posodablja uteži (weights) modela med treniranjem, da bi čim bolj zmanjšal funkcijo izgube (loss function).

model.compile(
    optimizer=optimizer, 
    loss='categorical_crossentropy',  # 'categorical_crossentropy' je pogosta izbira za večrazredno klasifikacijo
    metrics=['accuracy']  # Sledi natančnosti med treningom
)

# Augmentacija podatkov:
datagen = ImageDataGenerator(
    rotation_range=10,        # Naključna rotacija ±10 stopinj
    width_shift_range=0.1,    # Horizontalni premik ±10% širine
    height_shift_range=0.1,   # Vertikalni premik ±10% višine
    zoom_range=0.1            # Povečava/pomanjšava ±10%
)
# Augmentacija ustvarja nove učne primere z rahlim spreminjanjem originalnih slik
# Povečuje raznolikost podatkov in izboljšuje generalizacijo


# Povratni klici (callbacks):
early_stopping = EarlyStopping(
    monitor='val_accuracy',    # Spremlja validacijsko natančnost
    patience=10,              # Počaka 10 epoh brez izboljšanja
    verbose=1,                
    mode='max',               # Išče maksimalno natančnost
    restore_best_weights=True  # Obnovi najboljše uteži ob koncu
)
# Ustavi treniranje, če se model ne izboljšuje

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',    # Spremlja validacijsko izgubo
    factor=0.2,           # Zmanjša učno hitrost za 80% 
    patience=5,           # Počaka 5 epoh brez izboljšanja
    verbose=1,          
    mode='min',           # Išče minimalno izgubo
    min_lr=1e-6          # Najnižja možna učna hitrost
)
# Dinamično prilagaja učno hitrost med treningom
# Omogoča "fino nastavljanje" ko se učenje upočasni

model_checkpoint = ModelCheckpoint(
    BEST_MODEL_SAVE_PATH,    
    monitor='val_accuracy',   # Kriterij za "najboljši" model
    save_best_only=True,      # Shrani le, če je boljši od prejšnjega
    mode='max',               # Maksimizira natančnost
    verbose=1                 
)
# Avtomatsko shrani najboljšo različico modela

callbacks_list = [early_stopping, reduce_lr, model_checkpoint]
# Kombinira vse povratne klice

# Treniranje modela:
BATCH_SIZE = 128
# Število primerov, ki se procesira naenkrat
# Kompromis med stabilnostjo gradientov in hitrostjo

EPOCHS = 50 
# Maksimalno število epoh
# EarlyStopping bo verjetno ustavil prej

print(f"\nZačenjam treniranje modela. Najboljši model bo shranjen v: {BEST_MODEL_SAVE_PATH}")

history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
    # datagen.flow ustvarja augmentirane batche med treningom
    # Vsak batch vsebuje rahlo spremenjene različice originalnih slik
    
    epochs=EPOCHS,
    validation_data=(X_test, y_test_cat),
    # Podatki za validacijo (brez augmentacije)
    # Uporabljajo se za spremljanje napredka in callbacks
    
    callbacks=callbacks_list,
    steps_per_epoch=math.ceil(len(X_train) / BATCH_SIZE)
    # Izračuna števila korakov na epoho
)

print(f"\nTreniranje zaključeno. Najboljši model je bil shranjen v {BEST_MODEL_SAVE_PATH} (če je bil ModelCheckpoint uspešen).")

# Vizualizacija rezultatov:
plt.figure(figsize=(12, 5))

# Graf izgube
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Izguba (učenje)')
plt.plot(history.history['val_loss'], label='Izguba (validacija)')
plt.title('Izguba')
plt.xlabel('Epoha')
plt.legend()
# Prikaže, kako se izguba spreminja med treningom
# Razlika med učno in validacijsko izgubo kaže na overfitting

# Graf natančnosti
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Natančnost (učenje)')
plt.plot(history.history['val_accuracy'], label='Natančnost (validacija)')
plt.title('Natančnost')
plt.xlabel('Epoha')
plt.legend()
# Prikaže napredek natančnosti
# Validacijska natančnost je ključni pokazatelj uspešnosti

plt.tight_layout()
plt.show()

# Končna evaluacija modela:
print("\nKončna evaluacija modela na testnem nizu:")
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=1)
# Testira model na podatkih, ki jih ni videl med treningom
# Zagotavlja realno oceno uspešnosti

print(f"Testna natančnost: {accuracy*100:.2f}%")
print(f"Testna izguba: {loss:.4f}")

print("\n--- Celica 1 zaključena ---")