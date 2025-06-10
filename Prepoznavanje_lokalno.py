import tensorflow as tf               # Za nalaganje modela
import numpy as np
import matplotlib.pyplot as plt
import cv2                            # OpenCV knjižnica za procesiranje slik (spreminjanje velikosti, morfološke operacije itd.)
import os
import math                           

# Nastavitev poti in parametrov:
# Datoteki se morata nahajati v istem direktoriju kot ta skripta.
MODEL_FILENAME = 'emnist_model.keras'
MAPPING_FILENAME = 'emnist_byclass_mapping.npy'

# Parametri za predprocesiranje slike so definirani na enem mestu kot slovar.
PREPROCESSING_PARAMS = {
    "target_pixel_size": 22,        # Ciljna velikost znaka znotraj 28x28 platna (ohranja prazno obrobo).
    "use_adaptive_thresh": False,   # Če uporabimo Otsu metodo (False) ali adaptivno pragovno obdelavo (True).
    "opening_params": {             # Morfološka operacija "odpiranje": odstrani majhen šum.
        "kernel_size": (2,2),
        "iterations": 1
    },
    "closing_params_list": [        # "zapiranje": zapolni majhne luknje v znaku.
        {"kernel_size": (9,1), "iterations": 3},
        {"kernel_size": (3,3), "iterations": 2},
        {"kernel_size": (2,2), "iterations": 1}
    ],
    "dilation_params": {            # "dilacija": odebeli znak. (Emnist znaki so običajno debeli)
        "kernel_size": (5,5),
        "iterations": 4
    },
}

# Nalaganje modela in mapiranja:
model = None
mapping_loaded = None
num_classes_loaded = 0

if not os.path.exists(MODEL_FILENAME):
    print(f"NAPAKA: Datoteka modela '{MODEL_FILENAME}' ne obstaja v trenutnem direktoriju.")
elif not os.path.exists(MAPPING_FILENAME):
    print(f"NAPAKA: Datoteka mapiranja '{MAPPING_FILENAME}' ne obstaja v trenutnem direktoriju.")
else:
    print(f"Nalagam model iz: {MODEL_FILENAME}...")
    try:
        # Uporabimo `try-except` blok, da ujamemo morebitne napake med nalaganjem (npr. poškodovana datoteka).
        model = tf.keras.models.load_model(MODEL_FILENAME)
        mapping_loaded = np.load(MAPPING_FILENAME, allow_pickle=True)
        num_classes_loaded = mapping_loaded.shape[0]
        print("Model in mapiranje uspešno naložena.")
    except Exception as e:
        print(f"Napaka pri nalaganju modela ali mapiranja: {e}")
        model = None

# Definicija funkcij in izvedba prepoznavanja:
if model and mapping_loaded is not None:

    # Definicija funkcije, ki izvede vse korake predprocesiranja slike.
    # Spremenjeno: funkcija sedaj sprejme pot do datoteke namesto bajtov.
    def preprocess_user_image(image_path, config_params, debug=False):
        # Naložimo sliko neposredno s poti v sivinskem formatu.
        img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Preverimo, ali je bila slika uspešno naložena.
        if img_original is None:
            print(f"Napaka: Slike na poti '{image_path}' ni mogoče naložiti."); return None

        # Razpakiramo parametre iz slovarja
        target_pixel_size = config_params.get("target_pixel_size", 22)
        use_adaptive_thresh = config_params.get("use_adaptive_thresh", False)
        opening_params = config_params.get("opening_params")
        closing_params_list = config_params.get("closing_params_list", [])
        dilation_params = config_params.get("dilation_params")

        # Seznama za shranjevanje vmesnih slik in njihovih naslovov za prikaz (debug).
        debug_images_list, debug_titles_list = [], []
        # Če je debug vklopljen, shranimo originalno sliko.
        if debug: debug_images_list.append(img_original.copy()); debug_titles_list.append("1. Original")

        # CNN model je bil treniran na slikah, kjer je znak bel in ozadje črno.
        # Uporabniške slike so običajno obratne (črn znak, belo ozadje), zato sliko invertiamo.
        img_inverted = 255 - img_original
        if debug: debug_images_list.append(img_inverted.copy()); debug_titles_list.append("2. Invertirana")

        # Pragovna obdelava (thresholding) pretvori sivinsko sliko v črno-belo.
        if use_adaptive_thresh:
            # Adaptivna metoda izračuna prag za manjše regije, kar je dobro za slike z neenakomerno osvetlitvijo.
            adaptive_block_size = config_params.get("adaptive_block_size", 11)
            adaptive_C = config_params.get("adaptive_C", 2)
            adaptive_block_size = adaptive_block_size if adaptive_block_size % 2 == 1 and adaptive_block_size > 1 else 11
            img_thresh = cv2.adaptiveThreshold(img_inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, adaptive_block_size, adaptive_C)
            if debug: debug_images_list.append(img_thresh.copy()); debug_titles_list.append(f"3. Adaptivni Prag")
        else:
            # Otsu metoda samodejno določi optimalen globalni prag, ki loči ospredje od ozadja.
            _, img_thresh = cv2.threshold(img_inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if debug: debug_images_list.append(img_thresh.copy()); debug_titles_list.append("3. Otsu Prag")

        # Kopiramo sliko pred morfološkimi operacijami.
        img_morph = img_thresh.copy()
        
        # Pomožna notranja funkcija za lažje dodajanje slik v debug seznam.
        morph_idx_counter = 0
        def add_morph_debug_step(img, title_suffix):
            nonlocal morph_idx_counter
            if debug:
                morph_idx_counter += 1
                debug_images_list.append(img.copy())
                debug_titles_list.append(f"4.{chr(96+morph_idx_counter)} {title_suffix}")

        # Izvedemo operacijo "odpiranja" (erozija, nato dilacija) za odstranitev majhnih pik šuma.
        if opening_params and opening_params.get("kernel_size"):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, opening_params["kernel_size"])
            img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_OPEN, kernel, iterations=opening_params.get("iterations", 1))

        # Izvedemo zaporedje operacij "zapiranja" (dilacija, nato erozija) za zapolnitev lukenj.
        for closing_op in closing_params_list:
            if closing_op and closing_op.get("kernel_size"):
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, closing_op["kernel_size"])
                img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_CLOSE, kernel, iterations=closing_op.get("iterations", 1))
        
        # Prikažemo en združen rezultat čiščenja.
        if opening_params or closing_params_list:
             add_morph_debug_step(img_morph, "Čiščenje (Odpiranje in Zapiranje)")
        
        # Izvedemo operacijo "dilacije" (odebelitev), da poudarimo poteze znaka.
        if dilation_params and dilation_params.get("kernel_size"):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_params["kernel_size"])
            img_morph = cv2.dilate(img_morph, kernel, iterations=dilation_params.get("iterations", 1))
            add_morph_debug_step(img_morph, f"Dilacija")

        # Poiščemo vse zunanje konture (obrise) na obdelani sliki.
        contours, _ = cv2.findContours(img_morph.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Pripravimo prazno 28x28 platno, na katerega bomo postavili znak.
        final_28x28_image = np.zeros((28, 28), dtype=np.uint8)

        # Če nismo našli nobenih kontur (ali je slika po obdelavi popolnoma črna)...
        if not contours or np.sum(img_morph) == 0:
            print("Ni najdenih kontur, poskušam z osnovno spremembo velikosti.")
            # ...poskusimo rešiti situacijo tako, da spremenimo velikost slike pred morfološkimi operacijami.
            if np.sum(img_thresh) > 0:
                 final_28x28_image = cv2.resize(img_thresh, (28, 28), interpolation=cv2.INTER_AREA)
        # Če smo našli konture...
        else:
            # ...predpostavimo, da je največja kontura tista, ki predstavlja naš znak.
            contour = max(contours, key=cv2.contourArea)
            # Določimo pravokotnik, ki obdaja največjo konturo (Region of Interest - ROI).
            x, y, w, h = cv2.boundingRect(contour)
            # Izrežemo ta del slike.
            char_roi = img_morph[y:y+h, x:x+w]

            # Dodamo sliki z narisanim ROI in izrezanim ROI v debug seznam.
            if debug:
                img_roi_disp = cv2.cvtColor(img_morph.copy(), cv2.COLOR_GRAY2BGR)
                cv2.rectangle(img_roi_disp, (x,y), (x+w,y+h), (0,255,0),1)
                debug_images_list.append(img_roi_disp); debug_titles_list.append("5. Najdeni ROI")
                if char_roi.size > 0:
                    debug_images_list.append(char_roi.copy()); debug_titles_list.append("6. Izrezan ROI")

            # Če je izrezan del veljaven (ima širino in višino)...
            if char_roi.size > 0:
                cH, cW = char_roi.shape
                # Izračunamo razmerje stranic, da ga lahko ohranimo.
                ar = cW / cH
                # Izračunamo novo širino in višino tako, da daljša stranica ustreza `target_pixel_size`.
                if cW > cH: nW = target_pixel_size; nH = int(target_pixel_size / ar)
                else: nH = target_pixel_size; nW = int(target_pixel_size * ar)
                nH, nW = max(1, nH), max(1, nW)
                
                # Spremenimo velikost izrezanega znaka.
                char_res = cv2.resize(char_roi, (nW, nH), interpolation=cv2.INTER_AREA)
                # Izračunamo pozicijo, da bo znak na sredini 28x28 platna.
                pY = (28 - nH) // 2; pX = (28 - nW) // 2
                # Postavimo pomanjšan znak na prazno platno.
                final_28x28_image[pY:pY+nH, pX:pX+nW] = char_res

        # Če je debug vklopljen, prikažemo vse vmesne korake obdelave.
        if debug:
            debug_images_list.append(final_28x28_image.copy()); debug_titles_list.append("7. Vhod v model (28x28)")
            cols = 4; rows = math.ceil(len(debug_images_list) / cols)
            if rows > 0:
                plt.figure(figsize=(cols * 3.5, rows * 3.5))
                for i, (img, title) in enumerate(zip(debug_images_list, debug_titles_list)):
                    plt.subplot(rows, cols, i + 1)
                    plt.imshow(img, cmap='gray'); plt.title(title); plt.axis('off')
                plt.tight_layout(); plt.show()

        # Normaliziramo vrednosti pikslov na območje [0, 1].
        img_to_predict = final_28x28_image.astype('float32') / 255.0
        # Model pričakuje vhod v obliki (batch_size, višina, širina, kanali).
        # Dodamo dimenzijo za "batch" (ker obdelujemo eno sliko) in dimenzijo za "kanale" (ker je slika sivinska).
        return np.expand_dims(np.expand_dims(img_to_predict, axis=0), axis=-1)

    # Glavna zanka za izvedbo:
    while True:
        # Vprašamo uporabnika za pot do slike.
        image_path = input("\nVnesite ime slikovne datoteke (npr. 'crka_A.png') ali 'q' za izhod: ")
        
        # Preverimo, ali želi uporabnik končati.
        if image_path.lower() == 'q':
            print("Nasvidenje!")
            break

        # Preverimo, ali datoteka obstaja, preden jo poskusimo obdelati.
        if not os.path.exists(image_path):
            print(f"NAPAKA: Datoteka '{image_path}' ne obstaja. Poskusite znova.")
            continue

        print(f'\nObdelujem: {image_path}')

        # Kličemo funkcijo za predprocesiranje s potjo do slike.
        processed_img_data = preprocess_user_image(
            image_path,
            config_params=PREPROCESSING_PARAMS,
            debug=True # Pustimo debug vklopljen, da vidimo korake obdelave.
        )

        # Če je predprocesiranje vrnilo veljavno sliko...
        if processed_img_data is not None:
            # Izvedemo napoved z modelom.
            prediction = model.predict(processed_img_data, verbose=0)[0]
            
            # `np.argmax` poišče indeks z najvišjo vrednostjo (najvišjim zaupanjem).
            pred_idx = np.argmax(prediction)
            confidence = prediction[pred_idx]
            # Z indeksom poiščemo ustrezno ASCII vrednost v naloženem mapiranju.
            rec_char = chr(int(mapping_loaded[pred_idx, 1]))

            # Izpišemo končni rezultat.
            print(f"\n--- REZULTAT ZA '{image_path}' ---")
            print(f"Prepoznan znak: '{rec_char}' z zaupanjem {confidence*100:.2f}%")

            # Za dodatno analizo izpišemo še top 3 najverjetnejše napovedi.
            top_k = 3
            # `np.argsort` sortira indekse po verjetnosti, `[-top_k:]` vzame zadnje tri (najvišje),
            top_indices = np.argsort(prediction)[-top_k:][::-1]
            print(f"\nTop {top_k} napovedi:")
            for i, idx in enumerate(top_indices):
                char_k = chr(int(mapping_loaded[idx, 1]))
                prob_k = prediction[idx]
                print(f"  {i+1}. '{char_k}' (Verjetnost: {prob_k*100:.2f}%)")
        
        else:
            print(f"\nSlike '{image_path}' ni bilo mogoče obdelati.")

# Izpis, če se model ali mapiranje na začetku nista uspešno naložila.
else:
    print("\nModel ali mapiranje ni bilo uspešno naloženo. Preverite, ali datoteki obstajata.")