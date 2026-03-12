## PRD: Sistema di Identificazione e Classificazione di Tumori Cerebrali da MRI
 
### 1. Panoramica del Progetto
 
L'obiettivo è sviluppare un sistema informatico computerizzato per la classificazione precisa di immagini di Risonanza Magnetica (MRI). Il sistema dovrà eseguire la pre-elaborazione, la segmentazione, l'estrazione delle caratteristiche e la classificazione per distinguere le immagini cerebrali normali da quelle con tumore, evolvendo l'architettura originale per supportare la classificazione multi-classe fornita dal dataset Kaggle.
 
### 2. Specifiche del Dataset
 
* 
**Dataset Originale (nel paper):** 60 immagini MRI in formato .jpeg (10 normali, 50 anormali di cui 30 benigni e 20 maligni) raccolte in Bangladesh e dal web.
 
 
* **Dataset Target (Kaggle):** `sartajbhuvaji/brain-tumor-classification-mri`.
* 
**Adattamento Richiesto:** Il paper originale prevede una classificazione gerarchica (Normale vs Anormale $\rightarrow$ Benigno vs Maligno). Poiché il dataset Kaggle presenta 4 classi specifiche (Glioma, Meningioma, Pituitary tumor, No tumor), la pipeline dovrà mappare "No tumor" come *Normale* e le restanti tre classi come *Anormale*, per poi applicare una classificazione multi-classe al posto della semplice dicotomia benigno/maligno.
 
 
### 3. Pipeline Architetturale
 
Il sistema dovrà implementare rigorosamente le seguenti fasi sequenziali:
 
#### 3.1 Pre-processing (Miglioramento dell'Immagine)
 
* 
**Scopo:** Rimuovere il rumore e isolare la porzione del cranio tramite operazioni morfologiche.
 
 
* 
**Filtri da implementare:** * *Filtro Passa-Alto:* Per rimuovere piccole quantità di rumore a bassa frequenza.
 
 
* 
*Filtro Passa-Basso:* Per affilare l'immagine e definire meglio i bordi del tumore.
 
 
* 
*Filtro Mediano:* Fondamentale per ridurre il rumore ad alta frequenza preservando i bordi essenziali dell'immagine MRI.
 
 
 
#### 3.2 Segmentazione
 
* 
**Scopo:** Isolare o partizionare l'immagine in regioni con proprietà simili per estrarre la regione del tumore.
 
 
* 
**Algoritmi richiesti:** * *Expectation-Maximization (EM):* Per riprodurre l'immagine di input.
 
 
* 
*Level Set Method:* Per ottenere i valori accurati dei confini della massa.
 
 
* 
*Tecniche aggiuntive:* Segmentazione watershed e segmentazione basata su soglia del livello di grigio (gray-level edge segmentation).
 
 
 
#### 3.3 Estrazione e Selezione delle Feature
 
* 
**Scopo:** Convertire la regione isolata (misurata in base al numero di pixel bianchi) in dati statistici.
 
 
* 
**Algoritmo di ottimizzazione:** Utilizzare un Algoritmo Genetico (Genetic Algorithm) per l'estrazione.
 
 
* 
**Feature Statistiche da calcolare:** Valore medio, deviazione standard, entropia, momenti di ordine superiore, energia, contrasto, momento di differenza inverso (inverse difference moment), correlazione, fluidità (smoothness) e uniformità.
 
 
#### 3.4 Classificazione (SVM)
 
* 
**Scopo:** Apprendere dal set di feature estratte per classificare il tessuto.
 
 
* 
**Tecnologia:** Classificatore Support Vector Machine (SVM) basato sul concetto di sostituzione del kernel.
 
 
* 
**Requisito Kernel:** Implementare e valutare le prestazioni confrontando kernel differenti: RBF, lineare, poligonale (polinomiale) e quadratico.
 
 
### 4. Requisiti Funzionali e Interfaccia Utente (GUI)
 
Il software sviluppato in MATLAB dovrà avere una Graphical User Interface (GUI) che permetta all'utente di eseguire i seguenti passaggi:
 
1. Caricare le immagini MRI di input.
 
 
2. Avviare i passaggi di elaborazione per migliorare la qualità dell'immagine.
 
 
3. Eseguire la segmentazione per rimuovere il rumore ed estrarre la regione d'interesse.
 
 
4. Estrarre le feature dall'immagine segmentata.
 
 
5. Avviare il classificatore SVM per rilevare la tipologia di tumore.
 
 
6. Visualizzare a schermo l'immagine originale, l'immagine filtrata, i confini (Bounding Box) e il tumore rilevato (Detected Tumor).
 
 
### 5. Metriche di Valutazione e Obiettivi di Successo
 
Le prestazioni dell'algoritmo SVM dovranno essere valutate calcolando la matrice di confusione (True Positive, True Negative, False Positive, False Negative).
 
Il sistema dovrà calcolare automaticamente le seguenti metriche e tentare di eguagliare o superare i benchmark del paper:
 
* 
**Accuratezza (Target: $\ge 98.30\%$):** $\frac{TP + TN}{TP + FP + TN + FN}$
 
 
* 
**Sensibilità (Target: $\ge 98\%$):** $\frac{TP}{TP + FN}$
 
 
* 
**Specificità (Target: $100\%$):** $\frac{TN}{FP + TN}$

## Dataset
Nella cartella "/dataset" sono presenti tutte le immagini per il training e il testing distinguendo immagini senza tumore, quelle con tumore benigno e quelle con tumore maligno.

## Consigli di implementazione
Assicurati che l'implementazione sia efficiente e curata. Utilizza le best practices di programmazione e di clean code. Organizza i file in modo da avere componenti monolitici e favorisci la leggibilità del codice.
Aggiungi i commenti nel codice esclusivamente quando è strettamente necessario.
