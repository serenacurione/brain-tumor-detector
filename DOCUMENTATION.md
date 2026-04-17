# Documentazione Tecnica - Brain Tumor MRI Classifier

Questo documento illustra l'architettura, le istruzioni di installazione, l'esecuzione del codice e i dettagli per il training e testing del sistema di classificazione di tumori cerebrali da immagini MRI in MATLAB.

## 1. Requisiti e Installazione

Il progetto è stato interamente sviluppato in **MATLAB**. Per eseguire correttamente il codice, è fondamentale assicurarsi che siano attivi i giusti toolbox MATLAB.

### Requisiti di Sistema
- **MATLAB** (versione R2021a o successive raccomandata).
- Il dataset di immagini MRI deve essere posizionato all'interno della cartella `dataset/`, strutturato in due sottocartelle principali: `Training/` e `Testing/`. All'interno di queste, le immagini devono essere raggruppate in cartelle nominate in base alla propria classe (es. `glioma_tumor`, `meningioma_tumor`, `pituitary_tumor`, `no_tumor`).

### Toolbox MATLAB Necessari
Assicurarsi di aver installato i seguenti Add-on tramite l'**Add-On Explorer** di MATLAB:
1. **Image Processing Toolbox**: Utilizzato per tutte le operazioni di filtraggio, morfologia matematica, ridimensionamento e segmentazione.
2. **Statistics and Machine Learning Toolbox**: Utilizzato per l'addestramento SVM (ECOC), la Cross-Validation e il calcolo delle metriche.

## 2. Struttura del Progetto e Moduli

Ogni funzione del progetto è implementata in un file `.m` dedicato per favorire la modularità. Ecco un riepilogo del ruolo di ciascun script:

### File Principali di Esecuzione
- **`main.m`**: Punto di ingresso dell'applicazione. Configura il percorso del dataset e avvia la pipeline di training e testing chiamando `train_model`.
- **`save_segmentation_results.m`**: Script di utility per visualizzare e salvare i risultati intermedi (preprocessing, skull stripping, tumor extraction) per un campione di immagini dal dataset.

### Moduli della Pipeline di Elaborazione
- **`preprocessing.m` (Fase 1)**: Esegue il trattamento iniziale dell'immagine. Implementa:
    1. Ridimensionamento a 256x256 e conversione in scala di grigi.
    2. **Filtro Mediano (3x3)** per la rimozione del rumore impulsivo.
    3. **Filtro Passa-Basso (Gaussiano 3x3)** per lo smoothing.
    4. **Filtro Passa-Alto (Laplaciano)** per l'edge sharpening.
- **`skull_stripping.m` (Fase 2a)**: Rimuove le ossa del cranio e i tessuti non cerebrali tramite sogliatura fissa (threshold = 30) e operazioni morfologiche (erosione, riempimento buchi).
- **`tumor_extraction.m` (Fase 2b)**: Isola la massa tumorale all'interno dell'area cerebrale precedentemente estratta. Utilizza il metodo di **Otsu** per la sogliatura dinamica, seguito da operazioni morfologiche e analisi delle componenti connesse per identificare la regione principale.
- **`feature_extraction.m` (Fase 3)**: Converte l'area segmentata in dati statistici. Estrae **7 parametri (features)**: Media, Deviazione Standard, Entropia, 3° Momento, 4° Momento, Smoothness (Fluidità) e Uniformità.
- **`svm_classifier.m` (Fase 4)**: Implementa la classificazione multi-classe utilizzando l'algoritmo **ECOC (Error-Correcting Output Codes)** con approccio "one-vs-all". Supporta kernel: RBF, Lineare, Polinomiale e Quadratico (mappato come polinomiale di grado 2).
- **`svm_grid_search.m` (Ottimizzazione)**: Esegue una ricerca degli iper-parametri ottimali (`BoxConstraint` e `KernelScale`) tramite **Grid Search con 5-fold Cross-Validation stratificata** su tutti i kernel supportati.
- **`evaluate_metrics.m`**: Calcola le metriche di performance: Accuratezza, Sensibilità (Recall), Specificità, Precisione e F1-Score (medie multi-classe).

### Utilities di Supporto
- **`train_model.m`**: Automatizza l'intera pipeline. Carica il dataset, estrae le features, normalizza i dati (z-score), esegue la Grid Search per trovare i parametri migliori e addestra il modello finale valutandolo sul set di training.

---

## 3. Guida al Training e al Testing

### Eseguire il Training
Per avviare il processo completo di addestramento e valutazione, eseguire lo script principale:
```matlab
main
```

Il sistema eseguirà automaticamente:
1. Caricamento delle immagini dalle cartelle `Training` e `Testing`.
2. Estrazione delle feature per ogni immagine.
3. **Grid Search** per identificare la combinazione migliore di Kernel, C e Gamma.
4. Addestramento del modello finale con i parametri ottimali.
5. Valutazione sul set di Testing e stampa della Matrice di Confusione e delle metriche finali nella Command Window.

### Visualizzare i Risultati della Segmentazione
Per verificare visivamente come il sistema isola il tumore, eseguire:
```matlab
save_segmentation_results
```
Questo script salverà nella cartella `output_segmentations/` le immagini elaborate (originale, maschera cranica, cervello estratto, maschera tumorale) per un campione di immagini dal dataset.
