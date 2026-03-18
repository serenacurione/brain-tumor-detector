# Documentazione Tecnica - Brain Tumor MRI Classifier

Questo documento illustra l'architettura, le istruzioni di installazione, l'esecuzione del codice e i dettagli per il training e testing del sistema di classificazione di tumori cerebrali da immagini MRI in MATLAB.

## 1. Requisiti e Installazione

Il progetto è stato interamente sviluppato in **MATLAB**. Per eseguire correttamente il codice, non è necessario installare dipendenze o pacchetti esterni tramite terminale, ma è fondamentale assicurarsi che siano attivi i giusti toolbox MATLAB.

### Requisiti di Sistema
- **MATLAB** (versione R2021a o successive raccomandata, necessaria per utilizzare la nuova UI `uifigure` in modo nativo).
- Il dataset di immagini MRI deve essere posizionato all'interno della cartella temporanea `dataset/`, strutturato rigorosamente in due sottocartelle principali: `Training/` e `Testing/`. All'interno di queste, le immagini devono essere raggruppate in folder nominati in base alla propria classe (es. `glioma_tumor`, `meningioma_tumor`, `no_tumor`).

### Toolbox MATLAB Necessari
Assicurarsi di aver scaricato e installato i seguenti Add-on dall'**Add-On Explorer** di MATLAB (visibile nela barra "Home"):
1. **Image Processing Toolbox**: Obbligatorio per tutte le operazioni di filtraggio, morfologia matematica, ridimensionamento e segmentazione dell'immagine.
2. **Statistics and Machine Learning Toolbox**: Obbligatorio per l'algoritmo EM (Gaussian Mixture Models), l'addestramento SVM, la Cross-Validation e la predizione.

Nessuna ulteriore installazione via shell è richiesta (nessun framework Python o pacchetto addizionale).

## 2. Struttura del Progetto e Moduli

Ogni funzione del progetto è implementata in un file `.m` dedicato, seguendo paradigmi di Clean Code e favorendo la modularità e leggibilità. Ecco un riepilogo dettagliato del ruolo di ciascun script:

### File Principali di Esecuzione
- **`main.m`**: È il punto di ingresso dell'applicazione (Entry point). Permette di lanciare l'addestramento da command line con grid search automatica su tutti i kernel. Opzionalmente, accetta come parametro il path personalizzato del dataset. Se non viene fornito, usa la cartella `dataset` predefinita.

### Moduli della Pipeline di Elaborazione
- **`preprocessing.m` (Fase 1)**: Modulo per il trattamento primario dell'immagine MRI (convertita in scala di grigi). Implementa l'applicazione sequenziale del filtro Passa-Alto (Laplaciano) per l'edge-sharpening e delineamento differenziale, filtri Passa-Basso (Gaussiano 5x5) per ammorbidire imperfezioni non del tumore e filtro Mediano per ridurre pesantemente il rumore salt-and-pepper preservando i bordi essenziali.
- **`segmentation.m` (Fase 2)**: Isola l'area cerebrale di pertinenza. Implementa diversi algoritmi di Image Segmentation, incrociandone i risultati: Expectation-Maximization (EM basato su GMM dei tessuti) e sogliatura dinamica (Otsu Thresholding).
- **`feature_extraction.m` (Fase 3)**: Modulo matematico che converte le aree maschera della segmentazione in matrici e dati statistici. In totale computa ed estrae **7 parametri (features)**: feature di primo ordine (Mean, Std, Entropy, Smoothness, Uniformity) e feature di momento superiore (Normalizzate alla 3a e 4a dimensione).
- **`svm_classifier.m` (Fase 4)**: Funzione trainante di supporto vettoriale che materializza la predizione. Implementa il multi-classe basandosi sull'algoritmo Error-Correcting Output Codes (ECOC one-vs-all). Permette flessibilmente l'interscambio immediato fra 4 astrazioni concettuali differenti: kernel Radial Basis Function (RBF), Lineare, Polinomiale e Quadratico. Accetta opzionalmente i parametri `BoxConstraint` (C) e `KernelScale` (gamma) per l'uso diretto con i valori ottimali trovati dalla grid search.
- **`svm_grid_search.m` (Ottimizzazione)**: Modulo dedicato alla ricerca degli iper-parametri SVM ottimali tramite **Grid Search con k-fold Cross-Validation stratificata** combinata al **riconoscimento del Kernel migliore**. Esplora la griglia combinatoria di `BoxConstraint` (C) e `KernelScale` (γ) per tutti i kernel in input (es. tutti e 4 passati come `'auto'`), su tutti i fold del training set. Al termine restituisce la combinazione **(Kernel, C, γ)** massima assoluta, evitando data leakage dal test set. Produce una tabella ASCII completa dei risultati intermedi per ogni kernel valutato.
- **`evaluate_metrics.m`**: Motore inferenziale statistico di fine pipeline. Date in pasto target originali e classi predette restituisce la Matrice di Confusione con Accuratezza, Sensibilità, Specificità, Precisione ed F1-Score (medie multi-classe) da comparare con l'originale ricerca target.

### Utilities di Supporto
- **`train_model.m`**: Script che automatizza e coordina le procedure precedenti sui set integrali di un database (`Training` e `Testing`). Cicla ricorsivamente ed estrae le features normalizzandole statisticamente (z-score) per limitare l'overfitting. Esegue in automatico una Grid Search strutturata con parametri opzionali configurabili prima del training finale, valutando al termine il modello multi-classe con i pesi ideali direttamente sul set di Test calcolando diverse metriche. Salva infine i pesi del kernel e la media per la standardizzazione nel file consolidato `brain_tumor_model.mat`, e i risultati della grid search in `grid_search_results.mat`.

---

## 3. Guida al Training e al Testing

Il sistema è un programma ad _apprendimento supervisionato_. Non può compiere deduzioni e classificare correttamente (Classify) su immagini libere finché non è stato "addestrato" (Training) con i dati del dataset corretto.

### Fase A: Eseguire il Training (Addestramento del Modello con Grid Search)
L'addestramento della SVM avviene interamente da **Command Line (CLI)** di MATLAB. Il sistema esegue SEMPRE una **Grid Search con Cross-Validation** per individuare automaticamente il **Kernel migliore** tra tutti e 4 quelli disponibili (RBF, Lineare, Polinomiale, Quadratico) e i valori ottimali degli iper-parametri (`BoxConstraint` C e `KernelScale` γ).

Nel prompt di MATLAB esegui:
```matlab
% Esecuzione sul dataset di default (cerca la cartella "dataset" nel percorso corrente)
main()

% Esecuzione specificando un percorso personalizzato per il dataset
main('mio/percorso/al/dataset')
```

Il sistema:
1. Per ogni kernel selezionato, costruisce tutte le combinazioni (C, γ) nella griglia
2. Valuta ciascuna combinazione con k-fold Cross-Validation **sul solo training set** (nessun data leakage dal test set)
3. Stampa le tabelle complete delle accuratezze CV con la configurazione ottimale globale evidenziata
4. Addestra il modello ECOC multi-classe finale con la **combinazione vincente assoluta (Kernel + C + γ)**
5. Salva `grid_search_results.mat` (risultati di ogni tabella CV) e `brain_tumor_model.mat` (modello finale per la GUI o test)

È anche possibile invocare la grid search programmaticamente per massima flessibilità. Puoi testare stringhe come `'auto'`, o un singolo kernel in formato stringa, oppure una matrice di celle specifica `{'rbf', 'linear'}`:
```matlab
datasetPath = '/path/to/dataset';
[X_train, y_train, ~] = ...; % dopo aver estratto le features

% Testa ad esempio RBF e Linear passandoli come cell array
[bestParams, gsResults] = svm_grid_search(X_train, y_train, {'rbf', 'linear'}, ...
    'KFolds', 5, ...
    'CValues',     logspace(-2, 3, 6), ...
    'GammaValues', logspace(-3, 2, 6));
```

### Visualizzare le Metriche Finali
La valutazione delle performance sull'intero Testing Set avviene automaticamente al termine della procedura di addestramento e validazione gestita da `train_model`, e invocabile tramite `main()`.

Nel MATLAB Command Window verranno mostrate alla fine del calcolo le Matrici di Confusione e le seguenti metriche fondamentali per validare contro i target prescritti del paper: *Accuracy*, *Sensitivity / Rischio veri positivi*, *Specificity / Sicurezza falsi allarmi*, *Precision* e l'*F1-Score*. Tutte queste misurazioni derivano rigorosamente dall'esecuzione dei dati "invisibili" di Test tramite l'apparato SVM (supportato dalla validazione ECOC).
