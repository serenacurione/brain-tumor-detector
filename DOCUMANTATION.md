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
- **`main.m`**: È il punto di ingresso dell'applicazione (Entry point). Permette di lanciare la GUI visuale (se eseguito senza parametri) oppure di lanciare il training veloce da command line / CLI passando l'argomento `'train'`.

### Moduli della Pipeline di Elaborazione
- **`preprocessing.m` (Fase 1)**: Modulo per il trattamento primario dell'immagine MRI (convertita in scala di grigi). Implementa l'applicazione sequenziale del filtro Passa-Alto (Laplaciano) per l'edge-sharpening e delineamento differenziale, filtri Passa-Basso (Gaussiano 5x5) per ammorbidire imperfezioni non del tumore e filtro Mediano per ridurre pesantemente il rumore salt-and-pepper preservando i bordi essenziali.
- **`segmentation.m` (Fase 2)**: Isola l'area cerebrale di pertinenza. Implementa diversi algoritmi di Image Segmentation, incrociandone i risultati: Expectation-Maximization (EM basato su GMM dei tessuti) e sogliatura dinamica (Otsu Thresholding).
- **`feature_extraction.m` (Fase 3)**: Modulo matematico che converte le aree maschera della segmentazione in matrici e dati statistici. In totale computa ed estrae **11 parametri (features)**: feature di primo ordine (Mean, Std, Entropy, Smoothness, Uniformity), feature di momento superiore (Normalizzate alla 3a e 4a dimensione) e feature testurali (Energy, Contrast, Inverse Difference Moment IDM, Correlation).
- **`genetic_feature_selection.m` (Opzionale)**: Routine Euristica ispirata ai meccanismi biologici (Algoritmo Genetico). Ha lo scopo di minimizzare la funzione di predizione scartando feature irrilevanti provando casualmente maschere di selezione (cromosomi). Usa come fitness la funzione di loss derivata da una Cross Validation "5-fold", massimizzando logicamente l'Accuratezza SVM di validazione.
- **`svm_classifier.m` (Fase 4)**: Funzione trainante di supporto vettoriale che materializza la predizione. Implementa il multi-classe basandosi sull'algoritmo Error-Correcting Output Codes (ECOC one-vs-all). Permette flessibilmente l'interscambio immediato fra 4 astrazioni concettuali differenti: kernel Radial Basis Function (RBF), Lineare, Polinomiale e Quadratico. Accetta opzionalmente i parametri `BoxConstraint` (C) e `KernelScale` (gamma) per l'uso diretto con i valori ottimali trovati dalla grid search.
- **`svm_grid_search.m` (Ottimizzazione)**: Modulo dedicato alla ricerca degli iper-parametri SVM ottimali tramite **Grid Search con k-fold Cross-Validation stratificata** combinata al **riconoscimento del Kernel migliore**. Esplora la griglia combinatoria di `BoxConstraint` (C) e `KernelScale` (γ) per tutti i kernel in input (es. tutti e 4 passati come `'auto'`), su tutti i fold del training set. Al termine restituisce la combinazione **(Kernel, C, γ)** massima assoluta, evitando data leakage dal test set. Produce una tabella ASCII completa dei risultati intermedi per ogni kernel valutato.
- **`evaluate_metrics.m`**: Motore inferenziale statistico di fine pipeline. Date in pasto target originali e classi predette restituisce la Matrice di Confusione con Accuratezza, Sensibilità, Specificità, Precisione ed F1-Score (medie multi-classe) da comparare con l'originale ricerca target.

### Utilities di Supporto e Benchmark
- **`train_model.m`**: Script che automatizza e coordina le procedure precedenti sui set integrali di un database. Cicla ricorsivamente ed estrae le features dal folder di `Training` normalizzandole statisticamente per limitare l'overfitting. Supporta il flag opzionale `'GridSearch'` per eseguire automaticamente la Grid Search prima del training finale. Salva infine i pesi del kernel e la media per la standardizzazione nel file consolidato `brain_tumor_model.mat`, e i risultati della grid search in `grid_search_results.mat`.
- **`compare_kernels.m`**: Potente script di misurazione comparativa. Estrae a strascico l'intero dataset e lo immette nelle 4 configurazioni SVM in un unico respiro. Crea un elegante grafico a barre per mettere in luce l'impatto algoritmico del Kernel sulla metrica d'interesse. (Essendo lungo, salva a terra la cache di dati nativi estratti in `feature_cache.mat` al primo giro per potenziamento drastico della computazione alle run future).

---

## 3. Guida al Training e al Testing

Il sistema è un programma ad _apprendimento supervisionato_. Non può compiere deduzioni e classificare correttamente (Classify) su immagini libere finché non è stato "addestrato" (Training) con i dati del dataset corretto.

### Fase A: Eseguire il Training (Addestramento del Modello)
L'addestramento della SVM avviene interamente da **Command Line (CLI)** di MATLAB. Nel prompt di MATLAB esegui:
```matlab
% Forma Base con default (kernel RBF)
main('train')

% Specificando Kernel RBF (migliore su questo set)
main('train', 'rbf')
```
Questo genererà un log in tempo reale delle cartelle ispezionate (iterazioni batch per batch) che certificherà una volta giunto al termine, la completezza della procedura. Il file dei pesi vettoriali `brain_tumor_model.mat` verrà salvato nella cartella *dataset/*.

### Fase A-bis: Training con Grid Search + Cross-Validation (Migliore)
Per trovare automaticamente sia il **Kernel migliore** che i valori ottimali degli iper-parametri SVM (`BoxConstraint` C e `KernelScale` γ), esegui il training con Grid Search. Se non specifichi nulla, il programma testerà in automatico **tutti e 4 i kernel disponibili** (RBF, Lineare, Polinomiale, Quadratico):
```matlab
% Grid Search automatica globale (testa tutti i kernel su una griglia di default)
main('gridsearch')

% Grid Search specificando un solo kernel (per risparmiare tempo)
main('gridsearch', 'rbf')

% Personalizzando la griglia (testa tutti i kernel con le tue opzioni)
main('gridsearch', 'auto', 'KFolds', 10, ...
     'CValues',      [0.01 0.1 1 10 100], ...
     'GammaValues',  [0.001 0.01 0.1 1])
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

### Fase B: Ottenere le Statistiche del Paper Target (Testing dell'intero set)
Per testare l'algoritmo *su tutti i record* tenuti in disparte e rilegati per Validazione (Testing Folder), usare il comando dedicato all'analisi qualitativa di massa nel Terminal di MATLAB:
```matlab
compare_kernels('/path/to/dataset')
```
Nel MATLAB Command Window appariranno le Matrici di Confusione e le seguenti metriche cruciali per validare contro i target prescritti del paper: *Accuracy* (Obiettivo: $\ge 98.30\%$), *Sensitivity / Rischio veri positivi* (Obiettivo: $\ge 98\%$), e *Specificity / Sicurezza falsi allarmi* (Obiettivo: $100\%$). Verrà anche generata una UI figure aggiuntiva (Bar Chart) che riassumerà con codice coloro-centrico le prestazioni divise per il variare dell'iper-parametro di Supporto Vettoriale (le 4 varianti di kernel).
