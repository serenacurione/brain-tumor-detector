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
3. **Global Optimization Toolbox**: (*Opzionale ma raccomandato*) Necessario *solo* se si intende eseguire la Genetic Algorithm (GA) per la selezione delle feature statistiche ottimali. Se non si possiede questo toolbox, assicurarsi di **non** spuntare la casella "Use Genetic Algorithm" nella GUI, in modo da utilizzare di default tutte le feature.

Nessuna ulteriore installazione via shell è richiesta (nessun framework Python o pacchetto addizionale).

## 2. Struttura del Progetto e Moduli

Ogni funzione del progetto è implementata in un file `.m` dedicato, seguendo paradigmi di Clean Code e favorendo la modularità e leggibilità. Ecco un riepilogo dettagliato del ruolo di ciascun script:

### File Principali di Esecuzione
- **`main.m`**: È il punto di ingresso dell'applicazione (Entry point). Permette di lanciare la GUI visuale (se eseguito senza parametri) oppure di lanciare il training veloce da command line / CLI passando l'argomento `'train'`.
- **`BrainTumorGUI.m`**: Contiene l'intero codice dell'interfaccia grafica (Graphical User Interface) "dark-theme" costruita programmaticamente tramite le callback di `uifigure`. Gestisce in un design ordinato i bottoni, la visualizzazione delle aree della MRI nei vari stadi, e l'aggiornamento simultaneo tra lo status predittivo e quello d'interfaccia.

### Moduli della Pipeline di Elaborazione
- **`preprocessing.m` (Fase 1)**: Modulo per il trattamento primario dell'immagine MRI (convertita in scala di grigi). Implementa l'applicazione sequenziale del filtro Passa-Alto (Laplaciano) per l'edge-sharpening e delineamento differenziale, filtri Passa-Basso (Gaussiano 5x5) per ammorbidire imperfezioni non del tumore e filtro Mediano per ridurre pesantemente il rumore salt-and-pepper preservando i bordi essenziali.
- **`segmentation.m` (Fase 2)**: Isola l'area cerebrale di pertinenza. Implementa diversi algoritmi di Image Segmentation, incrociandone i risultati: Expectation-Maximization (EM basato su GMM dei tessuti), sogliatura dinamica (Otsu Thresholding), Level-Set (Chan-Vese Method) e infine tecnica ad immersione Watershed per distinguere tra materia con densità similare.
- **`feature_extraction.m` (Fase 3)**: Modulo matematico che converte le aree maschera della segmentazione in matrici e dati statistici. In totale computa ed estrae **11 parametri (features)**: feature di primo ordine (Mean, Std, Entropy, Smoothness, Uniformity), feature di momento superiore (Normalizzate alla 3a e 4a dimensione) e feature testurali basate sulle Gray Level Co-occurrence Matrix / GLCM (Energy, Contrast, Inverse Difference Moment IDM, Correlation).
- **`genetic_feature_selection.m` (Opzionale)**: Routine Euristica ispirata ai meccanismi biologici (Algoritmo Genetico). Ha lo scopo di minimizzare la funzione di predizione scartando feature irrilevanti provando casualmente maschere di selezione (cromosomi). Usa come fitness la funzione di loss derivata da una Cross Validation "5-fold", massimizzando logicamente l'Accuratezza SVM di validazione.
- **`svm_classifier.m` (Fase 4)**: Funzione trainante di supporto vettoriale che materializza la predizione. Implementa il multi-classe basandosi sull'algoritmo Error-Correcting Output Codes (ECOC one-vs-all). Permette flessibilmente l'interscambio immediato fra 4 astrazioni concettuali differenti: kernel Radial Basis Function (RBF), Lineare, Polinomiale e Quadratico.
- **`evaluate_metrics.m`**: Motore inferenziale statistico di fine pipeline. Date in pasto target originali e classi predette restituisce la Matrice di Confusione con Accuratezza, Sensibilità, Specificità, Precisione ed F1-Score (medie multi-classe) da comparare con l'originale ricerca target.

### Utilities di Supporto e Benchmark
- **`train_model.m`**: Script che automatizza e coordina le procedure precedenti sui set integrali di un database. Cicla ricorsivamente ed estrae le features dal folder di `Training` normalizzandole statisticamente per limitare l'overfitting. Salva infine i pesi del kernel e la media per la standardizzazione nel file consolidato `brain_tumor_model.mat`.
- **`compare_kernels.m`**: Potente script di misurazione comparativa. Estrae a strascico l'intero dataset e lo immette nelle 4 configurazioni SVM in un unico respiro. Crea un elegante grafico a barre per mettere in luce l'impatto algoritmico del Kernel sulla metrica d'interesse. (Essendo lungo, salva a terra la cache di dati nativi estratti in `feature_cache.mat` al primo giro per potenziamento drastico della computazione alle run future).

---

## 3. Guida al Training e al Testing

Il sistema è un programma ad _apprendimento supervisionato_. Non può compiere deduzioni e classificare correttamente (Classify) su immagini libere finché non è stato "addestrato" (Training) con i dati del dataset corretto.

### Fase A: Eseguire il Training (Addestramento del Modello)
Puoi procedere ad allenare la SVM dell'algoritmo in due modalità principali:

**1. Training Interattivo (Tramite GUI)**
1. Clicca sul file `main.m` ed eseguilo, oppure digita `main()` nel terminale di MATLAB e premi invio. Verrà mostrata a schermo l'interfaccia utente in stile "Dark Theme".
2. Nel pannello sinistro, alla voce **Dataset path:**, assicurati di posizionare il path corretto (es. `C:\Users\NOME\Desktop\progetto_IP\dataset`). Se la barra di testo è lasciata vuota, cliccare su "Train Model" aprirà una comoda finestra del sistema operativo per selezionare interattivamente la radice del dataset e la memorizzerà.
3. Seleziona dal selettore a tendina il Kernel da usare. Di default, **RBF** (Radial Basis Function) gestisce egregiamente le non-linearità dello shape neoplastico tumorale.
4. *Opzionale:* Clicca "Use Genetic Algorithm" se vuoi limitare le feature in entrata al classificatore, perdendo inizialmente in efficienza computazionale per guadagnare in prestazione SVM.
5. Clicca vigorosamente sul bottone **"Train Model"**. Essendo un'operazione avida di risorse, potresti dover aspettare alcuni minuti dipendentemente dalle specifiche del tuo processore. Il label in basso ("Status") ti terrà informato.
6. A processo finito, apparirà nella shell i calcoli finali e sulla memoria fisica si plasmerà il file dei pesi vettoriali "brain_tumor_model.mat" nella cartella *dataset/*.

**2. Training Veloce Automation (Tramite CLI)**
Lavorare in CLI permette ai thread di non doversi scindere tra interfacce UI e calcolo matematico. Nel prompt di MATLAB esegui semplicemente:
```matlab
% Forma Base con default
main('train')

% Specificando Kernel RBF (migliore su questo set) disabilitando Genetic Algorithm (0)
main('train', 'rbf', 0)
```
Questo genererà un log in tempo reale delle cartelle ispezionate (iterazioni batch per batch) che certificherà una volta giunto al termine, la completezza della procedura.

### Fase B: Testing, Singola Valutazione e Benchmark Metriche
Ora che il modello è stato addestrato al discriminante cellulare, il software è in uso:

- **Eseguire un Test Real-Time (Una Immagine Nuova)**:
  1. Aprire la GUI e cliccare su **"Load MRI Image"**. Scegli dal tuo sistema un'immagine (puoi usare un file dentro `dataset/Testing/...` per test di precisione).
  2. Per fare un batch processing unico, clicca in fondo al pannello blu **"Run Full Pipeline"**. L'immagine verrà simultaneamente Processata tramite matrice Gaussiana/Laplaciana, il tumore Segmentato visualizzando una Bounding box di localizzazione in layer verde, verranno carpite le feature testurali e per ultimo un label grande sentenzierà la predizione ("Meningioma", "No Tumor", ecc.).
  (Puoi altresì procedere in slow-motion manuale cliccando i pulsanti verdi uno ad uno per ispezionare singolarmente i risultati grafici).

- **Ottenere le Statistiche del Paper Target (Testing dell'intero set)**:
  1. Per testare l'algoritmo *su tutti i record* tenuti in disparte e rilegati per Validazione (Testing Folder), non usa la GUI. Nel Terminal di MATLAB usa il comando dedicato e preposto all'analisi qualitativa di massa:
  ```matlab
  compare_kernels('/path/to/dataset')
  ```
  2. Nel MATLAB Command Window appariranno le Matrici di Confusione e le seguenti metriche cruciali per validare contro i target prescritti del paper: *Accuracy* (Obiettivo: $\ge 98.30\%$), *Sensitivity / Rischio veri positivi* (Obiettivo: $\ge 98\%$), e *Specificity / Sicurezza falsi allarmi* (Obiettivo: $100\%$). Verrà anche lanciata una bellissima e utile UI figure aggiuntiva (Bar Chart) che riassumerà con codice coloro-centrico le prestazioni divise per il variare dell'iper-parametro di Supporto Vettoriale (le 4 varianti di kernel).
