# Multilingual computational models capture a shared meaning component in brain responses across 21 languages

Code, data and results for: de Varda, A. G., Malik-Moraleda, S., Tuckute, G., & Fedorenko, E. *Multilingual Computational Models Capture a Shared Meaning Component in Brain Responses across 21 Languages*. **Nature Neuroscience** (in press). Large files that cannot be redistributed through GitHub (audio stimuli, model embeddings for one model, a minimal working demo) are available on OSF: https://osf.io/uc6y7/.

## Overview

The paper reports four studies. The repository is organised around them:

| Study | What it does | Where the code lives |
|---|---|---|
| **Study I** | Encoding models predicting fMRI responses in the language network from multilingual language-model (MNNLM) embeddings, in 12 languages (data from Malik-Moraleda, Ayyash, et al., 2022). Models are trained and tested within each language (WITHIN) or trained on all-but-one language and transferred zero-shot to the held-out language (ACROSS). | repository root, `data/`, `transcribed/`, `results/`, `perplexity/` |
| **Study II** | Encoding models are trained on four independent datasets (Study I data, NatStories, Pereira2018, Tuckute2024) and transferred zero-shot to newly collected fMRI data in 9 additional languages. | `confirmatory/` (new data, evaluation) and `additional_analyses/{NaturalStories,pereira,control}/` (training datasets; `control` = Tuckute2024) |
| **Study III** | Targeted perturbations of the English training stimuli (information-loss, word-order and paraphrase manipulations) to test what supports cross-lingual transfer. | `additional_analyses/control/perturbation/`, `additional_analyses/pereira/perturbation/` |
| **Study IV** | Iterative Null-space Projection (INLP) removing meaning- or form-related information directly from the embeddings before training the encoding models. | `additional_analyses/residualize_info/` |

Supplementary analyses are indexed in the section [Supplementary information: where each analysis lives](#supplementary-information-where-each-analysis-lives).

## Repository layout

```
.
├── README.md
├── data/                          Study I fMRI time series (csv) and the pickled per-participant/per-fROI dictionaries used by the code
├── transcribed/                   Whisper-timestamped transcriptions of the Study I passages (one csv per language: word, onset, offset)
├── sound_data/                    [not tracked; on OSF] audio stimuli of Study I
├── embeddings/                    [not tracked; DistilmBERT embeddings on OSF] contextual word embeddings, one file per model x language
├── results/                       Study I encoding results (one pickle per model x fROI x condition; see below)
├── plots/                         figures (svg) produced by the plotting scripts, plus the assembled paper panels
├── perplexity/                    next-word-prediction results and the Figure 2D script
├── whisper_timestamped.py         speech-to-text transcription with word-level timestamps
├── get_model_embeddings.py        embedding extraction from the 20 MNNLMs
├── get_embeddings_chunk_context.py  embedding extraction with the context reset at fold boundaries (SI S11)
├── time_series_corr.py            time series extraction, reliability (Figure 4A) and language selection
├── fit_encoding.py                all Study I encoding models (WITHIN, ACROSS, baselines, RH/MD, native fROIs, split context)
├── encoding.sh                    example SLURM script for fit_encoding.py
├── plot_encoding_mono.py          significance testing, aggregation and figures for the WITHIN condition
├── plot_encoding_multi.py         same for the ACROSS condition (also writes results/mono_multi.csv)
├── plot_layerwise.py              layer-wise results (Figure 2B)
├── confirmatory/                  Study II: newly collected 9-language data, transcriptions, registered encoding models and evaluation
├── additional_analyses/
│   ├── NaturalStories/            NatStories training dataset (time series, transcriptions, embeddings, model training)
│   ├── pereira/                   Pereira2018 training dataset (+ Study III perturbations)
│   ├── control/                   Tuckute2024 training dataset (+ Study III perturbations and the Study III evaluation script)
│   └── residualize_info/          Study IV (INLP ablation)
└── other/                         supplementary analyses (baselines, similarity of materials, typology, syntax, descriptives)
```

Directories or files marked "not tracked" are excluded through `.gitignore` because of their size (the full set of embeddings is ~200 GB); see [What is not included](#what-is-not-included).

## Study I

### 1. fMRI time series (`time_series_corr.py`)

Input: `data/Alice_Story_TimeSeries.csv` (one row per participant x fROI; columns `T_1 … T_142` are the 142 volumes of the passage-listening run), plus the analogous files for the multiple-demand network (`_MD.csv`) and for fROIs defined with the native-language localizer (`_native.csv`). The script (i) drops the silence periods at the beginning and end of the run (130 TRs remain), (ii) shifts the time series by 3 TRs (6 s) to account for the hemodynamic delay, (iii) averages across voxels within each fROI and across the five language fROIs, (iv) computes the inter-participant correlations used to select the 12 reliable languages (Figure 4A, `plots/time-series-corr.svg`) and the within- vs. between-language comparison of SI S10, and (v) stores the data as pickled dictionaries `data/dict_fROI*` (per-fROI time series, used by `fit_encoding.py`) and `data/dict_fMRI*` (network-level averages). The suffixes `_rh`, `_md` and `_native` refer to the right-hemisphere homotopes, the MD network and the native-language fROIs.

### 2. Transcriptions (`whisper_timestamped.py`)

Transcribes the audio files with Whisper-timestamped (largest Whisper model) and writes `transcribed/<lang>.csv` with one row per word (`text`, `start`, `end`, in seconds). Transcriptions are provided for the 12 languages used in the paper and for the other languages of the original dataset that were not analysed (excluded on the basis of the reliability criterion).

### 3. Embeddings (`get_model_embeddings.py`)

Extracts hidden states from every layer of the 20 MNNLMs (Table 1 of the paper; HuggingFace identifiers are in the script) and derives word-level embeddings by averaging over sub-word tokens. Bidirectional models are run with a sliding window of 100 words so that each word's representation only depends on the preceding context. Output: `embeddings/<model>_<lang>` (pickled dict `layer -> array[words x units]`). Note that XGLM is only available for 5 of the 12 languages and mGPT for 10 (the remaining languages were absent from their pre-training data). `get_embeddings_chunk_context.py` is the variant used in SI S11, where the context is reset at every cross-validation fold boundary (output in `embeddings/split_context/`).

### 4. Encoding models (`fit_encoding.py`)

Aligns word embeddings with the fMRI time series (TR-level embeddings = mean of the words pronounced within each 2-s TR), z-scores predictors and responses with parameters estimated on the training data, and fits ridge regression (`himalaya` `KernelRidgeCV`, 10 log-spaced alphas chosen with nested cross-validation). Cross-validation is 10-fold with sequential (non-shuffled) splits and is always performed across participants (train on 90% of participant A, test on the held-out 10% of participant B, and vice versa). Run with `python fit_encoding.py --mode <mode>`:

| `--mode` | What it fits | Output files in `results/` |
|---|---|---|
| `within` | WITHIN encoding models + circular-shift baselines (offsets 26, 52, 78, 104 TRs) | `monolingual_<model>_<froi>`, `monolingual_<model>_<froi>_circshift` |
| `across` | ACROSS models trained on N-1 languages (main analysis) and models trained on a single language (SI S14) | `multilingual_multitrain_<model>_<froi>`, `multilingual_<model>_<froi>` |
| `across-random` | circular-shift baselines for the ACROSS models | `multilingual_multitrain_<model>_<froi>_circshift`, `multilingual_<model>_<froi>_circshift` |
| `RH`, `MD` | same as above for the right-hemisphere language fROIs and the MD fROIs (Figure 2C) | prefixes `monolingual_rh_`, `multilingual_multitrain_rh_`, `monolingual_md_`, … |
| `native-within`, `native-across` | fROIs defined with the native-language localizer (SI S7) | prefix `monolingual_native_`, `multilingual_multitrain_native_` |
| `within-split`, `across-split` | embeddings with context reset at fold boundaries (SI S11) | suffix `_SPLIT` |

`<froi>` is one of `Lang_LH_AntTemp`, `Lang_LH_IFG`, `Lang_LH_IFGorb`, `Lang_LH_MFG`, `Lang_LH_PostTemp` or `all` (average across the five fROIs); MD fROIs and RH fROIs use their own labels. Each results file is a pickled dict `layer -> DataFrame` with one row per (test) language and the Pearson correlation between predicted and observed responses. The raw regression weights that the script can optionally dump (`results/coefficients/`) are not tracked.

### 5. Statistics and figures

- `plot_encoding_mono.py` (WITHIN) and `plot_encoding_multi.py` (ACROSS): best-layer selection, significance against the circular-shift baselines (Fisher r-to-z differences per language combined with Stouffer's method, see Methods "Significance testing"), model-wise dot plots (Figure 2A: `plots/mono.svg`, `plots/multi_multitrain.svg`), results by language (SI S2: `plots/all_languages_*.svg`), spatial specificity (Figure 2C: `plots/spatial_specificity_*.svg`), median-layer results (SI S5), native-fROI results (SI S7) and the split-context comparison (SI S11: `plots/split_context_vs_standard_*.svg`). `plot_encoding_multi.py` also writes `results/mono_multi.csv` (best-layer WITHIN and ACROSS score per model), which several other scripts read.
- `plot_layerwise.py`: layer-wise curves per model family and across-model average (Figure 2B: `plots/layerwise*.svg`).
- `other/across_froi_pvalues.py`: pairwise comparisons between fROIs (SI S8); it uses the `load`/`get_best_layerwise` helpers of `plot_encoding_mono.py`, so run it in the same session.
- `perplexity/plot_perplexity.py`: relationship between next-word-prediction ability and encoding performance (Figure 2D: `plots/perplexity_*.svg`). It reads the raw perplexity estimates (`perplexity/PERPLEXITY_results_<model>.pkl`: per language, the cross-validated perplexity of a ridge classifier trained on the model's embeddings and of a randomized baseline), residualizes them against the random baseline as described in the Methods, and writes `results/perplexity_results.csv` (per model) and `perplexity/perplexity_results_by_language.csv` (per language, used in SI S15).

The assembled multi-panel figures of the paper (`plots/panel_1.svg`, `plots/panel_2_3_sig.svg`, `plots/reliability-panel.svg`, …) were composed manually in a vector-graphics editor from the individual svg files produced by the scripts.

## Study II

### Newly collected data and evaluation (`confirmatory/`)

- `confirmatory_get_timeseries.py`: extracts the language-network time series from `confirmatory/data/confirmatory_time_series.csv` (3 participants x 3 passages per language, 9 languages), computes the inter-participant correlations (Figure 4B: `plots/time-series-corr-confirmatory.svg`; SI S10), applies the reliability-based selection of participant pairs described in the Methods, averages across participants and writes `confirmatory/data/dict_fMRI` (network level) and `confirmatory/data/dict_fROI` (per fROI). Per-passage time series plots are in `confirmatory/data/plot_ts/`.
- `whisper_timestamped.py`, `get_model_embeddings.py`: as in Study I, for the three passages (`confirmatory/transcribed/Passage_<n>/<lang>.csv`; embeddings in `confirmatory/embeddings/`, not tracked).
- `registered_model.py`: trains the encoding models on the **Study I** data (all 12 languages) and stores them, together with the normalization parameters and the circular-shift baseline models, in `confirmatory/registered_models/main/`.
- `confirmatory_encoding.py`: loads the registered encoding models from the four training sources (`confirmatory/registered_models/{main,natstor,pereira,control}/`), applies them zero-shot to the new data, evaluates significance against the corresponding randomized models, and produces Figure 3A (`plots/study2_summary.svg`, `plots/sudy2_*.svg`) and the per-language results of SI S3 (`plots/study2_all_langs_*.svg`). Cached results: `confirmatory_results.csv`, `confirmatory_results_aggregated.csv`.
- `confirmatory_encoding_fROI.py`: same evaluation for the individual fROIs (SI S8; `plots/study2_froi_*.svg`; cached in `confirmatory_results_froi*.csv`).
- `other/within_encoding_study2.py`: WITHIN encoding models fit directly on the Study II data (SI S4; cached in `other/within_results_cv.csv` and `other/confirmatory_within_all.csv`).

The pickled models in `confirmatory/registered_models/<source>/` follow the naming `<model>[_<fROI>][_random_<k>]`; the corresponding scalers are in `normaliz_params/`.

### Training datasets (`additional_analyses/`)

Each subfolder contains the (pre-processed) fMRI data, the embedding-extraction and model-training scripts, and writes its registered encoding models into `confirmatory/registered_models/<source>/`.

- **NatStories** (`NaturalStories/`): `whisper_NatStories.py` (transcription of the 9 stories; `transcribed/`), `natstor_fmri_timeseries.py` (time series extraction from `NaturalStories_TimeSeries_20220615.csv` [not tracked], split-half reliability of SI S12, time series at different HRF lags saved in `response/d_shift_<k>`), `get_natstor_embeddings.py`, `fit_encoding_natstor.py` (cross-story encoding for the choice of the TR shift, SI S9, `results/cross_story_<k>shift_xglm_small`; training and saving of the registered models).
- **Pereira2018** (`pereira/`): `pereira_encoding_registered_model.py` (sentence-level responses averaged across participants in `pereira_averaged.csv`, derived from the original data `Pereira.parquet.gzip` [not tracked]; shuffled baselines with four seeds).
- **Tuckute2024** (`control/`): `control_encoding_register_model.py` (responses in `data/brain-lang-data_participant_20230728.csv` from Tuckute et al., 2024a; averaged responses in `tuckute_averaged.csv`).

## Study III (`additional_analyses/{control,pereira}/perturbation/`)

- `perturbation_materials.py`: builds the perturbed versions of the English sentences (information-loss manipulations based on NLTK POS tags, local word swaps with a seeded random generator, full reversal, and paraphrases generated with Llama-4 Maverick through the Together API; the API key has been removed from the script). The materials are stored in `perturbation_dict`.
- `perturbation_encoding.py`: extracts embeddings for each perturbation at each model's best layer and trains the encoding models (`registered_models/<model>/<perturbation>`, scalers in `registered_models/normaliz_params/<model>/`). For Pereira2018 the models are stored under `pereira/perturbation/perturbation/registered_models/`.
- `embeddings_sim_perturbation.py` / `embeddings_sim_perturbation_pereira.py`: cosine similarity between intact and perturbed sentence embeddings (`similarity_outputs*/`).
- `control/perturbation/perturbation_encoding_Pereira_Tuckute.py`: evaluates all perturbed models of **both** datasets on the Study II data, tests each perturbation against the intact model (Fisher r-to-z differences combined with Stouffer's method, Bonferroni-corrected), and draws Figure 3B (`plots/perturb.svg`) and the similarity-vs-transfer plot (`plots/similarity_vs_encoding.svg`). Cached results: `confirmatory/perturbation_results*.csv`.
- `perturb.sh`: example SLURM scripts.

## Study IV (`additional_analyses/residualize_info/`)

- `prepare_features.py`: sentence-level meaning features (seven human ratings from Tuckute et al., 2024a) and form features (PCFG surprisal, grammaticality and frequency ratings, dependency length via spaCy, word frequency via `wordfreq`, word length) for the Tuckute2024 sentences (`features/`).
- `residualize_encoding.py`: for each model, extracts the best-layer sentence embeddings, runs the regression variant of INLP to remove the meaning or the form features, trains the intact / meaning-ablated / form-ablated encoding models with 5-fold cross-validation (`cv_results/`, SI S17), and stores the models, projection matrices and decodability diagnostics (`registered_models/`, `diagnostics/`).
- `test_residualize.py`: zero-shot transfer of the three model types to the Study II data, statistics (`results/`) and the panels of Figure 3C and Supplementary Figure 13 (`plots/residualize*.svg`).
- `residualize.sh`: example SLURM script.

## Supplementary information: where each analysis lives

| SI section | Script(s) | Output |
|---|---|---|
| S2 Results in individual languages (Study I) | `plot_encoding_mono.py`, `plot_encoding_multi.py` | `plots/all_languages_*.svg` |
| S3 Results in individual languages (Study II) | `confirmatory/confirmatory_encoding.py` | `plots/study2_all_langs_*.svg` |
| S4 WITHIN encoding on Study II data | `confirmatory/other/within_encoding_study2.py` | `plots/study2_study1_*.svg` |
| S5 Median layer | `plot_encoding_mono.py`, `plot_encoding_multi.py` (median-layer option) | `plots/median_*.svg` |
| S6 Frequency, length and timing baselines | `other/FL_baseline.py` | printed statistics |
| S7 Native fROIs | `fit_encoding.py --mode native-*`, plotting scripts | `plots/*native*.svg` |
| S8 Individual fROIs | `other/across_froi_pvalues.py`, `confirmatory/confirmatory_encoding_fROI.py` | `plots/study2_froi_*.svg` |
| S9 Optimal TR shift | `additional_analyses/NaturalStories/fit_encoding_natstor.py` | `NaturalStories/results/cross_story_*` |
| S10 Within- vs. between-language time series correlation | `time_series_corr.py`, `confirmatory/confirmatory_get_timeseries.py` | `plots/time-series-corr*.svg` |
| S11 Context contamination | `get_embeddings_chunk_context.py`, `fit_encoding.py --mode *-split`, plotting scripts | `plots/split_context_*.svg` |
| S12 NatStories reliability | `additional_analyses/NaturalStories/natstor_fmri_timeseries.py` | printed / plotted |
| S13 Syntactic complexity | `other/dep_length/dep_length.py` (Study I passage text in `other/dep_length/alice_punct.txt`) | Supplementary Figure 11 |
| S14 Single-language training and typology | `other/transfer_single_languages.py` (pairwise transfer, `lang2vec` distances), `other/xling_matrix_all_models.py` (all models, reliability vs. transfer, within vs. across) | `plots/xling_matrix*.svg`, `plots/reliability_vs_*.svg`, `plots/within_vs_across_infoxlm.svg` |
| S15 Language-level analyses | `other/language_level_analysis.py` | printed statistics |
| S16 Language resources | `other/language_resources_mC4.py` | boxplot and statistics |
| S17 Ablation within English | `additional_analyses/residualize_info/test_residualize.py` | `residualize_info/plots/residualize_cv*.svg` |
| Figure 1C (dataset descriptives and similarity of materials) | `other/dataset_descriptives.py`, `other/similarity_materials/similarity_exp1.py` (TR-level embedding similarity), `other/similarity_materials/similarity_exp1_allwords.py` (word-level cosine similarity and Jaccard word overlap; cached in `sim_df/` and `study2_sim_data*.pkl`) | printed statistics |
| Table 1 (parameter counts) | `other/model_parameter_counts.py` | printed |

## Reproducibility notes

**Working directory.** The scripts were written for interactive use. When executed as scripts they resolve all paths relative to their location in the repository; when run interactively (e.g. cell by cell), set the working directory to the folder indicated in the comment at the top of each script (the repository root for the Study I scripts, `confirmatory/` for the Study II scripts, and so on).

**Order of execution.** Study I: `whisper_timestamped.py` → `get_model_embeddings.py` → `time_series_corr.py` → `fit_encoding.py` (all modes) → plotting scripts. Study II: the three training-dataset pipelines and `confirmatory/registered_model.py` → `confirmatory/confirmatory_get_timeseries.py` → `confirmatory/get_model_embeddings.py` → `confirmatory/confirmatory_encoding.py`. Studies III and IV depend on the Study II data and on the best layers identified in Study I (hard-coded in the scripts). Intermediate results are cached in the repository (results pickles, csv files, registered models), so the plotting and statistics scripts can be run without recomputing the embeddings or refitting the models.

**Environment.** The code was run on Ubuntu 22.04 with Python 3.9 (embedding extraction and model fitting were run on a GPU cluster; a GPU is optional but recommended). Main dependencies and the versions used: numpy 1.26, pandas 1.5, scipy 1.11, scikit-learn 1.2, statsmodels 0.14, torch 1.12, transformers 4.45.2, himalaya 0.4.6, matplotlib 3.6, seaborn 0.12, adjustText 0.8, tqdm, nltk (POS tagging, Study III), spacy with `en_core_web_sm` (Studies IV and SI S13), wordfreq 3.0 (SI S6, Study IV), lang2vec (SI S14), together (paraphrase generation, Study III), whisper_timestamped (https://github.com/linto-ai/whisper-timestamped). The SLURM scripts (`encoding.sh`, `perturb.sh`, `residualize.sh`) are examples and need to be adapted to your cluster.

**Pickle compatibility.** The results were written with numpy 2; the plotting scripts include a small shim (`sys.modules['numpy._core.numeric'] = np.core.numeric`) so that they can be loaded with numpy 1.x.

**Minimal demo.** A minimal working example of the Study I encoding approach (WITHIN and ACROSS) is available on OSF (`encoding_demo.py`, with the DistilmBERT embeddings and the expected output in `demo_output.txt`). It runs in about 5 minutes on a laptop.

### What is not included

- **Embeddings** (`embeddings/`, `confirmatory/embeddings/`, `additional_analyses/*/embeddings/`): ~200 GB in total. DistilmBERT embeddings for Study I are provided on OSF; all others can be recomputed with the `get_*embeddings*.py` scripts.
- **Audio stimuli** (`sound_data/`, `confirmatory/sound_data/`, `additional_analyses/NaturalStories/sound_data/`): on OSF.
- **Source fMRI datasets of the training sets** that are distributed by the original studies: the Pereira2018 data (`additional_analyses/pereira/Pereira.parquet.gzip`; Pereira et al., 2018) and the NatStories time series (`additional_analyses/NaturalStories/NaturalStories_TimeSeries_20220615.csv`). The derived files used by the encoding models (`pereira_averaged.csv`, `response/d_shift_*`) are included.
- **Raw regression weights** (`results/coefficients/`).
- **The script that computed the raw perplexity estimates** (`perplexity/PERPLEXITY_results_<model>.pkl`) from the Wikipedia corpus (see Methods, "Next-word prediction performance") is not part of this repository; the resulting estimates are included and all downstream steps are reproducible from them.

## Citation

de Varda, A. G., Malik-Moraleda, S., Tuckute, G., & Fedorenko, E. (in press). Multilingual computational models capture a shared meaning component in brain responses across 21 languages. *Nature Neuroscience*.

Study I fMRI data: Malik-Moraleda, S., Ayyash, D., et al. (2022). An investigation across 45 languages and 12 language families reveals a universal language network. *Nature Neuroscience*, 25(8), 1014–1019. Tuckute2024 data: Tuckute, G., et al. (2024). Driving and suppressing the human language network using large language models. *Nature Human Behaviour*, 8(3), 544–561. Pereira2018 data: Pereira, F., et al. (2018). Toward a universal decoder of linguistic meaning from brain activation. *Nature Communications*, 9(1), 963. NatStories: expanded from Blank, I., Kanwisher, N., & Fedorenko, E. (2014). *Journal of Neurophysiology*, 112(5), 1105–1118.
