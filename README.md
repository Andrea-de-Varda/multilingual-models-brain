# A shared meaning component across languages in the human brain

Code and data for

> de Varda, A. G., Malik-Moraleda, S., Tuckute, G., & Fedorenko, E. A shared meaning component across languages in the human brain. *Nature Neuroscience* (in press).

We fit encoding models that predict fMRI responses of the language network from the representations of 20 multilingual language models (MNNLMs), and show that these models transfer zero-shot to languages they were never trained on (21 languages in total). This repository contains everything needed to reproduce the analyses, figures and statistics of the paper: the processed fMRI time series, the stimulus transcriptions, the audio stimuli (downloaded separately, see [Data](#data)), all analysis code, the cached intermediate results, and the Source Data of the figures.

**Contents:** [Quick start](#quick-start) · [Installation](#installation) · [Data](#data) · [Repository layout](#repository-layout) · [Study I](#study-i) · [Study II](#study-ii) · [Study III](#study-iii) · [Study IV](#study-iv) · [Where each figure and supplementary analysis comes from](#where-each-figure-and-supplementary-analysis-comes-from) · [Reproducibility notes](#reproducibility-notes) · [Citation](#citation)

## Quick start

The fastest way to check the approach is the **replication demo** in [`demo/`](demo/): it recomputes, from the transcriptions and fMRI time series in this repository, the Study I WITHIN and ACROSS encoding results for one model (DistilmBERT) and compares them with the values in the paper (Figure 2A/B).

- Interactive notebook: [`demo/encoding_demo.ipynb`](demo/encoding_demo.ipynb) (runs locally or on Google Colab, [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Andrea-de-Varda/multilingual-models-brain/blob/revision/demo/encoding_demo.ipynb)).
- Command line: `python demo/run_demo.py` (add `--full` to evaluate all layers in the ACROSS condition).

It takes about 5 minutes on a laptop CPU to extract the embeddings and 1–10 minutes for the encoding models (GPU or CPU). Expected output: WITHIN mean r per layer 0.030, 0.084, 0.109, 0.131, 0.143, 0.156, 0.173 (best layer 6, Stouffer z = 3.32, p = 9.0e-04) and ACROSS r = 0.140 at layer 4, as in the paper (Source Data Fig. 2).

Every other number in the figures can be regenerated from the cached results in this repository without recomputing embeddings or refitting models, e.g. `python plot_encoding_mono.py` (Figure 2A top, 2C top) or `python source_data/extract_source_data.py` (the Source Data tables of Figures 2–3 and Extended Data Figure 1).

## Installation

The code was tested with Python 3.11 on Linux (Ubuntu). Install PyTorch for your hardware first (https://pytorch.org/get-started/locally/), then the other dependencies:

```
conda create -n mnnlm python=3.11 && conda activate mnnlm
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128   # or .../whl/cpu
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

A GPU is optional. It speeds up embedding extraction and the encoding models (the analyses of the paper were run on NVIDIA A100 GPUs); the demo and all plotting/statistics scripts run on a CPU. Installation takes a few minutes. See [Reproducibility notes](#reproducibility-notes) for the package versions used in the paper.

## Data

**In this repository (git):** fMRI time series of all datasets at the fROI level (`data/`, `confirmatory/data/`, `additional_analyses/*/`), word-by-word transcriptions with timestamps (`transcribed/`, `confirmatory/transcribed/`, `additional_analyses/NaturalStories/transcribed/`), perturbed stimuli of Study III, sentence features of Study IV, all fitted ("registered") encoding models and cached results, and the Source Data.

**Large files (GitHub release `data-v1`):** the audio stimuli and two large source files are attached to the [`data-v1` release](https://github.com/Andrea-de-Varda/multilingual-models-brain/releases/tag/data-v1) of this repository, because they exceed what git can reasonably hold. Download them with

```
python download_data.py          # everything, about 4.2 GB
python download_data.py --list   # what is available
python download_data.py audio_study1.zip   # one archive only
```

The script (Python standard library only) downloads each archive, extracts it to the right place in the repository and checks the SHA-256 checksum of every file against [`data_manifest.json`](data_manifest.json). `python download_data.py --verify-only` re-checks files already in place.

| Archive | Size | Content | Needed for |
|---|---|---|---|
| `audio_study1.zip` | 716 MB | Study I passages, 12 languages + English (`sound_data/<Language>/<Language>_Intact_Long/<Language>.wav`) | re-running the Study I transcriptions; English: Figure 1C word overlap |
| `audio_study2.zip` | 1.37 GB | Study II passages, 3 per language, 9 languages + English (`confirmatory/sound_data/Passage_<n>/<Language>.wav`) | re-running the Study II transcriptions |
| `audio_natstories.zip` | 267 MB | the 9 NatStories stories (`additional_analyses/NaturalStories/sound_data/<n>.wav`, numbered 1–7, 9, 10; story 8 has no fMRI data) | re-running the NatStories transcriptions |
| `natstories_timeseries.zip` | 50 MB | NatStories fROI time series of all participants | `natstor_fmri_timeseries.py` (NatStories responses and reliability, SI S12) |
| `pereira2018_parquet.zip` | 1.81 GB | Pereira2018 voxel-level responses in the fROIs | re-deriving `pereira_averaged.csv` and its reliability (needs ~6 GB of RAM) |

None of these files is needed to run the demo, the plotting scripts or the statistics: the derived files they feed (transcriptions, time series, averaged responses) are in git.

**Not distributed: model embeddings.** The contextual embeddings of the 20 models take about 200 GB. They are deterministic functions of the transcriptions and can be recomputed with the `get_*embeddings*.py` scripts (embeddings recomputed with the versions in `requirements.txt` match the ones used in the paper to within float32 precision). Scripts that need embeddings say so below.

**Source datasets.** Study I data are from Malik-Moraleda, Ayyash, et al. (2022), Tuckute2024 from Tuckute et al. (2024), Pereira2018 from Pereira et al. (2018), NatStories expanded from Blank et al. (2014); the Study II data were collected for this paper. Please cite the original studies when using these data.

## Repository layout

```
.
├── demo/                           replication demo for one model (notebook + command-line script)
├── download_data.py, data_manifest.json   download and verification of the large files (GitHub release)
├── requirements.txt
├── source_data/                    Source Data of Figures 1-3 and Extended Data Figure 1 (xlsx) + script that regenerates them
│
├── data/                           Study I fMRI time series (csv) and per-participant/per-fROI dictionaries
├── transcribed/                    Study I transcriptions (word, onset, offset), one file per language
├── whisper_timestamped.py          Study I: transcription of the audio with word timestamps
├── get_model_embeddings.py         Study I: embeddings from the 20 models
├── get_embeddings_chunk_context.py Study I: embeddings with the context reset at fold boundaries (SI S11)
├── time_series_corr.py             Study I: time series, reliability, language selection (Extended Data Fig. 1A, SI S10)
├── fit_encoding.py                 Study I: all encoding models (WITHIN, ACROSS, baselines, controls); encoding.sh = example SLURM job
├── results/                        Study I encoding results (one pickle per model x fROI x condition)
├── plot_encoding_mono.py           Study I: WITHIN statistics and figures
├── plot_encoding_multi.py          Study I: ACROSS statistics and figures
├── plot_layerwise.py               Study I: layer-wise figure (Figure 2B)
├── plot_supplementary_variants.py  Study I: median layer, native-language fROIs, single-language training (SI S5, S7, S14)
├── perplexity/                     next-word prediction (Figure 2D, SI S15): Colab notebook, results, figure script
│
├── confirmatory/                   Study II: new data in 9 languages, registered models, transfer evaluation
├── additional_analyses/
│   ├── NaturalStories/             Study II training dataset NatStories (+ TR-shift analysis, SI S9)
│   ├── pereira/                    Study II training dataset Pereira2018 (+ Study III perturbations)
│   ├── control/                    Study II training dataset Tuckute2024 (+ Study III perturbations and Figure 3B)
│   └── residualize_info/           Study IV: INLP ablation of meaning and form information
├── other/                          supplementary analyses (SI S6, S13, S14, S15, S16, Figure 1C, Table 1)
├── plots/                          figure panels written by the scripts; plots/final/ = published figures
└── tools/make_release_assets.py    (maintainers) builds the release archives and data_manifest.json
```

**Working directory.** The scripts were developed for interactive (cell-by-cell) use. Run as `python <script>.py`, each script sets its own working directory, so they can be launched from anywhere. When running them interactively, set the working directory indicated at the top of each script (the repository root for Study I, `confirmatory/` for Study II scripts, the script's folder for the other datasets).

**Caching.** Several scripts store intermediate results and reuse them when present: `fit_encoding.py` skips models whose results exist; `confirmatory/confirmatory_encoding.py` (`confirmatory_results.csv`), `perturbation_encoding_Pereira_Tuckute.py` (`confirmatory/perturbation_results*.csv`) and `test_residualize.py` (`residualize_info/results/test_results.csv`) read their cached transfer results and recompute only the statistics and figures. To recompute from scratch (this needs the Study II embeddings), delete or move the corresponding files first. `confirmatory_encoding_fROI.py` and `confirmatory/other/within_encoding_study2.py` recompute their results and then read the cached csv files written when the paper was prepared.

## Study I

12 languages, 2 participants each, one ~4.5-min passage of *Alice in Wonderland* (data from Malik-Moraleda, Ayyash, et al., 2022).

1. **fMRI time series** — `time_series_corr.py` reads `data/Alice_Story_TimeSeries.csv` (one row per participant x fROI, 142 volumes), removes the 12-s silent periods, shifts the responses by 3 TRs (6 s) for the hemodynamic delay (130 volumes remain), averages the five language fROIs into a network-level response (`all`), computes the inter-participant correlations used to select the 12 languages (Extended Data Figure 1A, `plots/time-series-corr.svg`) and the within- vs. between-language comparison (SI S10), and writes `data/dict_fROI` (per participant and fROI; used by the encoding models). The same is done for the right-hemisphere homotopes (`_rh`), the multiple-demand network (`_md`, from `Alice_Story_TimeSeries_MD.csv`) and fROIs defined with the native-language localizer (`_native`).
2. **Transcriptions** — `whisper_timestamped.py` (Whisper large; needs `audio_study1.zip` and `whisper-timestamped`) writes `transcribed/<lang>.csv` (word, start, end in seconds). The transcriptions used in the paper are in the repository.
3. **Embeddings** — `get_model_embeddings.py` extracts the hidden states of every layer of the 20 models (Table 1; HuggingFace identifiers in the script) and averages sub-word tokens into word embeddings; bidirectional models see a sliding window of the 100 preceding words. XGLM covers 5 of the 12 languages and mGPT 10. Output: `embeddings/<model>_<lang>` (not distributed, ~200 GB). `get_embeddings_chunk_context.py`: same, with the context reset at the boundaries of the 10 cross-validation folds (SI S11).
4. **Encoding models** — `fit_encoding.py --mode <mode> [--model <model>]` (`--model`, e.g. `xglm_small`, restricts the run to one of the 20 models, so that each model can be a separate cluster job). TR-level embeddings (mean of the words whose offset falls in each 2-s volume) are mapped onto the responses with ridge regression (himalaya, 10 log-spaced alphas chosen by nested 5-fold cross-validation), with 10-fold sequential cross-validation, always across participants. Needs the embeddings.

   | `--mode` | Analysis | Output in `results/` |
   |---|---|---|
   | `within` | WITHIN (train on one participant, test on the other, both directions) + circular-shift baselines (26, 52, 78, 104 TRs) | `monolingual_<model>_<fROI>`, `..._circshift` |
   | `across` | ACROSS (train on 11 languages, test on the 12th; Figure 2A) and single-language training (SI S14) | `multilingual_multitrain_<model>_<fROI>`, `multilingual_<model>_<fROI>` |
   | `across-random` | circular-shift baselines of both | `..._circshift` |
   | `RH`, `MD` | right-hemisphere and multiple-demand fROIs (Figure 2C) | `monolingual_{rh,md}_...`, `multilingual_multitrain_{rh,md}_...` |
   | `native-within`, `native-across` | fROIs from the native-language localizer (SI S7) | `monolingual_native_...`, `multilingual_multitrain_native_...` |
   | `native-random` | circular-shift baselines for the native-language fROIs (Supplementary Figure 4) | `..._native_..._circshift` |
   | `within-split`, `across-split` | embeddings with context reset at fold boundaries (SI S11) | `..._SPLIT` |

   `<fROI>` is `all` (network average) or one of `Lang_LH_{AntTemp,IFG,IFGorb,MFG,PostTemp}` (RH and MD fROIs have their own labels). Each file is a pickled dict `layer -> DataFrame` with one row per (test) language: `m1`, `m2` (the two training directions) and `m` (their mean) for WITHIN, `r` for ACROSS; circular-shift files are dicts `shift -> layer -> DataFrame`.
5. **Statistics and figures** (read the cached results; no embeddings needed):
   - `plot_encoding_mono.py` / `plot_encoding_multi.py`: best layer per model, significance against the circular-shift baselines (Fisher r-to-z difference per language with n = 130, combined with Stouffer's method across languages and then across the four shifts; the best layer is selected separately for the real and for each shifted model), Figure 2A (`plots/mono.svg`, `plots/multi_multitrain.svg`, legend `plots/fROI_legend_multicol.svg`), Figure 2C (`plots/spatial_specificity_*.svg`), Supplementary Figure 1A (`plots/all_languages_*.svg`), Supplementary Figure 7 (`plots/split_context_vs_standard_*.svg`). `plot_encoding_multi.py` also writes `results/mono_multi.csv` (best-layer scores per model, used by Figure 2D).
   - `plot_layerwise.py`: Figure 2B (`plots/layerwise*.svg`).
   - `plot_supplementary_variants.py`: Supplementary Figures 3 (median layer), 4 (native-language fROIs) and 10A (single-language training), with the numbers reported in SI S5, S7 and S14.
   - `perplexity/plot_perplexity.py`: Figure 2D (`plots/perplexity_*.svg`). The next-word-prediction estimates (`perplexity/PERPLEXITY_results_<model>.pkl`) were computed with the Google Colab notebook [`perplexity/Linear_readout.ipynb`](perplexity/Linear_readout.ipynb) (Wikipedia text in each language, last-layer embeddings, ridge classifier, held-out perplexity of the real and of a shuffled-label baseline); the notebook is kept as it was run on Colab, with its outputs, which are identical to the pickles. The script regresses out the baseline perplexity and writes `results/perplexity_results.csv` and `perplexity/perplexity_results_by_language.csv`.

## Study II

Encoding models trained on four datasets and transferred zero-shot to new fMRI data from 9 languages (3 participants, 3 passages each).

- **New data** (`confirmatory/`): `confirmatory_get_timeseries.py` reads `confirmatory/data/confirmatory_time_series.csv`, computes the inter-participant correlations (Extended Data Figure 1B, `plots/time-series-corr-confirmatory.svg`; SI S10), selects the reliable participants per passage, averages them and writes `confirmatory/data/dict_fMRI` (network level) and `dict_fROI` (per fROI). `whisper_timestamped.py` and `get_model_embeddings.py` as in Study I (`confirmatory/transcribed/Passage_<n>/`; embeddings in `confirmatory/embeddings/`, not distributed).
- **Training datasets** — each script trains, for each model, the encoding model at the model's best ACROSS layer from Study I, with its scalers and the four randomized baselines, and stores them in `confirmatory/registered_models/<source>/`:
  - Study I data: `confirmatory/registered_model.py` → `main/`;
  - NatStories: `additional_analyses/NaturalStories/` — `natstor_fmri_timeseries.py` (time series of the 9 stories and their split-half reliability, SI S12; needs `natstories_timeseries.zip`), `whisper_NatStories.py`, `get_natstor_embeddings.py`, `fit_encoding_natstor.py` (choice of the 3-TR shift, SI S9, `results/cross_story_<k>shift_xglm_small`; registered models → `natstor/`);
  - Pereira2018: `additional_analyses/pereira/pereira_encoding_registered_model.py` (sentence responses averaged across participants in `pereira_averaged.csv`, derived from the parquet file in `pereira2018_parquet.zip`; → `pereira/`);
  - Tuckute2024: `additional_analyses/control/control_encoding_register_model.py` (responses in `data/brain-lang-data_participant_20230728.csv`, from Tuckute et al., 2024; → `control/`).
- **Transfer** — `confirmatory/confirmatory_encoding.py`: applies the registered models to the new data, tests them against the randomized baselines and draws Figure 3A (`plots/study2_summary.svg`, `plots/sudy2_*.svg`) and Supplementary Figure 1B (`plots/study2_all_langs_*.svg`); cached in `confirmatory_results*.csv`. `confirmatory_encoding_fROI.py`: per fROI (SI S8, `plots/study2_froi_*.svg`). `confirmatory/other/within_encoding_study2.py`: WITHIN encoding on the new data (SI S4, Supplementary Figure 2).

## Study III

Perturbed versions of the English training sentences of Pereira2018 and Tuckute2024 (`additional_analyses/{control,pereira}/perturbation/`).

- `perturbation_materials.py` builds `perturbation_dict` (intact sentences + 13 perturbations: content words; nouns, verbs and adjectives; nouns and verbs; nouns; verbs; function words — from NLTK part-of-speech tags; 1–5 local swaps with a seeded random generator; full reversal; paraphrases generated with Llama-4-Maverick through the Together API). The paraphrases used in the paper are stored in `perturbation_dict` and reused unless `TOGETHER_API_KEY` is set.
- `perturbation_encoding.py` trains the encoding models on the perturbed sentences (best ACROSS layer of each model) → `registered_models/` (Pereira2018: `pereira/perturbation/perturbation/registered_models/`). Needs internet access to download the models.
- `control/perturbation/perturbation_encoding_Pereira_Tuckute.py` evaluates the perturbed models of both datasets on the Study II data, compares each perturbation with the intact models (Fisher r-to-z, Stouffer, Bonferroni across the 13 perturbations) and draws Figure 3B (`plots/perturb.svg`, `plots/legend_perturb.svg`); cached in `confirmatory/perturbation_results*.csv`.

## Study IV

INLP ablation of meaning- or form-related information from the embeddings (`additional_analyses/residualize_info/`).

- `prepare_features.py`: the 7 meaning and 7 form features of the 1,000 Tuckute2024 baseline sentences (`features/`).
- `residualize_encoding.py`: for each model, sentence embeddings at the best layer, INLP for each feature set, encoding models on intact / meaning-ablated / form-ablated embeddings with 5-fold cross-validation (`cv_results/`), registered models and projections (`registered_models/`), decodability across INLP steps (`diagnostics/`). GPU recommended; `residualize.sh` is an example SLURM job.
- `test_residualize.py`: transfer to the Study II data and statistics (`results/`), Figure 3C (`plots/residualize_cross_diagnostics.svg`, `plots/residualize_transfer.svg`) and Supplementary Figure 11 (`plots/residualize_cv.svg`; SI S17).

## Where each figure and supplementary analysis comes from

| Paper item | Script(s) | Output |
|---|---|---|
| Figure 1C, counts | `other/dataset_descriptives.py` | printed |
| Figure 1C, reliability | Study I: `time_series_corr.py`; Study II: `confirmatory/confirmatory_get_timeseries.py`; NatStories: `additional_analyses/NaturalStories/natstor_fmri_timeseries.py`; Pereira2018: `additional_analyses/pereira/pereira_encoding_registered_model.py`; Tuckute2024: `additional_analyses/control/control_encoding_register_model.py` | printed |
| Figure 1C, similarity and word overlap | `other/similarity_materials/similarity_exp1_allwords.py` (cached per model in `sim_df/`) | printed |
| Table 1 (parameters) | `other/model_parameter_counts.py` | printed |
| Figure 2A, 2C | `plot_encoding_mono.py`, `plot_encoding_multi.py` | `plots/mono.svg`, `plots/multi_multitrain.svg`, `plots/spatial_specificity_*.svg` |
| Figure 2B | `plot_layerwise.py` | `plots/layerwise*.svg` |
| Figure 2D | `perplexity/plot_perplexity.py` (estimates from `perplexity/Linear_readout.ipynb`) | `plots/perplexity_*.svg` |
| Figure 3A | `confirmatory/confirmatory_encoding.py` | `plots/study2_summary.svg`, `plots/sudy2_*.svg` |
| Figure 3B | `additional_analyses/control/perturbation/perturbation_encoding_Pereira_Tuckute.py` | `plots/perturb.svg` |
| Figure 3C | `additional_analyses/residualize_info/test_residualize.py` | `residualize_info/plots/residualize_{cross_diagnostics,transfer}.svg` |
| Extended Data Figure 1 | `time_series_corr.py` (A), `confirmatory/confirmatory_get_timeseries.py` (B) | `plots/time-series-corr*.svg` |
| Source Data (all figures) | `source_data/extract_source_data.py` | `source_data/csv/` |
| SI S2, Supplementary Figure 1A | `plot_encoding_mono.py`, `plot_encoding_multi.py` | `plots/all_languages_*.svg` |
| SI S3, Supplementary Figure 1B | `confirmatory/confirmatory_encoding.py` | `plots/study2_all_langs_*.svg` |
| SI S4, Supplementary Figure 2 | `confirmatory/other/within_encoding_study2.py` | `plots/within_encoding_study2.svg`, `plots/study2_study1_*.svg` |
| SI S5, Supplementary Figure 3 | `plot_supplementary_variants.py` | `plots/SI_median_layer.svg` |
| SI S6 | `other/FL_baseline.py` | printed |
| SI S7, Supplementary Figure 4 | `time_series_corr.py` (English vs native time series), `plot_encoding_multi.py` (model-wise correlations), `plot_supplementary_variants.py` | `plots/SI_native_frois.svg` |
| SI S8 | `plot_encoding_mono.py`, `plot_encoding_multi.py` (Study I); `confirmatory/confirmatory_encoding_fROI.py` (Study II) | `plots/study2_froi_*.svg` |
| SI S9, Supplementary Figure 5 | `additional_analyses/NaturalStories/fit_encoding_natstor.py` | `results/cross_story_*` |
| SI S10, Supplementary Figure 6 | `time_series_corr.py`, `confirmatory/confirmatory_get_timeseries.py` | |
| SI S11, Supplementary Figure 7 | `get_embeddings_chunk_context.py`, `fit_encoding.py --mode *-split`, `plot_encoding_*.py` | `plots/split_context_vs_standard_*.svg` |
| SI S12, Supplementary Figure 8 | `additional_analyses/NaturalStories/natstor_fmri_timeseries.py` | |
| SI S13, Supplementary Figure 9 | `other/dep_length/dep_length.py` (Study I English text in `alice_punct.txt`) | `plots/dependency_{length,types}.svg` |
| SI S14, Supplementary Figure 10 | `plot_supplementary_variants.py` (A), `other/transfer_single_languages.py` (B, typology; run it after `plot_encoding_multi.py` in the same session), `other/xling_matrix_all_models.py` (C, D, reliability, permutation test) | `plots/SI_single_language_training.svg`, `plots/xling_matrix.svg`, `plots/reliability_vs_pairwise_infoxlm_large.svg`, `plots/within_vs_across_infoxlm.svg` |
| SI S15 | `other/language_level_analysis.py` | printed |
| SI S16, Supplementary Table 1 | `other/language_resources_mC4.py` | printed |
| SI S17, Supplementary Figure 11 | `additional_analyses/residualize_info/test_residualize.py` | `residualize_info/plots/residualize_cv.svg` |

The published figures were assembled from these panels in a vector-graphics editor (`plots/final/`).

## Reproducibility notes

- **Package versions.** `requirements.txt` lists the versions with which this repository was tested end to end (Python 3.11, numpy 2.2, pandas 2.2, scikit-learn 1.7.1, himalaya 0.4.6, transformers 4.45.2, torch 2.7.1). The stored models and results were produced with numpy 2 and scikit-learn 1.6–1.7 (Study I embeddings with transformers 4.45.2); a few early files were written with older versions. Some scripts include `sys.modules['numpy._core.numeric'] = np.core.numeric`, which allows reading these pickles with numpy 1.x. PyTorch 1.x does not load on recent Linux systems and does not support recent GPUs; any PyTorch ≥ 2.1 gives the same embeddings up to floating-point precision.
- **Determinism.** Cross-validation splits are sequential and fixed; random procedures (shuffled baselines, swaps, bootstraps, permutations) are seeded. Recomputed embeddings match the stored ones up to float32 precision. The stored WITHIN (and single-language ACROSS) results were computed with himalaya's CPU (numpy) backend and are reproduced to ~1e-9 on CPU; with the GPU backend (the default of `fit_encoding.py` when a GPU is available) isolated folds can select a different alpha, so layer means agree to about 3 decimals. The stored N−1 ACROSS and split-context results were computed on GPU (float32). The demo reproduces the paper's DistilmBERT scores to 4 decimals. The INLP ablation (Study IV) runs in float32 and its ablated results vary slightly (|Δr| < 0.01) between runs.
- **Optional extras.** Transcription: `pip install git+https://github.com/linto-ai/whisper-timestamped` and the audio archives. Paraphrase generation: `pip install together` and a Together API key. `wordfreq[cjk]` (in `requirements.txt`) is needed for Korean and Mandarin word frequencies (SI S6). NLTK downloads its tagger models on first use.

## Citation

```
@article{deVarda2026shared,
  title   = {A shared meaning component across languages in the human brain},
  author  = {de Varda, Andrea Gregor and Malik-Moraleda, Saima and Tuckute, Greta and Fedorenko, Evelina},
  journal = {Nature Neuroscience},
  year    = {2026}
}
```

Data sources: Malik-Moraleda, S., Ayyash, D., et al. (2022). An investigation across 45 languages and 12 language families reveals a universal language network. *Nature Neuroscience*, 25, 1014–1019. · Tuckute, G., et al. (2024). Driving and suppressing the human language network using large language models. *Nature Human Behaviour*, 8, 544–561. · Pereira, F., et al. (2018). Toward a universal decoder of linguistic meaning from brain activation. *Nature Communications*, 9, 963. · Blank, I., Kanwisher, N., & Fedorenko, E. (2014). A functional dissociation between language and multiple-demand systems revealed in patterns of BOLD signal fluctuations. *Journal of Neurophysiology*, 112, 1105–1118.
