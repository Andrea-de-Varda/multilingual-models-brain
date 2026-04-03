> ⚠️ Note for Reviewers (Round 2)

This is the **`revision`** branch, which contains all updates for the **Round 3** revision of the manuscript. All changes relative to the Round 1 version (archived on OSF) can be tracked here through Git’s version history. All Round 1 materials will remain archived on OSF for reference.

***

## Brain encoding in 21 languages

This repository contains the code and data to reproduce the article *Multilingual Computational Models Capture a Shared Meaning Component in Brain Responses across 21 Languages*, currently under review.

The article includes three studies. Study I (12 languages) is based on previously collected data from Malik-Moraleda, Ayyash, et al. (2022), Study II (9 languages) is based on newly collected data, and Study III uses targeted perturbations to identify what linguistic features drive cross-lingual transfer. 

### Study I
Study I leverages existing fMRI data from a passage-listening task in 12 languages (Malik-Moraleda, Ayyash, et al., 2022). We trained fMRI encoding models to predict brain responses based on multiple languages and transferred them zero-shot to a new language on which they had not been trained.

The methodology is as follows:

 1. We extract fMRI responses from the language network (functionally defined; Fedorenko et al., 2010) at the fROI level, analyzing both individual fROIs and network-level averages. We analyze five core left-hemisphere language fROIs: posterior temporal, anterior temporal, inferior frontal gyrus, orbital part of inferior frontal gyrus, and middle frontal gyrus.
 2. We obtain written transcriptions for the passages that participants listened to with Whisper-timestamped, which also outputs word-by-word timestamps.
 3. We extract embeddings of the text in the various languages with multilingual neural network language models (MNNLMs, n = 20).
 4. We fit encoding models to predict fMRI activity from the MNNLM embeddings
    - In the *WITHIN* condition, we train and test the encoding models in each language, ensuring generalization across participants.
    - In the *ACROSS* condition, we train the encoding models in all languages but one, and test in that language, also ensuring generalization across participants.

The **code and data** supporting Study I are in the "level 0" of this repository, which includes the following scripts:

#### Code
- `time_series_corr.py` extracts averaged time-series, checks reliability, and stores the data for later use. It extracts time-series also for the right hemisphere language areas and the MD network, which are used as control areas.
- `whisper-timestamped.py` transcribes the audio (wav) files that were presented to the participants and produces csv files with all the words and timestamps. 
- `get_model_embeddings.py` generates contextual word embeddings from the 20 MNNLMs we considered. Note that some models (XGLM, mGPT) are not tested in all the 20 languages because some of them were absent from the models' pre-training data.
    - To simulate an auto-regressive setup and prevent access to future tokens, bidirectional models are tested with a sliding window of 100 words.
- `fit_encoding.py` fits linear encoding models (Ridge regression) predicting fMRI responses from the contextual word embeddings. To do so, it first aligns the embeddings with the fMRI responses based on the timestamps. Encoding models are either trained and tested within each language separately with cross-validation (*WITHIN* condition), or alternatively, the encoding models are fitted in all languages but one, and transferred zero-shot to that language (*ACROSS* condition). Both conditions ensure generalization across participants.
    - This is done for each model × layer × language combination
    - This is performed separately for individual fROIs, the (standard) LH language areas, the homotropic RH language areas, and the MD network.
- `fit_encoding_random.py` and `fit_encoding_chunk_context.py` do the same thing but either using circular shifts of the response variable (this is done to calculate statistical significance using multiple offsets to preserve autocorrelation) or re-setting the context at each fold boundary (this is for a control analysis in the Supplementary Information).
- `plot_encoding_mono.py` and `plot_encoding_multi.py` calculate statistical significance, aggregate, and plot the results for the *WITHIN* and the *ACROSS* condition, respectively.

#### Data
- The fMRI data is contained in the folder `data`, both as a csv files with the responses from the individual participants, and as pickle files post aggregation.
- The sound files that participants listened to are in the folder `sound_data`, while the transcriptions are in `transcribed`

----------

### Study II
In Study II, In our second study, we implemented a stricter test for the cross-lingual transferability of the encoding models. We fitted the encoding models employing fMRI data from Study I and from three
additional fMRI datasets, all using English stimuli presented via auditory or visual modalities. Then, we collected new fMRI data on 9 additional languages, and transferred zero-shot the encoding models to the new data.

The **code and data** supporting Study II are in the `confirmatory` and `additional_analyses` folders, that are organized this way:

    project_root/
    ├── confirmatory/ # similar to Study I
    └── additional_analyses/ # training data for the encoding models
        ├── control/
                ├── data/
                ├── embeddings/
                └── control_encoding_registered_model.py # code to train encoding models
        ├── NaturalStories/
        └── Pereira/

The `confirmatory` folder is structured in a very similar way to Study I: it includes code to calculate the time-series reliability, extracting embeddings, and evaluating the encoding models. Note that in this case, there are thee participants and three passages per language, so the code is somewhat different (e.g., in the way we pre-select passages based on the reliability). 

Critically, here, the encoding models are not trained on the Study II data, but on four other datasets: on the Study I data (the code for this is in `confirmatory/registered_model`) and on three other fMRI datasets:

 - *NatStories* (story listening in English)
 - *Pereira2018* (sentence reading in English)
 - *Tuckute2024* (sentence reading in English)


The code for training and storing the encoding models' weights is in the folder `additional_analyses`, where there is one subfolder for each dataset. The subfolder typically includes a sub-subfolder with the fMRI data, another sub-subfolder with the embeddings for that dataset, and a Python script to train the encoding models.

The encoding models based on those separate datasets (together with the normalization parameters) are stored in `confirmatory/registered_models`, and they are then transferred zero-shot to the new data in `confirmatory/confirmatory_encoding.py`.

----------

### Study III
Study III uses targeted perturbations to identify what linguistic features drive cross-lingual transfer. We applied various perturbations to English stimuli (removing function words, scrambling word order, paraphrasing, etc.) and trained encoding models on embeddings of these perturbed sentences. We then measured transfer to the other nine languages of Study II. The results show that cross-lingual transfer depends primarily on lexical-semantic content: paraphrases and content-word-only inputs preserve transfer, whereas function-word-only inputs substantially reduce it.

The **code and data** supporting Study III are split across two locations based on the training dataset used:

**For Tuckute2024 (control) dataset:**
- `additional_analyses/control/perturbation/` contains:
  - `perturbation_encoding.py` - main script for training encoding models on perturbed stimuli
  - `perturbation_materials.py` - code for generating the perturbed stimuli
  - `registered_models/` - trained encoding models based on perturbed stimuli from Tuckute2024 dataset

**For Pereira2018 dataset:**
- `additional_analyses/pereira/perturbation/` contains:
  - `perturbation_encoding.py` - script for training encoding models on perturbed Pereira stimuli
  - `perturbation_materials.py` - code for generating the perturbed stimuli
  - `perturbation/registered_models/` - trained encoding models based on perturbed stimuli from Pereira2018 dataset

----------
### Additional analyses
The `additional_analyses` folder contains code for training encoding models on additional datasets and supplementary analyses:

- **control/**: Contains code for training encoding models on the control dataset and Study III perturbation analyses
- **MECO/**: Contains code for training encoding models on the MECO dataset  
- **NaturalStories/**: Contains code for training encoding models on the NaturalStories dataset
- **pereira/**: Contains code for training encoding models on the Pereira2018 dataset

Additional supplementary analyses include:

 - **Next-word prediction:** In the folder `perplexity` - evaluates if the MNNLM's next-word-prediction abilities explain the models' performance in each language
 - **Low-level feature baselines:** In the folder `other/` - evaluates whether low-level features (word frequency, length, rate, onset) can account for the observed effects

----------

> ⚠️ Notes for reproducibility

The code was designed and ran in interactive environments; to run it this way, the working directories will need to be manually specified with `os.chdir("/path/to/your/wd")`

This repository does **NOT** include all the embeddings used in all the studies, but embeddings for distilmbert are included on OSF to allow for fast experimentation and reproduction. Embeddings take up a lot of space (the full folder is ~200 GB) so the embeddings for the other models need to be recomputed (with `get_model_embeddings.py`).

The code was tested on a computer with Ubuntu 22.04.4 LTS. Python version: 3.9.7. No non-standard hardware is required. Installation only requires installing the required dependencies (see below) and should only take a few minutes. 

    Package           Version
    adjustText:       0.8
    deep_translator:  1.11.0
    matplotlib:       3.6.0
    numpy:            1.26.4
    pandas:           1.5.1
    scipy:            1.11.4
    seaborn:          0.12.1
    sklearn:          1.2.1
    statsmodels:      0.14.4
    torch:            1.12.0
    tqdm:             4.64.0
    transformers:     4.45.2
    
    whisper_timestamped: to be installed from https://github.com/linto-ai/whisper-timestamped

A minimal working code for the Study I encoding approach (both "within" and "across") can be found on OSF in `encoding_demo.py`, where it's possible to fit encoding models based on distilmbert embeddings, which need not be recomputed and are already provided in the folder. The demo should take about 5 minutes or less to run. The expected output is provided in `demo_output.txt`.
    
