> ⚠️ Note for Reviewers (Round 2)

This is the **`revision`** branch, which contains all updates for the **Round 2** revision of the manuscript. All changes relative to the Round 1 version (archived on OSF) can be tracked here through Git’s version history. All Round 1 materials will remain archived on OSF for reference.

***

## Brain encoding in 21 languages

This repository contains the code and data to reproduce the article *Multilingual Computational Models Reveal Shared Brain Responses to 21 Languages*, currently under review.

The article includes two studies; Study I and Study II. Study I (12 languages) is based on previously collected data from Malik-Moraleda, Ayyash, et al. (2022), whereas Study II (9 languages) is based on newly collected data. 

### Study I
Study I leverages existing fMRI data from a passage-listening task in 12 languages (Malik-Moraleda, Ayyash, et al., 2022). We trained fMRI encoding models to predict brain responses based on multiple languages and transferred them zero-shot to a new language on which they had not been trained.

The methodology is as follows:

 1. We extract fMRI responses from the language network (functionally defined; Fedorenko et al., 2010), averaging across voxels, fROIs, and lastly, across participants, to obtain one single time-series per language.
 2. We obtain written transcriptions for the passages that participants listened to with Whisper-timestamped, which also outputs word-by-word timestamps.
 3. We extract embeddings of the text in the various languages with multilingual neural network language models (MNNLMs, n = 20).
 4. We fit encoding models to predict fMRI activity from the MNNLM embeddings
    - In the *WITHIN* condition, we train and test the encoding models in each language.
    - In the *ACROSS* condition, we train the encoding models in all languages but one, and test in that language.

The **code and data** supporting Study I are in the "level 0" of this repository, which includes the following scripts:

#### Code
- `time_series_corr.py` extracts averaged time-series, checks reliability, and stores the data for later use. It extracts time-series also for the right hemisphere language areas and the MD network, which are used as control areas.
- `whisper-timestamped.py` transcribes the audio (wav) files that were presented to the participants and produces csv files with all the words and timestamps. 
- `get_model_embeddings.py` generates contextual word embeddings from the 20 MNNLMs we considered. Note that some models (XGLM, mGPT) are not tested in all the 20 languages because some of them were absent from the models' pre-training data.
    - To simulate an auto-regressive setup and prevent access to future tokens, bidirectional models are tested with a sliding window of 100 words.
- `fit_encoding.py` fits linear encoding models (Ridge regression) predicting fMRI responses from the contextual word embeddings. To do so, it first aligns the embeddings with the fMRI responses based on the timestamps. Encoding models are either trained and tested within each language separately with cross-validation (*WITHIN* condition), or alternatively, the encoding models are fitted in all languages but one, and transferred zero-shot to that language (*ACROSS* condition). 
    - This is done for each model × layer × language combination
    - This is performed separately for the (standard) LH language areas, the homotropic RH language areas, and the MD network.
- `fit_encoding_random.py` and `fit_encoding_chunk_context.py` do the same thing but either randomizing the response variable (this is done to calculate statistical significance) or re-setting the context at each fold boundary (this is for a control analysis in the Supplementary Information).
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
### Additional analyses
There are two core additional analyses in the paper: one where we evaluate if the MNNLM's next-word-prediction abilities explain the models' performance in each language, and another where we evaluate whether the extent to which representations are aligned across languages predicts transfer performance. 

The code for those analyses can be found at:

 - **Next-word prediction:** In the folder `perplexity`
 - **Alignment and transfer:** In the folder `other/synonyms`

----------

> ⚠️ Notes for reproducibility

The code automatically sets the working directory to the script's location, so in principle there is no need to specify the paths. However, this doesn't work if the scripts are executed in interactive environments like IPython. The code was designed and ran in interactive environments; to run it this way, the working directories will need to be manually specified with `os.chdir("/path/to/your/wd")`

This repository does **NOT** include all the embeddings used in all the studies, but embeddings for distilmbert are included to allow for fast experimentation and reproduction. Embeddings take up a lot of space (the full folder is ~200 GB) so the embeddings for the other models need to be recomputed (with `get_model_embeddings.py`).

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

A minimal working code for the Study I encoding approach (both "within" and "across") can be found in `encoding_demo.py`, where it's possible to fit encoding models based on distilmbert embeddings, which need not be recomputed and are already provided in the folder. The demo should take about 5 minutes or less to run. The expected output is provided in `demo_output.txt`.
    
