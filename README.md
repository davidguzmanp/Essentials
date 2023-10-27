# Testing the Ability of Language Models to Interpret Figurative Language

## Table of Contents
* [Introduction](#introduction)
* [Overview directories](#overview-directories)
* [Dataset](#dataset)
* [References](#references)
* [Appendix](#Appendix)
<!-- * [Contact](#contact) -->


## Introduction
This repository contains the dataset and code for the documentation of our submission to the course [Essentials in Text and Speech Processing](https://studentservices.uzh.ch/uzh/anonym/vvz/?sap-language=EN&sap-ui-language=EN#/details/2023/003/SM/51079434) course at [University of Zurich](https://www.uzh.ch/en.html). 
This code repository contains the code used to evaluate the performance of FLAN T-5 on figurative language, using a dataset of 500 Winograd schema examples. The code can be used to reproduce the results of the paper, and to explore the model's performance on different figurative language tasks.

<!-- #### Why Figurative language?
Most of NLP (as of the publication date of this paper) focuses on literal interpretation of phrases. However, this isn't the only way in which humans use language. In most cases, people can readily interpret creative phrases such as "She thinks of herself as a particle of sand in the desert", even if they have not directly heard such a phrase before. Figurative language is prominent in colloquial text and literature, and correct inference regarding figurative language involves commonsense knowledge as well as flexibility in word meaning inference. -->

#### Why FLAN T-5
?

#### Examples?

## Organization
In this repository, we have primarily stored Jupyter notebooks for all stages of the experiment like baseline results, prompt engineering and finetuning. Addionally, some of the data for training or produced as a results are stored as well.

#### Overview directories
This is the directory structure that aims to be similar to the documentation structure:
````
├── baseLine:                           # Comparing the performance of quantized large language models on the fig-qa dataset
│   ├── guanaco
│   ├── mistral
│   ├── phi-1_5
│   ├── synthia7B
│   └── t5
│       ├── flan-t5-large
│       └── flan-t5-xl-sharded:             # Running the T5-XL with 3 billion parameters using a quantization configuration was unsuccessful.
│
├── Finetuning:                         # Comparing the performance of quantized large language models on the fig-qa dataset
│   ├── csvs:                               # data preprocessing for the seq2seq training
│   ├── Qlora finetuning
│   ├── Qlora seq2seq
│   ├── traditional finetuning
│   └── Traditional finetuning like paper
│
├── Human_Label
└── prompt
````

#### Dataset
This code relies on the dataset and code Fig-QA which was presented in the paper [Testing the Ability of Language Models to Interpret Figurative Language](https://arxiv.org/abs/2204.12632). Fig-QA consists of 10256 examples of human-written creative metaphors that are paired as a Winograd schema. It can be used to evaluate the commonsense reasoning of models. The metaphors themselves can also be used as training data for other tasks, such as metaphor detection or generation. 
You can also find the dataset on [Huggingface datasets](https://huggingface.co/datasets/nightingal3/fig-qa).


# References
- FLAN T5 language model: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://doi.org/10.48550/arXiv.1910.10683)
- Training quantized models: [QLoRA: Efficient Finetuning of Quantized LLMs](https://doi.org/10.48550/arXiv.2305.14314)
- Dataset Fig-QA: [Testing the Ability of Language Models to Interpret Figurative Language](https://doi.org/10.48550/arXiv.2204.12632)


# Appendix
#### All files
````bash
├── baseLine
│   ├── guanaco
│   │   ├── output_data_JosephusCheung-Guanaco.json
│   │   ├── output_df_JosephusCheung-Guanaco.csv
│   │   ├── pickle_output_data_JosephusCheung-Guanaco.pkl
│   │   └── qlora_guanaco.ipynb
│   ├── mistral
│   │   ├── mistral_base
│   │   │   ├── file.tsv
│   │   │   └── Mistral_Base.ipynb
│   │   └── mistral_chat_finetuned
│   │       ├── file.tsv
│   │       └── Mistral_FineTuned.ipynb
│   ├── phi-1_5
│   │   ├── migtissera_synthia_7b_v1_3.ipynb
│   │   ├── output_data_microsoft-phi-1_5.json
│   │   ├── output_df_microsoft-phi-1_5.csv
│   │   ├── pickle_output_data_microsoft-phi-1_5.pkl
│   │   └── qlora_phi_1_5.ipynb
│   ├── synthia7B
│   │   ├── migtissera-synthia-7b-v1-3.ipynb
│   │   ├── output_data_migtissera-SynthIA-7B-v1.3_finetuned.json
│   │   ├── output_data_migtissera-SynthIA-7B-v1.3.json
│   │   ├── output_df_migtissera-SynthIA-7B-v1.3.csv
│   │   ├── output_df_migtissera-SynthIA-7B-v1.3_finetuned.csv
│   │   ├── pickle_output_data_migtissera-SynthIA-7B-v1.3_finetuned.pkl
│   │   └── pickle_output_data_migtissera-SynthIA-7B-v1.3.pkl
│   └── t5
│       ├── flan-t5-large
│       │   ├── file.tsv
│       │   └── t5.ipynb
│       ├── flan-t5-xl-sharded
│       │   ├── file.tsv
│       │   └── t5_XL.ipynb
│       ├── flan_ul2_t5.ipynb
│       └── t5_XL.ipynb
├── Finetuning
│   ├── csvs
│   │   ├── fiqa-seq2seq-inference.-to-csv.ipynb
│   │   └── predictions_seq2seq-T5-Large.csv
│   ├── Qlora finetuning
│   │   ├── contrastive_finetuning_qlora_phi_1_5.ipynb
│   │   ├── finetuning-qlora-mistral-ipynb.ipynb
│   │   ├── finetuning qlora_phi_1_5.ipynb
│   │   ├── finetuning_qlora_phi_1_5.ipynb
│   │   ├── finetuningq-lora_phi_1_5.ipynb
│   │   ├── finetuning-synthia-7b-v1-3.ipynb
│   │   ├── f-qlora-tinyllama.ipynb
│   │   └── novel_finetuning_qlora_phi_1_5.ipynb
│   ├── Qlora seq2seq
│   │   ├── contrastive attempt.ipynb
│   │   ├── omg-t5-large-figqa-seq2seq-second-run.ipynb
│   │   ├── omg-t5-large-figqa-seq2seq-third-run-overfit.ipynb
│   │   ├── omg t5-large-seq2seq-fqa-qlora.ipynb
│   │   ├── omg-t5-large-seq2seq-fqa-qlora.ipynb
│   │   ├── peft-flan-t5-int8-summarization.ipynb
│   │   ├── seq2seq t5base worsens.ipynb
│   │   ├── t5-base-seq2seq-fqa-qlora.ipynb
│   │   ├── t5-large-figqa-seq2seq-batch-150.ipynb
│   │   ├── T5-Large-figQA-seq2seq-batch-50.ipynb
│   │   ├── t5-seq2seq-summary-qlora.ipynb
│   │   └── ThisIsToSay T5 large.ipynb
│   ├── traditional finetuning
│   │   ├── finetuning_qlora_phi_1_5.ipynb
│   │   ├── traditional-finetuning-t5base.ipynb
│   │   ├── traditional-finetuning-t5base-kaggle-ipynb.ipynb
│   │   └── traditional_finetuning_t5base_two_epochs.ipynb
│   └── Traditional finetuning like paper
│       ├── figqa-training-gpt-neo-sssm-2-epochs.ipynb
│       ├── figqa-training-t5-base-gets-worse-2-epochs.ipynb
│       └── figqa-training-t5-small-gets-worse-2-epochs.ipynb
├── fineTuning_Code_from_FigQA.ipynb
├── fineTuning_t5_qa.ipynb
├── Human_Label
│   ├── predictions_seq2seq-T5-Large.csv
│   └── predictions_seq2seq-T5-Large_labeld.csv
├── prompt
│   ├── partial_result_qlora_mistral_400_prompt.txt
│   ├── qlora_Mistral_400_prompt.ipynb
│   ├── qlora_t5_large_400_prompt.ipynb
│   ├── qlora_t5_large_75_complex_prompt.ipynb
│   ├── tradional_flan_t5_base_PomptEngineering_1000_Samples.ipynb
│   └── tradional_flan_t5_base_PomptEngineering.ipynb
├── qlora_and_model_choice.ipynb
├── qlora-guanaco.ipynb
├── qlora_Orca.ipynb
├── README.md
└── TopicModeling.ipynb

18 directories, 70 files
````