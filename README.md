# CrossLingual-Translation-Models

Exploring Cross-Lingual Capabilities exhibited by Pre-Trained Models for Indian Language Translation Tasks

## Overview

This repository contains the code and resources for exploring the cross-lingual capabilities of pre-trained language models in translating Indian languages. The primary focus is on evaluating how well these models can be fine-tuned to handle translations for low-resource languages, leveraging existing multilingual capabilities.

## Objectives

Evaluate Language Pair Flexibility: Assess the ability of models pre-trained on specific language pairs to adapt to new language pairs through fine-tuning.

Enhance Cross-Lingual Capabilities: Explore the performance improvements of multilingual models when fine-tuned on Indian languages.

Transfer Learning to Low-Resource Languages: Investigate the effectiveness of sequential fine-tuning in transferring cross-lingual capabilities to low-resource Indian languages.

## Models Used

**Helsinki-NLP/opus-mt:** A robust transformer-based model pre-trained on various language pairs.

**mBART50:** A multilingual sequence-to-sequence model pre-trained on 50 languages.

**Llama-2:** The second generation of the LLaMA family, featuring advanced pre-training techniques and optimized for diverse tasks.

## Datasets

**AI4Bharat/Samanantar:** The largest parallel corpus for Indian languages, covering Hindi, Tamil, Telugu, and more.

**CFILT IITB English-Hindi:** A rich dataset for English-Hindi translations, adding diversity to our training data.

**WMT News Dataset:** Domain-specific dataset for English-Gujarati, used to validate language pair flexibility and cross-lingual capability transfer.

## Methodologies

**Language Pair Flexibility**

Base Model: Helsinki-NLP/opus-mt-en-fr

Fine-Tuned Model: [finetuned-opusmt-en-fr-hi](https://huggingface.co/ritika-kumar/finetuned-opusmt-en-fr-hi)

Goal: Validate how well the model adapts to new language pairs through fine-tuning.

**Cross-Lingual Capabilities of Multilingual Models**

Base Model: Helsinki-NLP/opus-mt-en-mul

Fine-Tuned Models: 
- [finetuned-opusmt-en-to-hi](https://huggingface.co/Varsha00/finetuned-opusmt-en-to-hi)
- [finetuned-opusmt-en-to-ta](https://huggingface.co/Varsha00/finetuned-opusmt-en-to-ta)
- [finetuned-opusmt-en-to-te](https://huggingface.co/Varsha00/finetuned-opusmt-en-to-te)

Goal: Assess the performance improvements of multilingual models when fine-tuned on Indian languages.

Base Model: facebook/mbart-large-50-many-to-many-mmt

Fine-Tuned Models:
- [finetuned-mbart50-en-hi](https://huggingface.co/rahimunisab/finetuned-MBart50-en-hi)
- [finetuned-mbart50-en-tam](https://huggingface.co/rahimunisab/finetuned-MBart50-en-tam)
- [finetuned-mbart50-en-tel](https://huggingface.co/ritika-kumar/finetuned-mbart50-en-tel)

Goal: Assess the performance improvements of multilingual models when fine-tuned on Indian languages.

**Cross-Lingual Capability Transfer to Low-Resource Indian Languages**

Base Model: Helsinki-NLP/opus-mt-en-mul

Fine-Tuned Models: 
- [finetuned-opusmt-en-hi-gu](https://huggingface.co/Varsha00/finetuned-opusmt-en-hi-gu)
- [finetuned-opusmt-en-ta-gu](https://huggingface.co/Varsha00/finetuned-opusmt-en-ta-gu)

Goal: Investigate the effectiveness of sequential fine-tuning in transferring cross-lingual capabilities to low-resource languages.

Base Model: meta-llama/Llama-2-7b-hf

Fine-Tuned Model: [finetuned-llama2-7b-en-hi](https://huggingface.co/ritika-kumar/finetuned-llama2-7b-en-hi)

Goal: Investigate the effectiveness of sequential fine-tuning in transferring cross-lingual capabilities to low-resource languages using PEFT LORA.

## Benchmarks and Results

To evaluate the performance of the fine-tuned models, we utilized two rigorous benchmarks: Tatoeba and IN-22. These benchmarks provide a standardized way to measure the translation quality and cross-lingual capabilities of the models.

**Tatoeba Benchmark**

The Tatoeba benchmark is widely used for evaluating translation models and measures the BLEU score, which indicates how well the model-generated translations match the reference translations.

**IN-22 Benchmark**

The IN-22 benchmark is another critical evaluation metric specifically designed for assessing Indian languages. It helps in understanding the model's effectiveness in handling the unique linguistic features and complexities of Indian languages.

### Comparative Analysis on Benchmarks (Helsinki)

#### Tatoeba Challenge

| Model Name                     | Lang     | BLEU  | Tatoeba Baseline | Helsinki Baseline |
|--------------------------------|----------|-------|------------------|-------------------|
| finetuned-opusmt-en-fr-hi      | Hindi    | 21.93 | 16.1             | 13.0 (Hindi)      |
| finetuned-opusmt-en-to-hi      | Hindi    | 12.33 | 16.1             | 13.0              |
| finetuned-opusmt-en-to-ta      | Tamil    | 14.05 | 6.8              | 5.0               |
| finetuned-opusmt-en-to-te      | Telugu   | 24.44 | 4.1              | 4.7               |
| finetuned-opusmt-en-hi-gu      | Gujarati | 27.77 | 18.8             | 15.4              |
| finetuned-opusmt-en-ta-gu      | Gujarati | 26.26 | 18.8             | 15.4              |

### Comparative Analysis on Benchmarks (Helsinki)

#### IN22 Gen

| Model Name                     | Lang     | BLEU  | IN-22 Baseline | IN-22 Models     |
|--------------------------------|----------|-------|----------------|------------------|
| finetuned-opusmt-en-fr-hi      | Hindi    | 15.54 | 21.5           | mbart50          |
| finetuned-opusmt-en-to-hi      | Hindi    | 26.00 | 22.1           | mbart50, m2m100  |
| finetuned-opusmt-en-to-ta      | Tamil    |  5.88 |  1.4           | m2m100           |
| finetuned-opusmt-en-to-te      | Telugu   | 10.64 |  2.8           | mbart50          |
| finetuned-opusmt-en-hi-gu      | Gujarati | 16.43 |  3.9           | mbart50, m2m100  |
| finetuned-opusmt-en-ta-gu      | Gujarati |  5.81 |  3.9           | mbart50, m2m100  |

### Comparative Analysis on Benchmarks (mBART50 & Llama)

#### Tatoeba Challenge

| Model Name                | Language | BLEU Score | Tatoeba Baseline |
|---------------------------|----------|------------|------------------|
| finetuned-mbart50-en-hi   | Hindi    | 11.2       | 16.1             |
| finetuned-mbart50-en-tel  | Telugu   | 35.93      | 4.1              |
| finetuned-mbart50-en-tam  | Tamil    | 14.05      | 6.8              |
| finetuned-llama2-en-hi    | Hindi    | 12.60      | 16.1             |

### Comparative Analysis on Benchmarks (mBART50 & Llama)

#### IN22 Gen

| Model Name                | Language | BLEU Score | IN22 Baseline | IN22 Models        |
|---------------------------|----------|------------|---------------|--------------------|
| finetuned-mbart50-en-hi   | Hindi    | 26.06      | 21.5          | mbart50            |
| finetuned-mbart50-en-tel  | Telugu   | 14.75      | 2.8           | mbart50            |
| finetuned-mbart50-en-tam  | Tamil    | 10.16      | 8.3           | mbart50            |
| finetuned-llama2-en-hi    | Hindi    | 25.89      | 22.1          | mbart50, m2m100    |

The comparative analysis of benchmarks for the mBART50 and Llama models highlights their performance in translating low-resource Indian languages. Notably, the finetuned-mbart50-en-tel model achieved the highest BLEU score of 35.93 on the Tatoeba challenge, significantly outperforming the baseline. The results demonstrate the effectiveness of these models in enhancing translation quality, particularly when leveraging cross-lingual capabilities through fine-tuning.

## How to Navigate Through the Repository

This repository is organized to facilitate easy access to all necessary resources for exploring cross-lingual capabilities in Indian language translation tasks. The following sections will help you navigate through the contents:

**HF Links:** Direct links to the Hugging Face models are provided in the README, allowing you to access the fine-tuned models directly.

**EDA Code:** The exploratory data analysis (EDA) code is also available within the notebooks, providing insights into the datasets used for training and evaluation.

**Models:** The models folder includes the training code necessary for fine-tuning the models on various datasets.

**Benchmark Evaluations:** The benchmark_evaluations folder contains the evaluation code used to assess the performance of the models against the Tatoeba and IN-22 benchmarks.

By navigating through these sections, you can efficiently access and utilize the provided resources to replicate and extend the study.





