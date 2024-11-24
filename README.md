# CS322Final-ElectionNewsClassification-AI

A final project for CS 322: Natural Language Processing, by Dake Peng 25' at Carleton College, Northfield. MN, United States.

## Introduction

This is a project that I inherited from my work at DataSquad. Professor Barbara Allen from the Political Science Department at Carleton College has a project analyzing election news that has been going for 10+ years. During each election period (), She and her colloabortors collected news transcripts from major news sources and annotated them for election and non-election stories.

This project aims to use that annotation data to train a support vector machine model. To simplify the issue, the task is reformatted as such: given a piece of text, classify it as "NONE", "ELECTION" or "NON-ELECTION".

## Data

The data was scraped from 5 different Google Drives that Barbara and her students work in. Each piece of text is annotated with the following symbols:

- "˜" indicates the beginning of an election story
- "|" indicates the end of an election story
- "\\" indicates the beginning of a non-election story
- "//" indicates the ending of a non-election story

The text are downloaded as .txt files. They are processed in ./data_processing, majorly with partition_data.py. The function works as follows:

- remove headers, footnotes, and irrelevant []s and {}s
- parse through the text, mark each symbol above and line break
- split the text into chunks by those markers
- parse therough the chunks, mark each chunk with flags "election" or "non-election". The flags switch on and off when seeing the markers above. The flags also handle cases where the annotators forget to end the story.
- parse through the chunks again, merge adjacent chunks when applicable

Then, the data is converted to a .csv file. With columns Source, Date, Text, and Type. The Date column is specifically for the GPT model. The .csv file is split into training, test, and dev sets by the proportions 8:1:1. The size of the data sets are roughly 70000:8000:8000. The details are stored in ./data/dataset_stats.txt

## Evaluation Metrics

Evaluation is done in get_statistics.py. For each model, I printed a confusion matrix and calculated the Precision, Recall, and F1 scores. I also calculated an overall F1 scores.

## SVM Approach & Results

### Overview
The SVM approach goes through the following pipeline:
- get_tf_idf calculates tf-idf weights for each document in the training set
    - stopwords and word-initiial/final punctuations are removed
- get_training_vectors calculates document vectors for each document in the training set
    - stopwords and word-initiial/final punctuations are removed
    - the vector for each document is calculated by an weighted average of
        - word vectors, form gensim's word2vec-google-news-300
        - the tf-idf weights calculated above
- get_test_vectors calculates document vectors for each document in the test set, using tf from the test set and the pre-calculated idf from the training set.
- soft_linear_SVM 
    - trains 2 SVMs from the document vectors calcualted above
        -  the 1st SVM distinguishes story-nonstory
        - the 2nd SVM distinguishes election-nonelection 
    - runs the test vectors through the 2 SVMs
        - vectors that are classified as story by the 1st SVM enter the 2nd SVM for another classification
    - the results are stored in a .csv file in ./results/
    - the SVM 
- get_statistics: compares the observed column and predicted column in a csv and generate a txt file with metrics described above

### Results

## zero-shot GPT Approach & Results

### Overview
The GPT approach is implemented in 0shot_GPT4o.py using the GPT4o mini model (since it's the cheapest of all). The model is prompted with the following:

I will provide you with a date and a piece of excerpt from a news transcript. Please classify the excerpt. There are 3 types of excerpts:
• NONE: An excerpt is considered NONE if it does not contain a story at all.
• ELECTION: An excerpt is considered an ELECTION story if it either: 1) mentions an upcoming election; 2) mentions a candidate involved in an upcoming election by name, or 3) focuses on the current duties or actions of an incumbent who is running for re-election or different office — or on issues in the campaign, by explicitly noting that they are election issues.
• NONELECTION: An excerpt is considered a NONELECTION story if it contains a story but the story is not relavant to the upcoming election and does not mention upcoming candidates nor the election campaign.
Text: [TEXT]
Type: 

and the results are stored in a csv. The statistics are calculated and stored in the same fashion as above.

### Results
Some responses (˜30) of the model were a length line of text followed by the results. In those cases, I manually formatted them. 

#### Confusion Matrix (the column under NONE indicates documents whose ACTUAL values are none):
| Predicted/Actual | NONE  | ELECTION | NONELECTION |
|------------------|-------|----------|-------------|
| NONE             | 1372  | 1        | 65          |
| ELECTION         | 1018  | 651      | 107         |
| NONELECTION      | 2163  | 59       | 2413        |

#### Statistics:
| Type        | Precision            | Recall               | F1                    |
|-------------|----------------------|----------------------|-----------------------|
| NONE        | 0.9541029207232267   | 0.3010092145677929   | 0.4576384256170781    |
| ELECTION    | 0.3622704507512521   | 0.9156118143459916   | 0.5191387559808612    |
| NONELECTION | 0.5200431034482759   | 0.9334622823984526   | 0.6679584775086505    |
| Overall     |                      |                      | 0.5324288596670352    |


## Discussion

In general, the GPT model performed better than average with an F1 score of 0.55. However, if we look at the confusion matrix (./results/GPT_Statistcs.csv), it is very accurate to distinguish election and non election stories, as well as to identify non-election stories. Given tht the segmentation of the data set and the annotation process are not perfect (the task was not designed to be a classification task, but rather an identification task, we can imagine that many text segments that would have been a story are not identified; even more nuance here: I think that Barbara's team made a list of stories to identify from for the students) --- all these point to that the performance of the GPT model is much better than it seems on paper.

A discussion comparing the results of the two approaches, reflecting on their success at addressing your task or question, and identifying any shortcomings or things you might decide to try with more time. As you consider possible shortcomings, you should consider what potential sources of bias there might be in both approaches, along with what potential impacts such bias might have if employed in a real-world context.

## Prerequisites (For Windows Machines)
Install the following libraries:

```bash
pip install nltk, pandas, csv, openai, tqdm, numpy
```

If you want to run the GPT model, you would need to acquire an OpenAI API key and set that as a system variable

```bash
setx OPENAI_API_KEY "your_api_key_here"
```
The whole test set (~7000 queries) costs about $0.5 on the GPT4o-mini model

## Running the code (For Windows Machines):

To run all processes of the SVM model (with training, testing and development sets prepared), run the following programs in order:

```bash
python3 get_tf_idf.py
python3 get_training_vectors.py
python3 get_test_vectors.py
python3 basic_SVM.py
python3 get_statistics.py
```

The entire process should take 5-10 minutes (I tested on a M4 device). Each .py program will generate intermediate .json files in ./data, occupying a lot of space, so please make sure to remove them afterwards.

To run the GPT model, run

```bash
python3 0shot_GPT4o.py
```