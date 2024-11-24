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
The linear SVM approach goes through the following pipeline:
- get_tf_idf calculates tf-idf weights for each document in the training set
    - stopwords and word-initiial/final punctuations are removed
- get_training_vectors calculates document vectors for each document in the training set
    - stopwords and word-initiial/final punctuations are removed
    - the vector for each document is calculated by an weighted average of
        - word vectors, form gensim's word2vec-google-news-300
        - the tf-idf weights calculated above
- get_test_vectors calculates document vectors for each document in the test set, using tf from the test set and the pre-calculated idf from the training set.
- soft_linear_SVM 
    - trains 2 soft-margin, linear SVMs from the document vectors calcualted above
        -  the 1st SVM distinguishes story-nonstory
        - the 2nd SVM distinguishes election-nonelection 
    - runs the test vectors through the 2 SVMs
        - vectors that are classified as story by the 1st SVM enter the 2nd SVM for another classification
    - the results are stored in a .csv file in ./results/
    - the SVM 
- get_statistics: compares the observed column and predicted column in a csv and generate a txt file with metrics described above

### Results

#### Confusion Matrix:

| Predicted/Actual | NONE  | ELECTION | NONELECTION |
|------------------|-------|----------|-------------|
| NONE             | 3423  | 416      | 773         |
| ELECTION         | 216   | 32       | 155         |
| NONELECTION      | 919   | 289      | 1657        |

#### Statistics:

| Type        | Precision            | Recall               | F1                    |
|-------------|----------------------|----------------------|-----------------------|
| NONE        | 0.7421942758022549    | 0.750987275120667    | 0.7465648854961832    |
| ELECTION    | 0.0794044665012407    | 0.04341926729986431  | 0.056140350877192984  |
| NONELECTION | 0.5783595113438046    | 0.6410058027079304   | 0.6080733944954129    |
| Overall     |                      |                      | 0.6365593796267432    |


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

#### Confusion Matrix

| Predicted/Actual | NONE | ELECTION | NONELECTION |
| ---------------- | ---- | -------- | ----------- |
| NONE             | 1372 | 1        | 65          |
| ELECTION         | 1021 | 672      | 107         |
| NONELECTION      | 2164 | 64       | 2413        |

#### Statistics

| Type        | Precision        | Recall           | F1              |
| ----------- | ---------------- | ---------------- | --------------- |
| NONE        | 0.9541           | 0.3010           | 0.4576          |
| ELECTION    | 0.3733           | 0.9118           | 0.5298          |
| NONELECTION | 0.5199           | 0.9335           | 0.6679          |
| Overall     | -                | -                | 0.5333          |

## Discussion

The SVM model actually performed better than I expected, with an overall F1 score of 0.64. The confusion matrix, however, shows that it did not do quite what we expected. Though it performed fairly decently at distinguishing story and non-story, it is terrible at distinguishing election stories from the rest. I think the success of this model (especially at distinguishing nonstories) is largely due to the fact that I incorporated a very large word2vec model (googlenews300), which captured the words' meanings in larger contexts and allowed the model to better generalize. I am rather surprized that the model did poorly on election stories, though, since I expected election stories to have more indexical words like "elelction" or names of the candidates that may be directly mapped to the document vectors. I also don't think that it's the problem of the size of training data or the proportion of election stories --- the model may do better even if it just classify elecion-nonelection by chance. This is perhaps due to that the division between election and nonelection stories is not linear, as I'll elaborate next.

The shortcomings of SVM is also obvious if we consider how the linear SVM works. It tries to find a plane that best separates 2 different sets of document vectors. For one, the document vectors are a mere aggregation of word embeddings and word frequencies, and the structures of the text --- which is essential to the composition of stories, is not considered at all. On the other hand, it is hard to say that such a linear plan between stories and nonstories exist --- the boundary between the two sets may be curved or spherical. I'm certain that if I take more time to fine-tune the 2-layer SVM structure or use different SVM methods like kernels, the statistics would look even better than the current model.

This hints on some of the potential biases in using linear SVM in real life: its ignorance of word order and structures may cause poor performance on tasks that involve grammar or in-text word relationships (like a story), and the fact that it sets a linear boundary in the space of vectors (or even a curvy one) may make it fail at divisions that arn't like that, for example, a rule-based division.

On paper, the GPT model performed slightly better than average with an overall F1 score of 0.54. However, if we look at the confusion matrix (./results/GPT_Statistcs.csv), it is very accurate to distinguish election and non election stories, as well as to identify non-election stories. Given tht the segmentation of the data set and the annotation process are not perfect (the task was not designed to be a classification task, but rather an identification task) we can imagine that many text segments that would have been a story are not identified. When I go into the results ans selected documents that are labeled "NONE" but misidentified as "ELECTION" or "NONELECTION", many that I see mention "election" or the name of the presidents. Even more nuance here: I think that Barbara's team made a list of stories to identify from for the students) --- all these point to that the performance of the GPT model is much better than it seems.

GPY4o-mini is a model with about 8b parameters. It is also potentially based on (trained on?) GPT4, with 1.8 trillion parameters. With the attention mechanism, it is able to capture the relationships between words in a sentence, as well as some structural information about the text. Given that both my prompt and text are relatively short, there is a chance that the context window can cover my entire input --- which makes it capture strucutures across the text. Given the amount of data that is put into the GPT models, it's very likely that these news transcripts are a part of the training that the model recieved. I also used a Q&A format prompt to potentially mimic some patterns that it learned from online forums. All of these factors might have contributed to the model's success.

A minor problem I see, as I mentioned above, is that the model does not always consistently output the 3 categories, even when instructed to do so. This may require another human or programatic process of identifying the output. If I had more time, I would probably use more time to process the annotation data and ensure that each piece is labeled correctly. Moreover, it's also possible to directly train the GPT model by providing it all of the data. That may help resolve some imperfect classification.

I don't think this process revealed any potential problems

A discussion comparing the results of the two approaches, reflecting on their success at addressing your task or question, and identifying any shortcomings or things you might decide to try with more time. As you consider possible shortcomings, you should consider what potential sources of bias there might be in both approaches, along with what potential impacts such bias might have if employed in a real-world context.

## AI use and outside sources

ChatGPT was used in different pieces of code to lookup api calls and object (like numpy array) operations. Links to the conversations are included when applicable. The OpenAI GPT api with GPT4omini is used in the 0-shot, third party comparison case. The SVM implementation referenced heavily on 2 online tutorials, listed in soft_linear_SVM.py, though I tried to comment each part to indicate that I have understood the process.

It's worth noting that I also tried the SVM library from sklearn and a gradient descent method that, instead of adjusting the weights for each document, takes a step using the entire corpus each training loop. Both performed worse than the current version --- I wonder if the current implementation overfitted the dataset (which is quite homogeneous) that I have.

## Prerequisites (For Windows Machines)
Install the following libraries:

```bash
pip install nltk, pandas, csv, openai, tqdm, numpy
```

If you want to run the GPT model, you would need to acquire an OpenAI API key and set that as a system variable

```bash
setx OPENAI_API_KEY "your_api_key_here"
```
The whole test set (~7000 queries) costs about $0.5 and about 30 minutes on the GPT4o-mini model

## Running the code (For Windows Machines):
To rerun the data partitioning process, run:

```bash
cd ./data_processing/
python3 partition_data.py
```
Then move the 4 new files training_data.csv, test_data.csv, dev_data.csv, and dataset_stats.txt to ./data

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

Then, modify the parameters in get_statistics.py and run 

```bash
python3 get_statistics.py
```