# CS322Final-ElectionNewsClassification-AI

A final project for CS 322: Natural Language Processing, by Dake Peng 25' at Carleton College, Northfield. MN, United States.

# Prerequisites (For Windows Machines)

Install the following libraries:

```bash
pip install nltk, pandas, csv, openai, tqdm, numpy
```

If you want to run the GPT model, you would need to acquire an OpenAI API key and set that as a system variable

```bash
setx OPENAI_API_KEY "your_api_key_here"
```
The whole test set (~3000 queries) costs about $0.5 on the GPT4o-mini model

# Running the code (For Windows Machines):

To run all processes of the SVM model (with training, testing and development sets prepared), run the following programs in order:

```bash
python3 get_tf_idf.py
python3 get_document_vectors.py
python3 basic_SVM.py
python3 get_statistics.py
```

This will take pretty long for the first time, since the programs will download models from nltk and gensim, run extensive processing process to obtain the document vectors, and run a long training process for the SVM. The intermediate files (tf_idf weights, and document vectors) are stored in .json formats in ./data. To run a quick version of the SVM with these offline data, run:

```bash
python3 basic_SVM.py
python3 get_statistics.py
```

To run the GPT model, run

```bash
python3 0shot_GPT4o.py
```