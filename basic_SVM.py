# idea:
# get word vector -> weight with TF-IDF to get document vectors
# remove the category "BOTH", merge into "ELECTION"
# use support vector machines to classify, 2 layer structure
# 1. does the document contain a story? Y -> 2; N -> NONE
# 2. is the story an election story? -> Y -> ELECTION; N - > NONELECTION