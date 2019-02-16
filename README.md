# ammi-2019-nlp

Lets use the following structure:

- all workflow function to run experiments can be defined in ipynb

- all models, datasets, helper functions can be defined in standalone py files such that it can be easily imported in many different ipynb files.

## Datasets

### Tokenizer

Actually stanfordnlp is quite heavy, needs to download some tree-stuff even if I only require tokenization. Instead for now I made a simple regexp there which works very fast and do tokenization for us.

### Amazon reviews

AmazonReviewsDataset inherits from torchtext.TabularDataset. One can find some initial routines (download, initialize, load vectors etc) in 01-day-LM folder/amazon_dataset.ipynb
