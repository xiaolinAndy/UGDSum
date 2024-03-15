# UGDSum
the code of UGDSum

## Instructions

We use the CSDS dataset as example.
1. Download the CSDS dataset and put it into Dialogue_Generation_Model/data/CSDS and Summarization_Model/data/raw.
2. run Dialogue_Generation_Model/data/CSDS/process_data.py to process the dataset.
3. run Dialogue_Generation_Model/run.sh to train and inference. (change the path in run.sh into the trained dialogue model)
4. run Dialogue_Generation_Model/data/CSDS/process_results.py to calculate UGD scores.
5. run Summarization_Model/data/process_data.py to process summarization dataset.
6. run Summarization_Model/test.py to generate summarization results.

