from datasets import load_dataset
import pandas as pd


ds = load_dataset("sajjadhadi/disease-diagnosis-dataset")


train_df = pd.DataFrame(ds['train'])
test_df = pd.DataFrame(ds['test'])

# CSV হিসেবে সেভ করো
train_df.to_csv("symptom_disease_train.csv", index=False)
test_df.to_csv("symptom_disease_test.csv", index=False)
