import pandas as pd
import numpy as np
from datasets import load_dataset

def build_dataset():
    print("1. Downloading datasets from Hugging Face...")
    
    # Load raw datasets
    hf_twitter = load_dataset("karthikarunr/Cyberbullying-Toxicity-Tweets", split="train")
    hf_dota = load_dataset("dffesalbon/dota-2-toxic-chat-data", split="train")
    hf_reddit = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train")

    # Convert to Pandas
    df_twitter = hf_twitter.to_pandas()
    df_dota = hf_dota.to_pandas()
    df_reddit = hf_reddit.to_pandas()

    print("2. Aligning schemas...")

    # Twitter (0 = Safe, 1 = Toxic, 2 = Severe)
    t_text = 'text' if 'text' in df_twitter.columns else df_twitter.columns[0]
    t_label = 'label' if 'label' in df_twitter.columns else df_twitter.columns[1]
    df_t = pd.DataFrame({'text': df_twitter[t_text].astype(str), 'label': df_twitter[t_label].astype(int)})

    # Dota 2
    df_d = pd.DataFrame({'text': df_dota['message'].astype(str), 'label': df_dota['target'].astype(int)})

    # Reddit (Converting float scores to our 0, 1, 2 system)
    def align_reddit(score):
        if pd.isna(score): return np.nan
        elif score < -1.0: return 0 
        elif score < 1.0: return 1   
        else: return 2               

    df_r = pd.DataFrame({'text': df_reddit['text'].astype(str), 'label': df_reddit['hate_speech_score'].apply(align_reddit)})

    print("3. Merging into master dataset...")
    df_master = pd.concat([df_t, df_d, df_r], ignore_index=True)

    # Clean out garbage data
    df_master.dropna(subset=['text', 'label'], inplace=True)
    df_master = df_master[df_master['text'].str.strip().astype(bool)]
    df_master = df_master[df_master['text'].str.len() > 2] 
    df_master['label'] = df_master['label'].astype(int)

    return df_master

if __name__ == "__main__":
    final_df = build_dataset()
    print("\n--- Success! ---")
    print(f"Total Rows: {len(final_df)}")
    print("\nLabel Distribution:")
    print(final_df['label'].value_counts())
    
    final_df.to_csv("clean_data.csv", index=False)
    print("\nSaved as 'clean_data.csv'. Ready for PyTorch!")