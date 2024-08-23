import pandas as pd
import re

def extract_column(input, column_name, output):
    """Extract sentences from a specific column in a CSV file and save them to a text file."""
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input)
    
    
    # Extract the specified column
    sentences = df[column_name].dropna().astype(str)  # Drop NaN values and convert to string
    
    # Write each sentence to a new line in the output text file
    with open(output, 'w', encoding='utf-8') as file:
        for sentence in sentences:
            file.write(sentence + '\n')


def tokenize(amr):
    """
    Tokenize an AMR string, keeping parentheses, colons, and labels separate,
     and lowercasing the tokens.
    """
    tokens = re.findall(r'[\w\-/]+|[:\(\)]', amr)
    return [token.lower().strip() for token in tokens]

def clean_amr_for_row(en_amr, cn_amr, fa_amr):
    # tokenization
    tokens_en = tokenize(en_amr)
    tokens_cn = tokenize(cn_amr)
    tokens_fa = tokenize(fa_amr)
    
    # find the intersection 
    common_tokens = set(tokens_en) & set(tokens_cn) & set(tokens_fa)

    # reconstruct AMRs, keeping only common tokens and parenthesis
    def filter_tokens(tokens, common_tokens):
        return ' '.join([token for token in tokens if token in common_tokens or token in '()'])

    cleaned_amr_en = filter_tokens(tokens_en, common_tokens)
    cleaned_amr_cn = filter_tokens(tokens_cn, common_tokens)
    cleaned_amr_fa = filter_tokens(tokens_fa, common_tokens)
    
    return cleaned_amr_en, cleaned_amr_cn, cleaned_amr_fa

def clean_amr_file(input_csv, output_csv):

    df = pd.read_csv(input_csv)

    for index, row in df.iterrows():
        en_amr = row['en_amr']
        cn_amr = row['cn_amr']
        fa_amr = row['fa_amr']

        cleaned_en_amr, cleaned_cn_amr, cleaned_fa_amr = clean_amr_for_row(en_amr, cn_amr, fa_amr)
        
        df.at[index, 'en_amr'] = cleaned_en_amr
        df.at[index, 'cn_amr'] = cleaned_cn_amr
        df.at[index, 'fa_amr'] = cleaned_fa_amr

    df.to_csv(output_csv, index=False)

    
if __name__ == "__main__":
    input_csv = 'MR_Project\Data\\aligned_amr_data_new2.csv'
    output_csv = 'cleaned_amr_data2.csv'
    clean_amr_file(input_csv, output_csv)