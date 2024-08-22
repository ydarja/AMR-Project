import re
import csv
from collections import Counter

def first_clean(file_path):
    """Clean the file by removing lines starting with '#' that don't have '::' afterwards."""
    cleaned_lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("#") and not line.startswith("# ::"):
                continue
            cleaned_lines.append(line)
    return cleaned_lines

def extract_en_id(entry):
    match = re.search(r'# ::id lpp_1943\.(\d+)', entry)
    if match:
        return int(match.group(1))
    return None  

def extract_cn_id(entry):
    match = re.search(r'# ::id test_amr\.(\d+)', entry)
    if match:
        return int(match.group(1))
    return None

def extract_fa_id(entry):
    match = re.search(r'# ::id lpp_fa\.(\d+)', entry)
    if match:
        return int(match.group(1))
    return None  

def parse_amr_file(file_path, language):
    """Parse an AMR file and return a list of tuples containing sentence and AMR graph."""
    lines = first_clean(file_path)
    data = ''.join(lines)

    # regex to match the components
    sent_pattern = re.compile(r'# ::snt (.*?)\n')
    amr_pattern = re.compile(r'\(.*?\)(?=\n\n|$)', re.DOTALL)

    # extracting the ids
    if language == "en":
        id_function = extract_en_id
    elif language == "cn":
        id_function = extract_cn_id
    elif language == "fa":
        id_function = extract_fa_id

    ids = []
    sents = []
    amrs = []

    with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                id = id_function(line)
                if id is not None:
                    ids.append(id)
        
    sents = sent_pattern.findall(data)
    amrs = amr_pattern.findall(data)


    print(f"{language} IDs:", len(ids))  
    print(f"{language} Sentences:", len(sents))  
    print(f"{language} AMRs:", len(amrs))  

    # keeping chinese sentences for English annotations
    '''
    if language == "en":
        zh_pattern = re.compile(r'# ::zh (.*?)\n')
        zh_sents = zh_pattern.findall(data)
        return list(zip(ids, sents, zh_sents, amrs))
    '''

    return list(zip(ids, sents, amrs))

def clean_data(parsed_data):
    """Remove entries where the ID, sentence, or AMR graph is None or empty, and print the IDs of removed entries."""
    cleaned_data = []
    for i, s, a in parsed_data:
        if i and s and a:
            cleaned_data.append((i, s, a))
        else:
            print(f"Removed entry with ID: {i}, Sentence: {s}, AMR: {a}")
    return cleaned_data

def clean_en_data(parsed_data):
    """Remove entries where the sentence, Chinese sentence, or AMR graph is None or empty."""
    return [(i, s, z, a) for i, s, z, a in parsed_data if i and s and z and a]

def align_data(en_data, cn_data, fa_data):
    """Align data based on IDs."""
    # Convert lists to dictionaries with IDs as keys
    en_dict = {i: (s, a) for i, s, a in en_data}
    cn_dict = {i: (s, a) for i, s, a in cn_data}
    fa_dict = {i: (s, a) for i, s, a in fa_data}


    # Find common IDs
    common_ids = set(en_dict.keys()) & set(cn_dict.keys()) & set(fa_dict.keys())
    # print("Common IDs:", common_ids)


    aligned_data = []
    for i in common_ids:
        en_sent, en_amr = en_dict[i]
        cn_sent, cn_amr = cn_dict[i]
        fa_sent, fa_amr = fa_dict[i]
        aligned_data.append((i, en_sent, en_amr, cn_sent, cn_amr, fa_sent, fa_amr))
    return aligned_data    

def write_to_csv(aligned_data, output_file_path):
    """Write aligned data to a CSV file, including a common ID."""
    with open(output_file_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'en_sent', 'en_amr', 'cn_sent', 'cn_amr', 'fa_sent', 'fa_amr'])
        writer.writerows(aligned_data)


# input AMR files
en_file = 'MR_Project\Data\AMR\The-Little-Prince-AMR-en-full.txt'
cn_file = 'MR_Project\Data\AMR\The-Little-Prince-AMR-cn-full.txt'
fa_file = 'MR_Project\Data\AMR\The-Little-Prince-AMR-fa.txt'

        
# parsing
en_data = parse_amr_file(en_file, "en")
fa_data = parse_amr_file(fa_file, "fa")
cn_data = parse_amr_file(cn_file, "cn")

# cleaning
en_data = clean_data(en_data)
cn_data = clean_data(cn_data)
fa_data = clean_data(fa_data)
   

# Ensure all files have the same number of entries
# assert len(en_data) == len(cn_data) == len(fa_data), f"Files have different number of entries. English: {len(en_data)}, Chinese: {len(cn_data)}, Persian: {len(fa_data)}"

# Align the data
aligned_data = align_data(en_data, cn_data, fa_data)

# Output file path
output_file_path = 'aligned_amr_data_new2.csv'

# Write the aligned data to the CSV file
write_to_csv(aligned_data, output_file_path)

print("CSV file created successfully!")
