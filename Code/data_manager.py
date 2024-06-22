import re
import csv

def first_clean(file_path):
    """Clean the file by removing lines starting with '#' that don't have '::' afterwards."""
    cleaned_lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("#") and not line.startswith("# ::"):
                continue
            cleaned_lines.append(line)
    return cleaned_lines

def parse_amr_file(file_path, language):
    """Parse an AMR file and return a list of tuples containing sentence and AMR graph."""
    lines = first_clean(file_path)
    data = ''.join(lines)

    # Regular expressions to match the components
    sent_pattern = re.compile(r'# ::snt (.*?)\n')
    amr_pattern = re.compile(r'(\(.*?\n\n)', re.DOTALL)

    # Extracting the components
    sents = sent_pattern.findall(data)
    amrs = amr_pattern.findall(data)

    if language == "en":
        zh_pattern = re.compile(r'# ::zh (.*?)\n')
        zh_sents = zh_pattern.findall(data)
        return list(zip(sents, zh_sents, amrs))
    
    return list(zip(sents, amrs))

def clean_data(parsed_data):
    """Remove entries where the sentence or AMR graph is None or empty."""
    return [(s, a) for s, a in parsed_data if s and a]

def clean_en_data(parsed_data):
    """Remove entries where the sentence, Chinese sentence, or AMR graph is None or empty."""
    return [(s, z, a) for s, z, a in parsed_data if s and z and a]

def write_to_csv(aligned_data, output_file_path):
    """Write aligned data to a CSV file."""
    with open(output_file_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['en_sent', 'en_amr', 'zh_sent', 'zh_amr', 'fa_sent', 'fa_amr'])
        writer.writerows(aligned_data)

# Paths to the input AMR files
en_file_path = 'Data/AMR/The-Little-Prince-AMR-en-full.txt'
cn_file_path = 'Data/AMR/The-Little-Prince-AMR-cn-full.txt'
fa_file_path = 'Data/AMR/The-Little-Prince-AMR-fa.txt'

# Parse each file
en_data = parse_amr_file(en_file_path, "en")
fa_data = parse_amr_file(fa_file_path, "fa")
cn_data = parse_amr_file(cn_file_path, "cn")

# Clean the parsed data
en_data = clean_en_data(en_data)
cn_data = clean_data(cn_data)
fa_data = clean_data(fa_data)

# Ensure all files have the same number of entries
assert len(en_data) == len(cn_data) == len(fa_data), f"Files have different number of entries. English: {len(en_data)}, Chinese: {len(cn_data)}, Persian: {len(fa_data)}"

# Align the data by order
aligned_data = []
for (en_sent, zh_sent, en_amr), (cn_sent, cn_amr), (fa_sent, fa_amr) in zip(en_data, cn_data, fa_data):
    aligned_data.append((en_sent, en_amr.strip(), zh_sent, cn_amr.strip(), fa_sent, fa_amr.strip()))

# Output file path
output_file_path = 'aligned_amr_data.csv'

# Write the aligned data to the CSV file
write_to_csv(aligned_data, output_file_path)

print("CSV file created successfully!")
