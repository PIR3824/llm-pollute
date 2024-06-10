import base64
import csv
import json
import sys
import os
import datetime
import sentencepiece as spm
import tiktoken
from tiktoken import get_encoding
from tiktoken.model import MODEL_TO_ENCODING

def decode_file(file_path, decode_mode, output_filename, file_type):
    def read_json_file(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

    if file_type == 'json':
        contents = read_json_file(file_path)
        if contents is None:
            return
        
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['tokenid', 'decoded_string'])
            for token, tokenid in contents.items():
                csvwriter.writerow([tokenid, token])

        print(f"Converted data has been written to {output_filename}")
        return

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        contents = file.read()
        lines = contents.splitlines()

    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['tokenid', 'decoded_string'])

        if decode_mode == 'base64':
            for tokenid, line in enumerate(lines):
                if line.strip():
                    token, rank = line.split()
                    decoded_token = base64.b64decode(token)
                    try:
                        decoded_string = decoded_token.decode('utf-8', errors='replace')
                    except Exception as e:
                        decoded_string = repr(decoded_token)
                    csvwriter.writerow([tokenid, decoded_string])
        elif decode_mode == 'sentencepiece':
            sp = spm.SentencePieceProcessor()
            sp.load(file_path)
            vocab_size = sp.GetPieceSize()
            for tokenid in range(vocab_size):
                decoded_string = sp.IdToPiece(tokenid)
                try:
                    decoded_string = decoded_string.encode('utf-8').decode('utf-8', errors='replace')
                except Exception as e:
                    decoded_string = repr(decoded_string)
                csvwriter.writerow([tokenid, decoded_string])
        elif decode_mode == 'none':
            for tokenid, line in enumerate(lines):
                csvwriter.writerow([tokenid, line])

    print(f"Decoded data has been written to {output_filename}")

def process_models_from_csv():
    with open('models.csv', 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header
        for row in csvreader:
            model_name, _, file_type, encode_type = row
            tokenizer_dir = 'Tokenizers'
            matched = False
            for filename in os.listdir(tokenizer_dir):
                if filename.startswith(model_name):
                    file_path = os.path.join(tokenizer_dir, filename)
                    output_filename = f"Token-Decode/{model_name}-{datetime.datetime.now():%y%m%d%H%M}.csv"
                    decode_file(file_path, encode_type.lower(), output_filename, file_type)
                    matched = True
                    break
            if not matched:
                print(f"No matching file found for model prefix: {model_name}")

def process_tiktoken_models():
    encodings = list(dict.fromkeys(MODEL_TO_ENCODING.values()))
    for enc_name in encodings:
        print(f"Processing encoding: {enc_name}")
        try:
            encoder = get_encoding(enc_name)
        except Exception as e:
            print(f"Error getting encoding for {enc_name}: {e}")
            continue
        
        output_filename = f"Token-Decode/tiktoken-{enc_name}-{datetime.datetime.now():%y%m%d%H%M}.csv"
        try:
            with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['tokenid', 'decoded_string'])
                for i in range(encoder.max_token_value - 20):  # Skip last 20 tokens to avoid special tokens
                    try:
                        print(f"Decoding token {i} for encoding {enc_name}")
                        decoded_string = encoder.decode([i])
                        csvwriter.writerow([i, decoded_string])
                    except Exception as e:
                        print(f"Error decoding token {i} for encoding {enc_name}: {e}")
                        continue
            print(f"Decoded data for tiktoken {enc_name} has been written to {output_filename}")
        except Exception as e:
            print(f"Error writing to file for encoding {enc_name}: {e}")

if __name__ == "__main__":
    if not os.path.exists('Token-Decode'):
        os.makedirs('Token-Decode')

    process_models_from_csv()
    process_tiktoken_models()
