import os
import csv
from collections import defaultdict
from datetime import datetime

# 判断是否为中文字符（包括中文标点符号）
def is_chinese(ch: str):
    code_point = ord(ch)
    return (
        0x4E00 <= code_point <= 0x9FFF or
        0x3400 <= code_point <= 0x4DBF or
        0x20000 <= code_point <= 0x2A6DF or
        0x2A700 <= code_point <= 0x2B73F or
        0x2B740 <= code_point <= 0x2B81F or
        0x2B820 <= code_point <= 0x2CEAF or
        0x2CEB0 <= code_point <= 0x2EBEF or
        0xF900 <= code_point <= 0xFAFF or
        0x3000 <= code_point <= 0x303F or  # CJK Symbols and Punctuation
        0xFF00 <= code_point <= 0xFFEF  # Halfwidth and Fullwidth Forms
    )

def remove_nul_characters(file_path):
    with open(file_path, 'rb') as f:
        content = f.read().replace(b'\x00', b'')
    return content.decode('utf-8', errors='ignore')

def process_csv(file_path, output_dir, polluted_dir):
    content = remove_nul_characters(file_path)
    tokens = []

    csvreader = csv.reader(content.splitlines())
    next(csvreader, None)  # 跳过表头
    for row in csvreader:
        if len(row) < 2:
            continue  # 跳过不完整的行
        try:
            tokens.append([int(row[0]), row[1], len(row[1]), 0, "", "", 0, 0.0, ""])
        except ValueError:
            continue  # 跳过包含非法字符的行

    # 统计所有包含中文字符的Token记录
    cn_tokens = []
    for record in tokens:
        token_id, token, length, _, _, _, _, _, _ = record
        chn = ''.join(filter(is_chinese, token))
        if chn:
            record[3] = len(chn)
            cn_tokens.append(record)

    total_records = len(cn_tokens)

    # 按中文字符数量排序
    cn_tokens.sort(key=lambda x: -x[3])

    # 标记longword和subword
    longword_id = 1
    for i, (token_id, token, total_len, cn_len, longword, subword, _, _, _) in enumerate(cn_tokens):
        if not longword:
            cn_tokens[i][4] = longword_id
            subword_id = 1
            for j, (sub_token_id, sub_token, sub_total_len, sub_cn_len, sub_longword, sub_subword, _, _, _) in enumerate(cn_tokens):
                if sub_token in token and not sub_longword and sub_token != token:
                    cn_tokens[j][4] = longword_id
                    cn_tokens[j][5] = subword_id
                    subword_id += 1
            longword_id += 1

    # 统计字符串在其他记录中出现的次数
    token_occurrences = defaultdict(int)
    for token_id, token, total_len, cn_len, longword, subword, _, _, _ in cn_tokens:
        for other_token_id, other_token, other_total_len, other_cn_len, other_longword, other_subword, _, _, _ in cn_tokens:
            if token in other_token:
                token_occurrences[token] += 1

    # 更新出现次数
    for i in range(len(cn_tokens)):
        token = cn_tokens[i][1]
        cn_tokens[i][6] = token_occurrences[token]

    # 更新longword的occurrences和计算percentage
    longword_occurrences = defaultdict(int)
    for record in cn_tokens:
        token_id, token, total_len, cn_len, longword, subword, occurrences, _, _ = record
        if longword and not subword:  # longword记录
            for sub_record in cn_tokens:
                if sub_record[4] == longword:
                    longword_occurrences[longword] += sub_record[6]
    
    for i, record in enumerate(cn_tokens):
        token_id, token, total_len, cn_len, longword, subword, occurrences, _, _ = record
        if longword and not subword:  # longword记录
            cn_tokens[i][6] = longword_occurrences[longword]
        cn_tokens[i][7] = f"{(cn_tokens[i][6] / total_records) * 100:.4f}%"

    # 提取Subword ID为空的记录
    polluted_tokens = [record for record in cn_tokens if not record[5]]

    # 获取模型名称和时间戳
    filename = os.path.basename(file_path)
    model_name, timestamp = filename.rsplit('-', 1)
    timestamp = timestamp.split('.')[0]

    # 生成输出文件名
    current_time = datetime.now().strftime('%y%m%d%H%M')
    output_filename = f"{model_name}-zh-fq-{current_time}.csv"
    output_filepath = os.path.join(output_dir, output_filename)

    # 输出为CSV文件
    with open(output_filepath, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Token ID', 'Token', 'Total Length', 'Chinese Length', 'Longword ID', 'Subword ID', 'Occurrences', 'Percentage', 'Pollute Type'])
        csvwriter.writerows(cn_tokens)

    # 生成污染记录的输出文件名
    polluted_filename = f"{model_name}-plabel-{current_time}.csv"
    polluted_filepath = os.path.join(polluted_dir, polluted_filename)

    # 输出污染记录为CSV文件
    with open(polluted_filepath, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Token ID', 'Token', 'Total Length', 'Chinese Length', 'Longword ID', 'Subword ID', 'Occurrences', 'Percentage', 'Pollute Type'])
        csvwriter.writerows(polluted_tokens)

def main():
    input_dir = 'Token-Decode'
    output_dir = 'Token-Frequencies'
    polluted_dir = 'Polluted-Tokens'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(polluted_dir):
        os.makedirs(polluted_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_dir, filename)
            process_csv(file_path, output_dir, polluted_dir)

if __name__ == "__main__":
    main()
