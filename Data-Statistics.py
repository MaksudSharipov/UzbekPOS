from pathlib import Path
from collections import Counter, defaultdict
import statistics

# ====== FILE PATHS ======
DATA_PATH = Path("Dataset/upos-dataset-v7.txt")
OUT_PATH  = Path("Dataset/Dataset_stats_advanced.txt")

def parse_token_tag(pair: str):
    if "/" not in pair: return None, None
    token, tag = pair.rsplit("/", 1)
    return token.strip(), tag.strip()

tag_counts = Counter()
token_counts = Counter()
tag_token_map = defaultdict(Counter)
sentence_lengths = []
total_chars = 0
plus_sign_tokens = 0

if DATA_PATH.exists():
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            parts = line.split()
            sentence_lengths.append(len(parts))
            
            for p in parts:
                tok, tg = parse_token_tag(p)
                if tok and tg:
                    tag_counts[tg] += 1
                    token_counts[tok] += 1
                    tag_token_map[tg][tok] += 1
                    total_chars += len(tok)
                    if "+" in tok:
                        plus_sign_tokens += 1

total_pairs = sum(tag_counts.values())
sorted_tags = sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))

with OUT_PATH.open("w", encoding="utf-8") as out:
    out.write("=== Advanced Dataset Statistics ===\n\n")

    out.write("--- Sentence Metrics ---\n")
    out.write(f"Count: {len(sentence_lengths)}\n")
    out.write(f"Max Length: {max(sentence_lengths)} tokens\n")
    out.write(f"Min Length: {min(sentence_lengths)} tokens\n")
    out.write(f"Median Length: {statistics.median(sentence_lengths)}\n\n")

    out.write("--- Morphology/Complexity ---\n")
    out.write(f"Tokens with '+': {plus_sign_tokens} ({plus_sign_tokens/total_pairs:.2%})\n\n")

    out.write(f"{'TAG':<10} | {'COUNT':<8} | {'DENSITY':<8} | {'TTR*':<8} | {'TOP TOKENS'}\n")
    out.write("-" * 80 + "\n")

    for tag, count in sorted_tags:
        density = (count / total_pairs) * 100
        
        # Type-Token Ratio for this tag
        unique_toks_for_tag = len(tag_token_map[tag])
        ttr = unique_toks_for_tag / count
        
        top_5 = [f'"{t}"' for t, c in tag_token_map[tag].most_common(5)]
        
        out.write(f"{tag:<10} | {count:<8} | {density:>6.2f}% | {ttr:>7.3f} | {', '.join(top_5)}\n")

    out.write("\n*TTR (Type-Token Ratio): Lower means a small set of words is reused often.\n")

print(f"Advanced report generated at {OUT_PATH}")