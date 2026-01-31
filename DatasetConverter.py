# This script takes the final version of the dataset in its raw <token>/<tag> annotated format
# and converts into following formats:
# 1. Conll-U - best for NLP pipelines and parsers
# 2. TSV - best for POS dataset reading
# 3. JSONL - best for usage in HuggingFace, or any other usecases.


# Converts your "TOKEN/TAG TOKEN/TAG ..." (one sentence per line) dataset into:
#   1) CoNLL-U (.conllu)
#   2) TSV (.tsv)
#   3) JSONL (.jsonl)  [one sentence per line]
#
# Input example line:
#   Yer/NOUN yuzida/NOUN yashaydigan/VERB ... ./PUNCT
#
# Notes:
# - MWEs like "kerakli+bo‘lgan/VERB" are preserved as a single token.
# - Uses rsplit("/", 1) so tokens containing "/" earlier won't break unless multiple "/" exist.
# - Writes placeholder "_" for CoNLL-U fields you don't have.

import json
from pathlib import Path
from typing import List, Tuple, Optional

IN_PATH = Path("Dataset/upos-dataset-v7.txt")

OUT_CONLLU = Path("Dataset/uzpos.conllu")
OUT_TSV    = Path("Dataset/uzpos.tsv")
OUT_JSONL  = Path("Dataset/uzpos.jsonl")

def parse_item(item: str) -> Optional[Tuple[str, str]]:
    """Return (token, tag) if item is TOKEN/TAG with exactly one tag part; else None."""
    if "/" not in item:
        return None
    token, tag = item.rsplit("/", 1)
    if not token or not tag:
        return None
    return token, tag

def read_sentences(path: Path) -> List[List[Tuple[str, str]]]:
    """
    Reads dataset where each non-empty line is one sentence.
    Returns list of sentences; each sentence is list of (token, tag).
    Malformed items are skipped; if a line becomes empty, it is dropped.
    """
    sentences: List[List[Tuple[str, str]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            pairs: List[Tuple[str, str]] = []
            for item in line.split():
                parsed = parse_item(item)
                if parsed is None:
                    # If you want strict behavior, raise instead:
                    # raise ValueError(f"Malformed item on line {line_no}: {item}")
                    continue
                pairs.append(parsed)
            if pairs:
                sentences.append(pairs)
    return sentences

def write_conllu(sentences: List[List[Tuple[str, str]]], out_path: Path) -> None:
    """
    Writes CoNLL-U with minimal fields:
    ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC
    Unknown fields are '_'.
    """
    lines = []
    for sid, sent in enumerate(sentences, start=1):
        lines.append(f"# sent_id = {sid}")
        for i, (tok, tag) in enumerate(sent, start=1):
            cols = [
                str(i),
                tok,
                "_",
                tag,
                "_",
                "_",
                "_",
                "_",
                "_",
                "_",
            ]
            lines.append("\t".join(cols))
        lines.append("")  # sentence separator
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_tsv(sentences: List[List[Tuple[str, str]]], out_path: Path) -> None:
    """
    Writes a simple TSV:
    sent_id  token_id  form  upos
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("sent_id\ttoken_id\tform\tupos\n")
        for sid, sent in enumerate(sentences, start=1):
            for tid, (tok, tag) in enumerate(sent, start=1):
                f.write(f"{sid}\t{tid}\t{tok}\t{tag}\n")

def write_jsonl(sentences: List[List[Tuple[str, str]]], out_path: Path) -> None:
    """
    Writes one JSON record per sentence:
    {"sent_id": 1, "tokens": [...], "upos": [...]}
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sid, sent in enumerate(sentences, start=1):
            rec = {
                "sent_id": sid,
                "tokens": [t for t, _ in sent],
                "upos": [p for _, p in sent],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    sentences = read_sentences(IN_PATH)
    if not sentences:
        raise SystemExit("No sentences read. Check input path/format.")

    write_conllu(sentences, OUT_CONLLU)
    write_tsv(sentences, OUT_TSV)
    write_jsonl(sentences, OUT_JSONL)

    total_sents = len(sentences)
    total_tokens = sum(len(s) for s in sentences)

    print("✅ Conversion done.")
    print(f"Sentences: {total_sents}")
    print(f"Tokens:    {total_tokens}")
    print(f"CoNLL-U:   {OUT_CONLLU}")
    print(f"TSV:       {OUT_TSV}")
    print(f"JSONL:     {OUT_JSONL}")

if __name__ == "__main__":
    main()
