from pathlib import Path

IN_PATH  = Path("Dataset/uzpos.conllu")
OUT_PATH = Path("Dataset/uzpos_fixed.conllu")

def fix_mod_upos(in_path: Path, out_path: Path):
    out_lines = []
    
    count=0
    with in_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")

            # Keep comments and empty lines as-is
            if not line or line.startswith("#"):
                out_lines.append(line)
                continue

            cols = line.split("\t")

            # CoNLL-U token lines must have exactly 10 columns
            if len(cols) != 10:
                # keep malformed lines untouched (or raise if you prefer strict)
                print("Malformed line: ", cols)
                out_lines.append(line)
                continue

            upos = cols[3]
            xpos = cols[4]

            if upos == "MOD":
                count+=1
                cols[3] = "AUX"
                cols[4] = "MOD" if xpos == "_" else xpos

            out_lines.append("\t".join(cols))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    print(f"✅ MOD → AUX/XPOS fix applied")
    print("Total replaces:", count)
    print(f"📄 Input : {in_path}")
    print(f"📄 Output: {out_path}")

if __name__ == "__main__":
    fix_mod_upos(IN_PATH, OUT_PATH)
