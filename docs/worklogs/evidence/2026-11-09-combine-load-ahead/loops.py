import re, sys
path, mnem, ldg = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(path).read()
start = text.index(".text.t4a_cuda_reduce:")
end = text.find("\n\t.section", start + 10)
body = text[start:end if end > 0 else None].splitlines()
blocks, cur, label = [], [], None
for line in body:
    m = re.match(r"^(\.L_x_\d+):", line.strip())
    if m:
        if label: blocks.append((label, cur))
        label, cur = m.group(1), [line]
    elif label:
        cur.append(line)
if label: blocks.append((label, cur))
for label, lines in blocks:
    ins = [l for l in lines if re.search(r"/\*[0-9a-f]{4}\*/", l)]
    txt = "\n".join(ins)
    back = any(re.search(r"BRA\s+(`\()?" + re.escape(label) + r"\b", l) for l in ins)
    if back and re.search(r"\b" + mnem + r"\b", txt) and re.search(r"\b" + ldg + r"\b", txt) and "SHFL" not in txt:
        n_ldg = len(re.findall(r"\b" + ldg + r"\b", txt)); n_op = len(re.findall(r"\b" + mnem + r"\b", txt))
        print(f"## loop block {label}: {len(ins)} instructions, {n_ldg} {ldg}, {n_op} {mnem}")
        for l in lines:
            l = re.sub(r"\s*/\* 0x[0-9a-f]+ \*/", "", l)
            print(l.rstrip())
