import json
from collections import Counter

with open('project/data/output/eq_final.json', encoding='utf-8') as f:
    data = json.load(f)

# Check paper 2401.13506 — was the fanning example
paper = data.get('2401.13506', {})
print('=== 2401.13506 ===')
for eq_num in sorted(paper.keys(), key=lambda x: (int(x) if x.isdigit() else 999)):
    rels = paper[eq_num].get('relations', {})
    strong = [(k, v) for k, v in rels.items() if v.get('grade') == 'strong']
    if strong:
        print(f'  eq {eq_num} -> {len(strong)} strong rels:')
        for k, v in strong[:2]:
            desc = v["description"][:80]
            print(f'    -> eq {k}: {repr(desc)}')

# Count unique descriptions among all strong pairs
all_descs = []
for arxiv_id, eqs in data.items():
    for eq_num, entry in eqs.items():
        for k, v in entry.get('relations', {}).items():
            if v.get('grade') == 'strong':
                all_descs.append(v.get('description', '')[:60])

print()
print(f'Total strong pairs: {len(all_descs)}')
print(f'Unique descriptions: {len(set(all_descs))}')
print('Top repeated strong descriptions:')
for desc, cnt in Counter(all_descs).most_common(8):
    print(f'  count={cnt}  {repr(desc)}')

# Meaning spot-check — first 5 non-empty meanings
print()
print('=== Meaning samples ===')
shown = 0
for arxiv_id, eqs in data.items():
    for eq_num, entry in eqs.items():
        m = entry.get('meaning', '')
        if m:
            print(f'  {arxiv_id}/eq{eq_num}: {repr(m[:100])}')
            shown += 1
            if shown >= 5:
                break
    if shown >= 5:
        break
