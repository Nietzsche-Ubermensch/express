#!/usr/bin/env python3
import sys, json, re
try:
    import pytesseract
    from PIL import Image
except ImportError:
    print(json.dumps({"error": "pytesseract or Pillow not installed"}))
    sys.exit(0)

SETS = {'ALLURE','PYRO','PRIZM','SELECT','OPTIC','DONRUSS','CHROME','HERITAGE','MOSAIC','FINEST','LUMINANCE','OBSIDIAN','FLUX','CONTENDERS','IMMACULATE','NATIONAL TREASURES'}
MFRS = {'UPPER DECK':'Upper Deck','TOPPS':'Topps','PANINI':'Panini','LEAF':'Leaf'}
PROMOS = {'AEW':'AEW','ALL ELITE':'AEW','WWE':'WWE','NXT':'NXT','NBA':'NBA','NFL':'NFL','MLB':'MLB','NHL':'NHL','UFC':'UFC'}
SKIP = SETS | {'AEW','WWE','NXT','CONGRATULATIONS','SUPERSTAR','COMMEMORATIVE','UPPER DECK','TOPPS','PANINI','LOGO PATCH','PATCH CARD','RESERVED'}

def extract(path):
    meta = {}
    try: img = Image.open(path)
    except: return {"error": "cannot open image"}
    try: text = pytesseract.image_to_string(img, config='--psm 1')
    except:
        try: text = pytesseract.image_to_string(img, config='--psm 6')
        except Exception as e: return {"error": str(e)}
    lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 1]
    back_sigs = ['height:','from:','finishing move','congratulations','printed in','all rights reserved']
    for mk in MFRS: back_sigs.append(mk.lower())
    meta['is_back'] = sum(1 for s in back_sigs if s in text.lower()) >= 2
    for line in lines:
        lu = line.upper().strip()
        if 'year' not in meta:
            m = re.search(r'\b(20[12]\d)\b', line)
            if m: meta['year'] = m.group(1)
        for k,v in MFRS.items():
            if k in lu and 'manufacturer' not in meta: meta['manufacturer'] = v
        for s in SETS:
            if s in lu and 'set_name' not in meta: meta['set_name'] = s.title()
        for k,v in PROMOS.items():
            if k in lu and 'promotion' not in meta: meta['promotion'] = v
        m = re.search(r'(\d{1,4})\s*/\s*(\d{1,4})', line)
        if m and 'serial' not in meta: meta['serial'] = f"{m.group(1)}/{m.group(2)}"
        m = re.search(r'HEIGHT:\s*(.+)', line, re.I)
        if m and 'height' not in meta: meta['height'] = m.group(1).strip().rstrip('|').strip()
        m = re.search(r'FROM:\s*(.+)', line, re.I)
        if m and 'from_location' not in meta: meta['from_location'] = m.group(1).strip()
        m = re.search(r'FINISHING MOVE:\s*(.+)', line, re.I)
        if m and 'finishing_move' not in meta: meta['finishing_move'] = m.group(1).strip()
    for i, line in enumerate(lines):
        if 'HEIGHT' in line.upper() and i > 0:
            for j in range(i-1, max(i-4,-1), -1):
                c = lines[j].strip()
                if len(c) > 3 and any(ch.isalpha() for ch in c):
                    name = re.sub(r'[\u2122\u00ae\u00a9]','',c).strip()
                    if name.isupper(): name = name.title()
                    meta['player_name'] = name; break
            break
    if 'player_name' not in meta:
        for line in lines:
            lc = line.strip()
            if lc.isupper() and 4 <= len(lc) <= 35 and not any(sw in lc for sw in SKIP) and re.match(r"^[A-Z\s.\-']+$", lc) and ' ' in lc:
                meta['player_name'] = lc.title(); break
    if meta.get('is_back'):
        bio = []; in_bio = False
        for line in lines:
            lc = line.strip()
            if not in_bio and len(lc) > 30 and lc[0].isupper() and not lc.isupper() and any(c in lc for c in '.,'):
                in_bio = True
            if in_bio:
                if lc.isupper() or '\u00a9' in lc or len(lc) < 5: break
                bio.append(lc)
        if bio: meta['bio_text'] = ' '.join(bio)
    meta['ocr_confidence'] = min(100, sum(1 for k in ['player_name','year','manufacturer','set_name','promotion','serial','height','from_location'] if k in meta) * 12)
    return meta

if __name__ == '__main__':
    if len(sys.argv) < 2: print(json.dumps({"error": "usage: ocr_worker.py <image_path>"})); sys.exit(0)
    print(json.dumps(extract(sys.argv[1]), default=str))
