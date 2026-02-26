# Parsar Wookieepedia XML-dumpen och gör om den till ren text
# Kräver: pip install mwparserfromhell

import json
import re
import xml.etree.ElementTree as ET
import mwparserfromhell

XML_file = "starwars_pages_current.xml"
output = "wookieepedia_corpus.jsonl"
minlength = 500 

# MediaWiki XML namespace som behövs för att hitta rätt taggar
MW_NS = "{http://www.mediawiki.org/xml/export-0.11/}"


def clean_wikitext(wikitext):
    
    parsed = mwparserfromhell.parse(wikitext)
    for template in parsed.filter_templates():
        try:
            parsed.remove(template)
        except ValueError:
            pass

    # Konvertera till ren text
    text = parsed.strip_code()
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

# Går igenom alla sidor i XML-filen
includes = 0
skipped_short = 0
skipped_redirect = 0
total = 0

with open(output, 'w', encoding='utf-8') as ut:
    for event, elem in ET.iterparse(XML_file, events=('end',)):
        if elem.tag != f"{MW_NS}page":
            continue

        # Vi vill bara ha vanliga artiklar (namespace 0)
        ns = elem.find(f"{MW_NS}ns")
        if ns is None or ns.text != "0":
            elem.clear()
            continue

        total += 1
        print(total, end='\r')

        titel = elem.find(f"{MW_NS}title")
        revision = elem.find(f"{MW_NS}revision")
        if revision is None:
            elem.clear()
            continue

        text_elem = revision.find(f"{MW_NS}text")
        if text_elem is None or not text_elem.text:
            elem.clear()
            continue

        wikitext = text_elem.text

        # Skippar redirects
        if wikitext.lower().startswith("#redirect"):
            skipped_redirect += 1
            elem.clear()
            continue

        clean_text = clean_wikitext(wikitext)

        # Skippar korta artiklar
        if len(clean_text) < minlength:
            skipped_short += 1
            elem.clear()
            continue

        # Spara artikeln som JSON
        article = {"title": titel.text, "text": clean_text}
        ut.write(json.dumps(article, ensure_ascii=False) + '\n')
        includes += 1

        if total % 10000 == 0:
            print(f"  {total:,} pages cleaned, {includes:,} included...")

        elem.clear()

print(f"\ndone")
print(f"  total: {total:,}")
print(f"  included: {includes:,}")
print(f"  skipped (short): {skipped_short:,}")
print(f"  skipped (redirects): {skipped_redirect:,}")
print(f"  saved to: {output}")
