## Finetuning vs RAG - Kod till gymnasiearbete

Detta repo innehåller koden som användes i mitt gymnasiearbete
där jag jämförde RAG och Finetuning för att förbättra en
språkmodells kunskap om Star Wars.

### Data

Wookieepedia-dumpen laddas ner från:
https://starwars.fandom.com/wiki/Special:Statistics


### Filer

01_parse_data.py    - Parsar Wookieepedia XML-dump till ren text
02_finetune.py      - Finetunar modellen med LoRA (körs på Google Colab)
03_rag.py           - RAG-system som söker i artiklarna och svarar på frågor
04_evaluate.py      - Kör quizet på alla tre system och jämför resultaten
test_questions.json - 100 Star Wars-frågor med svar (från QuizBreaker)

Modell
------
Basmodellen är Qwen 2.5 0.5B som laddas ner automatiskt från
Hugging Face när man kör scripten.


Ordning
-------
1. Ladda ner Wookieepedia-dumpen och packa upp den
2. Kör 01_parse_data.py för att få wookieepedia_corpus.jsonl
3. Kör 02_finetune.py på Google Colab för att träna modellen
4. Kör 03_rag.py för att testa RAG-systemet
5. Kör 04_evaluate.py för att jämföra alla tre system
