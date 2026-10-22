# OCR: скриншоттар бойынша есеп

`python -m safenet.multimodal` арқылы синтетикалық скриншоттарда есептелген.

## Нақты OCR

**Іске қосылмады — OCR қозғалтқышы жоқ.** Tesseract binary not found; easyocr is not installed. Install Tesseract with Russian/Kazakh data (Windows: `winget install UB-Mannheim.TesseractOCR`, then put rus/kaz .traineddata into models/tessdata; Linux: `apt install tesseract-ocr tesseract-ocr-rus tesseract-ocr-kaz`) or `pip install easyocr`.

## Мінсіз OCR (суреттегі мәтіннің өзі)

Скриншот мәтіндерінің скорингін OCR қателерінсіз тексеру үшін:

- Схема скриншоттары `medium`/`high`: **90.0%** (n = 90)
- Қиын негатив скриншоттары `low`: **100.0%** (n = 40)

## Алдын ала өңдеу

Сұр түс → кіші суретті 2 есе үлкейту → қараңғы тақырыпты инверсиялау → медианалық сүзгі → Otsu бинаризациясы → тұтас түсті жолақтарды (тақырып) алып тастау → қисаюды түзету (±5°, жол профилінің өткірлігі бойынша) → ақ фонда қара мәтін.
