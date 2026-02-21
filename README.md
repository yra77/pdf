# PDF Bank Statement Parser (offline)

Локальний скрипт для **Windows** (також працює на Linux/macOS), який перетворює банківські PDF-виписки у структурований `CSV`/`Excel`.

## Що вміє
- Працює **офлайн** (без онлайн-сервісів).
- Підтримує:
  - **текстові PDF** (через `pdfplumber`),
  - **скани PDF** (через OCR `pytesseract`).
- Нормалізує результат до полів:
  - `date`
  - `description`
  - `amount`
  - `balance`
  - `operation_type`
- Має профілі банків через YAML-конфіг (`bank_profiles.example.yaml`) для адаптації regex/ключових слів.

## Підготовка (Windows)

1. Встановіть Python 3.10+.
2. Встановіть Tesseract OCR:
   - завантажте інсталятор: <https://github.com/UB-Mannheim/tesseract/wiki>
   - під час інсталяції додайте мови `Ukrainian` + `English`.
3. (Рекомендовано) Додайте `tesseract.exe` в `PATH`.
4. Встановіть залежності:

```bat
pip install -r requirements.txt
```

## Швидкий запуск

### Через BAT-обгортку (Windows)

```bat
run_parser.bat statement.pdf result.csv
```

Для скану:

```bat
run_parser.bat statement_scan.pdf result.xlsx --ocr
```

### Напряму через Python

```bat
python statement_parser.py statement.pdf result.csv
python statement_parser.py statement_scan.pdf result.xlsx --ocr
```

## Профілі банків (адаптація)

1. Скопіюйте приклад:

```bat
copy bank_profiles.example.yaml bank_profiles.yaml
```

2. Додайте/відредагуйте профіль у `bank_profiles.yaml`.
3. Запустіть:

```bat
python statement_parser.py statement.pdf result.csv --profile-config bank_profiles.yaml --profile default
```

## Як це працює

1. Скрипт пробує витягти таблиці/рядки з PDF (`pdfplumber`).
2. Якщо транзакції не знайдені або увімкнено `--ocr`, включається OCR (`pytesseract`).
3. Рядки нормалізуються регулярними виразами та перетворюються в таблицю `pandas`.
4. Експорт у `.csv` або `.xlsx`.

## Обмеження

- Якість OCR сильно залежить від якості скану.
- Різні банки мають різну структуру виписок, тому для найкращого результату потрібне налаштування профілю.
- Якщо у вас PDF з «нестандартною» версткою, варто розширити `row_to_transaction` під конкретний формат банку.
