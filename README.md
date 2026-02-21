# PDF Bank Statement Parser (offline)

Локальний скрипт для **Windows** (також працює на Linux/macOS), який перетворює банківські PDF/JPG/PNG-виписки у структурований `CSV`/`Excel`.

## Що вміє
- Працює **офлайн** (без онлайн-сервісів).
- Підтримує:
  - **текстові PDF** (через `pdfplumber`),
  - **скани PDF/JPG/PNG** (через OCR `pytesseract`).
- Нормалізує результат до полів:
  - `date`
  - `transaction_type`
  - `description`
  - `amount`
  - `balance`
  - `operation_type`
- Має профілі банків через YAML-конфіг (`bank_profiles.example.yaml`) для адаптації regex/ключових слів.

## Підготовка (Windows)

1. Встановіть Python 3.10+.
2. Встановіть Tesseract OCR:
   - завантажте інсталятор: <https://github.com/UB-Mannheim/tesseract/wiki>
   - під час інсталяції додайте мови `Ukrainian` + `English` + `German` (або налаштуйте `ocr_lang` у профілі).
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
python statement_parser.py statement_photo.jpg result.csv
```

### GUI (Tkinter)

Запуск графічного інтерфейсу:

```bat
python statement_parser_gui.py
```

У GUI можна:
- обрати вхідний PDF/JPG/PNG,
- вказати вихідний файл `.csv` або `.xlsx`,
- увімкнути OCR,
- задати YAML-конфіг і назву профілю.

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


## Підтримка двох типів PDF

- **Тип 1 (`amount_and_balance_pdf`)**: у рядку є одна сума операції (часто зі знаком `-`/`+`) і залишок.
- **Тип 2 (`debit_credit_pdf`)**: у рядку окремі колонки `debit` і `credit` (опційно ще `balance`).
- Якщо в конкретному банку порядок колонок `credit`, потім `debit` — використовуйте `credit_debit_pdf`.
- Для OCR мову можна задавати у профілі через `ocr_lang`, наприклад `ukr+eng+deu`.
- Багаторядкові описи транзакцій обʼєднуються автоматично в один запис, якщо рядок продовжується без нової дати.

## Як це працює

1. Для PDF скрипт пробує витягти таблиці/рядки (`pdfplumber`).
2. Для JPG/PNG (або якщо увімкнено `--ocr`) використовується OCR (`pytesseract`).
3. Рядки нормалізуються регулярними виразами та перетворюються в таблицю `pandas`.
4. Експорт у `.csv` або `.xlsx`.

## Обмеження

- Якість OCR сильно залежить від якості скану.
- Різні банки мають різну структуру виписок, тому для найкращого результату потрібне налаштування профілю.
- Якщо у вас PDF з «нестандартною» версткою, варто розширити `row_to_transaction` під конкретний формат банку.
