# 🧠 PDF Formula Extractor & Preprocessor

This Python program processes PDF files to extract text content and detect mathematical formulas. It's designed to prepare high-quality, clean text chunks (including formulas) for training NLP or LLM models.

---

## ✨ Features

- 🔍 **Scans PDFs** for mathematical content and regular text
- 🧪 **Identifies formulas** using regex and heuristics
- 🧼 **Cleans** text while preserving formula integrity
- 📦 **Chunks text** intelligently for training purposes
- 🧠 **OCR-enabled** (optional) to extract formulas from images using Tesseract
- 📁 Generates structured **metadata** for every processed file

---

## 📂 Folder Structure

```
your-project/
│
├── training_data/               # Put your source PDF files here
│
├── processed_training_data/    # Output folder for cleaned, chunked text
│
├── metadata.json               # Metadata of the processing
│
└── main_script.py              # The script you run (this one)
```

---

## 🛠 Installation

### 📦 Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```txt
PyMuPDF
spacy
tqdm
pytesseract
pillow
```

Don't forget to download the spaCy model:

```bash
python -m spacy download en_core_web_sm
```

### 🧠 Optional: OCR Support

For OCR of formulas inside images (embedded in PDFs), install Tesseract and Pillow:

```bash
sudo apt-get install tesseract-ocr     # or use brew install tesseract (macOS)
pip install pytesseract pillow
```

---

## 🚀 Usage

From the command line:

```bash
python main_script.py --search_dir "." --output_dir "processed_training_data" --max_chunk_size 1024
```

### Arguments:

| Argument         | Description                                                        | Default                   |
|------------------|--------------------------------------------------------------------|---------------------------|
| `--search_dir`   | Root directory to look for the `training_data` folder              | `.` (current directory)   |
| `--output_dir`   | Where to save the processed, chunked text                          | `processed_training_data` |
| `--max_chunk_size` | Maximum number of characters per text chunk                      | `1024`                    |

---

## 🧮 Formula Detection & Markup

Formulas are wrapped in special tags:
```
[FORMULA] E = mc^2 [/FORMULA]
```

This makes it easy to distinguish and process them during downstream training or data manipulation.

---

## 📊 Output Metadata

After processing, a `metadata.json` file is created with stats like:

- Number of chunks per file
- Number of formulas detected
- Original file path

---

## 📌 Notes

- Works on scanned PDFs if Tesseract is available.
- Tries to preserve sentence boundaries and not split formulas during chunking.
- Max spaCy input size increased to allow large files.

---

## 🧑‍💻 Author

**Your Name** — [@yourhandle](https://github.com/yourhandle)  
Feel free to contribute or open issues!

---

## 🧾 License

MIT License — see [LICENSE](./LICENSE) for details.

---

Would you like me to also generate the `requirements.txt` and `LICENSE` files?
