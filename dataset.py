import os
import glob
import json
import argparse
from pathlib import Path
import fitz
import re
import spacy
from tqdm import tqdm
import io

# Check if pytesseract is available
TESSERACT_AVAILABLE = False
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    print("Warning: pytesseract or PIL is not installed. OCR for formulas in images will be disabled.")
    print("To enable this feature, install with: pip install pytesseract pillow")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 10_000_000

def is_likely_formula(text_block):
    """Check if a text block is likely to contain a formula."""
    formula_indicators = [
        r'[=×∙÷+\-*/^√∫∑∏≈<>≤≥]',  # Mathematical operators
        r'[α-ωΑ-Ω]',  # Greek letters
        r'\b[a-zA-Z]\s*\^\s*[0-9]',  # Simple exponents like x^2
        r'\bsin\b|\bcos\b|\btan\b|\blog\b|\bln\b|\blim\b|\bmax\b|\bmin\b',  # Function names
        r'\bexp\b|\bsqrt\b|\bfrac\b',  # More function names
        r'\(.*[=+\-*/^].*\)'  # Expressions in parentheses with operators
    ]
    
    # Check if any of the formula indicators are present
    for pattern in formula_indicators:
        if re.search(pattern, text_block):
            return True
    
    return False

def process_formula(formula_text):
    """Process formula text to make it more suitable for training."""
    # Replace common math symbols with LaTeX equivalents
    replacements = {
        '×': ' \\times ',
        '÷': ' \\div ',
        '≈': ' \\approx ',
        '≤': ' \\leq ',
        '≥': ' \\geq ',
        '∑': ' \\sum ',
        '∫': ' \\int ',
        '∏': ' \\prod ',
        '√': ' \\sqrt ',
        'π': ' \\pi ',
        'θ': ' \\theta ',
        'α': ' \\alpha ',
        'β': ' \\beta ',
        'γ': ' \\gamma ',
        'δ': ' \\delta ',
        'ε': ' \\epsilon ',
        'λ': ' \\lambda ',
        'μ': ' \\mu ',
        'σ': ' \\sigma ',
        'τ': ' \\tau ',
        'ω': ' \\omega ',
        '∞': ' \\infty '
    }
    
    # Apply replacements
    for orig, repl in replacements.items():
        formula_text = formula_text.replace(orig, repl)
    
    # Convert superscripts (e.g., x² to x^2)
    formula_text = re.sub(r'([a-zA-Z0-9])²', r'\1^2', formula_text)
    formula_text = re.sub(r'([a-zA-Z0-9])³', r'\1^3', formula_text)
    
    # Wrap the formula in delimiters to indicate it's a formula
    return f"[FORMULA] {formula_text} [/FORMULA]"

def extract_text_and_formulas_from_pdf(pdf_path):
    """Extract text and formulas from a PDF file with special handling for formulas."""
    try:
        doc = fitz.open(pdf_path)
        result_text = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = ""
            
            # Extract all blocks of text
            blocks = page.get_text("blocks")
            
            for block in blocks:
                block_text = block[4]
                
                if is_likely_formula(block_text):
                    # Process and mark as formula
                    processed_formula = process_formula(block_text)
                    page_text += processed_formula + "\n"
                else:
                    # Regular text
                    page_text += block_text + "\n"
            
            # If Tesseract is available, try to process images that might contain formulas
            if TESSERACT_AVAILABLE:
                try:
                    # Get images that might contain formulas
                    image_list = page.get_images(full=True)
                    
                    # If there are images, check for formulas in them
                    if image_list:
                        for img_index, img_info in enumerate(image_list):
                            xref = img_info[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            # Convert image bytes to PIL Image
                            img = Image.open(io.BytesIO(image_bytes))
                            
                            # Try to extract text from the image (might be a formula)
                            img_text = pytesseract.image_to_string(img, config='--psm 6')
                            
                            # If the extracted text looks like a formula, add it
                            if img_text.strip() and is_likely_formula(img_text):
                                processed_formula = process_formula(img_text)
                                page_text += processed_formula + "\n"
                except Exception as e:
                    print(f"Error processing image in {pdf_path}, page {page_num+1}: {e}")
            
            result_text.append(page_text)
        
        return "\n".join(result_text)
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""

def clean_text(text):
    """Clean the extracted text while preserving formula markers."""
    # First, protect formula sections by replacing them temporarily
    formula_sections = []
    
    def replace_formula(match):
        formula_sections.append(match.group(0))
        return f"[FORMULA_PLACEHOLDER_{len(formula_sections)-1}]"
    
    # Find all formula sections and store them
    protected_text = re.sub(r'\[FORMULA\].*?\[/FORMULA\]', replace_formula, text, flags=re.DOTALL)
    
    # Now clean the text
    cleaned_text = protected_text
    # Replace multiple newlines with a single newline
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)
    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    # Remove any non-printable characters (except for formula placeholders)
    cleaned_text = re.sub(r'[^\x20-\x7E\n\[FORMULA_PLACEHOLDER_\d+\]]', '', cleaned_text)
    
    # Put back the formula sections
    for i in range(len(formula_sections)):
        placeholder = f"[FORMULA_PLACEHOLDER_{i}]"
        if placeholder in cleaned_text:
            cleaned_text = cleaned_text.replace(placeholder, formula_sections[i])
    
    return cleaned_text.strip()

def split_into_chunks_preserving_formulas(text, max_chunk_size=1024):
    """Split text into chunks, preserving formulas and trying to respect sentence boundaries."""
    # First, identify formula sections and protect them
    formula_pattern = r'\[FORMULA\].*?\[/FORMULA\]'
    formulas = re.finditer(formula_pattern, text, re.DOTALL)
    formula_positions = [(m.start(), m.end(), m.group(0)) for m in formulas]
    
    # Split text into sentences
    doc = nlp(text)
    sentences = []
    last_pos = 0

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        sent_start = text.find(sent_text, last_pos)
        if sent_start == -1:
            sentences.append(sent_text)
            continue
    
        sent_end = sent_start + len(sent)
        relevant_formulas = [f for f in formula_positions if 
                            (f[0] >= sent_start and f[0] < sent_end) or  
                            (f[1] > sent_start and f[1] <= sent_end) or  
                            (f[0] <= sent_start and f[1] >= sent_end)]
        
        sentences.append(sent_text)
        last_pos = sent_end
    
    # Now create chunks, being careful not to split formulas
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        new_chunk_len = len(current_chunk) + len(sentence)
        if new_chunk_len > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def process_pdfs(search_dir=".", output_dir="training_data", max_chunk_size=1024):
    """
    Search for training_data folder, process PDFs, and save text chunks with special handling for formulas.
    """
    # Find training_data directory
    training_data_dir = None
    for root, dirs, _ in os.walk(search_dir):
        if "training_data" in dirs:
            training_data_dir = os.path.join(root, "training_data")
            break
    
    if not training_data_dir:
        print("No 'training_data' directory found. Creating one in current directory.")
        training_data_dir = os.path.join(search_dir, "training_data")
        os.makedirs(training_data_dir, exist_ok=True)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find PDF files in training_data directory
    pdf_files = []
    for extension in ['pdf', 'PDF']:
        pdf_files.extend(glob.glob(os.path.join(training_data_dir, f"**/*.{extension}"), recursive=True))
    
    if not pdf_files:
        print(f"No PDF files found in {training_data_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files in {training_data_dir}")
    
    # Create a metadata file for tracking processed files
    metadata_file = os.path.join(output_dir, "metadata.json")
    metadata = {}
    chunk_count = 0
    formula_count = 0
    
    # Process each PDF file
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_name = os.path.basename(pdf_path)
        print(f"Processing {pdf_name}...")
        
        # Extract text with formula handling
        raw_text = extract_text_and_formulas_from_pdf(pdf_path)
        if not raw_text:
            print(f"Skipping {pdf_name} - no text extracted")
            continue
        
        # Count formulas
        formula_matches = re.findall(r'\[FORMULA\].*?\[/FORMULA\]', raw_text, re.DOTALL)
        current_formula_count = len(formula_matches)
        formula_count += current_formula_count
        
        # Clean text while preserving formulas
        cleaned_text = clean_text(raw_text)
        
        # Split into chunks, being careful with formulas
        chunks = split_into_chunks_preserving_formulas(cleaned_text, max_chunk_size)
        
        # Save chunks
        file_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_filename = f"{pdf_name.replace('.pdf', '').replace('.PDF', '')}_chunk_{i+1}.txt"
            chunk_path = os.path.join(output_dir, chunk_filename)
            
            with open(chunk_path, 'w', encoding='utf-8') as f:
                f.write(chunk)
            
            # Count formulas in this chunk
            chunk_formula_count = len(re.findall(r'\[FORMULA\].*?\[/FORMULA\]', chunk, re.DOTALL))
            
            file_chunks.append({
                "chunk_id": i+1,
                "filename": chunk_filename,
                "characters": len(chunk),
                "formulas": chunk_formula_count
            })
            chunk_count += 1
        
        # Update metadata
        metadata[pdf_name] = {
            "original_path": pdf_path,
            "chunks": file_chunks,
            "total_chunks": len(chunks),
            "total_formulas": current_formula_count
        }
    
    # Save metadata
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Processing complete! Created {chunk_count} text chunks from {len(pdf_files)} PDF files.")
    print(f"Detected and processed {formula_count} mathematical formulas.")
    print(f"Processed data saved to {output_dir}")
    print(f"Metadata saved to {metadata_file}")

def main():
    parser = argparse.ArgumentParser(description="Process PDF files into training data for language models with special handling for mathematical formulas")
    parser.add_argument("--search_dir", type=str, default=".", help="Directory to start searching for training_data folder")
    parser.add_argument("--output_dir", type=str, default="processed_training_data", help="Directory to save processed text files")
    parser.add_argument("--max_chunk_size", type=int, default=1024, help="Maximum size of text chunks in characters")

    args = parser.parse_args()
    process_pdfs(args.search_dir, args.output_dir, args.max_chunk_size)

if __name__ == "__main__":
    main()
