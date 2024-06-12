import fitz  # PyMuPDF
import pdfminer
from pdfminer.high_level import extract_text
from tika import parser
import pytesseract
from PIL import Image
import io
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import sys

import google.generativeai as genai
# Set your API key here
API_KEY = "AIzaSyBU9ZVq4OfPrHCxeWjvLtAgKFdHaWY6J_8"

genai.configure(api_key=API_KEY)
# Whitelist of additional characters to keep
WHITELIST = "Σ±÷×"

nltk.download('punkt')

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\PC\PycharmProjects\tesseract.exe'

def extract_text_from_pdf(pdf_path):
    try:
        # Try extracting text using PyMuPDF
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        if text.strip():
            return text
    except Exception as e:
        print(f"PyMuPDF extraction failed: {e}")

    try:
        # Try extracting text using PDFMiner
        text = extract_text(pdf_path)
        if text.strip():
            return text
    except Exception as e:
        print(f"PDFMiner extraction failed: {e}")

    try:
        # Try extracting text using Tika
        raw = parser.from_file(pdf_path)
        text = raw['content']
        if text.strip():
            return text
    except Exception as e:
        print(f"Tika extraction failed: {e}")

    return None

def ocr_image(image):
    return pytesseract.image_to_string(image)

def handle_scanned_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        # img = Image.open(io.BytesIO(pix.tobytes()))
        # text += ocr_image(img)
    return text

def extract_images_from_pdf(pdf_path):
    images = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img = Image.open(io.BytesIO(image_bytes))
            images.append(img)
    return images


def clean_text(text):
    # Remove irrelevant characters and whitespace
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(f'[^\x20-\x7E{WHITELIST}]+', '', text)
    return text.strip()

def segment_sentences(text):
    # Split text into sentences
    sentences = sent_tokenize(text)
    return sentences

def tokenize_sentence(sentence):
    # Tokenize sentence into words
    tokens = word_tokenize(sentence)
    return tokens

def preprocess_text(text):
    cleaned_text = clean_text(text)
    sentences = segment_sentences(cleaned_text)
    tokenized_sentences = [tokenize_sentence(sentence) for sentence in sentences]
    return tokenized_sentences


# def gpt4_process_text(text, task="summarize"):
#     if task == "summarize":
#         prompt = f"Please summarize the following text:\n\n{text}"
#     else:
#         prompt = f"{task}:\n\n{text}"

#     response = openai.Completion.create(
#         model="text-davinci",
#         prompt=prompt,
#         max_tokens=1024,
#         n=1,
#         stop=None,
#         temperature=0.7
#     )
#     return response.choices[0].text.strip()

"""
Install the Google AI Python SDK

$ pip install google-generativeai

See the getting started guide for more information:
https://ai.google.dev/gemini-api/docs/get-started/python
"""



# Create the model
# See https://ai.google.dev/api/python/google/generativeai/GenerativeModel

def get_response(text):
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        # safety_settings = Adjust safety settings
        # See https://ai.google.dev/gemini-api/docs/safety-settings
        )

    chat_session = model.start_chat(
        history=[
        ]
        )
    response = chat_session.send_message(f"Please summarize the following text:\n\n{text}")
    return response.text

def convert_pdf_to_txt(pdf_path, txt_path):
    text = extract_text_from_pdf(pdf_path)
    if text is None:
        print("Standard text extraction failed, trying OCR.")
        text = handle_scanned_pdf(pdf_path)

    images = extract_images_from_pdf(pdf_path)
    if images:
        for i, image in enumerate(images):
            image_text = ocr_image(image)
            text += f"\n[Image {i+1}]:\n{image_text}"
    
    # Preprocess text
    tokenized_sentences = preprocess_text(text)
    
    # Convert tokenized sentences back to text format
    final_text = "\n".join([" ".join(tokens) for tokens in tokenized_sentences])
        # Use GPT-4 to refine the processed text
    summarized_text = get_response(final_text)
    return summarized_text


if __name__ == "__main__":
    # Ensure correct usage
    if len(sys.argv) != 3:
        print("Usage: python test.py <input_pdf_path> <output_txt_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    txt_path = sys.argv[2]
    summarized_text = convert_pdf_to_txt(pdf_path, txt_path)

