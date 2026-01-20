import json
import pandas as pd
import matplotlib.pyplot as plt
from openai_function import configure_openai, generate_chat_completion
from dotenv import load_dotenv
from paddleocr import PaddleOCR

load_dotenv()

def load_paddleocr_model():
    ocr = PaddleOCR(
        use_doc_orientation_classify=False, 
        use_doc_unwarping=False, 
        use_textline_orientation=False
    )
    print("PaddleOCR model loaded successfully.")
    return ocr

def extract_text_from_image(ocr, image_path):
    result = ocr.predict(image_path)
    for res in result:
        res.save_to_img("output")
        res.save_to_json("output")
    receipt_text = str(result[0]['rec_texts'])
    return receipt_text

RECEIPT_SYSTEM_PROMPT="""Your task is to analyze the receipt text and extract relevant information. Return a JSON object with the following structure: 
{
"items": [
    {
        "name": "Item Name 1", 
        "quantity": 1, 
        "unit_price": 10000, 
        "total_price": 10000
    },
    {
        "name": "Item Name 2", 
        "quantity": 3, 
        "unit_price": 15000, 
        "total_price": 45000
    }
], 
    "subtotal": 20000, 
    "tax": 2000, 
    "ppn": 1000, 
    "service_charge": 500, 
    "other_fees": 0, 
    "total": 23500, 
    "date": "2025-07-08", 
    "merchant": "Merchant Name", 
    "address": "Merchant Address"
}. 

Consider that price and quantities can appear before or after the item names in the text (e.g., both 'Nasi Goreng 25000' and '25000 Nasi Goreng' should be correctly parsed). Normalize quantities (default to 1 if not specified). Prices are usually in thousand rupiahs. If any field is missing in the receipt, set its value to null in the JSON. Return only the JSON object, with no extra text or explanation."""

def parse_receipt(text):
    client = configure_openai()
    messages = [
        {"role": "system", "content": RECEIPT_SYSTEM_PROMPT},
        {"role": "user", "content": text}
    ]
    
    response = generate_chat_completion(messages)
    
    if response and response.choices:
        content = response.choices[0].message.content
        try:
            receipt_data = json.loads(content)
            return receipt_data
        except json.JSONDecodeError:
            print("Failed to parse JSON from the response.")
            return None
    else:
        print("No valid response from OpenAI.")
        return None

def save_as_json(data):
    merchant_name = data['merchant']
    with open(f'output/{merchant_name}.json', 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    image_path = 'sample-images/image_1.jpeg'
    ocr = load_paddleocr_model()
    receipt_text = extract_text_from_image(ocr, image_path)
    receipt_json = parse_receipt(receipt_text)
    save_as_json(receipt_json)