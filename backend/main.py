from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama
import json
import io
import re
from pypdf import PdfReader

app = FastAPI(title="Receipt & Invoice Sorter Pro")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InvoiceRequest(BaseModel):
    raw_text: str

@app.get("/")
def read_root():
    return {"status": "Receipt Sorter Backend Online", "model": "gemma:2b"}

@app.post("/api/parse-invoice")
def parse_invoice(req: InvoiceRequest):
    return process_with_gemma(req.raw_text)

@app.post("/api/upload-invoice")
async def upload_invoice(file: UploadFile = File(...)):
    filename = file.filename.lower()
    
    if not (filename.endswith('.pdf') or filename.endswith('.csv') or filename.endswith('.json')):
        raise HTTPException(status_code=400, detail="Currently, only PDF, CSV, and JSON files are supported.")
    
    try:
        contents = await file.read()
        raw_text = ""
        
        if filename.endswith('.pdf'):
            pdf_reader = PdfReader(io.BytesIO(contents))
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    raw_text += extracted + "\n"
        elif filename.endswith('.csv') or filename.endswith('.json'):
            raw_text = contents.decode('utf-8')
            
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text. The file might be empty or an unreadable image.")
            
        return process_with_gemma(raw_text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

def process_with_gemma(raw_text: str):
    # 1. Deterministic Chunking
    chunks = re.split(r'(?i)Invoice\s*\d+\s*[:\-]', raw_text)
    
    if len(chunks) <= 1:
        chunks = [raw_text]
    else:
        chunks = [c.strip() for c in chunks if len(c.strip()) > 20]

    all_invoices = []
    valid_categories = ["Software", "Travel", "Office Supplies", "Marketing", "Legal", "Meals", "Logistics/Freight", "Facility Maintenance", "Misc"]

    for chunk in chunks:
        # 2. OCR Sanitization: Stitch broken pipe-separated tables back together
        clean_chunk = re.sub(r'\n\s*\|', ' |', chunk)
        clean_chunk = re.sub(r'\|\s*\n', '| ', clean_chunk)

        prompt = f"""
        You are a strict data extraction engine. Extract the invoice details from the text below.
        Read carefully. Tables might use pipe '|' characters. Ignore extra columns like 'Item Code' or 'Disc', just extract the description, quantity, unit_price, and total.
        
        Raw Text:
        "{clean_chunk}"

        Return a strict JSON object with this exact structure:
        {{
          "invoices": [
            {{
              "vendor_name": "Name of vendor",
              "invoice_date": "YYYY-MM-DD",
              "grand_total": 0.00,
              "currency": "USD",
              "tax_category": "", 
              "line_items": [
                {{
                  "description": "Exact item name",
                  "quantity": 1,
                  "unit_price": 0.00,
                  "total": 0.00
                }}
              ]
            }}
          ]
        }}
        """

        try:
            response = ollama.chat(
                model='gemma:2b',
                messages=[{'role': 'user', 'content': prompt}],
                format='json',
                options={'temperature': 0.0, 'num_ctx': 4096}
            )
            
            content = response['message']['content'].strip()
            
            if content.startswith("```json"): content = content[7:]
            if content.startswith("```"): content = content[3:]
            if content.endswith("```"): content = content[:-3]
                
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx+1]
            
            parsed_json = json.loads(content)
            
            invoices = []
            if "invoices" in parsed_json:
                invoices = parsed_json["invoices"]
            elif isinstance(parsed_json, list):
                invoices = parsed_json
            else:
                invoices = [parsed_json]
                
            for inv in invoices:
                if not inv.get("vendor_name") or inv.get("vendor_name") == "Name of vendor":
                    continue 
                    
                # 3. Dynamic Category Router (takes over since prompt leaves it blank)
                vendor_lower = str(inv.get("vendor_name", "")).lower()
                if "logistics" in vendor_lower or "shipping" in vendor_lower or "freight" in vendor_lower:
                    inv["tax_category"] = "Logistics/Freight"
                elif "cafe" in vendor_lower or "catering" in vendor_lower:
                    inv["tax_category"] = "Meals"
                elif "cleaning" in vendor_lower or "maintenance" in vendor_lower:
                    inv["tax_category"] = "Facility Maintenance"
                else:
                    inv["tax_category"] = "Misc"
                    
                calculated_sum = 0
                
                for item in inv.get("line_items", []):
                    try:
                        qty = float(item.get("quantity", 1))
                        unit = float(item.get("unit_price", 0))
                        total = float(item.get("total", 0))
                        
                        if "discount" in str(item.get("description", "")).lower() or "rebate" in str(item.get("description", "")).lower() or "surcharge" in str(item.get("description", "")).lower():
                            if total > 0 and "-" in raw_text: item["total"] = -total
                            if unit > 0 and "-" in raw_text: item["unit_price"] = -unit
                            
                        if qty > 0 and total != 0 and unit == 0:
                            item["unit_price"] = round(total / qty, 2)
                        elif qty > 0 and unit != 0 and total == 0:
                            item["total"] = round(qty * unit, 2)
                            
                        calculated_sum += item.get("total", 0)
                            
                    except (ValueError, TypeError):
                        pass
                
                # 4. Grand Total Mathematical Fallback
                if float(inv.get("grand_total", 0)) == 0.0:
                    inv["grand_total"] = round(calculated_sum, 2)
                        
                all_invoices.append(inv)
                
        except Exception as e:
            print(f"Skipped a chunk due to error: {str(e)}")
            continue

    if not all_invoices:
        raise HTTPException(status_code=500, detail="The AI failed to extract any valid invoices from this document.")
        
    return all_invoices