from utils._pdf_full_pipeline import PDFToQdrantPipeline
from dotenv import load_dotenv
import os
load_dotenv()
 
# Initialize pipeline
pipeline = PDFToQdrantPipeline(
    embedding_model=os.getenv("EMBEDDING_MODEL"),
    qdrant_host=os.getenv("QDRANT_HOST"),
    qdrant_port=int(os.getenv("QDRANT_PORT"))
)

from pathlib import Path

# --- Configuration ---
INPUT_FOLDER = os.getenv("RFQ_FOLDER")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
print(INPUT_FOLDER)
# Convert string path to a Path object

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

# Resolve RFQ folder
RFQ_FOLDER = BASE_DIR / os.getenv("RFQ_FOLDER")

if not RFQ_FOLDER.exists():
    raise FileNotFoundError(f"RFQ folder not found: {RFQ_FOLDER}")

# .rglob("*.pdf") finds all PDFs in this folder and all subfolders
pdf_files = [p for p in BASE_DIR.rglob("*") if p.suffix.lower() == ".pdf"]

print(f"Found {len(pdf_files)} PDF files. Starting processing...")

all_chunks = [] # Create a master list

for pdf_path in pdf_files:
    doc_id = pdf_path.stem
    print(f"--- Processing: {pdf_path.name} ---")

    try:
        chunks = pipeline.process_pdf(
            pdf_path=str(pdf_path),
            doc_id=doc_id,
            metadata={"original_filename": pdf_path.name},
            # ... other params ...
        )

        if chunks:
            all_chunks.extend(chunks) # Add to the master list
            print(f"Prepared {len(chunks)} chunks from: {pdf_path.name}")

    except Exception as e:
        print(f"Failed to process {pdf_path.name}/: {e}")

# UPLOAD ALL CHUNKS AT THE END
if all_chunks:
    print(f"\nUploading total of {len(all_chunks)} chunks to Qdrant...")
    pipeline.upload_to_qdrant(all_chunks, collection_name=COLLECTION_NAME)
    print("Done!")
else:
    print("No chunks found to upload.")

print("\nAll files have been processed.")