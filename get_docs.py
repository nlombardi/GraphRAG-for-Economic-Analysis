import pdfplumber
from urllib3.exceptions import NewConnectionError, HTTPError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


class DocumentCollector:
    """
    Processes unstructured documents into chunks suitable for graph construction and vector embedding.

    There are various chunking strategies, here we are creating sizeable chunks to maintain semantic coherence within
    paragraphs while keeping them small enough for effective retrieval.
    """
    def __init__(self, docs_path: Path = Path.cwd() / 'docs'):
        # RecursiveCharacterTextSplitter tries different separators in order, this preservers document structure better
        # than simple character splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "! ",
                "? "
            ]
        )
        self.docs_path = docs_path

    def ocr_document(self, doc_path: Path, use_google_vision: bool = False, use_local_llm: bool = False) -> dict:
        """
        Loads pdf/word docs to parse the information. OCR the documents using either pdfplumber or a vision llm.

        I have provided a couple ways to OCR the documents, using pdfplumber and using vision LLMs (google and local).

        For google you need to have gcloud initialized. It is recommended to store the pdfs in the cloud and run
        async_batch_annotate_files. For local implementation we can only batch files in pages of 5.

        For the purposes of this tutorial pdfplumber is sufficient and the other OCR method is for example or if you are
        having issues parsing the pdf.

        :param doc_path: Path to document.
        :param use_google_vision: If you want to use google vision LLM to parse the pdf
        :param use_local_llm: if you want to use local llm to parse the pdf -> set model and BASE_URL in .env file
        :return: A dict representing one document containing the str of the doc and metadata
        """

        # Function to call google's cloud vision model for OCR
        def ocr_google_llm(doc_path: Path, page_range: range) -> str:
            # Need to have gcloud CLI initialized
            from google.cloud import vision_v1
            ocr = vision_v1.ImageAnnotatorClient()
            mime_type = "application/pdf"

            parsed_text = []
            with open(doc_path, "rb") as f:
                content = f.read()
            input_config = {"mime_type": mime_type, "content": content}
            features = [{"type_": vision_v1.Feature.Type.DOCUMENT_TEXT_DETECTION}]
            pages = list(page_range)
            requests = [{"input_config": input_config, "features": features, "pages": pages}]
            try:
                response = ocr.batch_annotate_files(requests=requests)
                for text in response.responses[0].responses:
                    parsed_text.append(text.full_text_annotation.text)
                return ''.join(text for text in parsed_text)
            except (NewConnectionError, HTTPError) as e:
                print("Could not connect to Google Cloud.")
                return ''

        # Local OCR with defined OLLAMA vision model
        def ocr_doc_local_llm(image_path: Path) -> str:
            from ollama_ocr import OCRProcessor
            ocr = OCRProcessor(model_name=os.getenv("OLLAMA_MODEL"),
                               base_url=f"{os.getenv("OLLAMA_BASE_URL")}/api/generate/")
            try:
                # Process a PDF file
                result = ocr.process_image(
                    image_path=str(image_path),  # Replace with your PDF file path
                    format_type="text",  # Options: markdown, text, json, structured, key_value, table
                    language="eng"  # Specify language if supported by the model
                    # custom_prompt= ""            # overwrite the default format type
                )
                return result
            except (HTTPError, NewConnectionError) as e:
                print(e)

        pdf_ocr = {'METADATA': {}, 'TEXT': []}
        with pdfplumber.open(doc_path) as pdf:
            total_pages = len(pdf.pages)
            pdf_ocr['METADATA'] = pdf.metadata

            if use_google_vision or use_local_llm:
                # Batch the OCR in pages of 5
                for i in range(0, total_pages, 5):
                    batch_end = i + 5 if i + 5 < total_pages else total_pages
                    batch_start = i
                    batch_num = int(i / 5)
                    print(f"Batch {batch_num} for Pages {batch_start} to {batch_end}")

                if use_google_vision:
                    ocr_text = ocr_google_llm(doc_path, page_range=range(batch_start, batch_end))
                    pdf_ocr['TEXT'].append(ocr_text)
                elif use_local_llm:
                    page_obj = pdf.pages[batch_start:batch_end]
                    for page in page_obj:
                        image = page.to_image(resolution=150)
                        image_path = Path.cwd() / 'temp' / 'temp.png'
                        image.save(image_path, format="PNG", quantize=True, colors=256, bits=8)
                        ocr_text = ocr_doc_local_llm(image_path)
                        pdf_ocr['TEXT'].append(ocr_text)
                        page.close()
            else:
                # PDF extraction with pdfplumber
                for page in pdf.pages:
                    pdf_ocr['TEXT'].append(page.extract_text())
                    page.close()

            pdf_ocr['TEXT'] = ''.join(text for text in pdf_ocr['TEXT'])

            # Try to locate TOC and store the preceding info in the metadata
            contents_idx = pdf_ocr['TEXT'].find('CONTENTS')
            if contents_idx != -1:
                pdf_ocr['METADATA']['INTRO'] = pdf_ocr['TEXT'][:contents_idx]

                # Try to locate the Authors
                authors_idx = pdf_ocr['METADATA']['INTRO'][-500:].find('Prepared by')
                if authors_idx != -1:
                    authors_idx = -500 + authors_idx
                    pdf_ocr['METADATA']['AUTHORS'] = pdf_ocr['METADATA']['INTRO'][authors_idx:]
                    pdf_ocr['METADATA']['INTRO'].replace(pdf_ocr['METADATA']['AUTHORS'], "")

        return pdf_ocr

    def process_document(self, text: str, doc_metadata: Dict) -> List[Dict]:
        """
        Split the document into chunks with preserved metadata.

        :param text: Raw document text content
        :param doc_metadata: Document information (source, title, date, author)
        :return: List of chunk dictionaries with content and metadata.
        """
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)

        processed_chunks = []
        for i, chunk in enumerate(chunks):
            # Create chunk with a unique ID and inhereited metadata
            chunk_data = {
                'content': chunk,
                'chunk_id': f"{doc_metadata['doc_id']}_chunk_{i}",
                'source': doc_metadata['source'],
                'title': doc_metadata['title'],
                'date': doc_metadata.get('date', ''),
                'author': doc_metadata.get('author', ''),
                'doc_type': doc_metadata.get('doc_type', 'report'),
            }
            processed_chunks.append(chunk_data)
        return processed_chunks


def main(docs_path: Path = None):
    # Initialize the document processing class and pass the documents path if defined, otherwise use default "docs"
    if docs_path:
        doc_collector = DocumentCollector(docs_path)
    doc_collector = DocumentCollector()

    # Loop through the folders in the documents path
    all_chunks = []
    for folder in doc_collector.docs_path.iterdir():
        source = folder.name

        # Loop through the documents in the folder
        for doc in folder.iterdir():
            doc_data = {'metadata': {}, 'content': str}
            doc_id = doc.name
            # OCR the document
            try:
                print(f"Trying to OCR the document: {doc_id}.")
                doc_ocr = doc_collector.ocr_document(doc)
                if not doc_ocr.get("TEXT"):
                    raise ValueError
                else:
                    print(f"Successfully OCR the document: {doc_id}.\n")
            except ValueError as e:
                try:
                    # If fails with pdfplumber use google vision
                    doc_ocr = doc_collector.ocr_document(doc, use_google_vision=True)
                    if not doc_ocr.get("TEXT"):
                        raise ValueError
                    else:
                        print(f"Successfully OCR the document: {doc_id}.\n")
                except ValueError as e:
                    print("Could not OCR the document using Google LLM, trying with Local LLM.\n\n")
                    try:
                        # If fails with google vision or gcloud isn't initialized use local llm
                        doc_ocr = doc_collector.ocr_document(doc, use_local_llm=True)
                        if not doc_ocr.get("TEXT"):
                            raise ValueError
                        else:
                            print(f"Successfully OCR the document: {doc_id}.\n")
                    except ValueError as e:
                        print("Could not OCR the document using Local LLM, trying next document.\n\n")
                        continue

            # Store the OCR data
            doc_data['metadata'] = {
                'source': source,
                'doc_id': doc_id,
                'title': doc_ocr.get("METADATA", {}).get("Title", ""),
                'date': doc_ocr.get("METADATA", {}).get("CreationDate", ""),
                'authors': doc_ocr.get("METADATA", {}).get("AUTHORS", ""),
                'doc_type': doc_ocr.get("METADATA", {}).get("Type", "Report"),
                'abstract': doc_ocr.get("METADATA", {}).get("INTRO", ""),

            }
            doc_data['content'] = doc_ocr.get("TEXT")
            del doc_ocr  # Free up RAM

            # Process the document into chunks
            print(f"Processing document {doc_id} into chunks.")
            chunks = doc_collector.process_document(doc_data.get("content", ""), doc_data.get("metadata", {}))
            all_chunks.extend(chunks)
            print(f"Finished chunking document {doc_id} into {len(chunks)} chunks.\n\n")
    return all_chunks

if __name__ == "__main__":
    chunks = main()
    print("\nSample chunk:")
    sample_chunk = chunks[0]
    for key, value in sample_chunk.items():
        if key == 'content':
            print(f"{key}: {value[:100]}...")  # Truncate content for display
        else:
            print(f"{key}: {value}")
