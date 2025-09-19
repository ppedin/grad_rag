import pandas as pd
import os
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import re
import json
import glob
import tiktoken
from datasets_schema import Dataset, Document, Question
from data_utils import extract_text_from_html


def download_narrativeqa_data(output_folder="narrativeqa"):
    """
    Download all data files from the NarrativeQA HuggingFace dataset.

    Args:
        output_folder (str): Folder to save downloaded files
    """
    base_url = "https://huggingface.co/datasets/deepmind/narrativeqa/tree/main/data"

    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(exist_ok=True)

    # Get the file listing page
    response = requests.get(base_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all file links in the data directory
    file_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if '/blob/main/data/' in href and not href.endswith('/'):
            file_links.append(href)

    print(f"Found {len(file_links)} files to download")

    # Download each file
    for link in file_links:
        # Extract filename from the link
        filename = link.split('/')[-1]

        # Construct download URL (raw file URL)
        download_url = f"https://huggingface.co/datasets/deepmind/narrativeqa/resolve/main/data/{filename}"

        output_path = Path(output_folder) / filename

        print(f"Downloading {filename}...")

        try:
            file_response = requests.get(download_url, stream=True)
            file_response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f" Downloaded {filename}")

        except Exception as e:
            print(f" Failed to download {filename}: {e}")

    print(f"Download complete. Files saved to {output_folder}/")


def convert_narrativeqa_to_schema(split="train", narrativeqa_folder="narrativeqa", output_file=None):
    """
    Convert NarrativeQA data to the schema format defined in datasets_schema.py

    Args:
        split (str): Data split to process ('train', 'validation', 'test')
        narrativeqa_folder (str): Path to NarrativeQA data folder
        output_file (str): Output JSON file path (auto-generated if None)

    Returns:
        str: Path to the generated JSON file
    """
    if output_file is None:
        output_file = f"narrativeqa_{split}.json"

    # Find all parquet files for the split
    parquet_pattern = f"{narrativeqa_folder}/{split}-*.parquet"
    parquet_files = sorted(glob.glob(parquet_pattern))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for split '{split}' in {narrativeqa_folder}")

    print(f"Processing {len(parquet_files)} parquet files for {split} split...")

    # Group data by document ID to collect all questions per document
    documents_data = {}

    for parquet_file in parquet_files:
        print(f"Processing {os.path.basename(parquet_file)}...")
        df = pd.read_parquet(parquet_file)

        for _, row in df.iterrows():
            doc_info = row['document']
            question_info = row['question']
            answers_info = row['answers']

            doc_id = doc_info['id']

            # Create question object
            question = Question(
                id=f"{doc_id}_{len(documents_data.get(doc_id, {}).get('questions', []))}",
                question=question_info['text'],
                answers=[answer['text'] for answer in answers_info],
                metadata={
                    'question_tokens': question_info['tokens'].tolist() if hasattr(question_info['tokens'], 'tolist') else list(question_info['tokens']),
                    'answer_tokens': [answer['tokens'].tolist() if hasattr(answer['tokens'], 'tolist') else list(answer['tokens']) for answer in answers_info]
                }
            )

            # Initialize document data if not exists
            if doc_id not in documents_data:
                # Load full text from .content file
                content_file = f"{narrativeqa_folder}/narrativeqa_full_text/{doc_id}.content"
                full_text = ""

                if os.path.exists(content_file):
                    with open(content_file, 'r', encoding='utf-8', errors='ignore') as f:
                        html_content = f.read()
                    # Extract plain text from HTML
                    full_text = extract_text_from_html(html_content)
                else:
                    print(f"Warning: Full text file not found for document {doc_id}")
                    full_text = doc_info.get('text', '')

                documents_data[doc_id] = {
                    'id': doc_id,
                    'text': full_text,
                    'questions': [],
                    'metadata': {
                        'kind': doc_info.get('kind'),
                        'url': doc_info.get('url'),
                        'file_size': doc_info.get('file_size'),
                        'word_count': doc_info.get('word_count'),
                        'start': doc_info.get('start'),
                        'end': doc_info.get('end'),
                        'summary': doc_info.get('summary')
                    }
                }

            # Add question to document
            documents_data[doc_id]['questions'].append(question)

    # Convert to schema format
    documents = []
    for doc_data in documents_data.values():
        document = Document(
            id=doc_data['id'],
            text=doc_data['text'],
            questions=doc_data['questions'],
            metadata=doc_data['metadata']
        )
        documents.append(document)

    # Create dataset
    dataset = Dataset(documents=documents)

    # Save to JSON - convert to dict first to handle any numpy arrays
    dataset_dict = dataset.model_dump()

    def convert_arrays(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_arrays(item) for item in obj]
        return obj

    dataset_dict = convert_arrays(dataset_dict)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_dict, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(documents)} documents with {sum(len(doc.questions) for doc in documents)} questions")
    print(f"Saved to {output_file}")

    return output_file


def convert_all_splits(narrativeqa_folder="narrativeqa"):
    """
    Convert all splits (train, validation, test) to schema format

    Args:
        narrativeqa_folder (str): Path to NarrativeQA data folder
    """
    splits = ['train', 'validation', 'test']

    for split in splits:
        try:
            output_file = convert_narrativeqa_to_schema(split, narrativeqa_folder)
            print(f"Completed {split} split: {output_file}")
        except Exception as e:
            print(f"Failed to process {split} split: {e}")


def validate_converted_data():
    """
    Validate the converted NarrativeQA JSON files by checking structure and showing statistics.
    """
    splits = ['train', 'validation', 'test']

    for split in splits:
        json_file = f"narrativeqa_{split}.json"

        if not os.path.exists(json_file):
            print(f"File {json_file} not found!")
            continue

        print(f"\n{'='*50}")
        print(f"VALIDATING {split.upper()} SPLIT")
        print(f"{'='*50}")

        # Load JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Basic structure validation
        if 'documents' not in data:
            print("Missing 'documents' key in JSON structure")
            continue

        documents = data['documents']
        total_documents = len(documents)
        total_questions = sum(len(doc['questions']) for doc in documents)
        avg_questions_per_doc = total_questions / total_documents if total_documents > 0 else 0

        # Calculate token statistics using tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        total_tokens = 0

        print(f"STATISTICS:")
        print(f"   - Number of documents: {total_documents}")
        print(f"   - Total questions: {total_questions}")
        print(f"   - Average questions per document: {avg_questions_per_doc:.2f}")

        # Calculate tokens for all documents
        print(f"   - Computing tokens for {total_documents} documents...")
        for doc in documents:
            try:
                doc_tokens = len(encoding.encode(doc['text']))
                total_tokens += doc_tokens
            except Exception as e:
                # Handle potential encoding errors
                print(f"     Warning: Could not tokenize document {doc['id']}: {e}")

        avg_tokens_per_doc = total_tokens / total_documents if total_documents > 0 else 0
        print(f"   - Total tokens: {total_tokens:,}")
        print(f"   - Average tokens per document: {avg_tokens_per_doc:.0f}")

        # Validate schema structure with first document
        if total_documents > 0:
            print(f"\nSTRUCTURE VALIDATION:")

            doc = documents[0]
            required_doc_fields = ['id', 'text', 'questions', 'metadata']

            print(f"   Document structure:")
            for field in required_doc_fields:
                if field in doc:
                    print(f"   - {field}: {type(doc[field])}")
                else:
                    print(f"   - Missing field: {field}")

            # Check questions structure
            if 'questions' in doc and len(doc['questions']) > 0:
                question = doc['questions'][0]
                required_q_fields = ['id', 'question', 'answers', 'metadata']

                print(f"   Question structure:")
                for field in required_q_fields:
                    if field in question:
                        print(f"   - {field}: {type(question[field])}")
                    else:
                        print(f"   - Missing field: {field}")

        # Show examples
        print(f"\nEXAMPLES:")

        if total_documents > 0:
            # Document example
            doc_example = documents[0]
            print(f"   Document ID: {doc_example['id']}")
            try:
                print(f"   Document text (first 200 chars): {doc_example['text'][:200]}...")
            except UnicodeEncodeError:
                print(f"   Document text (first 200 chars): [Text contains special characters - length: {len(doc_example['text'][:200])} chars]")
            print(f"   Document metadata keys: {list(doc_example['metadata'].keys())}")
            print(f"   Number of questions: {len(doc_example['questions'])}")

            # Question example
            if len(doc_example['questions']) > 0:
                q_example = doc_example['questions'][0]
                print(f"\n   Question ID: {q_example['id']}")
                print(f"   Question text: {q_example['question']}")
                print(f"   Number of answers: {len(q_example['answers'])}")
                print(f"   First answer: {q_example['answers'][0]}")
                print(f"   Question metadata keys: {list(q_example['metadata'].keys())}")

        # Additional validation
        print(f"\nADDITIONAL CHECKS:")

        # Check for duplicate document IDs
        doc_ids = [doc['id'] for doc in documents]
        unique_doc_ids = set(doc_ids)
        if len(doc_ids) == len(unique_doc_ids):
            print(f"   - All document IDs are unique")
        else:
            print(f"   - Found duplicate document IDs: {len(doc_ids) - len(unique_doc_ids)} duplicates")

        # Check for empty documents or questions
        empty_docs = sum(1 for doc in documents if not doc['text'].strip())
        empty_questions = sum(1 for doc in documents for q in doc['questions'] if not q['question'].strip())

        print(f"   - Documents with empty text: {empty_docs}")
        print(f"   - Questions with empty text: {empty_questions}")

        # Check average text length
        avg_doc_length = sum(len(doc['text']) for doc in documents) / total_documents if total_documents > 0 else 0
        print(f"   - Average document length: {avg_doc_length:.0f} characters")

        print(f"   - Validation completed for {split} split")

    print(f"\n{'='*50}")
    print("VALIDATION SUMMARY COMPLETED")
    print(f"{'='*50}")


#  Three columns: 'document', 'question', 'answers'
#  Answers contain a list (different possible answers). Each element has: 'text', 'tokens'