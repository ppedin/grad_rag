import os
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import re
import json
import tiktoken
from collections import Counter
from datasets_schema import Dataset, Document, Question


def download_squality_data(output_folder="squality"):
    """
    Download all data files from the SQuALITY dataset GitHub repository.

    Args:
        output_folder (str): Folder to save downloaded files
    """
    base_url = "https://github.com/nyu-mll/SQuALITY/tree/main/data/v1-3/txt"
    raw_base_url = "https://raw.githubusercontent.com/nyu-mll/SQuALITY/main/data/v1-3/txt"

    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(exist_ok=True)

    print(f"Downloading SQuALITY dataset files to {output_folder}/")

    # Get the file listing page
    response = requests.get(base_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all file links in the data directory
    file_links = []

    # Look for file links in the GitHub file browser
    for link in soup.find_all('a', href=True):
        href = link['href']
        # GitHub file links contain /blob/main/data/v1-3/txt/
        if '/blob/main/data/v1-3/txt/' in href and not href.endswith('/'):
            # Extract filename from the link
            filename = href.split('/')[-1]
            if filename and not filename.startswith('.'):  # Skip hidden files
                file_links.append(filename)

    # If the above method doesn't work well, try a more direct approach
    # by looking for specific file patterns or known file extensions
    if not file_links:
        print("Could not find files via page parsing. Trying known file patterns...")

        # Common file patterns for SQuALITY dataset (JSONL files)
        potential_files = [
            'train.jsonl',
            'dev.jsonl',
            'test.jsonl',
            'squality_train.jsonl',
            'squality_dev.jsonl',
            'squality_test.jsonl',
            'README.md'
        ]

        for filename in potential_files:
            file_url = f"{raw_base_url}/{filename}"
            try:
                head_response = requests.head(file_url)
                if head_response.status_code == 200:
                    file_links.append(filename)
                    print(f"Found file: {filename}")
            except:
                continue

    if not file_links:
        # Try another approach - look for span elements with file names
        print("Trying alternative parsing method...")
        for span in soup.find_all('span', class_='PRIVATE_TreeView-item-content-text'):
            text = span.get_text().strip()
            if text and '.' in text and not text.startswith('.'):
                file_links.append(text)

    # Remove duplicates and sort
    file_links = sorted(list(set(file_links)))

    print(f"Found {len(file_links)} files to download: {file_links}")

    if not file_links:
        print("Warning: No files found. The repository structure might have changed.")
        print("You may need to check the repository manually at:")
        print(base_url)
        return

    # Download each file
    for filename in file_links:
        # Construct download URL (raw file URL)
        download_url = f"{raw_base_url}/{filename}"
        output_path = Path(output_folder) / filename

        print(f"Downloading {filename}...")

        try:
            file_response = requests.get(download_url, stream=True)
            file_response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = os.path.getsize(output_path)
            print(f"  Downloaded {filename} ({file_size:,} bytes)")

        except Exception as e:
            print(f"  Failed to download {filename}: {e}")

    print(f"Download complete. Files saved to {output_folder}/")


def analyze_squality_structure():
    """
    Analyze the structure of SQuALITY dataset files and provide detailed report.
    """
    files = {
        'train': 'squality/train.jsonl',
        'dev': 'squality/dev.jsonl',
        'test': 'squality/test.jsonl'
    }

    # Also check for alternative naming
    alt_files = {
        'train': 'squality/squality_train.jsonl',
        'dev': 'squality/squality_dev.jsonl',
        'test': 'squality/squality_test.jsonl'
    }

    print("=" * 60)
    print("SQuALITY DATASET STRUCTURE ANALYSIS")
    print("=" * 60)

    # Initialize tiktoken encoder for token counting
    encoding = tiktoken.get_encoding("cl100k_base")

    for split_name in ['train', 'dev', 'test']:
        file_path = files[split_name]
        alt_file_path = alt_files[split_name]

        # Check which file exists
        if os.path.exists(file_path):
            actual_file_path = file_path
        elif os.path.exists(alt_file_path):
            actual_file_path = alt_file_path
        else:
            print(f"\n{split_name.upper()} SPLIT: FILE NOT FOUND")
            print(f"  Checked: {file_path}")
            print(f"  Checked: {alt_file_path}")
            continue

        print(f"\n{split_name.upper()} SPLIT")
        print("-" * 40)

        # File size
        file_size_mb = os.path.getsize(actual_file_path) / (1024 * 1024)
        print(f"File: {os.path.basename(actual_file_path)}")
        print(f"File size: {file_size_mb:.2f} MB")

        # Load data - assuming text format based on .txt extension
        print(f"Loading data...")

        try:
            with open(actual_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(actual_file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            except Exception as e:
                print(f"  Error reading file: {e}")
                continue

        print(f"File content length: {len(content)} characters")

        # Try to detect file format
        print(f"\nFORMAT DETECTION:")

        # Check if it's JSON Lines format
        lines = content.strip().split('\n')
        json_entries = []
        is_jsonl = True

        for line_num, line in enumerate(lines[:10]):  # Check first 10 lines
            if line.strip():
                try:
                    entry = json.loads(line.strip())
                    json_entries.append(entry)
                except json.JSONDecodeError:
                    is_jsonl = False
                    break

        if is_jsonl and json_entries:
            print(f"  - Format: JSON Lines (JSONL)")
            print(f"  - Number of entries (estimated): {len([l for l in lines if l.strip()])}")

            # Analyze structure using first entry
            sample_entry = json_entries[0]
            print(f"\nDATA STRUCTURE:")
            print(f"  - Entry keys: {list(sample_entry.keys())}")

            # Show sample data
            for key, value in sample_entry.items():
                if isinstance(value, str):
                    display_value = value[:100] + "..." if len(value) > 100 else value
                elif isinstance(value, list):
                    display_value = f"List with {len(value)} items"
                    if len(value) > 0:
                        if isinstance(value[0], str):
                            display_value += f", first: '{value[0][:50]}...'"
                        else:
                            display_value += f", first: {value[0]}"
                else:
                    display_value = str(value)

                print(f"    - {key}: {type(value).__name__} - {display_value}")

            # Calculate statistics across all entries
            print(f"\nSTATISTICS:")

            all_entries = []
            for line in lines:
                if line.strip():
                    try:
                        entry = json.loads(line.strip())
                        all_entries.append(entry)
                    except json.JSONDecodeError:
                        continue

            print(f"  - Total entries: {len(all_entries)}")

            # Analyze common fields
            field_counts = {}
            text_lengths = []
            query_lengths = []
            summary_lengths = []

            for entry in all_entries:
                for key in entry.keys():
                    field_counts[key] = field_counts.get(key, 0) + 1

                # Collect text statistics
                if 'article' in entry:
                    text_lengths.append(len(entry['article']))
                elif 'text' in entry:
                    text_lengths.append(len(entry['text']))

                if 'query' in entry:
                    query_lengths.append(len(entry['query']))
                elif 'question' in entry:
                    query_lengths.append(len(entry['question']))

                if 'summary' in entry:
                    summary_lengths.append(len(entry['summary']))
                elif 'answer' in entry:
                    summary_lengths.append(len(entry['answer']))

            print(f"  - Common fields: {sorted(field_counts.keys())}")

            if text_lengths:
                avg_text_length = sum(text_lengths) / len(text_lengths)
                print(f"  - Average text length: {avg_text_length:.0f} characters")

            if query_lengths:
                avg_query_length = sum(query_lengths) / len(query_lengths)
                print(f"  - Average query length: {avg_query_length:.0f} characters")

            if summary_lengths:
                avg_summary_length = sum(summary_lengths) / len(summary_lengths)
                print(f"  - Average summary length: {avg_summary_length:.0f} characters")

            # Token analysis - compute for ALL entries
            print(f"  - Computing tokens for ALL {len(all_entries)} entries...")
            total_document_tokens = 0
            total_response_tokens = 0
            document_token_lengths = []

            for entry in all_entries:
                try:
                    # Count tokens for document text
                    document_text = entry.get('document', '')
                    if document_text:
                        doc_tokens = len(encoding.encode(document_text))
                        total_document_tokens += doc_tokens
                        document_token_lengths.append(doc_tokens)

                    # Count tokens for all response texts in questions
                    questions = entry.get('questions', [])
                    for question in questions:
                        responses = question.get('responses', [])
                        for response in responses:
                            response_text = response.get('response_text', '')
                            if response_text:
                                response_tokens = len(encoding.encode(response_text))
                                total_response_tokens += response_tokens

                except Exception as e:
                    print(f"     Warning: Could not tokenize entry: {e}")
                    continue

            if document_token_lengths:
                avg_doc_tokens = total_document_tokens / len(document_token_lengths)
                min_doc_tokens = min(document_token_lengths)
                max_doc_tokens = max(document_token_lengths)
                print(f"  - Total document tokens: {total_document_tokens:,}")
                print(f"  - Average document tokens: {avg_doc_tokens:.0f}")
                print(f"  - Min document tokens: {min_doc_tokens:,}")
                print(f"  - Max document tokens: {max_doc_tokens:,}")

            if total_response_tokens > 0:
                total_responses = sum(len(q.get('responses', [])) for entry in all_entries for q in entry.get('questions', []))
                avg_response_tokens = total_response_tokens / total_responses if total_responses > 0 else 0
                print(f"  - Total response tokens: {total_response_tokens:,}")
                print(f"  - Average response tokens: {avg_response_tokens:.0f}")
                print(f"  - Total responses: {total_responses}")

            # Show additional examples
            print(f"\nADDITIONAL EXAMPLES:")
            if len(all_entries) > 1:
                second_entry = all_entries[1]
                print(f"  Second entry keys: {list(second_entry.keys())}")

                # Show a brief sample from second entry
                for key in ['article', 'text', 'query', 'question', 'summary', 'answer']:
                    if key in second_entry:
                        value = second_entry[key]
                        if isinstance(value, str) and len(value) > 50:
                            print(f"  {key} sample: {value[:100]}...")
                        break

        else:
            print(f"  - Format: Plain text or other format")
            print(f"  - Content preview (first 500 chars): {content[:500]}...")

    print(f"\n{'=' * 60}")
    print("SQuALITY ANALYSIS COMPLETED")
    print(f"{'=' * 60}")


def convert_squality_to_schema(split="train", output_file=None):
    """
    Convert SQuALITY data to the schema format defined in datasets_schema.py

    Args:
        split (str): Data split to process ('train', 'dev', 'test')
        output_file (str): Output JSON file path (auto-generated if None)

    Returns:
        str: Path to the generated JSON file
    """
    # File mapping
    file_mapping = {
        'train': 'squality/train.jsonl',
        'dev': 'squality/dev.jsonl',
        'test': 'squality/test.jsonl'
    }

    if split not in file_mapping:
        raise ValueError(f"Invalid split '{split}'. Must be one of: {list(file_mapping.keys())}")

    input_file = file_mapping[split]
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if output_file is None:
        if split == 'dev':
            output_file = f"squality_val.json"
        else:
            output_file = f"squality_{split}.json"

    print(f"Converting SQuALITY {split} split to schema format...")

    # Load SQuALITY data - JSONL format (one JSON per line)
    entries = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line.strip():
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"  Warning: Could not parse line {line_num + 1}: {e}")
                    continue

    print(f"Loaded {len(entries)} entries")

    documents = []

    for entry in entries:
        # Extract metadata
        metadata = entry.get('metadata', {})
        passage_id = metadata.get('passage_id', 'unknown')

        print(f"Processing document {passage_id}...")

        # Extract document content
        document_text = entry.get('document', '')

        # Process questions - SQuALITY is a summarization dataset
        questions = []
        entry_questions = entry.get('questions', [])

        for q_data in entry_questions:
            question_text = q_data.get('question_text', '')
            question_number = q_data.get('question_number', 0)
            question_id = f"{passage_id}_q{question_number}"

            # Get all responses (human-written summaries)
            responses = q_data.get('responses', [])
            answer_texts = []

            for response in responses:
                response_text = response.get('response_text', '')
                if response_text and response_text.strip():
                    answer_texts.append(response_text.strip())

            # For summarization datasets, we typically use the first response as the main answer
            # but preserve all responses in metadata
            if not answer_texts:
                continue

            # Create Question object
            question = Question(
                id=question_id,
                question=question_text,
                answers=[answer_texts[0]],  # Use first response as primary answer
                metadata={
                    'question_number': question_number,
                    'all_responses': answer_texts,  # Keep all human responses
                    'response_workers': [r.get('worker_id') for r in responses],
                    'response_uids': [r.get('uid') for r in responses],
                    'task_type': 'summarization',
                    'num_responses': len(answer_texts)
                }
            )
            questions.append(question)

        # Create Document object
        document = Document(
            id=passage_id,
            text=document_text,
            questions=questions,
            metadata={
                'uid': metadata.get('uid'),
                'license': metadata.get('license'),
                'source': 'Project Gutenberg',
                'dataset': 'SQuALITY',
                'task_type': 'query_based_summarization',
                'num_questions': len(questions),
                'document_tokens': len(tiktoken.get_encoding("cl100k_base").encode(document_text)) if document_text else 0
            }
        )
        documents.append(document)

    # Create Dataset
    dataset = Dataset(documents=documents)

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset.model_dump(), f, indent=2, ensure_ascii=False)

    print(f"Converted {len(documents)} documents with {sum(len(doc.questions) for doc in documents)} questions")
    print(f"Saved to {output_file}")

    return output_file


def convert_all_squality_splits():
    """
    Convert all SQuALITY splits (train, dev, test) to schema format
    """
    splits = ['train', 'dev', 'test']

    for split in splits:
        try:
            output_file = convert_squality_to_schema(split)
            print(f"Completed {split} split: {output_file}")
        except Exception as e:
            print(f"Failed to process {split} split: {e}")


def validate_squality_converted_data():
    """
    Validate the converted SQuALITY JSON files by checking structure and showing statistics.
    """
    splits = ['train', 'dev', 'test']
    encoding = tiktoken.get_encoding("cl100k_base")

    for split in splits:
        if split == 'dev':
            json_file = "squality_val.json"
        else:
            json_file = f"squality_{split}.json"

        if not os.path.exists(json_file):
            print(f"File {json_file} not found!")
            continue

        print(f"\n{'='*50}")
        print(f"VALIDATING SQuALITY {split.upper()} SPLIT")
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

        # Calculate token statistics
        total_tokens = 0
        total_response_tokens = 0
        print(f"STATISTICS:")
        print(f"   - Number of documents: {total_documents}")
        print(f"   - Total questions: {total_questions}")
        print(f"   - Average questions per document: {avg_questions_per_doc:.2f}")

        # Calculate tokens for all documents and responses
        print(f"   - Computing tokens for {total_documents} documents...")
        for doc in documents:
            try:
                doc_tokens = len(encoding.encode(doc['text']))
                total_tokens += doc_tokens

                # Count response tokens
                for question in doc['questions']:
                    for answer in question['answers']:
                        response_tokens = len(encoding.encode(answer))
                        total_response_tokens += response_tokens

            except Exception as e:
                print(f"     Warning: Could not tokenize document {doc['id']}: {e}")

        avg_tokens_per_doc = total_tokens / total_documents if total_documents > 0 else 0
        avg_response_tokens = total_response_tokens / total_questions if total_questions > 0 else 0

        print(f"   - Total document tokens: {total_tokens:,}")
        print(f"   - Average tokens per document: {avg_tokens_per_doc:.0f}")
        print(f"   - Total response tokens: {total_response_tokens:,}")
        print(f"   - Average response tokens: {avg_response_tokens:.0f}")

        # Validate schema structure
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

            # Show document text sample (first 300 chars)
            doc_text = doc_example['text']
            try:
                print(f"   Document text (first 300 chars): {doc_text[:300]}...")
            except UnicodeEncodeError:
                print(f"   Document text (first 300 chars): [Text contains special characters - length: {len(doc_text[:300])} chars]")

            print(f"   Document metadata keys: {list(doc_example['metadata'].keys())}")
            print(f"   Number of questions: {len(doc_example['questions'])}")

            # Question example
            if len(doc_example['questions']) > 0:
                q_example = doc_example['questions'][0]
                print(f"\n   Question ID: {q_example['id']}")
                print(f"   Question text: {q_example['question']}")
                print(f"   Number of answers: {len(q_example['answers'])}")
                if len(q_example['answers']) > 0:
                    answer_preview = q_example['answers'][0][:200] + "..." if len(q_example['answers'][0]) > 200 else q_example['answers'][0]
                    print(f"   First answer (preview): {answer_preview}")

                # Show summarization-specific metadata
                metadata = q_example['metadata']
                print(f"   Question metadata keys: {list(metadata.keys())}")
                print(f"   Task type: {metadata.get('task_type', 'unknown')}")
                print(f"   Number of human responses: {metadata.get('num_responses', 0)}")

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

        # Check task type consistency
        task_types = set()
        for doc in documents:
            for q in doc['questions']:
                task_types.add(q['metadata'].get('task_type', 'unknown'))

        print(f"   - Task types found: {sorted(task_types)}")

        print(f"   - Validation completed for {split} split")

    print(f"\n{'='*50}")
    print("SQuALITY VALIDATION SUMMARY COMPLETED")
    print(f"{'='*50}")


if __name__ == "__main__":
    download_squality_data()