import os
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import re
import json
import tiktoken
from collections import Counter
from datasets_schema import Dataset, Document, Question


def download_qmsum_data(output_folder="qmsum"):
    """
    Download all data files from the QMSum dataset GitHub repository.

    Args:
        output_folder (str): Folder to save downloaded files
    """
    base_url = "https://github.com/Yale-LILY/QMSum/tree/main/data/ALL/jsonl"
    raw_base_url = "https://raw.githubusercontent.com/Yale-LILY/QMSum/main/data/ALL/jsonl"

    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(exist_ok=True)

    print(f"Downloading QMSum dataset files to {output_folder}/")

    # Get the file listing page
    response = requests.get(base_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all file links in the data directory
    file_links = []

    # Look for file links in the GitHub file browser
    for link in soup.find_all('a', href=True):
        href = link['href']
        # GitHub file links contain /blob/main/data/ALL/jsonl/
        if '/blob/main/data/ALL/jsonl/' in href and not href.endswith('/'):
            # Extract filename from the link
            filename = href.split('/')[-1]
            if filename and not filename.startswith('.'):  # Skip hidden files
                file_links.append(filename)

    # If the above method doesn't work well, try a more direct approach
    # by looking for specific file patterns or known file extensions
    if not file_links:
        print("Could not find files via page parsing. Trying known file patterns...")

        # Common file patterns for QMSum dataset (JSONL files)
        potential_files = [
            'train.jsonl',
            'val.jsonl',
            'test.jsonl',
            'dev.jsonl',
            'qmsum_train.jsonl',
            'qmsum_val.jsonl',
            'qmsum_test.jsonl',
            'qmsum_dev.jsonl',
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


def analyze_qmsum_structure():
    """
    Analyze the structure of QMSum dataset files and provide detailed report.
    """
    files = {
        'train': 'qmsum/train.jsonl',
        'val': 'qmsum/val.jsonl',
        'test': 'qmsum/test.jsonl'
    }

    # Also check for alternative naming
    alt_files = {
        'train': 'qmsum/qmsum_train.jsonl',
        'val': 'qmsum/qmsum_val.jsonl',
        'test': 'qmsum/qmsum_test.jsonl'
    }

    print("=" * 60)
    print("QMSum DATASET STRUCTURE ANALYSIS")
    print("=" * 60)

    # Initialize tiktoken encoder for token counting
    encoding = tiktoken.get_encoding("cl100k_base")

    for split_name in ['train', 'val', 'test']:
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

        # Load data - assuming JSONL format based on expected structure
        print(f"Loading data...")

        entries = []
        try:
            with open(actual_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line.strip():
                        try:
                            entry = json.loads(line.strip())
                            entries.append(entry)
                        except json.JSONDecodeError as e:
                            print(f"  Error parsing line {line_num + 1}: {e}")
                            continue
        except UnicodeDecodeError:
            try:
                with open(actual_file_path, 'r', encoding='latin-1') as f:
                    for line_num, line in enumerate(f):
                        if line.strip():
                            try:
                                entry = json.loads(line.strip())
                                entries.append(entry)
                            except json.JSONDecodeError as e:
                                print(f"  Error parsing line {line_num + 1}: {e}")
                                continue
            except Exception as e:
                print(f"  Error reading file: {e}")
                continue

        print(f"Number of entries: {len(entries)}")

        if not entries:
            continue

        # Analyze structure using first entry
        sample_entry = entries[0]
        print(f"\nDATA STRUCTURE:")
        print(f"  - Entry keys: {list(sample_entry.keys())}")

        # Show sample data for each key
        for key, value in sample_entry.items():
            if isinstance(value, str):
                display_value = value[:100] + "..." if len(value) > 100 else value
            elif isinstance(value, list):
                display_value = f"List with {len(value)} items"
                if len(value) > 0:
                    if isinstance(value[0], str):
                        display_value += f", first: '{value[0][:50]}...'"
                    elif isinstance(value[0], dict):
                        display_value += f", first item keys: {list(value[0].keys())}"
                    else:
                        display_value += f", first: {value[0]}"
            elif isinstance(value, dict):
                display_value = f"Dict with keys: {list(value.keys())}"
            else:
                display_value = str(value)

            print(f"    - {key}: {type(value).__name__} - {display_value}")

        # Calculate statistics across all entries
        print(f"\nSTATISTICS:")

        print(f"  - Total entries: {len(entries)}")

        # Analyze common fields and content
        field_counts = {}
        text_lengths = []
        query_lengths = []
        summary_lengths = []
        meeting_lengths = []

        for entry in entries:
            for key in entry.keys():
                field_counts[key] = field_counts.get(key, 0) + 1

            # Collect statistics based on likely field names for meeting summarization
            for field_name in ['meeting_transcripts', 'transcript', 'input', 'source']:
                if field_name in entry:
                    if isinstance(entry[field_name], str):
                        meeting_lengths.append(len(entry[field_name]))
                    elif isinstance(entry[field_name], list):
                        # If it's a list of utterances, join them
                        combined_text = ' '.join([str(item) for item in entry[field_name]])
                        meeting_lengths.append(len(combined_text))
                    break

            for field_name in ['query', 'question', 'specific_query']:
                if field_name in entry:
                    if isinstance(entry[field_name], str):
                        query_lengths.append(len(entry[field_name]))
                    break

            for field_name in ['summary', 'answer', 'general_query_list', 'specific_query_list']:
                if field_name in entry:
                    if isinstance(entry[field_name], str):
                        summary_lengths.append(len(entry[field_name]))
                    elif isinstance(entry[field_name], list):
                        # Handle list of summaries
                        for item in entry[field_name]:
                            if isinstance(item, str):
                                summary_lengths.append(len(item))
                            elif isinstance(item, dict) and 'answer' in item:
                                summary_lengths.append(len(item['answer']))
                    break

        print(f"  - Common fields: {sorted(field_counts.keys())}")

        if meeting_lengths:
            avg_meeting_length = sum(meeting_lengths) / len(meeting_lengths)
            print(f"  - Average meeting/transcript length: {avg_meeting_length:.0f} characters")

        if query_lengths:
            avg_query_length = sum(query_lengths) / len(query_lengths)
            print(f"  - Average query length: {avg_query_length:.0f} characters")

        if summary_lengths:
            avg_summary_length = sum(summary_lengths) / len(summary_lengths)
            print(f"  - Average summary length: {avg_summary_length:.0f} characters")

        # Token analysis - compute for ALL entries
        print(f"  - Computing tokens for ALL {len(entries)} entries...")
        total_document_tokens = 0
        total_query_tokens = 0
        total_summary_tokens = 0
        document_token_lengths = []

        for entry in entries:
            try:
                # Find document/meeting text field
                document_text = ""
                for field_name in ['meeting_transcripts', 'transcript', 'input', 'source']:
                    if field_name in entry:
                        if isinstance(entry[field_name], str):
                            document_text = entry[field_name]
                        elif isinstance(entry[field_name], list):
                            document_text = ' '.join([str(item) for item in entry[field_name]])
                        break

                if document_text:
                    doc_tokens = len(encoding.encode(document_text))
                    total_document_tokens += doc_tokens
                    document_token_lengths.append(doc_tokens)

                # Count query tokens
                for field_name in ['query', 'question', 'specific_query']:
                    if field_name in entry and isinstance(entry[field_name], str):
                        query_tokens = len(encoding.encode(entry[field_name]))
                        total_query_tokens += query_tokens
                        break

                # Count summary tokens
                for field_name in ['summary', 'answer', 'general_query_list', 'specific_query_list']:
                    if field_name in entry:
                        if isinstance(entry[field_name], str):
                            summary_tokens = len(encoding.encode(entry[field_name]))
                            total_summary_tokens += summary_tokens
                        elif isinstance(entry[field_name], list):
                            for item in entry[field_name]:
                                if isinstance(item, str):
                                    summary_tokens = len(encoding.encode(item))
                                    total_summary_tokens += summary_tokens
                                elif isinstance(item, dict) and 'answer' in item:
                                    summary_tokens = len(encoding.encode(item['answer']))
                                    total_summary_tokens += summary_tokens
                        break

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

        if total_query_tokens > 0:
            avg_query_tokens = total_query_tokens / len(entries)
            print(f"  - Total query tokens: {total_query_tokens:,}")
            print(f"  - Average query tokens: {avg_query_tokens:.0f}")

        if total_summary_tokens > 0:
            print(f"  - Total summary tokens: {total_summary_tokens:,}")

        # Show additional examples
        print(f"\nADDITIONAL EXAMPLES:")
        if len(entries) > 1:
            second_entry = entries[1]
            print(f"  Second entry keys: {list(second_entry.keys())}")

            # Show a brief sample from second entry
            for key in ['meeting_transcripts', 'transcript', 'input', 'query', 'question', 'summary', 'answer']:
                if key in second_entry:
                    value = second_entry[key]
                    if isinstance(value, str) and len(value) > 50:
                        print(f"  {key} sample: {value[:100]}...")
                    elif isinstance(value, list) and len(value) > 0:
                        print(f"  {key} sample: {value[0] if isinstance(value[0], str) else str(value[0])[:100]}...")
                    break

    print(f"\n{'=' * 60}")
    print("QMSum ANALYSIS COMPLETED")
    print(f"{'=' * 60}")


def convert_qmsum_to_schema(split="train", output_file=None):
    """
    Convert QMSum data to the schema format defined in datasets_schema.py

    Args:
        split (str): Data split to process ('train', 'val', 'test')
        output_file (str): Output JSON file path (auto-generated if None)

    Returns:
        str: Path to the generated JSON file
    """
    # File mapping
    file_mapping = {
        'train': 'qmsum/train.jsonl',
        'val': 'qmsum/val.jsonl',
        'test': 'qmsum/test.jsonl'
    }

    if split not in file_mapping:
        raise ValueError(f"Invalid split '{split}'. Must be one of: {list(file_mapping.keys())}")

    input_file = file_mapping[split]
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if output_file is None:
        output_file = f"qmsum_{split}.json"

    print(f"Converting QMSum {split} split to schema format...")

    # Load QMSum data - JSONL format (one JSON per line)
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

    print(f"Loaded {len(entries)} meeting entries")

    documents = []

    for entry_idx, entry in enumerate(entries):
        meeting_id = f"meeting_{entry_idx + 1}"
        print(f"Processing {meeting_id}...")

        # Extract meeting transcript and convert to text
        meeting_transcripts = entry.get('meeting_transcripts', [])

        # Convert meeting transcript to formatted text
        transcript_lines = []
        for utterance in meeting_transcripts:
            speaker = utterance.get('speaker', 'Unknown')
            content = utterance.get('content', '')
            # Format as "Speaker: Content"
            transcript_lines.append(f"{speaker}: {content}")

        document_text = '\n'.join(transcript_lines)

        # Extract topics for context
        topics = entry.get('topic_list', [])
        topic_text = ""
        if topics:
            topic_descriptions = [topic.get('topic', '') for topic in topics if topic.get('topic')]
            if topic_descriptions:
                topic_text = f"Meeting Topics: {', '.join(topic_descriptions)}\n\n"

        # Combine topics and transcript
        full_document_text = topic_text + document_text

        # Process questions - QMSum has general and specific queries
        questions = []

        # Process general queries
        general_queries = entry.get('general_query_list', [])
        for q_idx, query_data in enumerate(general_queries):
            question_text = query_data.get('query', '')
            answer_text = query_data.get('answer', '')
            question_id = f"{meeting_id}_general_q{q_idx + 1}"

            if question_text and answer_text:
                question = Question(
                    id=question_id,
                    question=question_text,
                    answers=[answer_text],
                    metadata={
                        'query_type': 'general',
                        'question_index': q_idx,
                        'task_type': 'meeting_summarization',
                        'has_relevant_spans': False
                    }
                )
                questions.append(question)

        # Process specific queries
        specific_queries = entry.get('specific_query_list', [])
        for q_idx, query_data in enumerate(specific_queries):
            question_text = query_data.get('query', '')
            answer_text = query_data.get('answer', '')
            relevant_spans = query_data.get('relevant_text_span', [])
            question_id = f"{meeting_id}_specific_q{q_idx + 1}"

            if question_text and answer_text:
                question = Question(
                    id=question_id,
                    question=question_text,
                    answers=[answer_text],
                    metadata={
                        'query_type': 'specific',
                        'question_index': q_idx,
                        'task_type': 'meeting_summarization',
                        'has_relevant_spans': bool(relevant_spans),
                        'relevant_text_span': relevant_spans,
                        'num_relevant_spans': len(relevant_spans) if relevant_spans else 0
                    }
                )
                questions.append(question)

        # Create Document object
        document = Document(
            id=meeting_id,
            text=full_document_text,
            questions=questions,
            metadata={
                'source': 'QMSum',
                'dataset': 'QMSum',
                'task_type': 'query_based_meeting_summarization',
                'num_general_queries': len(general_queries),
                'num_specific_queries': len(specific_queries),
                'total_questions': len(questions),
                'num_utterances': len(meeting_transcripts),
                'num_topics': len(topics),
                'topics': [topic.get('topic', '') for topic in topics],
                'speakers': list(set([utt.get('speaker', '') for utt in meeting_transcripts if utt.get('speaker')])),
                'num_speakers': len(set([utt.get('speaker', '') for utt in meeting_transcripts if utt.get('speaker')])),
                'document_tokens': len(tiktoken.get_encoding("cl100k_base").encode(full_document_text)) if full_document_text else 0,
                'original_entry_index': entry_idx
            }
        )
        documents.append(document)

    # Create Dataset
    dataset = Dataset(documents=documents)

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset.model_dump(), f, indent=2, ensure_ascii=False)

    total_questions = sum(len(doc.questions) for doc in documents)
    general_questions = sum(len([q for q in doc.questions if q.metadata.get('query_type') == 'general']) for doc in documents)
    specific_questions = sum(len([q for q in doc.questions if q.metadata.get('query_type') == 'specific']) for doc in documents)

    print(f"Converted {len(documents)} meetings with {total_questions} questions")
    print(f"  - General questions: {general_questions}")
    print(f"  - Specific questions: {specific_questions}")
    print(f"Saved to {output_file}")

    return output_file


def convert_all_qmsum_splits():
    """
    Convert all QMSum splits (train, val, test) to schema format
    """
    splits = ['train', 'val', 'test']

    for split in splits:
        try:
            output_file = convert_qmsum_to_schema(split)
            print(f"Completed {split} split: {output_file}")
        except Exception as e:
            print(f"Failed to process {split} split: {e}")


def validate_qmsum_converted_data():
    """
    Validate the converted QMSum JSON files by checking structure and showing statistics.
    """
    splits = ['train', 'val', 'test']
    encoding = tiktoken.get_encoding("cl100k_base")

    for split in splits:
        json_file = f"qmsum_{split}.json"

        if not os.path.exists(json_file):
            print(f"File {json_file} not found!")
            continue

        print(f"\n{'='*50}")
        print(f"VALIDATING QMSum {split.upper()} SPLIT")
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

        # Calculate detailed statistics
        total_tokens = 0
        total_response_tokens = 0
        general_questions = 0
        specific_questions = 0
        total_speakers = 0
        total_utterances = 0

        print(f"STATISTICS:")
        print(f"   - Number of meetings: {total_documents}")
        print(f"   - Total questions: {total_questions}")
        print(f"   - Average questions per meeting: {avg_questions_per_doc:.2f}")

        # Calculate tokens and meeting-specific statistics
        print(f"   - Computing tokens for {total_documents} meetings...")
        for doc in documents:
            try:
                doc_tokens = len(encoding.encode(doc['text']))
                total_tokens += doc_tokens

                # Count response tokens
                for question in doc['questions']:
                    for answer in question['answers']:
                        response_tokens = len(encoding.encode(answer))
                        total_response_tokens += response_tokens

                # Count question types
                for question in doc['questions']:
                    q_type = question['metadata'].get('query_type', 'unknown')
                    if q_type == 'general':
                        general_questions += 1
                    elif q_type == 'specific':
                        specific_questions += 1

                # Meeting-specific statistics
                total_speakers += doc['metadata'].get('num_speakers', 0)
                total_utterances += doc['metadata'].get('num_utterances', 0)

            except Exception as e:
                print(f"     Warning: Could not tokenize document {doc['id']}: {e}")

        avg_tokens_per_doc = total_tokens / total_documents if total_documents > 0 else 0
        avg_response_tokens = total_response_tokens / total_questions if total_questions > 0 else 0
        avg_speakers_per_meeting = total_speakers / total_documents if total_documents > 0 else 0
        avg_utterances_per_meeting = total_utterances / total_documents if total_documents > 0 else 0

        print(f"   - Total meeting tokens: {total_tokens:,}")
        print(f"   - Average tokens per meeting: {avg_tokens_per_doc:.0f}")
        print(f"   - Total response tokens: {total_response_tokens:,}")
        print(f"   - Average response tokens: {avg_response_tokens:.0f}")
        print(f"   - General questions: {general_questions}")
        print(f"   - Specific questions: {specific_questions}")
        print(f"   - Average speakers per meeting: {avg_speakers_per_meeting:.1f}")
        print(f"   - Average utterances per meeting: {avg_utterances_per_meeting:.0f}")

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
            print(f"   Meeting ID: {doc_example['id']}")

            # Show meeting text sample (first 300 chars)
            doc_text = doc_example['text']
            try:
                print(f"   Meeting text (first 300 chars): {doc_text[:300]}...")
            except UnicodeEncodeError:
                print(f"   Meeting text (first 300 chars): [Text contains special characters - length: {len(doc_text[:300])} chars]")

            print(f"   Meeting metadata keys: {list(doc_example['metadata'].keys())}")
            print(f"   Number of questions: {len(doc_example['questions'])}")
            print(f"   Number of speakers: {doc_example['metadata'].get('num_speakers', 0)}")
            print(f"   Topics: {doc_example['metadata'].get('topics', [])}")

            # Question examples
            if len(doc_example['questions']) > 0:
                # Show general question if available
                general_q = None
                specific_q = None

                for q in doc_example['questions']:
                    if q['metadata'].get('query_type') == 'general' and general_q is None:
                        general_q = q
                    elif q['metadata'].get('query_type') == 'specific' and specific_q is None:
                        specific_q = q

                if general_q:
                    print(f"\n   GENERAL QUESTION EXAMPLE:")
                    print(f"   Question ID: {general_q['id']}")
                    print(f"   Question: {general_q['question']}")
                    if len(general_q['answers']) > 0:
                        answer_preview = general_q['answers'][0][:200] + "..." if len(general_q['answers'][0]) > 200 else general_q['answers'][0]
                        print(f"   Answer (preview): {answer_preview}")

                if specific_q:
                    print(f"\n   SPECIFIC QUESTION EXAMPLE:")
                    print(f"   Question ID: {specific_q['id']}")
                    print(f"   Question: {specific_q['question']}")
                    if len(specific_q['answers']) > 0:
                        answer_preview = specific_q['answers'][0][:200] + "..." if len(specific_q['answers'][0]) > 200 else specific_q['answers'][0]
                        print(f"   Answer (preview): {answer_preview}")
                    print(f"   Has relevant spans: {specific_q['metadata'].get('has_relevant_spans', False)}")

        # Additional validation
        print(f"\nADDITIONAL CHECKS:")

        # Check for duplicate document IDs
        doc_ids = [doc['id'] for doc in documents]
        unique_doc_ids = set(doc_ids)
        if len(doc_ids) == len(unique_doc_ids):
            print(f"   - All meeting IDs are unique")
        else:
            print(f"   - Found duplicate meeting IDs: {len(doc_ids) - len(unique_doc_ids)} duplicates")

        # Check for empty documents or questions
        empty_docs = sum(1 for doc in documents if not doc['text'].strip())
        empty_questions = sum(1 for doc in documents for q in doc['questions'] if not q['question'].strip())

        print(f"   - Meetings with empty text: {empty_docs}")
        print(f"   - Questions with empty text: {empty_questions}")

        # Check average text length
        avg_doc_length = sum(len(doc['text']) for doc in documents) / total_documents if total_documents > 0 else 0
        print(f"   - Average meeting length: {avg_doc_length:.0f} characters")

        # Check task type consistency
        task_types = set()
        query_types = set()
        for doc in documents:
            for q in doc['questions']:
                task_types.add(q['metadata'].get('task_type', 'unknown'))
                query_types.add(q['metadata'].get('query_type', 'unknown'))

        print(f"   - Task types found: {sorted(task_types)}")
        print(f"   - Query types found: {sorted(query_types)}")

        print(f"   - Validation completed for {split} split")

    print(f"\n{'='*50}")
    print("QMSum VALIDATION SUMMARY COMPLETED")
    print(f"{'='*50}")


if __name__ == "__main__":
    download_qmsum_data()