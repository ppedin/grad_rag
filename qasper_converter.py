import json
import os
import tiktoken
from collections import Counter
from datasets_schema import Dataset, Document, Question


def analyze_qasper_structure():
    """
    Analyze the structure of QASPER dataset files and provide detailed report.
    """
    files = {
        'train': 'qasper/qasper-train-dev-v0.3/qasper-train-v0.3.json',
        'dev': 'qasper/qasper-train-dev-v0.3/qasper-dev-v0.3.json',
        'test': 'qasper/qasper-test-and-evaluator-v0.3/qasper-test-v0.3.json'
    }

    print("=" * 60)
    print("QASPER DATASET STRUCTURE ANALYSIS")
    print("=" * 60)

    # Initialize tiktoken encoder for token counting
    encoding = tiktoken.get_encoding("cl100k_base")

    for split_name, file_path in files.items():
        if not os.path.exists(file_path):
            print(f"\n{split_name.upper()} SPLIT: FILE NOT FOUND - {file_path}")
            continue

        print(f"\n{split_name.upper()} SPLIT")
        print("-" * 40)

        # File size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")

        # Load data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        num_papers = len(data)
        print(f"Number of papers: {num_papers}")

        # Analyze first paper structure
        sample_paper_id = list(data.keys())[0]
        sample_paper = data[sample_paper_id]

        print(f"\nPAPER STRUCTURE:")
        print(f"  - Paper ID example: {sample_paper_id}")
        print(f"  - Top-level keys: {list(sample_paper.keys())}")

        # Analyze title and abstract
        print(f"\n  TITLE & ABSTRACT:")
        print(f"    - Title: {sample_paper['title'][:100]}...")
        print(f"    - Abstract length: {len(sample_paper['abstract'])} chars")

        # Analyze full_text structure
        full_text = sample_paper['full_text']
        print(f"\n  FULL TEXT STRUCTURE:")
        print(f"    - Number of sections: {len(full_text)}")
        print(f"    - Section structure example:")

        if len(full_text) > 0:
            sample_section = full_text[0]
            print(f"      - Section name: {sample_section['section_name']}")
            print(f"      - Number of paragraphs: {len(sample_section['paragraphs'])}")
            if len(sample_section['paragraphs']) > 0:
                print(f"      - First paragraph (100 chars): {sample_section['paragraphs'][0][:100]}...")

        # Analyze QA structure
        qas = sample_paper.get('qas', [])
        print(f"\n  QUESTIONS & ANSWERS:")
        print(f"    - Number of QA pairs: {len(qas)}")

        if len(qas) > 0:
            sample_qa = qas[0]
            print(f"    - QA structure keys: {list(sample_qa.keys())}")
            print(f"    - Question: {sample_qa.get('question', '')[:100]}...")

            if 'answers' in sample_qa:
                answers = sample_qa['answers']
                print(f"    - Number of answers: {len(answers)}")
                if len(answers) > 0:
                    sample_answer = answers[0]
                    print(f"    - Answer structure keys: {list(sample_answer.keys())}")
                    if 'answer' in sample_answer:
                        print(f"    - Answer text: {sample_answer['answer'].get('free_form_answer', '')[:100]}...")

        # Analyze figures_and_tables
        figures_tables = sample_paper.get('figures_and_tables', [])
        print(f"\n  FIGURES & TABLES:")
        print(f"    - Number of figures/tables: {len(figures_tables)}")

        # Calculate statistics across all papers
        print(f"\nSTATISTICS ACROSS ALL PAPERS:")

        total_questions = 0
        total_sections = 0
        total_paragraphs = 0
        total_figures_tables = 0
        total_text_length = 0
        total_tokens = 0

        section_names = []
        question_types = []

        print(f"  - Computing statistics for {num_papers} papers...")

        for paper_id, paper in data.items():
            # Count QAs
            paper_qas = paper.get('qas', [])
            total_questions += len(paper_qas)

            # Collect question types
            for qa in paper_qas:
                question_types.append(qa.get('question_type', 'unknown'))

            # Count sections and paragraphs
            full_text = paper.get('full_text', [])
            total_sections += len(full_text)

            paper_text = ""
            for section in full_text:
                section_names.append(section.get('section_name', ''))
                paragraphs = section.get('paragraphs', [])
                total_paragraphs += len(paragraphs)
                paper_text += " ".join(paragraphs) + " "

            # Add title and abstract to text
            paper_text = paper.get('title', '') + " " + paper.get('abstract', '') + " " + paper_text

            total_text_length += len(paper_text)

            # Count tokens
            try:
                paper_tokens = len(encoding.encode(paper_text))
                total_tokens += paper_tokens
            except:
                pass

            # Count figures and tables
            total_figures_tables += len(paper.get('figures_and_tables', []))

        # Print aggregated statistics
        avg_questions_per_paper = total_questions / num_papers if num_papers > 0 else 0
        avg_sections_per_paper = total_sections / num_papers if num_papers > 0 else 0
        avg_paragraphs_per_paper = total_paragraphs / num_papers if num_papers > 0 else 0
        avg_figures_tables_per_paper = total_figures_tables / num_papers if num_papers > 0 else 0
        avg_text_length_per_paper = total_text_length / num_papers if num_papers > 0 else 0
        avg_tokens_per_paper = total_tokens / num_papers if num_papers > 0 else 0

        print(f"  - Total questions: {total_questions}")
        print(f"  - Average questions per paper: {avg_questions_per_paper:.2f}")
        print(f"  - Average sections per paper: {avg_sections_per_paper:.2f}")
        print(f"  - Average paragraphs per paper: {avg_paragraphs_per_paper:.2f}")
        print(f"  - Average figures/tables per paper: {avg_figures_tables_per_paper:.2f}")
        print(f"  - Average text length per paper: {avg_text_length_per_paper:.0f} characters")
        print(f"  - Average tokens per paper: {avg_tokens_per_paper:.0f}")

        # Analyze question types
        question_type_counts = Counter(question_types)
        print(f"\n  QUESTION TYPES DISTRIBUTION:")
        for q_type, count in question_type_counts.most_common():
            percentage = (count / total_questions) * 100 if total_questions > 0 else 0
            print(f"    - {q_type}: {count} ({percentage:.1f}%)")

        # Analyze section names
        section_name_counts = Counter(section_names)
        print(f"\n  MOST COMMON SECTION NAMES:")
        for section, count in section_name_counts.most_common(10):
            print(f"    - {section}: {count}")

    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETED")
    print(f"{'=' * 60}")


def convert_qasper_to_schema(split="train", output_file=None):
    """
    Convert QASPER data to the schema format defined in datasets_schema.py

    Args:
        split (str): Data split to process ('train', 'dev', 'test')
        output_file (str): Output JSON file path (auto-generated if None)

    Returns:
        str: Path to the generated JSON file
    """
    # File mapping
    file_mapping = {
        'train': 'qasper/qasper-train-dev-v0.3/qasper-train-v0.3.json',
        'dev': 'qasper/qasper-train-dev-v0.3/qasper-dev-v0.3.json',
        'test': 'qasper/qasper-test-and-evaluator-v0.3/qasper-test-v0.3.json'
    }

    if split not in file_mapping:
        raise ValueError(f"Invalid split '{split}'. Must be one of: {list(file_mapping.keys())}")

    input_file = file_mapping[split]
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if output_file is None:
        # Handle dev/val naming
        if split == 'dev':
            output_file = "qasper_val.json"
        else:
            output_file = f"qasper_{split}.json"

    print(f"Converting QASPER {split} split to schema format...")

    # Load QASPER data
    with open(input_file, 'r', encoding='utf-8') as f:
        qasper_data = json.load(f)

    documents = []

    for paper_id, paper_data in qasper_data.items():
        print(f"Processing paper {paper_id}...")

        # Extract and concatenate full text from all sections
        full_text_parts = []

        # Add title and abstract
        title = paper_data.get('title', '')
        abstract = paper_data.get('abstract', '')

        full_text_parts.append(f"Title: {title}")
        full_text_parts.append(f"Abstract: {abstract}")

        # Add all sections
        full_text_sections = paper_data.get('full_text', [])
        for section in full_text_sections:
            section_name = section.get('section_name', '')
            paragraphs = section.get('paragraphs', [])

            if section_name:
                full_text_parts.append(f"Section: {section_name}")

            for paragraph in paragraphs:
                if paragraph.strip():  # Skip empty paragraphs
                    full_text_parts.append(paragraph)

        # Concatenate all text
        document_text = '\n\n'.join(full_text_parts)

        # Process questions and answers
        questions = []
        qas_data = paper_data.get('qas', [])

        for qa_idx, qa in enumerate(qas_data):
            question_text = qa.get('question', '')
            question_id = qa.get('question_id', f"{paper_id}_q{qa_idx}")

            # Extract answers
            answers_data = qa.get('answers', [])
            answer_texts = []

            for answer in answers_data:
                answer_obj = answer.get('answer', {})
                if isinstance(answer_obj, dict):
                    # Handle different answer types
                    if 'free_form_answer' in answer_obj:
                        answer_text = answer_obj['free_form_answer']
                    elif 'extractive_spans' in answer_obj:
                        spans = answer_obj['extractive_spans']
                        if spans:
                            answer_text = '; '.join(spans)
                        else:
                            answer_text = "No extractive spans provided"
                    elif 'yes_no' in answer_obj:
                        answer_text = str(answer_obj['yes_no'])
                    else:
                        answer_text = str(answer_obj)
                else:
                    answer_text = str(answer_obj)

                if answer_text and answer_text.strip():
                    answer_texts.append(answer_text)

            # If no valid answers found, skip this question
            if not answer_texts:
                continue

            # Create Question object
            question = Question(
                id=question_id,
                question=question_text,
                answers=answer_texts,
                metadata={
                    'nlp_background': qa.get('nlp_background'),
                    'topic_background': qa.get('topic_background'),
                    'paper_read': qa.get('paper_read'),
                    'search_query': qa.get('search_query'),
                    'question_writer': qa.get('question_writer'),
                    'original_qa_index': qa_idx
                }
            )
            questions.append(question)

        # Create Document object
        document = Document(
            id=paper_id,
            text=document_text,
            questions=questions,
            metadata={
                'title': title,
                'abstract': abstract,
                'num_sections': len(full_text_sections),
                'num_figures_tables': len(paper_data.get('figures_and_tables', [])),
                'figures_and_tables': paper_data.get('figures_and_tables', [])
            }
        )
        documents.append(document)

    # Create Dataset
    dataset = Dataset(documents=documents)

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset.model_dump(), f, indent=2, ensure_ascii=False)

    print(f"Converted {len(documents)} papers with {sum(len(doc.questions) for doc in documents)} questions")
    print(f"Saved to {output_file}")

    return output_file


def convert_all_qasper_splits():
    """
    Convert all QASPER splits (train, dev, test) to schema format
    """
    splits = ['train', 'dev', 'test']

    for split in splits:
        try:
            output_file = convert_qasper_to_schema(split)
            print(f"Completed {split} split: {output_file}")
        except Exception as e:
            print(f"Failed to process {split} split: {e}")


def validate_qasper_converted_data():
    """
    Validate the converted QASPER JSON files by checking structure and showing statistics.
    """
    splits = ['train', 'dev', 'test']
    encoding = tiktoken.get_encoding("cl100k_base")

    for split in splits:
        # Handle dev/val naming inconsistency
        if split == 'dev':
            json_file = "qasper_val.json"
        else:
            json_file = f"qasper_{split}.json"

        if not os.path.exists(json_file):
            print(f"File {json_file} not found!")
            continue

        print(f"\n{'='*50}")
        print(f"VALIDATING QASPER {split.upper()} SPLIT")
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
                    print(f"   First answer: {q_example['answers'][0]}")
                print(f"   Question metadata keys: {list(q_example['metadata'].keys())}")

            # Show second document example if available
            if total_documents > 1:
                doc_example2 = documents[1]
                print(f"\n   SECOND DOCUMENT EXAMPLE:")
                print(f"   Document ID: {doc_example2['id']}")
                print(f"   Title: {doc_example2['metadata'].get('title', 'N/A')[:100]}...")
                print(f"   Abstract: {doc_example2['metadata'].get('abstract', 'N/A')[:150]}...")
                print(f"   Number of questions: {len(doc_example2['questions'])}")

                if len(doc_example2['questions']) > 0:
                    q_example2 = doc_example2['questions'][0]
                    print(f"   Sample question: {q_example2['question']}")
                    print(f"   Sample answer: {q_example2['answers'][0] if q_example2['answers'] else 'No answers'}")

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
        docs_without_questions = sum(1 for doc in documents if len(doc['questions']) == 0)

        print(f"   - Documents with empty text: {empty_docs}")
        print(f"   - Questions with empty text: {empty_questions}")
        print(f"   - Documents without questions: {docs_without_questions}")

        # Check average text length
        avg_doc_length = sum(len(doc['text']) for doc in documents) / total_documents if total_documents > 0 else 0
        print(f"   - Average document length: {avg_doc_length:.0f} characters")

        # Analyze question metadata
        metadata_keys = set()
        for doc in documents:
            for q in doc['questions']:
                metadata_keys.update(q['metadata'].keys())

        print(f"   - Question metadata fields found: {sorted(metadata_keys)}")

        print(f"   - Validation completed for {split} split")

    print(f"\n{'='*50}")
    print("QASPER VALIDATION SUMMARY COMPLETED")
    print(f"{'='*50}")


if __name__ == "__main__":
    analyze_qasper_structure()