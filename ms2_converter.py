import pandas as pd
import json
import tiktoken
from typing import Dict, List, Tuple
from pathlib import Path
from datasets_schema import Dataset, Document, Question

def load_ms2_data(split: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load MS² dataset files for a given split.

    Args:
        split: One of 'train', 'dev', 'test'

    Returns:
        Tuple of (inputs_df, targets_df, info_df)
        Note: targets_df will be None for test split
    """
    base_path = Path("ms2")

    # Load inputs (always available)
    inputs_df = pd.read_csv(base_path / f"{split}-inputs.csv")

    # Load targets (not available for test split)
    targets_df = None
    if split != 'test':
        targets_df = pd.read_csv(base_path / f"{split}-targets.csv")

    # Load review info (always available)
    info_df = pd.read_csv(base_path / f"{split}-reviews-info.csv")

    return inputs_df, targets_df, info_df

def analyze_ms2_structure(split: str = 'train'):
    """
    Analyze the structure of MS² dataset for a given split.
    Focus on concatenated document statistics.
    """
    print(f"\n=== MS² Dataset Analysis - {split.upper()} Split ===")

    inputs_df, targets_df, info_df = load_ms2_data(split)

    print(f"\nFile sizes:")
    print(f"- {split}-inputs.csv: {len(inputs_df):,} rows")
    if targets_df is not None:
        print(f"- {split}-targets.csv: {len(targets_df):,} rows")
    print(f"- {split}-reviews-info.csv: {len(info_df):,} rows")

    # Analyze unique reviews
    unique_reviews_inputs = inputs_df['ReviewID'].nunique()
    unique_reviews_info = info_df['ReviewID'].nunique()
    if targets_df is not None:
        unique_reviews_targets = targets_df['ReviewID'].nunique()
        print(f"\nUnique ReviewIDs:")
        print(f"- In inputs: {unique_reviews_inputs:,}")
        print(f"- In targets: {unique_reviews_targets:,}")
        print(f"- In info: {unique_reviews_info:,}")
    else:
        print(f"\nUnique ReviewIDs:")
        print(f"- In inputs: {unique_reviews_inputs:,}")
        print(f"- In info: {unique_reviews_info:,}")

    # Articles per review statistics
    articles_per_review = inputs_df.groupby('ReviewID').size()
    print(f"\nArticles per review statistics:")
    print(f"- Mean: {articles_per_review.mean():.1f}")
    print(f"- Median: {articles_per_review.median():.1f}")
    print(f"- Min: {articles_per_review.min()}")
    print(f"- Max: {articles_per_review.max()}")

    # Token analysis using tiktoken - FULL DATASET ANALYSIS
    enc = tiktoken.get_encoding("cl100k_base")

    print(f"\nAnalyzing document tokens (concatenated abstracts) for ALL reviews...")

    # Calculate tokens for concatenated documents (all abstracts per review)
    document_tokens = []

    for review_id, review_inputs in inputs_df.groupby('ReviewID'):
        # Concatenate all abstracts for this review
        abstracts = []
        for _, article in review_inputs.iterrows():
            abstract = str(article['Abstract']) if pd.notna(article['Abstract']) else ""
            if abstract.strip():  # Only add non-empty abstracts
                abstracts.append(abstract)

        # Join abstracts with space
        concatenated_text = " ".join(abstracts)
        tokens = len(enc.encode(concatenated_text))
        document_tokens.append(tokens)

    print(f"\nDocument token statistics (concatenated abstracts):")
    print(f"- Number of documents: {len(document_tokens):,}")
    print(f"- Mean tokens per document: {sum(document_tokens)/len(document_tokens):.1f}")
    print(f"- Median tokens per document: {sorted(document_tokens)[len(document_tokens)//2]:.1f}")
    print(f"- Min tokens per document: {min(document_tokens):,}")
    print(f"- Max tokens per document: {max(document_tokens):,}")
    print(f"- Total tokens in dataset: {sum(document_tokens):,}")

    # Analyze other components if available
    if targets_df is not None:
        target_tokens = []
        background_tokens = []

        for _, row in targets_df.iterrows():
            target = str(row['Target']) if pd.notna(row['Target']) else ""
            background = str(row['Background']) if pd.notna(row['Background']) else ""

            if target.strip():
                target_tokens.append(len(enc.encode(target)))
            if background.strip():
                background_tokens.append(len(enc.encode(background)))

        if target_tokens:
            print(f"\nTarget (answer) token statistics:")
            print(f"- Mean: {sum(target_tokens)/len(target_tokens):.1f}")
            print(f"- Max: {max(target_tokens):,}")

        if background_tokens:
            print(f"\nBackground (question) token statistics:")
            print(f"- Mean: {sum(background_tokens)/len(background_tokens):.1f}")
            print(f"- Max: {max(background_tokens):,}")

    # Show examples
    print(f"\nExample data:")
    if targets_df is not None:
        example_review_id = targets_df['ReviewID'].iloc[0]
        example_target = targets_df[targets_df['ReviewID'] == example_review_id].iloc[0]
        example_inputs = inputs_df[inputs_df['ReviewID'] == example_review_id]

        # Create concatenated document
        abstracts = []
        for _, article in example_inputs.iterrows():
            abstract = str(article['Abstract']) if pd.notna(article['Abstract']) else ""
            if abstract.strip():
                abstracts.append(abstract)
        concatenated_doc = " ".join(abstracts)

        print(f"\nReview ID: {example_review_id}")
        print(f"Background (question): {example_target['Background'][:200]}...")
        print(f"Target (answer): {example_target['Target'][:200]}...")
        print(f"Number of input articles: {len(example_inputs)}")
        print(f"Concatenated document tokens: {len(enc.encode(concatenated_doc)):,}")
        print(f"Concatenated document preview: {concatenated_doc[:400]}...")
    else:
        example_review_id = info_df['ReviewID'].iloc[0]
        example_info = info_df[info_df['ReviewID'] == example_review_id].iloc[0]
        example_inputs = inputs_df[inputs_df['ReviewID'] == example_review_id]

        # Create concatenated document
        abstracts = []
        for _, article in example_inputs.iterrows():
            abstract = str(article['Abstract']) if pd.notna(article['Abstract']) else ""
            if abstract.strip():
                abstracts.append(abstract)
        concatenated_doc = " ".join(abstracts)

        print(f"\nReview ID: {example_review_id}")
        print(f"Background (question): {example_info['Background'][:200]}...")
        print(f"Number of input articles: {len(example_inputs)}")
        print(f"Concatenated document tokens: {len(enc.encode(concatenated_doc)):,}")
        print(f"Concatenated document preview: {concatenated_doc[:400]}...")

def convert_ms2_to_schema(split: str) -> Dataset:
    """
    Convert MS² dataset to the unified schema format.
    Document = concatenated abstracts, Question = background, Answer = target review

    Args:
        split: One of 'train', 'dev', 'test'

    Returns:
        Dataset object following the unified schema
    """
    inputs_df, targets_df, info_df = load_ms2_data(split)

    documents = []

    # Group inputs by ReviewID to create documents
    for review_id, review_inputs in inputs_df.groupby('ReviewID'):
        # Concatenate all abstracts for this review into one document
        abstracts = []
        article_metadata = []

        for _, article in review_inputs.iterrows():
            abstract = str(article['Abstract']) if pd.notna(article['Abstract']) else ""
            title = str(article['Title']) if pd.notna(article['Title']) else ""
            pmid = str(article['PMID']) if pd.notna(article['PMID']) else ""

            if abstract.strip():  # Only add non-empty abstracts
                abstracts.append(abstract)
                article_metadata.append({
                    'pmid': pmid,
                    'title': title
                })

        # Create document text from concatenated abstracts
        document_text = " ".join(abstracts)

        # Create questions for this review
        questions = []

        if targets_df is not None:
            # Get targets for this review
            review_targets = targets_df[targets_df['ReviewID'] == review_id]

            for idx, target_row in review_targets.iterrows():
                target_text = str(target_row['Target']) if pd.notna(target_row['Target']) else ""
                background_text = str(target_row['Background']) if pd.notna(target_row['Background']) else ""

                # Question is the background, answer is the target
                question = Question(
                    id=f"{review_id}_{idx}",
                    question=background_text,  # Background becomes the question
                    answers=[target_text],     # Target becomes the answer
                    metadata={
                        'review_id': str(review_id),
                        'task_type': 'systematic_review_summarization',
                        'domain': 'medical_literature',
                        'num_input_articles': len(review_inputs)
                    }
                )
                questions.append(question)
        else:
            # For test split, use background from info but no answers
            review_info = info_df[info_df['ReviewID'] == review_id]
            if len(review_info) > 0:
                background_text = str(review_info.iloc[0]['Background']) if pd.notna(review_info.iloc[0]['Background']) else ""

                question = Question(
                    id=f"{review_id}_test",
                    question=background_text,  # Background becomes the question
                    answers=[],  # No answers for test split
                    metadata={
                        'review_id': str(review_id),
                        'task_type': 'systematic_review_summarization',
                        'domain': 'medical_literature',
                        'num_input_articles': len(review_inputs)
                    }
                )
                questions.append(question)

        # Create document
        document = Document(
            id=str(review_id),
            text=document_text,  # Concatenated abstracts
            questions=questions,
            metadata={
                'review_id': str(review_id),
                'num_articles': len(review_inputs),
                'articles': article_metadata
            }
        )
        documents.append(document)

    # Create dataset
    dataset = Dataset(
        documents=documents
    )

    return dataset

def save_ms2_dataset(split: str):
    """
    Convert and save MS² dataset split to JSON format.

    Args:
        split: One of 'train', 'dev', 'test'
    """
    print(f"Converting MS² {split} split...")

    dataset = convert_ms2_to_schema(split)

    # Save to JSON
    output_file = f"ms2_{split}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset.model_dump(), f, indent=2, ensure_ascii=False)

    print(f"Saved {len(dataset.documents)} reviews to {output_file}")

    # Calculate file size
    file_size = Path(output_file).stat().st_size / (1024 * 1024)  # MB
    print(f"File size: {file_size:.2f} MB")

def validate_ms2_conversion(split: str):
    """
    Validate the converted MS² dataset.

    Args:
        split: One of 'train', 'dev', 'test'
    """
    print(f"\n=== Validating MS² {split} conversion ===")

    # Load the converted dataset
    json_file = f"ms2_{split}.json"
    if not Path(json_file).exists():
        print(f"Error: {json_file} not found. Run save_ms2_dataset('{split}') first.")
        return

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset = Dataset(**data)

    # Basic statistics
    print(f"Dataset split: {split}")
    print(f"Number of documents: {len(dataset.documents)}")

    total_questions = sum(len(doc.questions) for doc in dataset.documents)
    print(f"Total questions: {total_questions}")

    # Token analysis
    enc = tiktoken.get_encoding("cl100k_base")

    doc_tokens = []
    question_tokens = []
    answer_tokens = []

    for doc in dataset.documents:
        doc_tokens.append(len(enc.encode(doc.text)))

        for q in doc.questions:
            question_tokens.append(len(enc.encode(q.question)))
            if q.answers:  # Only for splits with answers
                answer_tokens.extend([len(enc.encode(ans)) for ans in q.answers])

    print(f"\nToken statistics:")
    print(f"Document tokens - Mean: {sum(doc_tokens)/len(doc_tokens):.1f}, Max: {max(doc_tokens):,}")
    print(f"Question tokens - Mean: {sum(question_tokens)/len(question_tokens):.1f}, Max: {max(question_tokens):,}")

    if answer_tokens:  # Only for splits with answers
        print(f"Answer tokens - Mean: {sum(answer_tokens)/len(answer_tokens):.1f}, Max: {max(answer_tokens):,}")

    # Show example
    print(f"\nExample document:")
    example_doc = dataset.documents[0]
    print(f"ID: {example_doc.id}")
    print(f"Text length: {len(example_doc.text)} characters")
    print(f"Number of questions: {len(example_doc.questions)}")
    print(f"Text preview: {example_doc.text[:500]}...")

    if example_doc.questions:
        example_q = example_doc.questions[0]
        print(f"\nExample question:")
        print(f"ID: {example_q.id}")
        print(f"Question: {example_q.question[:300]}...")
        if example_q.answers:
            print(f"Answer: {example_q.answers[0][:300]}...")

    print(f"\nValidation completed successfully!")

if __name__ == "__main__":
    # Analyze all splits
    for split in ['train', 'dev', 'test']:
        analyze_ms2_structure(split)

    # Convert and save all splits
    for split in ['train', 'dev', 'test']:
        save_ms2_dataset(split)
        validate_ms2_conversion(split)