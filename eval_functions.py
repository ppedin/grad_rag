"""
Evaluation functions for comparing system-generated answers with reference answers.
"""

from typing import List, Set
import re
from datasets_schema import Question


def tokenize(text: str) -> List[str]:
    """
    Simple tokenization function that splits on whitespace and removes punctuation.

    Args:
        text (str): Input text to tokenize

    Returns:
        List[str]: List of tokens (words)
    """
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Split on whitespace and filter empty strings
    tokens = [token for token in text.split() if token.strip()]
    return tokens


def compute_rouge_l(candidate: str, reference: str) -> float:
    """
    Compute ROUGE-L score between candidate and reference text.
    ROUGE-L measures the longest common subsequence between the two texts.

    Args:
        candidate (str): System-generated answer
        reference (str): Reference answer

    Returns:
        float: ROUGE-L F1 score
    """
    def lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        if m == 0 or n == 0:
            return 0

        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    # Tokenize both texts
    candidate_tokens = tokenize(candidate)
    reference_tokens = tokenize(reference)

    if not candidate_tokens and not reference_tokens:
        return 1.0
    if not candidate_tokens or not reference_tokens:
        return 0.0

    # Compute LCS length
    lcs_len = lcs_length(candidate_tokens, reference_tokens)

    # Compute precision and recall
    precision = lcs_len / len(candidate_tokens) if len(candidate_tokens) > 0 else 0.0
    recall = lcs_len / len(reference_tokens) if len(reference_tokens) > 0 else 0.0

    # Compute F1 score
    if precision + recall == 0:
        return 0.0

    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score


def compute_f1_unigram_overlap(candidate: str, reference: str) -> float:
    """
    Compute F1 score based on unigram (word) overlap between candidate and reference text.

    Args:
        candidate (str): System-generated answer
        reference (str): Reference answer

    Returns:
        float: F1 score based on unigram overlap
    """
    # Tokenize both texts
    candidate_tokens = set(tokenize(candidate))
    reference_tokens = set(tokenize(reference))

    if not candidate_tokens and not reference_tokens:
        return 1.0
    if not candidate_tokens or not reference_tokens:
        return 0.0

    # Compute overlap
    overlap = candidate_tokens.intersection(reference_tokens)

    # Compute precision and recall
    precision = len(overlap) / len(candidate_tokens) if len(candidate_tokens) > 0 else 0.0
    recall = len(overlap) / len(reference_tokens) if len(reference_tokens) > 0 else 0.0

    # Compute F1 score
    if precision + recall == 0:
        return 0.0

    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score


def evaluate_rouge_score(question: Question, system_answer: str) -> float:
    """
    Evaluate a system answer against reference answers using ROUGE-L score.
    Returns the maximum ROUGE-L score across all reference answers.

    Args:
        question (Question): Question object containing reference answers
        system_answer (str): System-generated answer to evaluate

    Returns:
        float: Maximum ROUGE-L score across all reference answers
    """
    if not question.answers:
        return 0.0

    rouge_scores = []
    for reference_answer in question.answers:
        score = compute_rouge_l(system_answer, reference_answer)
        rouge_scores.append(score)

    return max(rouge_scores)


def evaluate_f1_unigram_score(question: Question, system_answer: str) -> float:
    """
    Evaluate a system answer against reference answers using F1 unigram overlap.
    Returns the maximum F1 unigram overlap score across all reference answers.

    Args:
        question (Question): Question object containing reference answers
        system_answer (str): System-generated answer to evaluate

    Returns:
        float: Maximum F1 unigram overlap score across all reference answers
    """
    if not question.answers:
        return 0.0

    f1_scores = []
    for reference_answer in question.answers:
        score = compute_f1_unigram_overlap(system_answer, reference_answer)
        f1_scores.append(score)

    return max(f1_scores)