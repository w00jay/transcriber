import argparse
import json
import math
from collections import Counter
from difflib import SequenceMatcher

import jiwer
import nltk
import numpy as np
from nltk.metrics.distance import edit_distance
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt_tab")

# Load a pre-trained Sentence-BERT model for semantic similarity
model = SentenceTransformer("all-MiniLM-L6-v2")


def preprocess(text):
    # Basic preprocessing: Lowercase and tokenize
    return " ".join(word_tokenize(text.lower()))


def string_similarity(text1, text2):
    # String similarity using SequenceMatcher
    return SequenceMatcher(None, text1, text2).ratio()


def token_similarity(text1, text2):
    # Token similarity using Jaccard similarity on bag-of-words vectors
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    jaccard = (
        np.minimum(vectors[0], vectors[1]).sum()
        / np.maximum(vectors[0], vectors[1]).sum()
    )
    return jaccard


def semantic_similarity(text1, text2):
    # Semantic similarity using Sentence-BERT embeddings
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding1, embedding2).item()


def are_texts_similar(
    text1, text2, string_threshold=0.8, token_threshold=0.7, semantic_threshold=0.85
):
    text1, text2 = preprocess(text1), preprocess(text2)

    # Compute similarities
    string_sim = string_similarity(text1, text2)
    token_sim = token_similarity(text1, text2)
    semantic_sim = semantic_similarity(text1, text2)

    # Print similarities for insight
    print("===== Text similarty results =====")
    print(f"String Similarity: {string_sim:.2f}")
    print(f"Token Similarity: {token_sim:.2f}")
    print(f"Semantic Similarity: {semantic_sim:.2f}")

    # Check if all similarities exceed their thresholds
    result = (
        (string_sim > string_threshold)
        and (token_sim > token_threshold)
        and (semantic_sim > semantic_threshold)
    )
    if result:
        print("The texts are similar.")
    else:
        print("The texts are not similar.")

    return result


def interpret_cosine_similarity(score):
    if score >= 0.7:
        result = f"Score: {score:.2f} - Highly similar"
    elif 0.4 <= score < 0.7:
        result = f"Score: {score:.2f} - Moderately similar"
    elif 0 < score < 0.4:
        result = f"Score: {score:.2f} - Low similarity"
    else:
        result = "Score: 0.00 - No similarity"
    print(result)


def compute_similarity(text1, text2):
    """
    Compute cosine similarity between two text strings.

    Args:
        text1 (str): First text string
        text2 (str): Second text string

    Returns:
        float: Cosine similarity score between 0 and 1
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    result = cosine_sim[0][0]

    print("===== Cosine similarty results =====")
    print(f"Cosine Similarity: {result:.4f}")
    interpret_cosine_similarity(result)

    return result


# WER = Word Error Rate:
# Number of words in the reference that are not in the hypothesis divided by the number of words in the reference
def calculate_wer(reference, hypothesis):
    result = jiwer.wer(reference, hypothesis)
    print(f"WER: {result:.2f}")
    return result


# Levenshtein Distance = Edit Distance
# Number of single-character edits (i.e. insertions, deletions, or substitutions) required to change one word into another
def calculate_levenshtein(reference, hypothesis):
    result = edit_distance(reference, hypothesis)
    print(f"Levenshtein Distance: {result}")
    return result


# CER = Character Error Rate
# Measures the error rate at the character level
def calculate_cer(reference, hypothesis):
    result = jiwer.cer(reference, hypothesis)
    print(f"CER: {result}")
    return result


# BLEU Score = Bilingual Evaluation Understudy Score
# Measures the similarity between a candidate translation and a reference translation
def calculate_bleu(reference, hypothesis):
    reference = [reference.split()]
    hypothesis = hypothesis.split()
    result = sentence_bleu(reference, hypothesis)
    print(f"BLEU Score: {result:.2f}")
    return result


# ROUGE Score = Recall-Oriented Understudy for Gisting Evaluation
# Measures the similarity between a candidate translation and a reference translation
def calculate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    result = scorer.score(reference, hypothesis)
    print(f"ROUGE Score: {json.dumps(result, indent=4)}")
    return result


# Precision and Recall
def precision_recall(reference, hypothesis):
    ref_words = Counter(reference.split())
    hyp_words = Counter(hypothesis.split())

    common_words = ref_words & hyp_words
    true_positives = sum(common_words.values())
    false_positives = sum((hyp_words - ref_words).values())
    false_negatives = sum((ref_words - hyp_words).values())

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )

    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")
    return precision, recall


def resource_allocation_similarity(text1, text2):
    """
    Adapts resource allocation index concept to text similarity.
    Measures similarity based on shared word frequencies.
    """
    words1 = Counter(text1.lower().split())
    words2 = Counter(text2.lower().split())

    shared_words = set(words1.keys()) & set(words2.keys())
    score = sum(
        1 / math.log(words1[w] + words2[w])
        for w in shared_words
        if words1[w] + words2[w] > 1
    )

    print(f"Resource Allocation Similarity: {score:.4f}")
    return score


def token_sort_ratio(text1, text2):
    """
    Sorts words in both texts before comparing.
    Handles word order differences.
    """
    sorted1 = " ".join(sorted(text1.lower().split()))
    sorted2 = " ".join(sorted(text2.lower().split()))

    ratio = SequenceMatcher(None, sorted1, sorted2).ratio()
    print(f"Token Sort Ratio: {ratio:.4f}")
    return ratio


def partial_ratio(text1, text2):
    """
    Finds best matching substring and computes similarity.
    Adapts RapidFuzz's partial_ratio concept.
    """
    if len(text1) <= len(text2):
        shorter, longer = text1, text2
    else:
        shorter, longer = text2, text1

    best_ratio = 0
    for i in range(len(longer) - len(shorter) + 1):
        substring = longer[i : i + len(shorter)]
        ratio = SequenceMatcher(None, shorter, substring).ratio()
        best_ratio = max(best_ratio, ratio)

    print(f"Partial Ratio: {best_ratio:.4f}")
    return best_ratio


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Compare similarity of two text files."
    )
    parser.add_argument("file1", type=str, help="Path to the first text file.")
    parser.add_argument("file2", type=str, help="Path to the second text file.")

    args = parser.parse_args()

    # Read the contents of the files
    with open(args.file1, "r") as f1, open(args.file2, "r") as f2:
        text1 = f1.read()
        text2 = f2.read()

    # Compare texts w/ string, token, and semantic similarity
    _ = are_texts_similar(text1, text2)

    # Compute cosine similarity
    _ = compute_similarity(text1, text2)

    # Compute WER
    _ = calculate_wer(text1, text2)

    # Compute Levenshtein Distance
    _ = calculate_levenshtein(text1, text2)

    # Compute CER
    _ = calculate_cer(text1, text2)

    # Compute BLEU Score
    _ = calculate_bleu(text1, text2)

    # Compute ROUGE Score
    _ = calculate_rouge(text1, text2)

    # Compute Precision and Recall
    _ = precision_recall(text1, text2)

    # Additional similarity metrics
    _ = resource_allocation_similarity(text1, text2)
    _ = partial_ratio(text1, text2)
    _ = token_sort_ratio(text1, text2)


if __name__ == "__main__":
    main()

# python text-diff.py test/data/text-source.txt test/data/text-transcribed.txt
