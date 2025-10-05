import heapq
import random
from collections import defaultdict
import spacy
import string
import re
from spacy.cli import download
import nltk
from nltk.tokenize import sent_tokenize

# Constants for alignment operations to improve readability
ALIGN = 0
INSERT = 1
DELETE = 2


def load_text_from_file(file_path):
    """Reads the entire content of a specified file."""
    with open(file_path, "r") as file:
        content = file.read()
    return content


def tokenize_into_sentences(text_content):
    """Splits a block of text into a list of sentences using spaCy."""
    text_content = text_content.lower()
    nlp_model = spacy.load("en_core_web_sm")
    doc = nlp_model(text_content)
    sentence_list = [sent.text for sent in doc.sents]
    return sentence_list


def strip_punctuation(sentence_text):
    """Removes punctuation and newline characters from a sentence."""
    return sentence_text.translate(str.maketrans("\n", " ", string.punctuation))


def prepare_document(file_path):
    """Processes a text file by reading, sentence-tokenizing, and cleaning it."""
    file_content = load_text_from_file(file_path)
    sentences = tokenize_into_sentences(file_content)
    cleaned_sentences = [strip_punctuation(sent) for sent in sentences]
    return cleaned_sentences


class SearchNode:
    """Represents a node in the A* search space for document alignment."""

    def __init__(self, position, predecessor=None, path_cost=0, heuristic_cost=0):
        self.position = position  # A tuple (doc1_idx, doc2_idx, move_type)
        self.predecessor = predecessor
        self.path_cost = path_cost  # g(n): Cost from start to this node
        self.heuristic_cost = heuristic_cost  # h(n): Estimated cost to goal
        self.total_cost = self.path_cost + self.heuristic_cost  # f(n)

    def __lt__(self, other):
        """Comparator for the priority queue (heapq)."""
        return self.total_cost < other.total_cost


def compute_levenshtein_distance(str1, str2):
    """Calculates the character-level edit distance between two strings using dynamic programming."""
    len1, len2 = len(str1), len(str2)
    cost_matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(1, len1 + 1):
        cost_matrix[i][0] = i  # Cost of deletions
    for j in range(1, len2 + 1):
        cost_matrix[0][j] = j  # Cost of insertions

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1  # Substitution cost
            cost_matrix[i][j] = min(
                cost_matrix[i - 1][j] + 1,  # Deletion
                cost_matrix[i][j - 1] + 1,  # Insertion
                cost_matrix[i - 1][j - 1] + cost,  # Substitution
            )
    return cost_matrix[len1][len2]


def calculate_move_cost(position):
    """Calculates the cost g(n) for a single alignment, insertion, or deletion."""
    doc1_idx, doc2_idx, move_type = position
    cost = 0

    if move_type == ALIGN:
        sent1 = source_doc[doc1_idx - 1]
        sent2 = target_doc[doc2_idx - 1]
        cost = compute_levenshtein_distance(sent1, sent2)
    elif move_type == INSERT:
        sent2 = target_doc[doc2_idx - 1]
        cost = len(sent2)
    elif move_type == DELETE:
        sent1 = source_doc[doc1_idx - 1]
        cost = len(sent1)
    return cost

def estimate_remaining_cost(position, goal_position):
    """Heuristic function h(n) to estimate the cost from the current state to the goal."""
    # This is a simplified heuristic; a more accurate one could be used.
    doc1_idx, doc2_idx, _ = position
    goal_doc1_idx, goal_doc2_idx, _ = goal_position
    
    remaining_sents1 = len(source_doc) - doc1_idx
    remaining_sents2 = len(target_doc) - doc2_idx
    
    # A basic heuristic: the number of remaining sentences.
    return max(remaining_sents1, remaining_sents2)


def generate_next_states(current_node):
    """Generates all possible successor nodes from the current node."""
    successors = []
    current_pos = current_node.position

    # Defines the possible operations: (d_idx1, d_idx2, move_type)
    alignment_operations = [(1, 1, ALIGN), (0, 1, INSERT), (1, 0, DELETE)]

    for op in alignment_operations:
        new_position = (current_pos[0] + op[0], current_pos[1] + op[1], op[2])
        new_node = SearchNode(new_position, current_node)
        successors.append(new_node)
    return successors


def find_optimal_alignment(start_pos, goal_pos):
    """Performs the A* search to find the minimum-cost alignment path."""
    start_node = SearchNode(start_pos)
    priority_queue = []
    heapq.heappush(priority_queue, (start_node.total_cost, start_node))
    explored_positions = set()
    nodes_visited_count = 0

    while priority_queue:
        _, current_node = heapq.heappop(priority_queue)

        if current_node.position in explored_positions:
            continue
        explored_positions.add(current_node.position)
        nodes_visited_count += 1

        doc1_idx, doc2_idx, _ = current_node.position
        goal_doc1_idx, goal_doc2_idx, _ = goal_pos

        # Check if the goal is reached (end of both documents).
        if doc1_idx > goal_doc1_idx and doc2_idx > goal_doc2_idx:
            path = []
            temp_node = current_node
            while temp_node:
                path.append(temp_node.position)
                temp_node = temp_node.predecessor
            print(f"Search complete. Nodes explored: {nodes_visited_count}")
            return path

        for successor_node in generate_next_states(current_node):
            succ_doc1_idx, succ_doc2_idx, _ = successor_node.position
            
            # Pruning: Do not explore beyond the document lengths.
            if succ_doc1_idx <= goal_doc1_idx + 1 and succ_doc2_idx <= goal_doc2_idx + 1:
                successor_node.path_cost = current_node.path_cost + calculate_move_cost(successor_node.position)
                successor_node.heuristic_cost = estimate_remaining_cost(successor_node.position, goal_pos)
                successor_node.total_cost = successor_node.path_cost + successor_node.heuristic_cost
                heapq.heappush(priority_queue, (successor_node.total_cost, successor_node))
                
    print(f"Search failed. Nodes explored: {nodes_visited_count}")
    return None


def reconstruct_alignment(path, start_pos):
    """Builds the aligned document from the solution path."""
    aligned_sentences = []
    for position in path:
        doc1_idx, doc2_idx, move_type = position
        if position == start_pos:
            continue
        
        if move_type == ALIGN:
            aligned_sentences.append(source_doc[doc1_idx - 1])
        elif move_type == INSERT:
            # An insertion in doc1 means we take the sentence from doc2.
            aligned_sentences.append(target_doc[doc2_idx - 1])
        # Deletions from doc1 are skipped in the final alignment.

    return aligned_sentences


if __name__ == "__main__":
    source_doc = prepare_document("doc1.txt")
    target_doc = prepare_document("doc2.txt")

    start_position = (0, 0, -1)  # Using -1 for the initial dummy move
    end_position = (len(source_doc), len(target_doc), -1)

    print("Source Document Sentences:", source_doc)
    print("Target Document Sentences:", target_doc)
    print("-" * 30)

    solution_path = find_optimal_alignment(start_position, end_position)

    if solution_path:
        solution_path.reverse()
        reconstructed_doc = reconstruct_alignment(solution_path, start_position)
        
        print("\n--- Alignment Results ---")
        total_edit_distance = 0
        for i in range(len(reconstructed_doc)):
            s1 = reconstructed_doc[i]
            s2 = target_doc[i]
            distance = compute_levenshtein_distance(s1, s2)
            total_edit_distance += distance
            print(f'Aligned Sentence: "{s1}"')
            print(f'Target Sentence:  "{s2}"')
            print(f"Edit Distance: {distance}\n")
        
        print(f"--- Total Edit Distance for Alignment: {total_edit_distance} ---")