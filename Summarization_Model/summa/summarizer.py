import pdb

import nltk

from math import log10

from .pagerank_weighted import pagerank_weighted_scipy as _pagerank
from .preprocessing.textcleaner import clean_text_by_sentences as _clean_text_by_sentences
from .commons import build_graph as _build_graph
from .commons import remove_unreachable_nodes as _remove_unreachable_nodes
import pdb


def _set_graph_edge_weights(graph, nll=None, weight=None, pair_weight=None):
    #pdb.set_trace()
    for sentence_1 in graph.nodes():
        for sentence_2 in graph.nodes():

            edge = (sentence_1, sentence_2)
            if nll is None:
                if sentence_1 != sentence_2 and not graph.has_edge(edge):
                    similarity = _get_similarity(sentence_1, sentence_2)
                    # if weight is not None:
                    #     similarity += 0.1 * (weight[graph.node_attr[sentence_1]] + weight[graph.node_attr[sentence_2]])
                    if similarity != 0:
                        graph.add_edge(edge, similarity)
            else:
                if sentence_1 != sentence_2 and not graph.has_edge(edge):
                    # similarity = _get_similarity(sentence_1, sentence_2)
                    similarity = nll[graph.node_attr[sentence_1]][graph.node_attr[sentence_2]]
                    if nll[graph.node_attr[sentence_1]][graph.node_attr[sentence_2]] > 0:
                        # if weight is not None:
                        #     similarity *= weight[graph.node_attr[sentence_1]] * weight[graph.node_attr[sentence_2]]
                            # similarity += 0.1 * (
                            #             weight[graph.node_attr[sentence_1]] + weight[graph.node_attr[sentence_2]])
                        # if similarity > 0:
                        graph.add_edge(edge, similarity)# + similarity)

    # Handles the case in which all similarities are zero.
    # The resultant summary will consist of random sentences.
    if all(graph.edge_weight(edge) == 0 for edge in graph.edges()):
        _create_valid_graph(graph)
    # _create_valid_graph(graph)

# def _set_graph_edge_weights(graph, nll=None, weight=None, pair_weight=None, nll_reverse=None):
#     for sentence_1 in graph.nodes():
#         for sentence_2 in graph.nodes():
#
#             edge = (sentence_1, sentence_2)
#             if nll is None:
#                 if sentence_1 != sentence_2 and not graph.has_edge(edge):
#                     similarity = _get_similarity(sentence_1, sentence_2)
#                     # if weight is not None:
#                     #     similarity += 0.1 * (weight[graph.node_attr[sentence_1]] + weight[graph.node_attr[sentence_2]])
#                     if similarity != 0:
#                         graph.add_edge(edge, similarity)
#             else:
#                 if sentence_1 != sentence_2 and not graph.has_edge(edge):
#                     pos = max(0, nll[graph.node_attr[sentence_1]][graph.node_attr[sentence_2]])
#                     neg = max(0, nll_reverse[graph.node_attr[sentence_1]][graph.node_attr[sentence_2]])
#                     similarity = [pos, neg]
#                     graph.add_edge(edge, similarity)

    # Handles the case in which all similarities are zero.
    # The resultant summary will consist of random sentences.
    # if all(graph.edge_weight(edge) == 0 for edge in graph.edges()):
    #     _create_valid_graph(graph)
    # _create_valid_graph(graph)


def _create_valid_graph(graph):
    nodes = graph.nodes()

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue

            edge = (nodes[i], nodes[j])

            if graph.has_edge(edge):
                graph.del_edge(edge)

            graph.add_edge(edge, 1)


def _get_similarity(s1, s2):
    words_sentence_one = s1.split()
    words_sentence_two = s2.split()

    common_word_count = _count_common_words(words_sentence_one, words_sentence_two)

    log_s1 = log10(len(words_sentence_one) + 1e-12)
    log_s2 = log10(len(words_sentence_two) + 1e-12)

    if log_s1 + log_s2 == 0:
        return 0

    return common_word_count / (log_s1 + log_s2)


def _count_common_words(words_sentence_one, words_sentence_two):
    return len(set(words_sentence_one) & set(words_sentence_two))


def _format_results(extracted_sentences, split, score):
    if score:
        return [(sentence.text, sentence.score) for sentence in extracted_sentences]
    if split:
        return [sentence.text for sentence in extracted_sentences]
    return [sentence.text for sentence in extracted_sentences]


def _add_scores_to_sentences(sentences, scores, weights=None):
    for sentence in sentences:
        # Adds the score to the object if it has one.
        if sentence.token in scores:
            sentence.score = scores[sentence.token]
        else:
            sentence.score = 0


def _get_sentences_with_word_count(sentences, words):
    """ Given a list of sentences, returns a list of sentences with a
    total word count similar to the word count provided.
    """
    word_count = 0
    selected_sentences = []
    # Loops until the word count is reached.
    for sentence in sentences:
        words_in_sentence = len(sentence.text.split())

        # Checks if the inclusion of the sentence gives a better approximation
        # to the word parameter.
        if abs(words - word_count - words_in_sentence) > abs(words - word_count):
            return selected_sentences

        selected_sentences.append(sentence)
        word_count += words_in_sentence

    return selected_sentences


def _extract_most_important_sentences(sentences, ratio, words, length, language, num, mode):
    scores = [s.score for s in sentences]
    select_indexes = []
    map = {sentences[i].text: i for i in range(len(sentences))}
    sentences.sort(key=lambda s: s.score, reverse=True)

    # If no "words" option is selected, the number of sentences is
    # reduced by the provided ratio.
    if words is None:
        cands = []
        if length != 0:
            tmp_length = 0
            index = 0
            while index < len(sentences):
                if mode == 'user' and sentences[index].role != 'user' or mode == 'agent' and sentences[index].role != 'agent':
                    index += 1
                    continue
                if language == 'chinese':
                    tmp_length += len(''.join(sentences[index].text.split()))
                else:
                    #tmp_length += len(sentences[index].text.split())
                    tmp_length += len(nltk.word_tokenize(sentences[index].text))
                # if tmp_length > length and index > 0:
                #     break
                cands.append(sentences[index])
                select_indexes.append(map[sentences[index].text])
                index += 1
                if tmp_length > length and index > 0:
                    break
        else:
            cands = sentences[:num]
        return cands, scores, select_indexes

    # Else, the ratio is ignored.
    else:
        return _get_sentences_with_word_count(sentences, words)

def get_ugdg_lm(graph, sentences, weight, lamb):
    scores = {}
    nodes = graph.nodes()
    for i in range(len(nodes)):
        tmp_score = 0
        for j in range(len(nodes)):
            #if i == j:
            if graph.node_attr[nodes[i]] < graph.node_attr[nodes[j]]:
                edge = (nodes[i], nodes[j])
                if graph.has_edge(edge):
                    edge = (nodes[i], nodes[j])
                    tmp_score += graph.edge_weight(edge)
            # elif graph.node_attr[nodes[i]] > graph.node_attr[nodes[j]]:
            #     edge = (nodes[i], nodes[j])
            #     if graph.has_edge(edge):
            #         edge = (nodes[i], nodes[j])
            #         tmp_score += graph.edge_weight(edge)[0]
        scores[nodes[i]] = lamb * tmp_score + (1 - lamb) * weight[graph.node_attr[nodes[i]]]
        #scores[nodes[i]] = tmp_score
    for sentence in sentences:
        if sentence.token in scores:
            sentence.score = scores[sentence.token]



def summarize(text, ratio=0.2, words=None, language="english", split=False, scores=False, additional_stopwords=None, length=84
              , nll=None, num=3, weight=None, pair_weight=None, lamb=0.1, nll_re=None, mode='overall'):
    if not isinstance(text, str):
        raise ValueError("Text parameter must be a Unicode object (str)!")

    # Gets a list of processed sentences.
    sentences = _clean_text_by_sentences(text, language, additional_stopwords)

    # Creates the graph and calculates the similarity coefficient for every pair of nodes.
    graph = _build_graph([sentence.token for sentence in sentences])
    _set_graph_edge_weights(graph, nll, weight, pair_weight)

    # Remove all nodes with all edges weights equal to zero.
    _remove_unreachable_nodes(graph)

    # PageRank cannot be run in an empty graph.
    if len(graph.nodes()) == 0:
        return [] if split else ""

    # Ranks the tokens using the PageRank algorithm. Returns dict of sentence -> score
    #pagerank_scores = _pagerank(graph)

    # Adds the summa scores to the sentence objects.
    #_add_scores_to_sentences(sentences, pagerank_scores)
    get_ugdg_lm(graph, sentences, weight, lamb)

    # for s in sentences:
    #     print(round(s.score,3))
    # exit()

    # Extracts the most important sentences with the selected criterion.
    extracted_sentences, sentence_scores, select_indexes = _extract_most_important_sentences(sentences, ratio, words, length, language, num, mode)

    # Sorts the extracted sentences by apparition order in the original text.
    extracted_sentences.sort(key=lambda s: s.index)

    return _format_results(extracted_sentences, split, scores), sentence_scores, select_indexes


def get_graph(text, language="english"):
    sentences = _clean_text_by_sentences(text, language)

    graph = _build_graph([sentence.token for sentence in sentences])
    _set_graph_edge_weights(graph)

    return graph
