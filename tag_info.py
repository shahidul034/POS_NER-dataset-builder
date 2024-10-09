
pos_tags_list={'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '।': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11, 'DT': 12,
 'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23,
 'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33,
 'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43,
 'WP': 44, 'WP$': 45, 'WRB': 46}
ner_tags_list={'O': 0, 'B-ADJP': 1, 'I-ADJP': 2, 'B-ADVP': 3, 'I-ADVP': 4, 'B-CONJP': 5, 'I-CONJP': 6, 'B-INTJ': 7, 'I-INTJ': 8,
 'B-LST': 9, 'I-LST': 10, 'B-NP': 11, 'I-NP': 12, 'B-PP': 13, 'I-PP': 14, 'B-PRT': 15, 'I-PRT': 16, 'B-SBAR': 17,
 'I-SBAR': 18, 'B-UCP': 19, 'I-UCP': 20, 'B-VP': 21, 'I-VP': 22}
# POS tags mapping
pos_tags_list_descr = {
    0: '" (Quotation mark)',
    1: "'' (Closing quotation mark)",
    2: '# (Number sign)',
    3: '$ (Dollar sign)',
    4: '( (Left parenthesis)',
    5: ') (Right parenthesis)',
    6: ', (Comma)',
    7: '। (Bengali punctuation for period)',
    8: ': (Colon)',
    9: '` (Opening quotation mark)',
    10: 'CC (Coordinating conjunction)',
    11: 'CD (Cardinal number)',
    12: 'DT (Determiner)',
    13: 'EX (Existential there)',
    14: 'FW (Foreign word)',
    15: 'IN (Preposition/subordinating conjunction)',
    16: 'JJ (Adjective)',
    17: 'JJR (Comparative adjective)',
    18: 'JJS (Superlative adjective)',
    19: 'LS (List item marker)',
    20: 'MD (Modal verb)',
    21: 'NN (Noun, singular/mass)',
    22: 'NNP (Proper noun, singular)',
    23: 'NNPS (Proper noun, plural)',
    24: 'NNS (Noun, plural)',
    25: 'NN|SYM (Noun or symbol)',
    26: 'PDT (Predeterminer)',
    27: 'POS (Possessive ending)',
    28: 'PRP (Personal pronoun)',
    29: 'PRP$ (Possessive pronoun)',
    30: 'RB (Adverb)',
    31: 'RBR (Comparative adverb)',
    32: 'RBS (Superlative adverb)',
    33: 'RP (Particle)',
    34: 'SYM (Symbol)',
    35: 'TO (Preposition or infinitive marker)',
    36: 'UH (Interjection)',
    37: 'VB (Verb, base form)',
    38: 'VBD (Verb, past tense)',
    39: 'VBG (Verb, gerund/present participle)',
    40: 'VBN (Verb, past participle)',
    41: 'VBP (Verb, non-3rd person singular present)',
    42: 'VBZ (Verb, 3rd person singular present)',
    43: 'WDT (Wh-determiner)',
    44: 'WP (Wh-pronoun)',
    45: 'WP$ (Possessive wh-pronoun)',
    46: 'WRB (Wh-adverb)',
}

# NER tags mapping
ner_tags_list_descr = {
    0: 'O (Outside any named entity)',
    1: 'B-ADJP (Beginning of adjective phrase)',
    2: 'I-ADJP (Inside adjective phrase)',
    3: 'B-ADVP (Beginning of adverb phrase)',
    4: 'I-ADVP (Inside adverb phrase)',
    5: 'B-CONJP (Beginning of conjunction phrase)',
    6: 'I-CONJP (Inside conjunction phrase)',
    7: 'B-INTJ (Beginning of interjection)',
    8: 'I-INTJ (Inside interjection)',
    9: 'B-LST (Beginning of list item)',
    10: 'I-LST (Inside list item)',
    11: 'B-NP (Beginning of noun phrase)',
    12: 'I-NP (Inside noun phrase)',
    13: 'B-PP (Beginning of prepositional phrase)',
    14: 'I-PP (Inside prepositional phrase)',
    15: 'B-PRT (Beginning of particle phrase)',
    16: 'I-PRT (Inside particle phrase)',
    17: 'B-SBAR (Beginning of subordinate clause)',
    18: 'I-SBAR (Inside subordinate clause)',
    19: 'B-UCP (Beginning of unlike coordinated phrase)',
    20: 'I-UCP (Inside unlike coordinated phrase)',
    21: 'B-VP (Beginning of verb phrase)',
    22: 'I-VP (Inside verb phrase)',
}

# Function to get POS tag description
def get_pos_tag_description(tag_number):
    return pos_tags_list_descr.get(tag_number, "Unknown POS tag")

# Function to get NER tag description
def get_ner_tag_description(tag_number):
    return ner_tags_list_descr.get(tag_number, "Unknown NER tag")

# Example usage
# pos_tag_number = 21
# ner_tag_number = 11
# print(f"POS Tag {pos_tag_number}: {get_pos_tag_description(pos_tag_number)}")
# print(f"NER Tag {ner_tag_number}: {get_ner_tag_description(ner_tag_number)}")
