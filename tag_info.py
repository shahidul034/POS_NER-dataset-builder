
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

markdown_data="""
## POS Tags Documentation

- `"` : 0 - Quotation mark (open or close)
- `''` : 1 - Closing quotation mark
- `#` : 2 - Number sign (e.g., hashtags)
- `$` : 3 - Dollar sign
- `(` : 4 - Left parenthesis
- `)` : 5 - Right parenthesis
- `,` : 6 - Comma
- `।` : 7 - Bengali punctuation mark for a full stop (period)
- `:` : 8 - Colon
- ``` ` ``` : 9 - Opening quotation mark
- `CC` : 10 - Coordinating conjunction (e.g., *and*, *but*)
- `CD` : 11 - Cardinal number (e.g., *one*, *two*)
- `DT` : 12 - Determiner (e.g., *the*, *a*)
- `EX` : 13 - Existential "there" (e.g., *there is*)
- `FW` : 14 - Foreign word
- `IN` : 15 - Preposition or subordinating conjunction (e.g., *in*, *on*, *because*)
- `JJ` : 16 - Adjective (e.g., *big*, *old*)
- `JJR` : 17 - Adjective, comparative (e.g., *bigger*, *older*)
- `JJS` : 18 - Adjective, superlative (e.g., *biggest*, *oldest*)
- `LS` : 19 - List item marker (e.g., *1.*, *2.*)
- `MD` : 20 - Modal verb (e.g., *can*, *should*)
- `NN` : 21 - Noun, singular or mass (e.g., *cat*, *laughter*)
- `NNP` : 22 - Proper noun, singular (e.g., *John*, *Paris*)
- `NNPS` : 23 - Proper noun, plural (e.g., *Americans*)
- `NNS` : 24 - Noun, plural (e.g., *cats*)
- `NN|SYM` : 25 - Noun or symbol (rare case)
- `PDT` : 26 - Predeterminer (e.g., *all*, *both*)
- `POS` : 27 - Possessive ending (e.g., *'s*)
- `PRP` : 28 - Personal pronoun (e.g., *I*, *he*, *they*)
- `PRP$` : 29 - Possessive pronoun (e.g., *my*, *his*, *their*)
- `RB` : 30 - Adverb (e.g., *quickly*, *never*)
- `RBR` : 31 - Adverb, comparative (e.g., *faster*)
- `RBS` : 32 - Adverb, superlative (e.g., *fastest*)
- `RP` : 33 - Particle (e.g., *up*, *off*)
- `SYM` : 34 - Symbol (e.g., math or currency symbols)
- `TO` : 35 - "To" as a preposition or infinitive marker (e.g., *to go*)
- `UH` : 36 - Interjection (e.g., *uh*, *well*)
- `VB` : 37 - Verb, base form (e.g., *run*, *eat*)
- `VBD` : 38 - Verb, past tense (e.g., *ran*, *ate*)
- `VBG` : 39 - Verb, gerund or present participle (e.g., *running*, *eating*)
- `VBN` : 40 - Verb, past participle (e.g., *run*, *eaten*)
- `VBP` : 41 - Verb, non-3rd person singular present (e.g., *I run*)
- `VBZ` : 42 - Verb, 3rd person singular present (e.g., *he runs*)
- `WDT` : 43 - Wh-determiner (e.g., *which*, *that*)
- `WP` : 44 - Wh-pronoun (e.g., *who*, *what*)
- `WP$` : 45 - Possessive wh-pronoun (e.g., *whose*)
- `WRB` : 46 - Wh-adverb (e.g., *how*, *where*)

---

## NER Tags Documentation

- `O` : 0 - Outside of any named entity
- `B-ADJP` : 1 - Beginning of an adjective phrase
- `I-ADJP` : 2 - Inside an adjective phrase
- `B-ADVP` : 3 - Beginning of an adverb phrase
- `I-ADVP` : 4 - Inside an adverb phrase
- `B-CONJP` : 5 - Beginning of a conjunction phrase
- `I-CONJP` : 6 - Inside a conjunction phrase
- `B-INTJ` : 7 - Beginning of an interjection
- `I-INTJ` : 8 - Inside an interjection
- `B-LST` : 9 - Beginning of a list item
- `I-LST` : 10 - Inside a list item
- `B-NP` : 11 - Beginning of a noun phrase (e.g., *the big dog*)
- `I-NP` : 12 - Inside a noun phrase
- `B-PP` : 13 - Beginning of a prepositional phrase (e.g., *in the house*)
- `I-PP` : 14 - Inside a prepositional phrase
- `B-PRT` : 15 - Beginning of a particle phrase (e.g., *up* in *give up*)
- `I-PRT` : 16 - Inside a particle phrase
- `B-SBAR` : 17 - Beginning of a subordinate clause (e.g., *that he left*)
- `I-SBAR` : 18 - Inside a subordinate clause
- `B-UCP` : 19 - Beginning of an unlike coordinated phrase
- `I-UCP` : 20 - Inside an unlike coordinated phrase
- `B-VP` : 21 - Beginning of a verb phrase (e.g., *ran quickly*)
- `I-VP` : 22 - Inside a verb phrase


"""