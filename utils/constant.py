
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN]

# https://spacy.io/models/en#en_core_web_sm
# https://raw.githubusercontent.com/clir/clearnlp-guidelines/master/md/specifications/dependency_labels.md

DEP = [
    'ACL',           # Clausal modifier of noun
	'ACOMP',         # Adjectival complement             # 1.0.0 #
	'ADVCL',         # Adverbial clause modifier         # 1.0.0 #
	'ADVMOD',        # Adverbial modifier                # 1.0.0 #
	'AGENT',         # Agent                             # 1.0.0 #
	'AMOD',          # Adjectival modifier               # 1.0.0 #
	'APPOS',         # Appositional modifier             # 1.0.0 #
	'ATTR',          # Attribute                         # 1.0.0 #``
	'AUX',           # Auxiliary                         # 1.0.0 #
	'AUXPASS',       # Auxiliary (passive)               # 1.0.0 #
	'CASE',          # Case marker                       # 3.1.0 #
	'CC',            # Coordinating conjunction          # 1.0.0 #
	'CCOMP',         # Clausal complement                # 1.0.0 #
	'COMPOUND',      # Compound modifier                 # 3.1.0 #
	'CONJ',          # Conjunct                          # 1.0.0 #
	'CSUBJ',         # Clausal subject                   # 1.0.0 #
	'CSUBJPASS',     # Clausal subject (passive)         # 1.0.0 #
	'DATIVE',        # Dative                            # 3.1.0 #
	'DEP',           # Unclassified dependent            # 1.0.0 #
	'DET',           # Determiner                        # 1.0.0 #
	'DOBJ',          # Direct Object                     # 1.0.0 #
	'EXPL',          # Expletive                         # 1.0.0 #
	'INTJ',          # Interjection                      # 1.0.0 #
	'MARK',          # Marker                            # 1.0.0 #
	'META',          # Meta modifier                     # 1.0.0 #
	'NEG',           # Negation modifier                 # 1.0.0 #
	'NOUNMOD',       # Modifier of nominal               # 3.1.0 #
	'NPMOD',         # Noun phrase as adverbial modifier # 3.1.0 #
	'NSUBJ',         # Nominal subject                   # 1.0.0 #
	'NSUBJPASS',     # Nominal subject (passive)         # 1.0.0 #
	'NUMMOD',        # Number modifier                   # 3.1.0 #
	'OPRD',          # Object predicate                  # 1.0.0 #
	'PARATAXIS',     # Parataxis                         # 1.0.0 #
	'PCOMP',         # Complement of preposition         # 1.0.0 #
	'POBJ',          # Object of preposition             # 1.0.0 #
	'POSS',          # Possession modifier               # 1.0.0 #
	'PRECONJ',       # Pre-correlative conjunction       # 1.0.0 #
	'PREDET',        # Pre-determiner                    # 1.0.0 #
	'PREP',          # Prepositional modifier            # 1.0.0 #
	'PRT',           # Particle                          # 1.0.0 #
	'PUNCT',         # Punctuation                       # 1.0.0 #
	'QUANTMOD',      # Modifier of quantifier            # 1.0.0 #
	'RELCL',         # Relative clause modifier          # 3.1.0 #
	'ROOT',          # Root                              # 1.0.0 #
	'XCOMP'         # Open clausal complement           # 1.0.0 #
]

# Part-Of-Speech tag
# https://universaldependencies.org/u/pos/
POS = [
    '',
    'ADJ', #: adjective
    'ADP', #: adposition
    'ADV', #: adverb
    'AUX', #: auxiliary
    'CCONJ', #: coordinating conjunction
    'DET', #: determiner
    'INTJ', #: interjection
    'NOUN', #: noun
    'NUM', #: numeral
    'PART', #: particle
    'PRON', #: pronoun
    'PROPN', #: proper noun
    'PUNCT', #: punctuation
    'SCONJ', #: subordinating conjunction
    'SYM', #: symbol
    'VERB', #: verb
    'X', #: other
	'SPACE', # space
]

NER = [
    '',
    'PERSON',
    'NORP', # nationalities, religious and political groups
    'FAC', # (buildings, airports etc.)
    'ORG', # organizations
    'GPE', # (countries, cities etc.)
    'LOC', # (mountain ranges, water bodies etc.)
    'PRODUCT', # (products)
    'EVENT', # (event names)
    'WORK_OF_ART', # (books, song titles)
    'LAW', # (legal document titles) 
    'LANGUAGE', # (named languages)
    'DATE',
    'TIME',
    'PERCENT',
    'MONEY',
    'QUANTITY',
    'ORDINAL',
    'CARDINAL'
]

dep_dict = {a: i for i, a in enumerate(DEP)}
pos_dict = {a: i for i, a in enumerate(POS)}
ner_dict = {a: i for i, a in enumerate(NER)}

EDGE_OFFSET = {
    'self': 0,
    'counter': 1000,
	}