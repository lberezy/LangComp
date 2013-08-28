# L&C Project 1
# Lucas BEREZY
# Student Number: 588236
# Python 3.2.5

# ~$ 'python3 588236.py build' to create new data file.

import re
import sys
import pickle
from nltk.corpus import words, brown, treebank
from nltk.probability import ConditionalFreqDist
from nltk.tokenize import word_tokenize, sent_tokenize

""" <approximately 400 words of discussion about what your program does, and
what problems you tacked and solved (or could not solve). Are there other
sources of linguistic information that would have been useful? Briefly describe
a possible application of this software.

This program aims to programmatically determine the correct casing for the
sentences in a text. In order for this to be effective the program must be able
to do more than simply capitalise the first word of a sentence.

The main method of improving accuracy is to tokenize each word in the sentence
and check for its exclusion or inclusion in various sample sets, e.g. a
set may contain a large number of comsmonly found lowercase words. If the tested
word does not exist in this set then the likelyhood that it is a potentially
capitalised word increases.

The personal pronoun 'I' and the article 'a' are usually capitalised as stated,
so checkes are made on those and capitalised accordingly (as long as the 'a'
does not start a sentence or is inside a quote block).

Quote blocks are another thing to watch out for. The first word inside a quoted
piece of text is usually capitalised, so a check is made for that. e.g. 'Word
word word "A phrase!", she said.'

The capitalisation of the first word of a sentence was slightly less trivial
than usually imagined as there exists corner cases, like an entire sentence
being quoted and thus the first alphabet character of the string needs to be
capitalised.

This program also includes a 'build' function that attempts to build some
probability distributions of words and their capitalisation frequency from a
user-supplied piece of text (currently still broken). This


Issues encountered:
    -   The word tokenizer behaves oddly some quoted texts,
    sometimes returning `` and '' tokens and at other times
    returning the
    expected "" tokens.
    -   Having truecase() return a string was tricky, and I'm not sure how
    good a job my regex does on it.
    -   Some sentences with quoted text embedded in them tokenize to two
    sentences, making it hard to know if the first word of a sentence was
    really that. Overcome by capitalising the first alphabet character seen
    in a sentence.


One method that is used to increase the probability of success is a dictionary
file containing the lowercased words and their relative probability of appearing
capitalised in a text. This data will need to be gathered by a helper function
in advance. This will work quite well for acronyms and other words that are
fully capitalised and names/proper nouns, however, it is expected to work less
effectively for words that have no default case preference. In order to combat
this, the NLTK's Part of Speech (POS) tagging functions will be used to better
guess the class for each word and this data can then be used to augment the
trained tool heuristic.

Other sources of useful linguistic information would be a distribution of word-
lengths and probability of title casing. A very useful, but probably
unobtainable source of information would be a perfect part of speech tagging
system that could perfectly infer the correct word type based on context in a
sentence.

A use of this type of program would be in customer-facing natural language query
systems where it cannot be guaranteed the use will use correct casing. If
correct casing can be guessed from the input it makes discovering contextual
meaning of their input easier and may make certain backend queries simpler
(correctly casing the name of a movie in a natural language request for
information on the movie, and the movies' actor allows simpler extraction of the
key information as it can be tagged as a proper noun easier). Could also be used
as part of a spellchecker that automatically corrected capitalisation. """

# PROBLEMS:
# nltk.tokenizer.word_tokenizer() tokenizes "" as `` ''



# ------
# optionally import NLTK modules here (e.g. tokenizer or corpus)


# how to tokenize "and then he said ""where are you?" said someone."
# ? and then . in a single sentence.

# when building the data file, don't be tricked by title cased words at the start of
# the text




# need some global data variables

wordlist = set(words.words() + treebank.words())
common_words_lower = set([w for w in wordlist if w.islower()])
common_words_titlecase = set([w.lower() for w in wordlist if (w.istitle() and w not in common_words_lower)])

def truecase(s, threshold = 0.5):
    '''Attempts to correctly capitalise words in a sentence.
    Returns a string.'''
    print(s)
    # capitalise the first alphabet character, not simply the first
    s = s.capitalize()
    for i, c in enumerate(s):
        if c == ' ':  # potentially broken quote, don't capitalise in this case
            break
        if c.isalpha():
            s = s[:i] + s[i:].capitalize()
            break

    new_sent = []
    words = word_tokenize(s)
    new_words = []

    for i in range(1, len(words)-1):   # begin at second item
        # capitalise the first word inside any weird quotes
        # would like to try to fix contractions being tokenized
        # into their meaning parts ('do' , '`nt' -> 'don`t')
        # but it turned into more of a mess
        if words[i-1] in set(['"', "''", '''“''']):
            try:
                words[i] = words[i].capitalize()
            except:
                pass


    for word in words:
        # if 'long-ish' word and most commonly found capitalised, capitalise it
        if word.lower() in cfd and len(word) >= 3:
            candidate_word = cfd[word.lower()].max()
            if cfd[word.lower()].freq(candidate_word) >= threshold:
                word = cfd[word.lower()].max()
        else:  # word not found, it's probably a proper noun
            if len(word) > 3 and word not in common_words_lower:
                word = word.capitalize()

        #if word.lower() in common_words_titlecase:
        #    word = word.capitalize()
        if word =='i':    # any occurance of a lone 'I' should be capitalised
            word = word.capitalize()
        if word == 'A':    # any occurance of a lone a should be lowercase
            word = word.lower()
        new_sent.append(word)

        # fixes a strange issue where the very first sentence of a text
        # would not get capitalised
        new_sent[0] = new_sent[0].capitalize()
    return build_sentence(new_sent)


def evaluate(s):
    '''Helper function to evaluate the truecase function.
    Calls the  truecase() function on each sentence found in the (lowercased)
    text, compares the accuracy of the truecasing function to the originally
    correct cased texts and returns a floating point percentage of words
    cased correctly.'''

    t = s.lower()
    score = 0.0

    orig_sents = sent_tokenize(s)
    lower_sents = sent_tokenize(t)

    # note: tokenization can sometimes get out of alignment between
    # the two sentences which causes massive accuracy drops.
    for sent_orig, sent_lower in zip(orig_sents, lower_sents):
        score += rate_similarity(truecase(sent_lower), sent_orig)

    return score/len(orig_sents)*100


def rate_similarity(sent1, sent2):
    '''Compares two sentences, word for word and returns the percentage
    of words that are identical.'''

    sent1_words = word_tokenize(sent1)
    sent2_words = word_tokenize(sent2)
    count = 0
    if not len(sent1) or not len(sent2):
        print("Warning: Something's going wrong!")
        return 0.0

    print("\nNew: " + sent1)
    print("Old: " + sent2)
    for w1, w2 in zip(sent1_words, sent2_words):
        if w1 == w2:
            count += 1

    return count/len(sent2_words)


def build_sentence(sent_list):
    '''Takes a list of strings that constitute a tokenized sentence
    and attempts to rebuild them into a string with some regex hackery.
    Doesn't always play nicely when re-tokenized.'''
    sentence = ' '.join(sent_list)  # join lists with space separators
    # find any "space then punctuation" and remove the space
    sentence = re.sub(""" (?=[?'`“!.,;:@-])""", '', sentence)
    sentence = re.sub("""(``)|('')""", '"', sentence)
    return sentence


def load_data(data_filename):
    try:
        data_file = open(data_filename, 'rb')
    except IOError:
        exit("Cannot open: " + data_filename + "\n Make sure it has been\
            created with the 'build' argument.")

    global cfd  # so cfd can be quickly accessed from other fucntions
    cfd = pickle.load(data_file)
    # global common_words_lower
    # common_words_lower = pickle.load(data_file)
    # global common_words_titlecase
    # common_words_titlecase = pickle.load(data_file)
    # global common_words_FULLCAPS
    # common_words_FULLCAPS = pickle.load(data_file)


def build_index(out_filename, in_filename = None):
    '''Builds data files for word lookup. Can take an optional input file
    to add to the data pool which is processed. Data is then dumped to a
    pickle file.'''

    sents_data = []
    try:
        in_file = open(in_filename).read()
        sents_data += sent_tokenize(in_file)
        in_file.close()
    except:
        print("Warning: Failed to load external file for building.")

    sents_data += brown.sents() + treebank.sents()

    # get sentences, chop of rtheir ambiguous heads, and look at their words!
    mysents = [sent[1:] for sent in sents_data]
    # flatten sublists of words to list of words
    mywords = [word for word in mysents for word in word]
    cfd = ConditionalFreqDist((word.lower(), word) for word in mywords)
    # look up most frequent form of lowercase word by doing cfd['word'].max()
    # but need to check for existance of word in cfd first

    # made pickle file too large and slow
    # wordlist = set(words.words())
    # wordlist.update(brown.words())
    # wordlist.update(treebank.words())
    # common_words_lower = set([w for w in wordlist if w.islower()])
    # common_words_titlecase = set([w.lower() for w in wordlist if (w.istitle() and w not in common_words_lower)])
    
    out_file = open(out_filename, 'wb')
    pickle.dump(cfd, out_file, 2)
    # pickle.dump(common_words_lower, out_file, 2)
    # pickle.dump(common_words_titlecase, out_file, 2)
    out_file.close()


# Main program: read in the data, process it, and print the result
if __name__ == '__main__':

    if len(sys.argv) == 2 and sys.argv[1] != 'build':
        filename = sys.argv[1]
        # get the data out of the file
        try:
            input = open(filename).read()
        except IOError:
            exit("Cannot open: " + filename)
        # process the data
        data_filename = (sys.argv[0].split('.')[0] + '.data')
        print(data_filename)
        load_data(data_filename)
        result = evaluate(input)
        print(result)
        exit()

    if sys.argv[1] == 'build':  # build argument
        if len(sys.argv) == 3:
                    print("Built index from " + sys.argv[2])
                    filename = sys.argv[2]
        else:
            print("Building index...")
            filename = None
        build_index(sys.argv[0].split('.')[0] + '.data', filename)

        exit("Finished building index.")
    else:
        exit("Usage:" + sys.argv[0] + " [build] filename")
