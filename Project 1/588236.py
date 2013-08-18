# L&C Project 1
# Lucas BEREZY
# 588236

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import re
import pickle
from nltk.corpus import words

"""
<approximately 400 words of discussion about what your program does,
and what problems you tacked and solved (or could not solve).
Are there other sources of linguistic information that would have
been useful? Briefly describe a possible application of this software.

This program aims to programmatically determine the correct casing for
the sentences in a text. In order for this to be effective the program
must be able to do more than simply capitalise the first word of a sentence.
One method that is used to increase the probability of success is a dictionary
file containing the lowercased words and their relative probability of appearing
capitalised in a text. This data will need to be gathered by a helper function in
advance. This will work quite well for acronyms and other words that are fully
capitalised and names/proper nouns, however, it is expected to work less effectively
for words that have no default case preference. In order to compbat this, the NLTK's
Part of Speech (POS) tagging functions will be used to better guess the class for each
word and this data can then be used to augment the trained tool heuristic.

Other sources of useful linguistic information would be a distribution of wordlengths
and probability of title casing. A very useful, but probably unobtainable source of
information would be a perfect part of speech tagging system that could perfectly infer
the correct word type based on context in a sentence.

A use of this type of program would be in customer-facing natural language query systems
where it cannot be guaranteed the use will use correct casing. If correct casing can be
guessed from the input it makes discovering contextual meaning of their input easier and
may make certain backend queries simpler (correctly casing the name of a movie in a natural
language request for information on the movie, and the movies actor allows simpler
extraction of the key information as it can be tagged as a propper noun easier). Could also be used as
part of a spellchecker that automatically corrected capitalisation.
"""

# PROBLEMS:
# nltk.tokenizer.word_tokenizer() tokenizes "" as `` ''



# ------
# optionally import NLTK modules here (e.g. tokenizer or corpus)


# how to tokenize "and then he said ""where are you?" said someone."
# ? and then . in a single sentence.

# when building the data file, don't be tricked by title cased words at the start of
# the text




# need some global data variables

wordlist = set(words.words())
common_words_lower = set([w for w in wordlist if w.islower()])
common_words_titlecase = set([w.lower() for w in wordlist if w.istitle()])
common_words_fullcaps = set([w.lower() for w in wordlist if w.isupper()])
def truecase(s, threshold=0.8):
    '''Attempts to correctly capitalise words in a sentence through a heuristcs
    of comparing words with their relative probability of casing (data pickle
    obtained by training on #text). Also makes use of NLTK's Part Of Speech (POS)
    tagging functions to attempt to improve accuracy. The POS and the trained data
    are evaluated to find a probability that a word should be capitalised, and if
    the combined heuristic reaches a threshold, then then word becomes capitalised'''

    

    '''d_file = open(datafile,'rb')
    wcf = pickle.load(d_file)
    lcf = piclke.load(d_file)
    fullcaps = pickle.load(d_file)
    d_file.close()

    s_original = s[:]  # shallow copy'''
    s.capitalize()
    words = word_tokenize(s)

    words = [w.capitalize() if w not in common_words_lower and len(w) > 2 else w for w in words]
    #words = [w.capitalize() if w in common_words_titlecase and len(w) > 2 else w for w in words]
    #words = [w.upper() if w in common_words_fullcaps and len(w) > 2 else w for w in words]

    # check to capitalise anything a .!? etc and the first thing inside "quotes"
    for i in range(1,len(words)):   # begin at second item
        if words[i-1] in ('"',"'",'''“''','.','!','?'):
            words[i].capitalize()

    sentence_string = build_sentence(words)
    return sentence_string


def evaluate(s):
    '''Helper function to evaluate the truecase function.
    Calls the  truecase() function on each sentence found in the (lowercased)
    text, compares the accuracy of the truecasing function to the originally
    correct cased texts and returns a floating point percentage of words
    cased correctly.'''


    t = s.lower()
    score = 0.0

    orig_sents  = sent_tokenize(s)
    lower_sents = sent_tokenize(t)

    for sent_orig, sent_lower in zip(orig_sents,lower_sents):
        sent_lower = truecase(sent_lower)
        # print(sent_lower)
        # print(sent_orig)
        # print("-")
        score += rate_similarity(sent_lower, sent_orig)


    # put your code here

    return score/len(orig_sents)


def rate_similarity(sent1, sent2):
    ''' compares two sentences, word for word and returns the percentage
    of words that are identical.'''

    sent1 = word_tokenize(sent1)
    sent2 = word_tokenize(sent2)
    count = 0
    if not len(sent1) or not len(sent2):
        return 0.0
    # print(list(zip(sent1,sent2)))
    for w1,w2 in zip(sent1, sent2):
        if w1 == w2:
            count += 1
        else:
            print(w1,w2)
    return count/len(sent1)

def build_sentence(sent_list):
    '''takes a list of strings that constitute a tokenized sentence
    and attempts to rebuild them into a string with some regex hackery.'''
    sentence = ' '.join(sent_list) # join lists with space separators
    return re.sub(""" (?=[?'`!.,;:@-])""", '', sentence) # find any "space then punctuation" errors and remove the space
    # I probably missed some punctuation that needs replacing




def to_capitalise(word):
    '''to capitalise, or not to capitalise...:
    looks up a word in the index pickle file built by build_index and
    returns a (somewhat arbitrary) 'capitalisation score', which is a
    function of the frequency of that word occuring capitalised in text
    and the frequency of a word of that length occuring capitalised in
    the database.'''

    result = 0.0

    import cpickle as pickle

    d_file = open(datafile,'rb')
    wcf = pickle.load(d_file)
    lcf = piclke.load(d_file)
    fullcaps = pickle.load(d_file)
    d_file.close()

    if word in fullcaps and wcf[word] > 0.1:
        return 1.0

    result = wcf[word]

    return result


def build_index(in_file, out_file):
    '''builds an in index datafile based on some training text. builds
    dictionaries with word capitalisation frequency and wordlength and capitalisation
    frequency.'''

    import pickle  # cPickle doesn't do unicode, not that it should matter for this
    from collections import defaultdict
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.probability import FreqDist

    input_f = open(in_file)
    text = input_f.read()
    wcf = defaultdict(int)  # word: capitalisation frequency dict
    lcf = defaultdict(int)  # length of word: capitalisation frequency dict
    fullcaps = {}           # used as a list, but with constant existance checking

    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    wordfreq = FreqDist([word.lower() for word in words if word.isalpha()])  # word: count distribution
    wordlenfreq = FreqDist([len(word.lower()) for word in words if word.isalpha()])

    # build capitalisation count dictionaries
    # if a word is found capitalised, an entry is icrememented for it
    for sent in sentences:
        for word in word_tokenize(sent):
            if word.isupper() or word.istitle():
                wcf[word.lower()] += 1  # increment capitalisation count
                lcf[len(word)] += 1     # incremenet capitalisation count
            if word.isupper():  # build a dictionary of words found in full caps
                fullcaps[word.lower()] = 1

    # count/total appearances = frequency
    print(wcf.items())
    wcf = {word: (wcf[word] / wordfreq[word]) for word in wordfreq.keys()}
    
    print(wcf.items())
    print(wordlenfreq.items())
    print('blah')
    print(lcf.items())
    
    lcf = {word: (lcf[word]/wordlenfreq[len(word)]) for word in wordfreq.keys()}

    print(lcf.items())
    print(fullcaps)
    
    out = open(out_file, 'wb')
    pickle.dump(wcf, out, 0)
    pickle.dump(lcf, out, 0)
    pickle.dump(fullcaps, out, 0)
    out.close()
    input_f.close()
    return


# Main program: read in the data, process it, and print the result
if __name__ == '__main__':

    # get the filename
    import sys
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    elif len(sys.argv) == 3 and sys.argv[1] == 'build':  # oh boy I hope the input isn't called this! :C
        filename = sys.argv[2]
        print("Building index...")
        build_index(filename, sys.argv[0].split('.')[0] + '.data')
        print("Built index from " + sys.argv[2])
    else:
        exit("Usage: " + sys.argv[0] + " filename")

    # get the data out of the file
    try:
        input = open(filename).read()
    except IOError:
        exit("Cannot open: " + filename)

    # process the data
    result = evaluate(input)
    print(result)
