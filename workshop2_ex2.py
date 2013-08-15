from nltk.corpus import brown
from nltk.probability import FreqDist

text = brown.words(categories='romance')

fd = FreqDist([w for w in text if w[:2] in ["wh","Wh"]])
for word in fd.items():
    print("{0}: {1}".format(word[0],word[1]))
