from nltk.corpus import brown
from nltk.probability import ConditionalFreqDist



genres = ['news', 'romance']
days = set("Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split())


cfd = ConditionalFreqDist(
	(genre,word)
	for genre in genres
	for word in brown.words(categories=genre)
	if word in days
	)

cfd.tabulate()