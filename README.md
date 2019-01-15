# VisualConcept
### blabels
all words freq > 5 except stopword
binary label

### slabels
all words freq > 5except stopword
freq / max freq

### slabels_noun
only nouns and freq > 5
then for tokens in each cap:
    sigmoid(freq)

get all vocab which freq > 5

train_labels.pkl contains word whose freq less than 5
