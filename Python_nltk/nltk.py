
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-Ntural Language Processing NLTK nltk #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
'''
GETTING STARTED WITH NATURAL LANGUAGE PROCESSING
Text Preprocessing
"You never know what you have... until you clean your data."
~ Unknown (or possibly made up)

Cleaning and preparation are crucial for many tasks, and NLP is no exception. Text preprocessing is usually the first step you’ll take when faced with an NLP task.

Without preprocessing, your computer interprets "the", "The", and "<p>The" as entirely different words. There is a LOT you can do here, depending on the formatting you need. Lucky for you, Regex and NLTK will do most of it for you! Common tasks include:

Noise removal — stripping text of formatting (e.g., HTML tags).

Tokenization — breaking text into individual words.

Normalization — cleaning text data in any other way:

Stemming is a blunt axe to chop off word prefixes and suffixes. “booing” and “booed” become “boo”, but “sing” may become “s” and “sung” would remain “sung.”
Lemmatization is a scalpel to bring words down to their root forms. For example, NLTK’s savvy lemmatizer knows “am” and “are” are related to “be.”
Other common tasks include lowercasing, stopwords removal, spelling correction, etc.'''

'''
1.
We used NLTK’s PorterStemmer to normalize the text — run the code to see how it does. (It may take a few seconds for the code to run.)

Checkpoint 2 Passed
2.
In the output terminal you’ll see our program counts "go" and "went" as different words! Also, what’s up with "mani" and "hardli"? A lemmatizer will fix this. Let’s do it.

Where lemmatizer is defined, replace None with WordNetLemmatizer().

Where we defined lemmatized, replace the empty list with a list comprehension that uses lemmatizer to lemmatize() each token in tokenized.

(Don’t know Python that well? No problem. Just check the hints for help throughout the lesson.)

Checkpoint 3 Passed

Hint
lemmatized = [lemmatizer.lemmatize(token) for token in tokenized]
3.
Why are the lemmatized verbs like "went" still conjugated? By default lemmatize() treats every word as a noun.

Give lemmatize() a second argument: get_part_of_speech(token). This will tell our lemmatizer what part of speech the word is.

Run your code again to see the result!

Checkpoint 4 Passed

Hint
lemmatized = [lemmatizer.lemmatize(token, get_part_of_speech(token)) for token in tokenized]'''


# regex for removing punctuation!
import re
# nltk preprocessing magic
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# grabbing a part of speech function:
from part_of_speech import get_part_of_speech

text = "So many squids are jumping out of suitcases these days that you can barely go anywhere without seeing one burst forth from a tightly packed valise. I went to the dentist the other day, and sure enough I saw an angry one jump out of my dentist's bag within minutes of arriving. She hardly even noticed."

cleaned = re.sub('\W+', ' ', text)
tokenized = word_tokenize(cleaned)

stemmer = PorterStemmer()
stemmed = [stemmer.stem(token) for token in tokenized]

## -- CHANGE these -- ##
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(token, get_part_of_speech(token)) for token in tokenized]

print("Stemmed text:")
print(stemmed)
print("\nLemmatized text:")
print(lemmatized)



#-- result ----------------

Stemmed text:
['So', 'mani', 'squid', 'are', 'jump', 'out', 'of', 'suitcas', 'these', 'day', 'that', 'you', 'can', 'bare', 'go', 'anywher', 'without', 'see', 'one', 'burst', 'forth', 'from', 'a', 'tightli', 'pack', 'valis', 'I', 'went', 'to', 'the', 'dentist', 'the', 'other', 'day', 'and', 'sure', 'enough', 'I', 'saw', 'an', 'angri', 'one', 'jump', 'out', 'of', 'my', 'dentist', 's', 'bag', 'within', 'minut', 'of', 'arriv', 'she', 'hardli', 'even', 'notic']

Lemmatized text:
['So', 'many', 'squid', 'be', 'jump', 'out', 'of', 'suitcase', 'these', 'day', 'that', 'you', 'can', 'barely', 'go', 'anywhere', 'without', 'see', 'one', 'burst', 'forth', 'from', 'a', 'tightly', 'pack', 'valise', 'I', 'go', 'to', 'the', 'dentist', 'the', 'other', 'day', 'and', 'sure', 'enough', 'I', 'saw', 'an', 'angry', 'one', 'jump', 'out', 'of', 'my', 'dentist', 's', 'bag', 'within', 'minute', 'of', 'arrive', 'She', 'hardly', 'even', 'notice']




'''

GETTING STARTED WITH NATURAL LANGUAGE PROCESSING
Parsing Text
You now have a preprocessed, clean list of words. Now what? It may be helpful to know how the words relate to each other and the underlying syntax (grammar). Parsing is a stage of NLP concerned with segmenting text based on syntax.

You probably do not want to be doing any parsing by hand and NLTK has a few tricks up its sleeve to help you out:

Part-of-speech tagging (POS tagging) identifies parts of speech (verbs, nouns, adjectives, etc.). NLTK can do it faster (and maybe more accurately) than your grammar teacher.

Named entity recognition (NER) helps identify the proper nouns (e.g., “Natalia” or “Berlin”) in a text. This can be a clue as to the topic of the text and NLTK captures many for you.

Dependency grammar trees help you understand the relationship between the words in a sentence. It can be a tedious task for a human, so the Python library spaCy is at your service, even if it isn’t always perfect.

In English we leave a lot of ambiguity, so syntax can be tough, even for a computer program. Take a look at the following sentence:

I saw a cow under a tree with binoculars.
Do I have the binoculars? Does the cow have binoculars? Does the tree have binoculars?

Regex parsing, using Python’s re library, allows for a bit more nuance. When coupled with POS tagging, you can identify specific phrase chunks. On its own, it can find you addresses, emails, and many other common patterns within large chunks of text.

Instructions
1.
Run the code to see the silly squid sentences parsed into dependency trees visually!

Checkpoint 2 Passed
2.
Change my_sentence to a sentence of your choosing and run the code again to see it parsed out as a tree!'''


#------ script.py --------------

import spacy
from nltk import Tree
from squids import squids_text

dependency_parser = spacy.load('en')

parsed_squids = dependency_parser(squids_text)

# Assign my_sentence a new value:
my_sentence = "Your sentence goes here!"
my_parsed_sentence = dependency_parser(my_sentence)

def to_nltk_tree(node):
  if node.n_lefts + node.n_rights > 0:
    parsed_child_nodes = [to_nltk_tree(child) for child in node.children]
    return Tree(node.orth_, parsed_child_nodes)
  else:
    return node.orth_

for sent in parsed_squids.sents:
  to_nltk_tree(sent.root).pretty_print()
  
for sent in my_parsed_sentence.sents:
 to_nltk_tree(sent.root).pretty_print()

 #-----------squids.py -------------

 squids_text = "So many squids are jumping out of suitcases these days. You can barely go anywhere without seeing one. I went to the dentist the other day. Sure enough, I saw an angry one jump out of my dentist's bag. She hardly even noticed."


#-------result --------------


          went               
  _________|_________         
 |   |     to        |       
 |   |     |         |        
 |   |  dentist     day      
 |   |     |      ___|____    
 I   .    the   the     other

             saw                                     
  ____________|___________________                    
 |   |   |    |                  jump                
 |   |   |    |          _________|__________         
 |   |   |    |         |                   out      
 |   |   |    |         |                    |        
 |   |   |    |         |                    of      
 |   |   |    |         |                    |        
 |   |   |    |         |                   bag      
 |   |   |    |         |                    |        
 |   |   |  enough     one                dentist    
 |   |   |    |      ___|____           _____|_____   
 ,   I   .   Sure   an     angry       my          's

    noticed         
  _____|__________   
She  hardly even  . 

     goes         
  ____|______      
 |    |   sentence
 |    |      |     
here  !     Your  

'''

GETTING STARTED WITH NATURAL LANGUAGE PROCESSING
Language Models - Bag-of-Words Approach
How can we help a machine make sense of a bunch of word tokens? We can help computers make predictions about language by training a language model on a corpus (a bunch of example text).

Language models are probabilistic computer models of language. We build and use these models to figure out the likelihood that a given sound, letter, word, or phrase will be used. Once a model has been trained, it can be tested out on new texts.

One of the most common language models is the unigram model, a statistical language model commonly known as bag-of-words. As its name suggests, bag-of-words does not have much order to its chaos! What it does have is a tally count of each instance for each word. Consider the following text example:

The squids jumped out of the suitcases.
Provided some initial preprocessing, bag-of-words would result in a mapping like:'''

{"the": 2, "squid": 1, "jump": 1, "out": 1, "of": 1, "suitcase": 1}'''
Now look at this sentence and mapping: “Why are your suitcases full of jumping squids?”'''

{"why": 1, "be": 1, "your": 1, "suitcase": 1, "full": 1, "of": 1, "jump": 1, "squid": 1}'''
You can see how even with different word order and sentence structures, “jump,” “squid,” and “suitcase” are shared topics between the two examples. Bag-of-words can be an excellent way of looking at language when you want to make predictions concerning topic or sentiment of a text. When grammar and word order are irrelevant, this is probably a good model to use.

Instructions
1.
We’ve turned a passage from Through the Looking Glass by Lewis Carroll into a list of words (aside from stopwords, which we’ve removed) using nltk preprocessing. Run your code to see the full list.

Checkpoint 2 Passed
2.
Now let’s turn this list into a bag-of-words using Counter()!

Comment out the print statement and set bag_of_looking_glass_words equal to a call of Counter() on normalized. Print bag_of_looking_glass_words. What are the most common words?


Hint
bag_of_looking_glass_words = Counter(normalized)
3.
Try changing text to another string of your choosing and see what happens!

'''

#-------------- script.py -------------------

# importing regex and nltk
import re, nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# importing Counter to get word counts for bag of words
from collections import Counter
# importing a passage from Through the Looking Glass
from looking_glass import looking_glass_text
# importing part-of-speech function for lemmatization
from part_of_speech import get_part_of_speech

# Change text to another string:
text = looking_glass_text

cleaned = re.sub('\W+', ' ', text).lower()
tokenized = word_tokenize(cleaned)

stop_words = stopwords.words('english')
filtered = [word for word in tokenized if word not in stop_words]

normalizer = WordNetLemmatizer()
normalized = [normalizer.lemmatize(token, get_part_of_speech(token)) for token in filtered]
# Comment out the print statement below
#print(normalized)

# Define bag_of_looking_glass_words & print:
bag_of_looking_glass_words = Counter(normalized)
print(bag_of_looking_glass_words)

#--------------------- looking_glass.py
looking_glass_text = """
 However, the egg only got larger and larger, and more and more human: when she had come within a few yards of it, she saw that it had eyes and a nose and mouth; and when she had come close to it, she saw clearly that it was HUMPTY DUMPTY himself. It cant be anybody else! she said to herself. Im as certain of it, as if his name were written all over his face.

It might have been written a hundred times, easily, on that enormous face. Humpty Dumpty was sitting with his legs crossed, like a Turk, on the top of a high wallsuch a narrow one that Alice quite wondered how he could keep his balanceand, as his eyes were steadily fixed in the opposite direction, and he didnt take the least notice of her, she thought he must be a stuffed figure after all.

And how exactly like an egg he is! she said aloud, standing with her hands ready to catch him, for she was every moment expecting him to fall.

Its very provoking, Humpty Dumpty said after a long silence, looking away from Alice as he spoke, to be called an eggVery!

I said you looked like an egg, Sir, Alice gently explained. And some eggs are very pretty, you know she added, hoping to turn her remark into a sort of a compliment.

Some people, said Humpty Dumpty, looking away from her as usual, have no more sense than a baby!

Alice didnt know what to say to this: it wasnt at all like conversation, she thought, as he never said anything to her; in fact, his last remark was evidently addressed to a treeso she stood and softly repeated to herself:

     Humpty Dumpty sat on a wall:
     Humpty Dumpty had a great fall.
     All the Kings horses and all the Kings men
     Couldnt put Humpty Dumpty in his place again.

That last line is much too long for the poetry, she added, almost out loud, forgetting that Humpty Dumpty would hear her.

Dont stand there chattering to yourself like that, Humpty Dumpty said, looking at her for the first time, but tell me your name and your business.

My name is Alice, but

Its a stupid enough name! Humpty Dumpty interrupted impatiently. What does it mean?

Must a name mean something? Alice asked doubtfully.

Of course it must, Humpty Dumpty said with a short laugh: my name means the shape I amand a good handsome shape it is, too. With a name like yours, you might be any shape, almost.

Why do you sit out here all alone? said Alice, not wishing to begin an argument.

Why, because theres nobody with me! cried Humpty Dumpty. Did you think I didnt know the answer to that? Ask another.

Dont you think youd be safer down on the ground? Alice went on, not with any idea of making another riddle, but simply in her good-natured anxiety for the queer creature. That wall is so very narrow!

What tremendously easy riddles you ask! Humpty Dumpty growled out. Of course I dont think so! Why, if ever I did fall offwhich theres no chance ofbut if I did Here he pursed his lips and looked so solemn and grand that Alice could hardly help laughing. If I did fall, he went on, The King has promised mewith his very own mouthtoto

To send all his horses and all his men, Alice interrupted, rather unwisely.

Now I declare thats too bad! Humpty Dumpty cried, breaking into a sudden passion. Youve been listening at doorsand behind treesand down chimneysor you couldnt have known it!

I havent, indeed! Alice said very gently. Its in a book.

Ah, well! They may write such things in a book, Humpty Dumpty said in a calmer tone. Thats what you call a History of England, that is. Now, take a good look at me! Im one that has spoken to a King, I am: mayhap youll never see such another: and to show you Im not proud, you may shake hands with me! And he grinned almost from ear to ear, as he leant forwards (and as nearly as possible fell off the wall in doing so) and offered Alice his hand. She watched him a little anxiously as she took it. If he smiled much more, the ends of his mouth might meet behind, she thought: and then I dont know what would happen to his head! Im afraid it would come off!

Yes, all his horses and all his men, Humpty Dumpty went on. Theyd pick me up again in a minute, they would! However, this conversation is going on a little too fast: lets go back to the last remark but one.

Im afraid I cant quite remember it, Alice said very politely.

In that case we start fresh, said Humpty Dumpty, and its my turn to choose a subject (He talks about it just as if it was a game! thought Alice.) So heres a question for you. How old did you say you were?

Alice made a short calculation, and said Seven years and six months.

Wrong! Humpty Dumpty exclaimed triumphantly. You never said a word like it!

I though you meant How old are you? Alice explained.

If Id meant that, Id have said it, said Humpty Dumpty. 
"""

#-------------- part_of_speech.py----------
from nltk.corpus import wordnet
from collections import Counter
def get_part_of_speech(word):
  probable_part_of_speech = wordnet.synsets(word)
  pos_counts = Counter()
  pos_counts["n"] = len(  [ item for item in probable_part_of_speech if item.pos()=="n"]  )
  pos_counts["v"] = len(  [ item for item in probable_part_of_speech if item.pos()=="v"]  )
  pos_counts["a"] = len(  [ item for item in probable_part_of_speech if item.pos()=="a"]  )
  pos_counts["r"] = len(  [ item for item in probable_part_of_speech if item.pos()=="r"]  )
  
  most_likely_part_of_speech = pos_counts.most_common(1)[0][0]
  return most_likely_part_of_speech

#------------ result -------------

 Counter({'humpty': 19, 'dumpty': 19, 'say': 19, 'alice': 16, 'name': 7, 'like': 7, 'think': 7, 'look': 6, 'im': 5, 'know': 5, 'mean': 5, 'go': 5, 'egg': 4, 'fall': 4, 'king': 4, 'would': 4, 'dont': 4, 'come': 3, 'write': 3, 'might': 3, 'sit': 3, 'one': 3, 'didnt': 3, 'take': 3, 'must': 3, 'stand': 3, 'hand': 3, 'remark': 3, 'never': 3, 'last': 3, 'wall': 3, 'horse': 3, 'men': 3, 'almost': 3, 'ask': 3, 'shape': 3, 'good': 3, 'another': 3, 'however': 2, 'large': 2, 'saw': 2, 'eye': 2, 'mouth': 2, 'cant': 2, 'face': 2, 'time': 2, 'narrow': 2, 'quite': 2, 'could': 2, 'long': 2, 'away': 2, 'speak': 2, 'call': 2, 'gently': 2, 'explain': 2, 'add': 2, 'turn': 2, 'conversation': 2, 'couldnt': 2, 'much': 2, 'interrupt': 2, 'course': 2, 'short': 2, 'laugh': 2, 'there': 2, 'cry': 2, 'make': 2, 'riddle': 2, 'thats': 2, 'behind': 2, 'book': 2, 'may': 2, 'ear': 2, 'little': 2, 'afraid': 2, 'old': 2, 'id': 2, 'get': 1, 'human': 1, 'within': 1, 'yard': 1, 'nose': 1, 'close': 1, 'clearly': 1, 'anybody': 1, 'else': 1, 'certain': 1, 'hundred': 1, 'easily': 1, 'enormous': 1, 'leg': 1, 'cross': 1, 'turk': 1, 'top': 1, 'high': 1, 'wallsuch': 1, 'wonder': 1, 'keep': 1, 'balanceand': 1, 'steadily': 1, 'fix': 1, 'opposite': 1, 'direction': 1, 'least': 1, 'notice': 1, 'stuff': 1, 'figure': 1, 'exactly': 1, 'aloud': 1, 'ready': 1, 'catch': 1, 'every': 1, 'moment': 1, 'expect': 1, 'provoke': 1, 'silence': 1, 'eggvery': 1, 'sir': 1, 'pretty': 1, 'hop': 1, 'sort': 1, 'compliment': 1, 'people': 1, 'usual': 1, 'sense': 1, 'baby': 1, 'wasnt': 1, 'anything': 1, 'fact': 1, 'evidently': 1, 'address': 1, 'treeso': 1, 'softly': 1, 'repeat': 1, 'great': 1, 'put': 1, 'place': 1, 'line': 1, 'poetry': 1, 'loud': 1, 'forget': 1, 'hear': 1, 'chatter': 1, 'first': 1, 'tell': 1, 'business': 1, 'stupid': 1, 'enough': 1, 'impatiently': 1, 'something': 1, 'doubtfully': 1, 'amand': 1, 'handsome': 1, 'alone': 1, 'wish': 1, 'begin': 1, 'argument': 1, 'nobody': 1, 'answer': 1, 'youd': 1, 'safe': 1, 'grind': 1, 'idea': 1, 'simply': 1, 'natured': 1, 'anxiety': 1, 'queer': 1, 'creature': 1, 'tremendously': 1, 'easy': 1, 'growl': 1, 'ever': 1, 'offwhich': 1, 'chance': 1, 'ofbut': 1, 'purse': 1, 'lip': 1, 'solemn': 1, 'grand': 1, 'hardly': 1, 'help': 1, 'promise': 1, 'mewith': 1, 'mouthtoto': 1, 'send': 1, 'rather': 1, 'unwisely': 1, 'declare': 1, 'bad': 1, 'break': 1, 'sudden': 1, 'passion': 1, 'youve': 1, 'listen': 1, 'doorsand': 1, 'treesand': 1, 'chimneysor': 1, 'havent': 1, 'indeed': 1, 'ah': 1, 'well': 1, 'thing': 1, 'calm': 1, 'tone': 1, 'history': 1, 'england': 1, 'mayhap': 1, 'youll': 1, 'see': 1, 'show': 1, 'proud': 1, 'shake': 1, 'grin': 1, 'lean': 1, 'forward': 1, 'nearly': 1, 'possible': 1, 'fell': 1, 'offer': 1, 'watch': 1, 'anxiously': 1, 'smile': 1, 'end': 1, 'meet': 1, 'happen': 1, 'head': 1, 'yes': 1, 'theyd': 1, 'pick': 1, 'minute': 1, 'fast': 1, 'let': 1, 'back': 1, 'remember': 1, 'politely': 1, 'case': 1, 'start': 1, 'fresh': 1, 'choose': 1, 'subject': 1, 'talk': 1, 'game': 1, 'here': 1, 'question': 1, 'calculation': 1, 'seven': 1, 'year': 1, 'six': 1, 'month': 1, 'wrong': 1, 'exclaim': 1, 'triumphantly': 1, 'word': 1, 'though': 1})

'''
GETTING STARTED WITH NATURAL LANGUAGE PROCESSING
Language Models - N-Grams and NLM
For parsing entire phrases or conducting language prediction, you will want to use a model that pays attention to each word’s neighbors. Unlike bag-of-words, the n-gram model considers a sequence of some number (n) units and calculates the probability of each unit in a body of language given the preceding sequence of length n. Because of this, n-gram probabilities with larger n values can be impressive at language prediction.

Take a look at our revised squid example: “The squids jumped out of the suitcases. The squids were furious.”

A bigram model (where n is 2) might give us the following count frequencies:

{('', 'the'): 2, ('the', 'squids'): 2, ('squids', 'jumped'): 1, ('jumped', 'out'): 1, ('out', 'of'): 1, ('of', 'the'): 1, ('the', 'suitcases'): 1, ('suitcases', ''): 1, ('squids', 'were'): 1, ('were', 'furious'): 1, ('furious', ''): 1}
There are a couple problems with the n gram model:

How can your language model make sense of the sentence “The cat fell asleep in the mailbox” if it’s never seen the word “mailbox” before? During training, your model will probably come across test words that it has never encountered before (this issue also pertains to bag of words). A tactic known as language smoothing can help adjust probabilities for unknown words, but it isn’t always ideal.

For a model that more accurately predicts human language patterns, you want n (your sequence length) to be as large as possible. That way, you will have more natural sounding language, right? Well, as the sequence length grows, the number of examples of each sequence within your training corpus shrinks. With too few examples, you won’t have enough data to make many predictions.

Enter neural language models (NLM)! Much recent work within NLP has involved developing and training neural networks to approximate the approach our human brains take towards language. This deep learning approach allows computers a much more adaptive tack to processing human language.

Instructions
1.
If you run the code, you’ll see the 10 most commonly used words in Through the Looking Glass parsed with NLTK’s ngrams module — if you’re thinking this looks like a bag of words, that’s because it is one!

2.
What do you think the most common phrases in the text are? Let’s find out…

Where looking_glass_bigrams is defined, change the second argument to 2 to see bigrams. Change n to 3 for looking_glass_trigrams to see trigrams.


Hint
The ngrams() function takes two arguments: the text you want to use and the n value.

3.
Change n to a number greater than 3 for looking_glass_ngrams. Try increasing the number.

At what n are you just getting lines from poems repeated in the text? This is where there may be too few examples of each sequence within your training corpus to make any helpful predictions.'''

#--------------script.py

import nltk, re
from nltk.tokenize import word_tokenize
# importing ngrams module from nltk
from nltk.util import ngrams
from collections import Counter
from looking_glass import looking_glass_full_text

cleaned = re.sub('\W+', ' ', looking_glass_full_text).lower()
tokenized = word_tokenize(cleaned)

# Change the n value to 2:
looking_glass_bigrams = ngrams(tokenized, 2)
looking_glass_bigrams_frequency = Counter(looking_glass_bigrams)

# Change the n value to 3:
looking_glass_trigrams = ngrams(tokenized, 3)
looking_glass_trigrams_frequency = Counter(looking_glass_trigrams)

# Change the n value to a number greater than 3:
looking_glass_ngrams = ngrams(tokenized, 5)
looking_glass_ngrams_frequency = Counter(looking_glass_ngrams)

print("Looking Glass Bigrams:")
print(looking_glass_bigrams_frequency.most_common(10))

print("\nLooking Glass Trigrams:")
print(looking_glass_trigrams_frequency.most_common(10))

print("\nLooking Glass n-grams:")
print(looking_glass_ngrams_frequency.most_common(10))

#--------------- result -------------

Looking Glass Bigrams:
[(('of', 'the'), 101), (('said', 'the'), 98), (('in', 'a'), 97), (('in', 'the'), 90), (('as', 'she'), 82), (('you', 'know'), 72), (('a', 'little'), 68), (('the', 'queen'), 67), (('said', 'alice'), 67), (('to', 'the'), 66)]

Looking Glass Trigrams:
[(('the', 'red', 'queen'), 54), (('the', 'white', 'queen'), 31), (('said', 'in', 'a'), 21), (('she', 'went', 'on'), 18), (('said', 'the', 'red'), 17), (('thought', 'to', 'herself'), 16), (('the', 'queen', 'said'), 16), (('said', 'to', 'herself'), 14), (('said', 'humpty', 'dumpty'), 14), (('the', 'knight', 'said'), 14)]

Looking Glass n-grams:
[(('one', 'and', 'one', 'and', 'one'), 8), (('and', 'one', 'and', 'one', 'and'), 7), (('for', 'a', 'minute', 'or', 'two'), 6), (('the', 'lion', 'and', 'the', 'unicorn'), 6), (('as', 'well', 'as', 'she', 'could'), 5), (('is', 'worth', 'a', 'thousand', 'pounds'), 4), (('the', 'walrus', 'and', 'the', 'carpenter'), 4), (('said', 'to', 'herself', 'as', 'she'), 4), (('twas', 'brillig', 'and', 'the', 'slithy'), 3), (('brillig', 'and', 'the', 'slithy', 'toves'), 3)]


'''
GETTING STARTED WITH NATURAL LANGUAGE PROCESSING
Topic Models
We’ve touched on the idea of finding topics within a body of language. But what if the text is long and the topics aren’t obvious?

Topic modeling is an area of NLP dedicated to uncovering latent, or hidden, topics within a body of language. For example, one Codecademy curriculum developer used topic modeling to discover patterns within Taylor Swift songs related to love and heartbreak over time.

A common technique is to deprioritize the most common words and prioritize less frequently used terms as topics in a process known as term frequency-inverse document frequency (tf-idf). Say what?! This may sound counter-intuitive at first. Why would you want to give more priority to less-used words? Well, when you’re working with a lot of text, it makes a bit of sense if you don’t want your topics filled with words like “the” and “is.” The Python libraries gensim and sklearn have modules to handle tf-idf.

Whether you use your plain bag of words (which will give you term frequency) or run it through tf-idf, the next step in your topic modeling journey is often latent Dirichlet allocation (LDA). LDA is a statistical model that takes your documents and determines which words keep popping up together in the same contexts (i.e., documents). We’ll use sklearn to tackle this for us.

If you have any interest in visualizing your newly minted topics, word2vec is a great technique to have up your sleeve. word2vec can map out your topic model results spatially as vectors so that similarly used words are closer together. In the case of a language sample consisting of “The squids jumped out of the suitcases. The squids were furious. Why are your suitcases full of jumping squids?”, we might see that “suitcase”, “jump”, and “squid” were words used within similar contexts. This word-to-vector mapping is known as a word embedding.

Instructions
1.
Check out how the bag of words model and tf-idf models stack up when faced with a new Sherlock Holmes text!

Run the code as is to see what topics they uncover…

Checkpoint 2 Passed
2.
Tf-idf has some interesting findings, but the regular bag of words is full of words that tell us very little about the topic of the texts!

Let’s fix this. Add some words to stop_list that don’t tell you much about the topic and then run your code again. Do this until you have at least 10 words in stop_list so that the bag of words LDA model has some interesting topics.

Checkpoint 3 Passed

Hint
Some words you may want to add to the stop_list:

"say", "see", "holmes", "shall", "say", 
"man", "upon", "know", "quite", "one", 
"well", "could", "would", "take", "may", 
"think", "come", "go", "little", "must", 
"look"'''

#--------------script.py-----------------


import nltk, re
from sherlock_holmes import bohemia_ch1, bohemia_ch2, bohemia_ch3, boscombe_ch1, boscombe_ch2, boscombe_ch3
from preprocessing import preprocess_text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# preparing the text
corpus = [bohemia_ch1, bohemia_ch2, bohemia_ch3, boscombe_ch1, boscombe_ch2, boscombe_ch3]
preprocessed_corpus = [preprocess_text(chapter) for chapter in corpus]

# Update stop_list:
stop_list = ["say", "see", "holmes", "shall", "say", 
"man", "upon", "know", "quite", "one", 
"well", "could", "would", "take", "may", 
"think", "come", "go", "little", "must", 
"look"]
# filtering topics for stop words
def filter_out_stop_words(corpus):
  no_stops_corpus = []
  for chapter in corpus:
    no_stops_chapter = " ".join([word for word in chapter.split(" ") if word not in stop_list])
    no_stops_corpus.append(no_stops_chapter)
  return no_stops_corpus
filtered_for_stops = filter_out_stop_words(preprocessed_corpus)

# creating the bag of words model
bag_of_words_creator = CountVectorizer()
bag_of_words = bag_of_words_creator.fit_transform(filtered_for_stops)

# creating the tf-idf model
tfidf_creator = TfidfVectorizer(min_df = 0.2)
tfidf = tfidf_creator.fit_transform(preprocessed_corpus)

# creating the bag of words LDA model
lda_bag_of_words_creator = LatentDirichletAllocation(learning_method='online', n_components=10)
lda_bag_of_words = lda_bag_of_words_creator.fit_transform(bag_of_words)

# creating the tf-idf LDA model
lda_tfidf_creator = LatentDirichletAllocation(learning_method='online', n_components=10)
lda_tfidf = lda_tfidf_creator.fit_transform(tfidf)

print("~~~ Topics found by bag of words LDA ~~~")
for topic_id, topic in enumerate(lda_bag_of_words_creator.components_):
  message = "Topic #{}: ".format(topic_id + 1)
  message += " ".join([bag_of_words_creator.get_feature_names()[i] for i in topic.argsort()[:-5 :-1]])
  print(message)

print("\n\n~~~ Topics found by tf-idf LDA ~~~")
for topic_id, topic in enumerate(lda_tfidf_creator.components_):
  message = "Topic #{}: ".format(topic_id + 1)
  message += " ".join([tfidf_creator.get_feature_names()[i] for i in topic.argsort()[:-5 :-1]])
  print(message)


#---------preprocessing.py
import nltk, re
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

stop_words = stopwords.words('english')
normalizer = WordNetLemmatizer()

def get_part_of_speech(word):
  probable_part_of_speech = wordnet.synsets(word)
  pos_counts = Counter()
  pos_counts["n"] = len(  [ item for item in probable_part_of_speech if item.pos()=="n"]  )
  pos_counts["v"] = len(  [ item for item in probable_part_of_speech if item.pos()=="v"]  )
  pos_counts["a"] = len(  [ item for item in probable_part_of_speech if item.pos()=="a"]  )
  pos_counts["r"] = len(  [ item for item in probable_part_of_speech if item.pos()=="r"]  )
  most_likely_part_of_speech = pos_counts.most_common(1)[0][0]
  return most_likely_part_of_speech

def preprocess_text(text):
  cleaned = re.sub(r'\W+', ' ', text).lower()
  tokenized = word_tokenize(cleaned)
  normalized = [normalizer.lemmatize(token, get_part_of_speech(token)) for token in tokenized]
  filtered = [word for word in normalized if word not in stop_words]
  return " ".join(filtered)

#----------result ---------------------
'''
~~~ Topics found by bag of words LDA ~~~
Topic #1: mr mccarthy sherlock find
Topic #2: mccarthy hand find hear
Topic #3: behind remark escape word
Topic #4: lodge woman much worm
Topic #5: mccarthy father case mr
Topic #6: note paper write eye
Topic #7: find mccarthy hand case
Topic #8: hand young find indeed
Topic #9: find mccarthy father room
Topic #10: mccarthy turner young lestrade


~~~ Topics found by tf-idf LDA ~~~
Topic #1: forward paper limb wear
Topic #2: appearance small interest gentleman
Topic #3: say holmes man mccarthy
Topic #4: biography attempt sink little
Topic #5: morning employ slip shall
Topic #6: holmes majesty king photograph
Topic #7: turn wind new old
Topic #8: holmes say upon lead
Topic #9: respectable guinea save anyone
Topic #10: whether lounge towards flush'''

'''
GETTING STARTED WITH NATURAL LANGUAGE PROCESSING
Text Similarity
Most of us have a good autocorrect story. Our phone’s messenger quietly swaps one letter for another as we type and suddenly the meaning of our message has changed (to our horror or pleasure). However, addressing text similarity — including spelling correction — is a major challenge within natural language processing.

Addressing word similarity and misspelling for spellcheck or autocorrect often involves considering the Levenshtein distance or minimal edit distance between two words. The distance is calculated through the minimum number of insertions, deletions, and substitutions that would need to occur for one word to become another. For example, turning “bees” into “beans” would require one substitution (“a” for “e”) and one insertion (“n”), so the Levenshtein distance would be two.

Phonetic similarity is also a major challenge within speech recognition. English-speaking humans can easily tell from context whether someone said “euthanasia” or “youth in Asia,” but it’s a far more challenging task for a machine! More advanced autocorrect and spelling correction technology additionally considers key distance on a keyboard and phonetic similarity (how much two words or phrases sound the same).

It’s also helpful to find out if texts are the same to guard against plagiarism, which we can identify through lexical similarity (the degree to which texts use the same vocabulary and phrases). Meanwhile, semantic similarity (the degree to which documents contain similar meaning or topics) is useful when you want to find (or recommend) an article or book similar to one you recently finished.

Instructions
1.
Assign the variable three_away_from_code a word with a Levenshtein distance of 3 from “code”. Assign two_away_from_chunk a word with a Levenshtein distance of 2 from “chunk”.'''

#-------------script.py
import nltk
# NLTK has a built-in function
# to check Levenshtein distance:
from nltk.metrics import edit_distance

def print_levenshtein(string1, string2):
  print("The Levenshtein distance from '{0}' to '{1}' is {2}!".format(string1, string2, edit_distance(string1, string2)))

# Check the distance between
# any two words here!
print_levenshtein("fart", "target")

# Assign passing strings here:
three_away_from_code = "kudi"

two_away_from_chunk = "chnuk"

print_levenshtein("code", three_away_from_code)
print_levenshtein("chunk", two_away_from_chunk)


#---------result ------------

The Levenshtein distance from 'fart' to 'target' is 3!
The Levenshtein distance from 'code' to 'kudi' is 3!
The Levenshtein distance from 'chunk' to 'chnuk' is 2!


'''
GETTING STARTED WITH NATURAL LANGUAGE PROCESSING
Language Prediction & Text Generation
How does your favorite search engine complete your search queries? How does your phone’s keyboard know what you want to type next? Language prediction is an application of NLP concerned with predicting text given preceding text. Autosuggest, autocomplete, and suggested replies are common forms of language prediction.

Your first step to language prediction is picking a language model. Bag of words alone is generally not a great model for language prediction; no matter what the preceding word was, you will just get one of the most commonly used words from your training corpus.

If you go the n-gram route, you will most likely rely on Markov chains to predict the statistical likelihood of each following word (or character) based on the training corpus. Markov chains are memory-less and make statistical predictions based entirely on the current n-gram on hand.

For example, let’s take a sentence beginning, “I ate so many grilled cheese”. Using a trigram model (where n is 3), a Markov chain would predict the following word as “sandwiches” based on the number of times the sequence “grilled cheese sandwiches” has appeared in the training data out of all the times “grilled cheese” has appeared in the training data.

A more advanced approach, using a neural language model, is the Long Short Term Memory (LSTM) model. LSTM uses deep learning with a network of artificial “cells” that manage memory, making them better suited for text prediction than traditional neural networks.

Instructions
1.
Add three short stories by your favorite author or the lyrics to three songs by your favorite artist to document1.py, document2.py, and document3.py. Then run script.py to see a short example of text prediction.

Does it look like something by your favorite author or artist?

If you accidentally close one of the files, just click the file folder in the top left corner of the code editor to find the file and re-open it.'''

#----------- script.py--------------------
import nltk, re, random
from nltk.tokenize import word_tokenize
from collections import defaultdict, deque
from document1 import training_doc1
from document2 import training_doc2
from document3 import training_doc3

class MarkovChain:
  def __init__(self):
    self.lookup_dict = defaultdict(list)
    self._seeded = False
    self.__seed_me()

  def __seed_me(self, rand_seed=None):
    if self._seeded is not True:
      try:
        if rand_seed is not None:
          random.seed(rand_seed)
        else:
          random.seed()
        self._seeded = True
      except NotImplementedError:
        self._seeded = False
    
  def add_document(self, str):
    preprocessed_list = self._preprocess(str)
    pairs = self.__generate_tuple_keys(preprocessed_list)
    for pair in pairs:
      self.lookup_dict[pair[0]].append(pair[1])
  
  def _preprocess(self, str):
    cleaned = re.sub(r'\W+', ' ', str).lower()
    tokenized = word_tokenize(cleaned)
    return tokenized

  def __generate_tuple_keys(self, data):
    if len(data) < 1:
      return

    for i in range(len(data) - 1):
      yield [ data[i], data[i + 1] ]
      
  def generate_text(self, max_length=50):
    context = deque()
    output = []
    if len(self.lookup_dict) > 0:
      self.__seed_me(rand_seed=len(self.lookup_dict))
      chain_head = [list(self.lookup_dict)[0]]
      context.extend(chain_head)
      
      while len(output) < (max_length - 1):
        next_choices = self.lookup_dict[context[-1]]
        if len(next_choices) > 0:
          next_word = random.choice(next_choices)
          context.append(next_word)
          output.append(context.popleft())
        else:
          break
      output.extend(list(context))
    return " ".join(output)

my_markov = MarkovChain()
my_markov.add_document(training_doc1)
my_markov.add_document(training_doc2)
my_markov.add_document(training_doc3)
generated_text = my_markov.generate_text()
print(generated_text)


#------------- document1.py ------------------

training_doc1 = """


"""

#------------- document2.py ------------------

training_doc1 = """


"""

#------------- document3.py ------------------

training_doc1 = """


"""

#------------ result --------------

'''
GETTING STARTED WITH NATURAL LANGUAGE PROCESSING
Advanced NLP Topics
Believe it or not, you’ve just scratched the surface of natural language processing. There are a slew of advanced topics and applications of NLP, many of which rely on deep learning and neural networks.

Naive Bayes classifiers are supervised machine learning algorithms that leverage a probabilistic theorem to make predictions and classifications. They are widely used for sentiment analysis (determining whether a given block of language expresses negative or positive feelings) and spam filtering.

We’ve made enormous gains in machine translation, but even the most advanced translation software using neural networks and LSTM still has far to go in accurately translating between languages.

Some of the most life-altering applications of NLP are focused on improving language accessibility for people with disabilities. Text-to-speech functionality and speech recognition have improved rapidly thanks to neural language models, making digital spaces far more accessible places.

NLP can also be used to detect bias in writing and speech. Feel like a political candidate, book, or news source is biased but can’t put your finger on exactly how? Natural language processing can help you identify the language at issue.
'''

'''
.
Assign review a string with a brief review of this lesson so far. Next, run your code. Is the Naive Bayes Classifier accurately classifying your review?'''


#----------------------script.py -----------------------
from reviews import counter, training_counts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Add your review:
review = "this one is too much general"
review_counts = counter.transform([review])

classifier = MultinomialNB()
training_labels = [0] * 1000 + [1] * 1000

classifier.fit(training_counts, training_labels)
  
neg = (classifier.predict_proba(review_counts)[0][0] * 100).round()
pos = (classifier.predict_proba(review_counts)[0][1] * 100).round()

if pos > 50:
  print("Thank you for your positive review!")
elif neg > 50:
  print("We're sorry this hasn't been the best possible lesson for you! We're always looking to improve.")
else:
  print("Naive Bayes cannot determine if this is negative or positive. Thank you or we're sorry?")

  
print("\nAccording to our trained Naive Bayes classifier, the probability that your review was negative was {0}% and the probability it was positive was {1}%.".format(neg, pos))

# -------------reviews.py-------------------
import pickle
counter = pickle.load( open( "count_vect.p", "rb" ) )
training_counts =  pickle.load( open( "train.p", "rb" ) )

# count_vect.p and train.p are too big to import

# result 

We're sorry this hasn't been the best possible lesson for you! We're always looking to improve.

According to our trained Naive Bayes classifier, the probability that your review was negative was 74.0% and the probability it was positive was 26.0%.




'''
GETTING STARTED WITH NATURAL LANGUAGE PROCESSING
Challenges and Considerations
As you’ve seen, there are a vast array of applications for NLP. However, as they say, “with great language processing comes great responsibility” (or something along those lines). When working with NLP, we have several important considerations to take into account:

Different NLP tasks may be more or less difficult in different languages. Because so many NLP tools are built by and for English speakers, these tools may lag behind in processing other languages. The tools may also be programmed with cultural and linguistic biases specific to English speakers.
What if your Amazon Alexa could only understand wealthy men from coastal areas of the United States? English itself is not a homogeneous body. English varies by person, by dialect, and by many sociolinguistic factors. When we build and train NLP tools, are we only building them for one type of English speaker?
You can have the best intentions and still inadvertently program a bigoted tool. While NLP can limit bias, it can also propagate bias. As an NLP developer, it’s important to consider biases, both within your code and within the training corpus. A machine will learn the same biases you teach it, whether intentionally or unintentionally.
As you become someone who builds tools with natural language processing, it’s vital to take into account your users’ privacy. There are many powerful NLP tools that come head-to-head with privacy concerns. Who is collecting your data? How much data is being collected and what do those companies plan to do with your data?
Instructions
1.
Test out different slang on the Naive Bayes Classifier! What happens when you use the word “lit” to mean “wonderful” or “fun”?

Is the sentiment prediction accurate? Test out different slang.'''

# code is save above

'''
GETTING STARTED WITH NATURAL LANGUAGE PROCESSING
NLP Review
Check out how much you’ve learned about natural language processing!

Natural language processing combines computer science, linguistics, and artificial intelligence to enable computers to process human languages.
NLTK is a Python library used for NLP.
Text preprocessing is a stage of NLP focused on cleaning and preparing text for other NLP tasks.
Parsing is a stage of NLP concerned with breaking up text based on syntax.
Language models are probabilistic machine models of language use for NLP comprehension tasks. Common models include bag-of-words, n-gram models, and neural language modeling.
Topic modeling is the NLP process by which hidden topics are identified given a body of text.
Text similarity is a facet of NLP concerned with semblance between instances of language.
Language prediction is an application of NLP concerned with predicting language given preceding language.
There are many social and ethical considerations to take into account when designing NLP tools.''

Instructions
You can build a lot of fun tools with NLP knowledge and a bit of Python. This is just the beginning.

Feel free to test out the plagiarism classifier we built in the code editor (does it work?) or use the space to play around with other NLP code you’ve encountered in this lesson!
'''

#----------- script.py -------------------

import nltk
# Levenshtein distance:
from nltk.metrics import edit_distance

# an arbitrary plagiarism classifier:
def is_plagiarized(text1, text2):
  n = 7
  if edit_distance(text1.lower(), text2.lower()) > ((len(text1) + len(text2)) / n):
    return False
  return True

doc1 = "is this plagiarized"
doc2 = "maybe it's plagiarized"

print(is_plagiarized(doc1, doc2))

'''
NATURAL LANGUAGE PARSING WITH REGULAR EXPRESSIONS
Introduction
Discovering new code words in declassified CIA documents may seem like a mission for a foreign intelligence service, and detecting gender biases in the Harry Potter novels a task for a literature professor. Yet by utilizing natural language parsing with regular expressions, the power to perform such analyses is in your own hands!

While you may not put much explicit thought into the structure of your sentences as you write, the syntax choices you make are critical in ensuring your writing has meaning. Analyzing such sentence structure as well as word choice can not only provide insights into the connotation of a piece text, but can also highlight the biases of its author or uncover additional insights that even a deep, rigorous reading of the text might not reveal.

By using Python’s regular expression modulere and the Natural Language Toolkit, known as NLTK, you can find keywords of interest, discover where and how often they are used, and discern the parts-of-speech patterns in which they appear to understand the sometimes hidden meaning in a piece of writing. Let’s get started!

Instructions
1.
The code in the workspace performs natural language parsing with regular expressions on L. Frank Baum’s classic novel The Wonderful Wizard of Oz! . Run the code to view the output, which gives the frequency of different phrases that appear in the text.

Proceed to the next exercise when you are ready to learn how to perform such parsing yourself!'''

from nltk import RegexpParser
from pos_tagged_oz import pos_tagged_oz
from np_chunk_counter import np_chunk_counter

# define noun-phrase chunk grammar here
chunk_grammar = "NP: {<DT>?<JJ>*<NN>}"

# create RegexpParser object here
chunk_parser = RegexpParser(chunk_grammar)

# create a list to hold noun-phrase chunked sentences
np_chunked_oz = list()

# create a for-loop through each pos-tagged sentence in pos_tagged_oz here
for pos_tagged_sentence in pos_tagged_oz:
  # chunk each sentence and append to np_chunked_oz here
  np_chunked_oz.append(chunk_parser.parse(pos_tagged_sentence))

# store and print the most common np-chunks here
most_common_np_chunks = np_chunk_counter(np_chunked_oz)
print(most_common_np_chunks)

#-----------------result -----------------

[((('i', 'NN'),), 326), ((('dorothy', 'NN'),), 222), ((('the', 'DT'), ('scarecrow', 'NN')), 213), ((('the', 'DT'), ('lion', 'NN')), 148), ((('the', 'DT'), ('tin', 'NN')), 123), ((('woodman', 'NN'),), 112), ((('oz', 'NN'),), 86), ((('toto', 'NN'),), 73), ((('head', 'NN'),), 59), ((('the', 'DT'), ('woodman', 'NN')), 59), ((('the', 'DT'), ('wicked', 'JJ'), ('witch', 'NN')), 58), ((('the', 'DT'), ('emerald', 'JJ'), ('city', 'NN')), 51), ((('the', 'DT'), ('witch', 'NN')), 49), ((('the', 'DT'), ('girl', 'NN')), 46), ((('the', 'DT'), ('road', 'NN')), 41), ((('room', 'NN'),), 29), ((('nothing', 'NN'),), 29), ((('the', 'DT'), ('air', 'NN')), 29), ((('the', 'DT'), ('country', 'NN')), 26), ((('the', 'DT'), ('land', 'NN')), 24), ((('a', 'DT'), ('heart', 'NN')), 24), ((('the', 'DT'), ('west', 'NN')), 23), ((('axe', 'NN'),), 23), ((('the', 'DT'), ('sun', 'NN')), 22), ((('the', 'DT'), ('little', 'JJ'), ('girl', 'NN')), 22), ((('course', 'NN'),), 22), ((('the', 'DT'), ('cowardly', 'JJ'), ('lion', 'NN')), 21), ((('aunt', 'NN'),), 21), ((('the', 'DT'), ('house', 'NN')), 21), ((('the', 'DT'), ('door', 'NN')), 21)]


'''
NATURAL LANGUAGE PARSING WITH REGULAR EXPRESSIONS
Compiling and Matching
Before you dive into more complex syntax parsing, you’ll begin with basic regular expressions in Python using the re module as a regex refresher.

The first method you will explore is .compile(). This method takes a regular expression pattern as an argument and compiles the pattern into a regular expression object, which you can later use to find matching text. The regular expression object below will exactly match 4 upper or lower case characters.
'''
regular_expression_object = re.compile("[A-Za-z]{4}")'''
Regular expression objects have a .match() method that takes a string of text as an argument and looks for a single match to the regular expression that starts at the beginning of the string. To see if your regular expression matches the string "Toto" you can do the following:
'''
result = regular_expression_object.match("Toto")'''
If .match() finds a match that starts at the beginning of the string, it will return a match object. The match object lets you know what piece of text the regular expression matched, and at what index the match begins and ends. If there is no match, .match() will return None.

With the match object stored in result, you can access the matched text by calling result.group(0). If you use a regex containing capture groups, you can access these groups by calling .group() with the appropriately numbered capture group as an argument.

Instead of compiling the regular expression first and then looking for a match in separate lines of code, you can simplify your match to one line:
'''
result = re.match("[A-Za-z]{4}","Toto")'''
With this syntax, re‘s .match() method takes a regular expression pattern as the first argument and a string as the second argument.'''

import re

# characters are defined
character_1 = "Dorothy"
character_2 = "Henry"

# compile your regular expression here
regular_expression = re.compile("[A-Za-z]{7}")

# check for a match to character_1 here
result_1 = regular_expression.match(character_1)

# store and print the matched text here
print(result_1)
match_1 = result_1.group(0)
print(match_1)


# compile a regular expression to match a 7 character string of word characters and check for a match to character_2 here
result_2 = re.match('[A-Za-z]{7}',character_2)

#----------------- result --------------------------------

<_sre.SRE_Match object; span=(0, 7), match='Dorothy'>
Dorothy

'''

NATURAL LANGUAGE PARSING WITH REGULAR EXPRESSIONS
Searching and Finding
You can make your regular expression matches even more dynamic with the help of the .search() method. Unlike .match() which will only find matches at the start of a string, .search() will look left to right through an entire piece of text and return a match object for the first match to the regular expression given. If no match is found, .search() will return None. For example, to search for a sequence of 8 word characters in the string Are you a Munchkin?:
'''
result = re.search("\w{8}","Are you a Munchkin?")'''
Using .search() on the string above will find a match of "Munchkin", while using .match() on the same string would return None!

So far you have used methods that only return one piece of matching text. What if you want to find all the occurrences of a word or keyword in a piece of text to determine a frequency count? Step in the .findall() method!

Given a regular expression as its first argument and a string as its second argument, .findall() will return a list of all non-overlapping matches of the regular expression in the string. Consider the below piece of text:
'''
text = "Everything is green here, while in the country of the Munchkins blue was the favorite color. But the people do not seem to be as friendly as the Munchkins, and I'm afraid we shall be unable to find a place to pass the night."'''
To find all non-overlapping sequences of 8 word characters in the sentence you can do the following:
'''
list_of_matches = re.findall("\w{8}",text)'''
.findall() will thus return the list ['Everythi', 'Munchkin', 'favorite', 'friendly', 'Munchkin'].'''

import re

# import L. Frank Baum's The Wonderful Wizard of Oz
oz_text = open("the_wizard_of_oz_text.txt",encoding='utf-8').read().lower()

# search oz_text for an occurrence of 'wizard' here
found_wizard = re.search('wizard',oz_text)
print(found_wizard)

# find all the occurrences of 'lion' in oz_text here
all_lions = re.findall('lion', oz_text)
print(all_lions)

# store and print the length of all_lions here
number_lions = len(all_lions)
print(number_lions)

#----------- result ------------------------

<_sre.SRE_Match object; span=(14, 20), match='wizard'>
['lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion']
183

'''

NATURAL LANGUAGE PARSING WITH REGULAR EXPRESSIONS
Part-of-Speech Tagging
While it is useful to match and search for patterns of individual characters in a text, you can often find more meaning by analyzing text on a word-by-word basis, focusing on the part of speech of each word in a sentence. This process of identifying and labeling the part of speech of words is known as part-of-speech tagging!

It may have been a while since you’ve been in English class, so let’s review the nine parts of speech with an example:

Wow! Ramona and her class are happily studying the new textbook she has on NLP.

Noun: the name of a person (Ramona,class), place, thing (textbook), or idea (NLP)
Pronoun: a word used in place of a noun (her,she)
Determiner: a word that introduces, or “determines”, a noun (the)
Verb: expresses action (studying) or being (are,has)
Adjective: modifies or describes a noun or pronoun (new)
Adverb: modifies or describes a verb, an adjective, or another adverb (happily)
Preposition: a word placed before a noun or pronoun to form a phrase modifying another word in the sentence (on)
Conjunction: a word that joins words, phrases, or clauses (and)
Interjection: a word used to express emotion (Wow)
You can automate the part-of-speech tagging process with nltk‘s pos_tag() function! The function takes one argument, a list of words in the order they appear in a sentence, and returns a list of tuples, where the first entry in the tuple is a word and the second is the part-of-speech tag.

Given the sentence split into a list of words below:

word_sentence = ['do', 'you', 'suppose', 'oz', 'could', 'give', 'me', 'a', 'heart', '?']
you can tag the parts of speech as follows:

part_of_speech_tagged_sentence = pos_tag(word_sentence)
The call to pos_tag() will return the following:

[('do', 'VB'), ('you', 'PRP'), ('suppose', 'VB'), ('oz', 'NNS'), ('could', 'MD'), ('give', 'VB'), ('me', 'PRP'), ('a', 'DT'), ('heart', 'NN'), ('?', '.')]
Abbreviations are given instead of the full part of speech name. Some common abbreviations include: NN for nouns, VB for verbs, RB for adverbs, JJ for adjectives, and DT for determiners. A complete list of part-of-speech tags and their abbreviations can be found here.
'''
'''
Alphabetical list of part-of-speech tags used in the Penn Treebank Project:
Number
Tag
Description
1.	CC	Coordinating conjunction
2.	CD	Cardinal number
3.	DT	Determiner
4.	EX	Existential there
5.	FW	Foreign word
6.	IN	Preposition or subordinating conjunction
7.	JJ	Adjective
8.	JJR	Adjective, comparative
9.	JJS	Adjective, superlative
10.	LS	List item marker
11.	MD	Modal
12.	NN	Noun, singular or mass
13.	NNS	Noun, plural
14.	NNP	Proper noun, singular
15.	NNPS	Proper noun, plural
16.	PDT	Predeterminer
17.	POS	Possessive ending
18.	PRP	Personal pronoun
19.	PRP$	Possessive pronoun
20.	RB	Adverb
21.	RBR	Adverb, comparative
22.	RBS	Adverb, superlative
23.	RP	Particle
24.	SYM	Symbol
25.	TO	to
26.	UH	Interjection
27.	VB	Verb, base form
28.	VBD	Verb, past tense
29.	VBG	Verb, gerund or present participle
30.	VBN	Verb, past participle
31.	VBP	Verb, non-3rd person singular present
32.	VBZ	Verb, 3rd person singular present
33.	WDT	Wh-determiner
34.	WP	Wh-pronoun
35.	WP$	Possessive wh-pronoun
36.	WRB	Wh-adverb

'''
import nltk
from nltk import pos_tag
from word_tokenized_oz import word_tokenized_oz

# save and print the sentence stored at index 100 in word_tokenized_oz here

witches_fate = word_tokenized_oz[100]
print(witches_fate)

# create a list to hold part-of-speech tagged sentences here
pos_tagged_oz = []


# create a for loop through each word tokenized sentence in word_tokenized_oz here

for word_tokenized_sentence in word_tokenized_oz:
  
  # part-of-speech tag each sentence and append to pos_tagged_oz here
  pos_tagged_oz.append(pos_tag(word_tokenized_sentence))
  

# store and print the 101st part-of-speech tagged sentence here
witches_fate_pos = pos_tagged_oz[100]
print(witches_fate_pos)

#----------------result ------------------------

['``', 'the', 'house', 'must', 'have', 'fallen', 'on', 'her', '.']
[('``', '``'), ('the', 'DT'), ('house', 'NN'), ('must', 'MD'), ('have', 'VB'), ('fallen', 'VBN'), ('on', 'IN'), ('her', 'PRP'), ('.', '.')]

'''
NATURAL LANGUAGE PARSING WITH REGULAR EXPRESSIONS
Introduction to Chunking
You have made it to the juicy stuff! Given your part-of-speech tagged text, you can now use regular expressions to find patterns in sentence structure that give insight into the meaning of a text. This technique of grouping words by their part-of-speech tag is called chunking.

With chunking in nltk, you can define a pattern of parts-of-speech tags using a modified notation of regular expressions. You can then find non-overlapping matches, or chunks of words, in the part-of-speech tagged sentences of a text.

The regular expression you build to find chunks is called chunk grammar. A piece of chunk grammar can be written as follows:

chunk_grammar = "AN: {<JJ><NN>}"
AN is a user-defined name for the kind of chunk you are searching for. You can use whatever name makes sense given your chunk grammar. In this case AN stands for adjective-noun
A pair of curly braces {} surround the actual chunk grammar
<JJ> operates similarly to a regex character class, matching any adjective
<NN> matches any noun, singular or plural
The chunk grammar above will thus match any adjective that is followed by a noun.

To use the chunk grammar defined, you must create a nltk RegexpParser object and give it a piece of chunk grammar as an argument.

chunk_parser = RegexpParser(chunk_grammar)
You can then use the RegexpParser object’s .parse() method, which takes a list of part-of-speech tagged words as an argument, and identifies where such chunks occur in the sentence!

Consider the part-of-speech tagged sentence below:

pos_tagged_sentence = [('where', 'WRB'), ('is', 'VBZ'), ('the', 'DT'), ('emerald', 'JJ'), ('city', 'NN'), ('?', '.')]
You can chunk the sentence to find any adjectives followed by a noun with the following:

chunked = chunk_parser.parse(pos_tagged_sentence)'''


from nltk import RegexpParser, Tree
from pos_tagged_oz import pos_tagged_oz

# define adjective-noun chunk grammar here
chunk_grammar = "AN:{<JJ><NN>}"

# create RegexpParser object here
chunk_parser = RegexpParser(chunk_grammar)

# chunk the pos-tagged sentence at index 282 in pos_tagged_oz here
scaredy_cat = chunk_parser.parse(pos_tagged_oz[282])
print(scaredy_cat)

# pretty_print the chunked sentence here
Tree.fromstring(str(scaredy_cat)).pretty_print()

#--------------- RESULT --------------

(S ``/`` where/WRB is/VBZ the/DT (AN emerald/JJ city/NN) ?/. ''/'')
                         S                                    
   ______________________|__________________________           
  |       |       |      |     |    |               AN        
  |       |       |      |     |    |        _______|_____     
``/`` where/WRB is/VBZ the/DT ?/. ''/'' emerald/JJ     city/NN

'''
NATURAL LANGUAGE PARSING WITH REGULAR EXPRESSIONS
Chunking Noun Phrases
While you are able to chunk any sequence of parts of speech that you like, there are certain types of chunking that are linguistically helpful for determining meaning and bias in a piece of text. One such type of chunking is NP-chunking, or noun phrase chunking. A noun phrase is a phrase that contains a noun and operates, as a unit, as a noun.

A popular form of noun phrase begins with a determiner DT, which specifies the noun being referenced, followed by any number of adjectives JJ, which describe the noun, and ends with a noun NN.

Consider the part-of-speech tagged sentence below:

[('we', 'PRP'), ('are', 'VBP'), ('so', 'RB'), ('grateful', 'JJ'), ('to', 'TO'), ('you', 'PRP'), ('for', 'IN'), ('having', 'VBG'), ('killed', 'VBN'), ('the', 'DT'), ('wicked', 'JJ'), ('witch', 'NN'), ('of', 'IN'), ('the', 'DT'), ('east', 'NN'), (',', ','), ('and', 'CC'), ('for', 'IN'), ('setting', 'VBG'), ('our', 'PRP$'), ('people', 'NNS'), ('free', 'VBP'), ('from', 'IN'), ('bondage', 'NN'), ('.', '.')]
Can you spot the three noun phrases of the form described above? They are:

(('the', 'DT'), ('wicked', 'JJ'), ('witch', 'NN'))
(('the', 'DT'), ('east', 'NN'))
(('bondage', 'NN'))
With the help of a regular expression defined chunk grammar, you can easily find all the non-overlapping noun phrases in a piece of text! Just like in normal regular expressions, you can use quantifiers to indicate how many of each part of speech you want to match.

The chunk grammar for a noun phrase can be written as follows:

chunk_grammar = "NP: {<DT>?<JJ>*<NN>}"
NP is the user-defined name of the chunk you are searching for. In this case NP stands for noun phrase
<DT> matches any determiner
? is an optional quantifier, matching either 0 or 1 determiners
<JJ> matches any adjective
* is the Kleene star quantifier, matching 0 or more occurrences of an adjective
<NN> matches any noun, singular or plural
By finding all the NP-chunks in a text, you can perform a frequency analysis and identify important, recurring noun phrases. You can also use these NP-chunks as pseudo-topics and tag articles and documents by their highest count NP-chunks! Or perhaps your analysis has you looking at the adjective choices an author makes for different nouns.

It is ultimately up to you, with your knowledge of the text you are working with, to interpret the meaning and use-case of the NP-chunks and their frequency of occurrence.

'''

from nltk import RegexpParser
from pos_tagged_oz import pos_tagged_oz
from np_chunk_counter import np_chunk_counter

# define noun-phrase chunk grammar here
chunk_grammar = "NP: {<DT>?<JJ>*<NN>}"

# create RegexpParser object here
chunk_parser = RegexpParser(chunk_grammar)

# create a list to hold noun-phrase chunked sentences
np_chunked_oz = list()

# create a for loop through each pos-tagged sentence in pos_tagged_oz here
for pos_tagged_sentence in pos_tagged_oz:
  # chunk each sentence and append to np_chunked_oz here
  np_chunked_oz.append(chunk_parser.parse(pos_tagged_sentence))

# store and print the most common np-chunks here
most_common_np_chunks = np_chunk_counter(np_chunked_oz)
print(most_common_np_chunks)

#----------------RESULT --------------------------

[((('i', 'NN'),), 326), ((('dorothy', 'NN'),), 222), ((('the', 'DT'), ('scarecrow', 'NN')), 213), ((('the', 'DT'), ('lion', 'NN')), 148), ((('the', 'DT'), ('tin', 'NN')), 123), ((('woodman', 'NN'),), 112), ((('oz', 'NN'),), 86), ((('toto', 'NN'),), 73), ((('head', 'NN'),), 59), ((('the', 'DT'), ('woodman', 'NN')), 59), ((('the', 'DT'), ('wicked', 'JJ'), ('witch', 'NN')), 58), ((('the', 'DT'), ('emerald', 'JJ'), ('city', 'NN')), 51), ((('the', 'DT'), ('witch', 'NN')), 49), ((('the', 'DT'), ('girl', 'NN')), 46), ((('the', 'DT'), ('road', 'NN')), 41), ((('room', 'NN'),), 29), ((('nothing', 'NN'),), 29), ((('the', 'DT'), ('air', 'NN')), 29), ((('the', 'DT'), ('country', 'NN')), 26), ((('the', 'DT'), ('land', 'NN')), 24), ((('a', 'DT'), ('heart', 'NN')), 24), ((('the', 'DT'), ('west', 'NN')), 23), ((('axe', 'NN'),), 23), ((('the', 'DT'), ('sun', 'NN')), 22), ((('the', 'DT'), ('little', 'JJ'), ('girl', 'NN')), 22), ((('course', 'NN'),), 22), ((('the', 'DT'), ('cowardly', 'JJ'), ('lion', 'NN')), 21), ((('aunt', 'NN'),), 21), ((('the', 'DT'), ('house', 'NN')), 21), ((('the', 'DT'), ('door', 'NN')), 21)]


'''
NATURAL LANGUAGE PARSING WITH REGULAR EXPRESSIONS
Chunking Verb Phrases
Another popular type of chunking is VP-chunking, or verb phrase chunking. A verb phrase is a phrase that contains a verb and its complements, objects, or modifiers.

Verb phrases can take a variety of structures, and here you will consider two. The first structure begins with a verb VB of any tense, followed by a noun phrase, and ends with an optional adverb RB of any form. The second structure switches the order of the verb and the noun phrase, but also ends with an optional adverb.

Both structures are considered because verb phrases of each form are essentially the same in meaning. For example, consider the part-of-speech tagged verb phrases given below:

(('said', 'VBD'), ('the', 'DT'), ('cowardly', 'JJ'), ('lion', 'NN'))
('the', 'DT'), ('cowardly', 'JJ'), ('lion', 'NN')), (('said', 'VBD'),
The chunk grammar to find the first form of verb phrase is given below:

chunk_grammar = "VP: {<VB.*><DT>?<JJ>*<NN><RB.?>?}"
VP is the user-defined name of the chunk you are searching for. In this case VP stands for verb phrase
<VB.*> matches any verb using the . as a wildcard and the * quantifier to match 0 or more occurrences of any character. This ensures matching verbs of any tense (ex. VB for present tense, VBD for past tense, or VBN for past participle)
<DT>?<JJ>*<NN> matches any noun phrase
<RB.?> matches any adverb using the . as a wildcard and the optional quantifier to match 0 or 1 occurrence of any character. This ensures matching any form of adverb (regular RB, comparative RBR, or superlative RBS)
? is an optional quantifier, matching either 0 or 1 adverbs
The chunk grammar for the second form of verb phrase is given below:

chunk_grammar = "VP: {<DT>?<JJ>*<NN><VB.*><RB.?>?}"
Just like with NP-chunks, you can find all the VP-chunks in a text and perform a frequency analysis to identify important, recurring verb phrases. These verb phrases can give insight into what kind of action different characters take or how the actions that characters take are described by the author.

Once again, this is the part of the analysis where you get to be creative and use your own knowledge about the text you are working with to find interesting insights!'''


from nltk import RegexpParser
from pos_tagged_oz import pos_tagged_oz
from vp_chunk_counter import vp_chunk_counter

# define verb phrase chunk grammar here
chunk_grammar = "VP: {<VB.*><DT>?<JJ>*<NN><RB.?>?}"
#chunk_grammar = "VP: {<DT>?<JJ>*<NN><VB.*><RB.?>?}"

# create RegexpParser object here
chunk_parser = RegexpParser(chunk_grammar)

# create a list to hold verb-phrase chunked sentences
vp_chunked_oz = list()

# create for loop through each pos-tagged sentence in pos_tagged_oz here
for pos_tagged_sentence in pos_tagged_oz:
  # chunk each sentence and append to vp_chunked_oz here
  vp_chunked_oz.append(chunk_parser.parse(pos_tagged_sentence))
  
# store and print the most common vp-chunks here
most_common_vp_chunks = vp_chunk_counter(vp_chunked_oz)
print(most_common_vp_chunks)

#-------------RESULT------------------------

[((('said', 'VBD'), ('the', 'DT'), ('scarecrow', 'NN')), 33), ((('said', 'VBD'), ('dorothy', 'NN')), 31), ((('asked', 'VBN'), ('dorothy', 'NN')), 20), ((('said', 'VBD'), ('the', 'DT'), ('tin', 'NN')), 19), ((('said', 'VBD'), ('the', 'DT'), ('lion', 'NN')), 15), ((('said', 'VBD'), ('the', 'DT'), ('girl', 'NN')), 10), ((('asked', 'VBN'), ('the', 'DT'), ('scarecrow', 'NN')), 10), ((('answered', 'VBD'), ('the', 'DT'), ('scarecrow', 'NN')), 8), ((('said', 'VBD'), ('the', 'DT'), ('cowardly', 'JJ'), ('lion', 'NN')), 8), ((('said', 'VBD'), ('oz', 'NN')), 8), ((('said', 'VBD'), ('the', 'DT'), ('woodman', 'NN')), 7), ((('pass', 'VB'), ('the', 'DT'), ('night', 'NN')), 6), ((('asked', 'VBN'), ('the', 'DT'), ('girl', 'NN')), 6), ((('see', 'VB'), ('the', 'DT'), ('great', 'JJ'), ('oz', 'NN')), 6), ((('answered', 'VBD'), ('oz', 'NN')), 6), ((('replied', 'VBD'), ('oz', 'NN')), 6), ((('cried', 'VBN'), ('dorothy', 'NN')), 5), ((('asked', 'VBN'), ('the', 'DT'), ('tin', 'NN')), 5), ((('asked', 'VBN'), ('the', 'DT'), ('lion', 'NN')), 5), ((('remarked', 'VBD'), ('the', 'DT'), ('lion', 'NN')), 5), ((('answered', 'VBD'), ('dorothy', 'NN')), 5), ((('replied', 'VBD'), ('the', 'DT'), ('lion', 'NN')), 5), ((('killed', 'VBN'), ('the', 'DT'), ('wicked', 'JJ'), ('witch', 'NN')), 4), ((('said', 'VBD'), ('the', 'DT'), ('witch', 'NN')), 4), ((('replied', 'VBD'), ('the', 'DT'), ('scarecrow', 'NN')), 4), ((('answered', 'VBD'), ('the', 'DT'), ('girl', 'NN')), 4), ((('said', 'VBD'), ('the', 'DT'), ('farmer', 'NN')), 4), ((('thought', 'VBD'), ('i', 'NN')), 4), ((('answered', 'VBD'), ('the', 'DT'), ('woodman', 'NN')), 4), ((('have', 'VBP'), ('no', 'DT'), ('heart', 'NN')), 4)]

'''
NATURAL LANGUAGE PARSING WITH REGULAR EXPRESSIONS
Chunk Filtering
Another option you have to find chunks in your text is chunk filtering. Chunk filtering lets you define what parts of speech you do not want in a chunk and remove them.

A popular method for performing chunk filtering is to chunk an entire sentence together and then indicate which parts of speech are to be filtered out. If the filtered parts of speech are in the middle of a chunk, it will split the chunk into two separate chunks! The chunk grammar you can use to perform chunk filtering is given below:

chunk_grammar = """NP: {<.*>+}
                       }<VB.?|IN>+{"""
NP is the user-defined name of the chunk you are searching for. In this case NP stands for noun phrase
The brackets {} indicate what parts of speech you are chunking. <.*>+ matches every part of speech in the sentence
The inverted brackets }{ indicate which parts of speech you want to filter from the chunk. <VB.?|IN>+ will filter out any verbs or prepositions
Chunk filtering provides an alternate way for you to search through a text and find the chunks of information useful for your analysis!'''

from nltk import RegexpParser, Tree
from pos_tagged_oz import pos_tagged_oz

# define chunk grammar to chunk an entire sentence together
grammar = "Chunk: {<.*>+}"

# create RegexpParser object
parser = RegexpParser(grammar)

# chunk the pos-tagged sentence at index 230 in pos_tagged_oz
chunked_dancers = parser.parse(pos_tagged_oz[230])
print(chunked_dancers)

# define noun phrase chunk grammar using chunk filtering here
chunk_grammar = """NP: {<.*>+}
                       }<VB.?|IN>+{"""

# create RegexpParser object here
chunk_parser = RegexpParser(chunk_grammar)

# chunk and filter the pos-tagged sentence at index 230 in pos_tagged_oz here
filtered_dancers = chunk_parser.parse(pos_tagged_oz[230])
print(filtered_dancers)

# pretty_print the chunked and filtered sentence here
Tree.fromstring(str(filtered_dancers)).pretty_print()

#------------RESULT -------------------------

'''
(S
  (Chunk
    then/RB
    she/PRP
    sat/VBD
    upon/IN
    a/DT
    settee/NN
    and/CC
    watched/VBD
    the/DT
    people/NNS
    dance/NN
    ./.))
(S
  (NP then/RB she/PRP)
  sat/VBD
  upon/IN
  (NP a/DT settee/NN and/CC)
  watched/VBD
  (NP the/DT people/NNS dance/NN ./.))
                                                 S                                                  
    _____________________________________________|_______________________________                    
   |       |         |               NP                  NP                      NP                 
   |       |         |          _____|_____       _______|_______        ________|________________   
sat/VBD upon/IN watched/VBD then/RB     she/PRP a/DT settee/NN and/CC the/DT people/NNS dance/NN ./.'''

'''
NATURAL LANGUAGE PARSING WITH REGULAR EXPRESSIONS
Review
And there you go! Now you have the toolkit to dig into any piece of text data and perform natural language parsing with regular expressions. What insights will you gain, or what bias may you uncover? Let’s review what you have learned:

The re module’s .compile() and .match() methods allow you to enter any regex pattern and look for a single match at the beginning of a piece of text
The re module’s .search() method lets you find a single match to a regex pattern anywhere in a string, while the .findall() method finds all the matches of a regex pattern in a string
Part-of-speech tagging identifies and labels the part of speech of words in a sentence, and can be performed in nltk using the pos_tag() function
Chunking groups together patterns of words by their part-of-speech tag. Chunking can be performed in nltk by defining a piece of chunk grammar using regular expression syntax and calling a RegexpParser‘s .parse() method on a word tokenized sentence
NP-chunking chunks together an optional determiner DT, any number of adjectives JJ, and a noun NN to form a noun phrase. The frequency of different NP-chunks can identify important topics in a text or demonstrate how an author describes different subjects
VP-chunking chunks together a verb VB, a noun phrase, and an optional adverb RB to form a verb phrase. The frequency of different VP-chunks can give insight into what kind of action different subjects take or how the actions that different subjects take are described by an author, potentially indicating bias
Chunk filtering provides an alternative means of chunking by specifying what parts of speech you do not want in a chunk and removing them'''


from nltk import RegexpParser
from pos_tagged_oz import pos_tagged_oz
from chunk_counter import chunk_counter

# define your own chunk grammar here
chunk_grammar = '''Chunk: {}
													}{'''

# create RegexpParser object
chunk_parser = RegexpParser(chunk_grammar)

# create a list to hold chunked sentences
chunked_oz = list()

# create a for loop through each pos-tagged sentence in pos_tagged_oz
for pos_tagged_sentence in pos_tagged_oz:
  # chunk each sentence and append to chunked_oz
  chunked_oz.append(chunk_parser.parse(pos_tagged_sentence))

# store and print the most common chunks
most_common_chunks = chunk_counter(chunked_oz)
print(most_common_chunks)

'''
BAG-OF-WORDS LANGUAGE MODEL
Intro to Bag-of-Words
“A bag-of-words is all you need,” some NLPers have decreed.

The bag-of-words language model is a simple-yet-powerful tool to have up your sleeve when working on natural language processing (NLP). The model has many, many use cases including:

determining topics in a song
filtering spam from your inbox
finding out if a tweet has positive or negative sentiment
creating word clouds'''

#------------- script.py ------------------------

from spam_data import training_spam_docs, training_doc_tokens, training_labels
from sklearn.naive_bayes import MultinomialNB
from preprocessing import preprocess_text

# Add your email text to test_text between the triple quotes:
test_text = """
Dear recipient,
Avangar Technologies announces the beginning of a new unprecendented global employment campaign.
reviser yeller winers butchery twenties
Due to company's exploding growth Avangar is expanding business to the European region.
During last employment campaign over 1500 people worldwide took part in Avangar's business
and more than half of them are currently employed by the company. And now we are offering you
one more opportunity to earn extra money working with Avangar Technologies.
druggists blame classy gentry Aladdin

We are looking for honest, responsible, hard-working people that can dedicate 2-4 hours of their
time per day and earn extra 拢300-500 weekly. All offered positions are currently part-time
and give you a chance to work mainly from home.
lovelies hockey Malton meager reordered

Please visit Avangar's corporate web site (http://www.avangar.com/sta/home/0077.htm) for more details regarding these vacancies.


"""
test_tokens = preprocess_text(test_text)

def create_features_dictionary(document_tokens):
  features_dictionary = {}
  index = 0
  for token in document_tokens:
    if token not in features_dictionary:
      features_dictionary[token] = index
      index += 1
  return features_dictionary

def tokens_to_bow_vector(document_tokens, features_dictionary):
  bow_vector = [0] * len(features_dictionary)
  for token in document_tokens:
    if token in features_dictionary:
      feature_index = features_dictionary[token]
      bow_vector[feature_index] += 1
  return bow_vector

bow_sms_dictionary = create_features_dictionary(training_doc_tokens)
training_vectors = [tokens_to_bow_vector(training_doc, bow_sms_dictionary) for training_doc in training_spam_docs]
test_vectors = [tokens_to_bow_vector(test_tokens, bow_sms_dictionary)]

spam_classifier = MultinomialNB()
spam_classifier.fit(training_vectors, training_labels)

predictions = spam_classifier.predict(test_vectors)

print("Looks like a normal email!" if predictions[0] == 0 else "You've got spam!")

'''
BAG-OF-WORDS LANGUAGE MODEL
Bag-of-What?
Bag-of-words (BoW) is a statistical language model based on word count. Say what?

Let’s start with that first part: a statistical language model is a way for computers to make sense of language based on probability. For example, let’s say we have the text:

“Five fantastic fish flew off to find faraway functions. Maybe find another five fantastic fish?”

A statistical language model focused on the starting letter for words might take this text and predict that words are most likely to start with the letter “f” because 11 out of 15 words begin that way. A different statistical model that pays attention to word order might tell us that the word “fish” tends to follow the word “fantastic.”

Bag-of-words does not give a flying fish about word starts or word order though; its sole concern is word count — how many times each word appears in a document.

If you’re already familiar with statistical language models, you may also have heard BoW referred to as the unigram model. It’s technically a special case of another statistical model, the n-gram model, with n (the number of words in a sequence) set to 1.

If you have no idea what n-grams are, don’t worry — we’ll dive deeper into them in another lesson.'''

'''
BAG-OF-WORDS LANGUAGE MODEL
BoW Dictionaries
One of the most common ways to implement the BoW model in Python is as a dictionary with each key set to a word and each value set to the number of times that word appears. Take the example below:

The squids jumped out of the suitcases.
The words from the sentence go into the bag-of-words and come out as a dictionary of words with their corresponding counts. For statistical models, we call the text that we use to build the model our training data. Usually, we need to prepare our text data by breaking it up into documents (shorter strings of text, generally sentences).

Let’s build a function that converts a given training text into a bag-of-words!'''

from preprocessing import preprocess_text
# Define text_to_bow() below:
def text_to_bow(some_text):
  bow_dictionary = {}
  tokens = preprocess_text(some_text)
  for token in tokens:
    if token in bow_dictionary:
      bow_dictionary[token] += 1
    else:
      bow_dictionary[token] = 1
  return bow_dictionary

print(text_to_bow("I love fantastic flying fish. These flying fish are just ok, so maybe I will find another few fantastic fish..."))

#resut 
{'i': 2, 'love': 1, 'fantastic': 2, 'fly': 2, 'fish': 3, 'these': 1, 'be': 1, 'just': 1, 'ok': 1, 'so': 1, 'maybe': 1, 'will': 1, 'find': 1, 'another': 1, 'few': 1}


'''
BAG-OF-WORDS LANGUAGE MODEL
Introducing BoW Vectors
Sometimes a dictionary just won’t fit the bill. Topic modelling applications, for example, require an implementation of bag-of-words that is a bit more mathematical: feature vectors.

A feature vector is a numeric representation of an item’s important features. Each feature has its own column. If the feature exists for the item, you could represent that with a 1. If the feature does not exist for that item, you could represent that with a 0. A few monsters could be represented as vectors like so:

has_fangs	melts_in_water	hates_sunlight	has_fur
vampire	1	0	1	0
werewolf	1	0	0	1
witch	0	1	0	0

For bag-of-words, instead of monsters you would have documents and the features would be different words. And we don’t just care if a word is present in a document; we want to know how many times it occurred! Turning text into a BoW vector is known as feature extraction or vectorization.

But how do we know which vector index corresponds to which word? When building BoW vectors, we generally create a features dictionary of all vocabulary in our training data (usually several documents) mapped to indices.

For example, with “Five fantastic fish flew off to find faraway functions. Maybe find another five fantastic fish?” our dictionary might be:

{'five': 0,
'fantastic': 1,
'fish': 2,
'fly': 3,
'off': 4,
'to': 5,
'find': 6,
'faraway': 7,
'function': 8,
'maybe': 9,
'another': 10}
Using this dictionary, we can convert new documents into vectors using a vectorization function. For example, we can take a brand new sentence “Another five fish find another faraway fish.” — test data — and convert it to a vector that looks like:

[1, 0, 2, 0, 0, 0, 1, 1, 0, 0, 2]
The word ‘another’ appeared twice in the test data. If we look at the feature dictionary for ‘another’, we find that its index is 10. So when we go back and look at our vector, we’d expect the number at index 10 to be 2.'''

'''BAG-OF-WORDS LANGUAGE MODEL
Building a Features Dictionary
Now that you know what a bag-of-words vector looks like, you can create a function that builds them!

First, we need a way of generating a features dictionary from a list of training documents. We can build a Python function to do that for us…
'''

from preprocessing import preprocess_text
# Define create_features_dictionary() below:
def create_features_dictionary(documents):
  features_dictionary = {}
  merged = " ".join(documents)
  tokens = preprocess_text(merged)
  index = 0
  for token in tokens:
    if token not in features_dictionary:
      features_dictionary[token] = index
      index += 1
  return features_dictionary, tokens

training_documents = ["Five fantastic fish flew off to find faraway functions.", "Maybe find another five fantastic fish?", "Find my fish with a function please!"]

print(create_features_dictionary(training_documents)[0])

#-----------result------------
{'five': 0, 'fantastic': 1, 'fish': 2, 'fly': 3, 'off': 4, 'to': 5, 'find': 6, 'faraway': 7, 'function': 8, 'maybe': 9, 'another': 10, 'my': 11, 'with': 12, 'a': 13, 'please': 14}


'''
BAG-OF-WORDS LANGUAGE MODEL
Building a BoW Vector
Nice work! Time to put that dictionary of vocabulary to good use and build a bag-of-words vector from a new document.

In Python, we can use a list to represent a vector. Each index in the list will correspond to a word and be set to its count.

'''
from preprocessing import preprocess_text
# Define text_to_bow_vector() below:
def text_to_bow_vector(some_text, features_dictionary):
  bow_vector = [0] * len(features_dictionary)
  tokens = preprocess_text(some_text)
  for token in tokens:
    feature_index = features_dictionary[token]
    bow_vector[feature_index] += 1
  return bow_vector, tokens

features_dictionary = {'function': 8, 'please': 14, 'find': 6, 'five': 0, 'with': 12, 'fantastic': 1, 'my': 11, 'another': 10, 'a': 13, 'maybe': 9, 'to': 5, 'off': 4, 'faraway': 7, 'fish': 2, 'fly': 3}

text = "Another five fish find another faraway fish."
print(text_to_bow_vector(text, features_dictionary)[0])

#----------- result --------------------------

[1, 0, 2, 0, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0, 0]

'''

It's All in the Bag
Phew! That was a lot of work.

It’s time to put create_features_dictionary() and tokens_to_bow_vector() together and use them in a spam filter we created that uses a Naive Bayes classifier. We’ve slightly modified the two functions for this use case, but they should still look familiar.

Let’s see create_features_dictionary() and tokens_to_bow_vector() in action with real test data, helping fend off spam!'''

from spam_data import training_spam_docs, training_doc_tokens, training_labels, test_labels, test_spam_docs, training_docs, test_docs
from sklearn.naive_bayes import MultinomialNB

def create_features_dictionary(document_tokens):
  features_dictionary = {}
  index = 0
  for token in document_tokens:
    if token not in features_dictionary:
      features_dictionary[token] = index
      index += 1
  return features_dictionary

def tokens_to_bow_vector(document_tokens, features_dictionary):
  bow_vector = [0] * len(features_dictionary)
  for token in document_tokens:
    if token in features_dictionary:
      feature_index = features_dictionary[token]
      bow_vector[feature_index] += 1
  return bow_vector

# Define bow_sms_dictionary:
bow_sms_dictionary = create_features_dictionary(training_doc_tokens)

# Define training_vectors:
training_vectors = [tokens_to_bow_vector(training_doc, bow_sms_dictionary) for training_doc in training_spam_docs]

# Define test_vectors:
test_vectors = [tokens_to_bow_vector(test_doc, bow_sms_dictionary) for test_doc in test_spam_docs]


spam_classifier = MultinomialNB()

def spam_or_not(label):
  return "spam" if label else "not spam"

# Uncomment the code below when you're done:
spam_classifier.fit(training_vectors, training_labels)

predictions = spam_classifier.score(test_vectors, test_labels)

print("The predictions for the test data were {0}% accurate.\n\nFor example, '{1}' was classified as {2}.\n\nMeanwhile, '{3}' was classified as {4}.".format(predictions * 100, test_docs[0], spam_or_not(test_labels[0]), test_docs[10], spam_or_not(test_labels[10])))

#------------result-------------
The predictions for the test data were 99.0% accurate.

For example, 'well obviously not because all the people in my cool college life go home _' was classified as not spam.

Meanwhile, 'urgent we be try to contact you last weekend draw show u have win a 1000 prize guarantee call 09064017295 claim code k52 valid 12hrs 150p pm' was classified as spam.


'''BAG-OF-WORDS LANGUAGE MODEL
Spam A Lot No More
Amazing work! As is the case with many tasks in Python, there’s already a library that can do all of that work for you.

For text_to_bow(), you can approximate the functionality with the collections module’s Counter() function:
'''
from collections import Counter

tokens = ['another', 'five', 'fish', 'find', 'another', 'faraway', 'fish']
print(Counter(tokens))

# Counter({'fish': 2, 'another': 2, 'find': 1, 'five': 1, 'faraway': 1})
'''
For vectorization, you can use CountVectorizer from the machine learning library scikit-learn. You can use fit() to train the features dictionary and then transform() to transform text into a vector:
'''
from sklearn.feature_extraction.text import CountVectorizer

training_documents = ["Five fantastic fish flew off to find faraway functions.", "Maybe find another five fantastic fish?", "Find my fish with a function please!"]
test_text = ["Another five fish find another faraway fish."]
bow_vectorizer = CountVectorizer()
bow_vectorizer.fit(training_documents)
bow_vector = bow_vectorizer.transform(test_text)
print(bow_vector.toarray())
# [[2 0 1 1 2 1 0 0 0 0 0 0 0 0 0]]'''


from spam_data import training_spam_docs, training_doc_tokens, training_labels, test_labels, test_spam_docs, training_docs, test_docs
from sklearn.naive_bayes import MultinomialNB
# Import CountVectorizer from sklearn:
from sklearn.feature_extraction.text import CountVectorizer

# Define bow_vectorizer:
bow_vectorizer = CountVectorizer()

# Define training_vectors:
training_vectors = bow_vectorizer.fit_transform(training_docs)
# Define test_vectors:
test_vectors = bow_vectorizer.transform(test_docs)

spam_classifier = MultinomialNB()

def spam_or_not(label):
  return "spam" if label else "not spam"

# Uncomment the code below when you're done:
spam_classifier.fit(training_vectors, training_labels)

predictions = spam_classifier.score(test_vectors, test_labels)

print("The predictions for the test data were {0}% accurate.\n\nFor example, '{1}' was classified as {2}.\n\nMeanwhile, '{3}' was classified as {4}.".format(predictions * 100, test_docs[7], spam_or_not(test_labels[7]), test_docs[15], spam_or_not(test_labels[15])))

#-------------- result ------------------------

The predictions for the test data were 100.0% accurate.

For example, 'really do hope the work doesnt get stressful have a gr8 day' was classified as not spam.

Meanwhile, '2p per min to call germany 08448350055 from your bt line just 2p per min check planettalkinstant com for info t s c s text stop to opt out' was classified as spam.

'''
BAG-OF-WORDS LANGUAGE MODEL
BoW Wow
As you can see, bag-of-words is pretty useful! BoW also has several advantages over other language models. For one, it’s an easier model to get started with and a few Python libraries already have built-in support for it.

Because bag-of-words relies on single words, rather than sequences of words, there are more examples of each unit of language in the training corpus. More examples means the model has less data sparsity (i.e., it has more training knowledge to draw from) than other statistical models.

Imagine you want to make a shirt to sell to people. If you have the shirt exactly tailored to someone’s body, it probably won’t fit that many people. But if you make a shirt that is just a giant bag with arm holes, you know that no one will buy it. What do you do? You loosely fit the shirt to someone’s body, leaving some extra room for different body shapes.

Overfitting (adapting a model too strongly to training data, akin to our highly tailored shirt) is a common problem for statistical language models. While BoW still suffers from overfitting in terms of vocabulary, it overfits less than other statistical models, allowing for more flexibility in grammar and word choice.

The combination of low data sparsity and less overfitting makes the bag-of-words model more reliable with smaller training data sets than other statistical models.'''

from preprocessing import preprocess_text
from nltk.util import ngrams
from collections import Counter

text = "It's exciting to watch flying fish after a hard day's work. I don't know why some fish prefer flying and other fish would rather swim. It seems like the fish just woke up one day and decided, 'hey, today is the day to fly away.'"
tokens = preprocess_text(text)

# Bigram approach:
bigrams_prepped = ngrams(tokens, 2)
bigrams = Counter(bigrams_prepped)
print("Three most frequent word sequences and the number of occurrences according to Bigrams:")
print(bigrams.most_common(3))

# Bag of Words approach:
# Define bag_of_words here:
bag_of_words = Counter(tokens)
print("\nThree most frequent words and number of occurrences according to Bag of Words:")
most_common_three = bag_of_words.most_common(3)
print(most_common_three)

#--------------- result ---------------
'''
Three most frequent word sequences and the number of occurrences according to Bigrams:
[(('it', 's'), 1), (('s', 'excite'), 1), (('excite', 'to'), 1)]

Three most frequent words and number of occurrences according to Bag-of-Words:

BAG-OF-WORDS LANGUAGE MODEL
BoW Ow
Alas, there is a trade-off for all the brilliance BoW brings to the table.

Unless you want sentences that look like “the a but for the”, BoW is NOT a great primary model for text prediction. If that sort of “sentence” isn’t your bag, it’s because bag-of-words has high perplexity, meaning that it’s not a very accurate model for language prediction. The probability of the following word is always just the most frequently used words.

If your BoW model finds “good” frequently occurring in a text sample, you might assume there’s a positive sentiment being communicated in that text… but if you look at the original text you may find that in fact every “good” was preceded by a “not.”

Hmm, that would have been helpful to know. The BoW model’s word tokens lack context, which can make a word’s intended meaning unclear.

Perhaps you are wondering, “What happens if the model comes across a new word that wasn’t in the training data?” As mentioned, like all statistical models, BoW suffers from overfitting when it comes to vocabulary.

There are several ways that NLP developers have tackled this issue. A common approach is through language smoothing in which some probability is siphoned from the known words and given to unknown words.
'''
import nltk, re, random
from nltk.tokenize import word_tokenize
from collections import defaultdict, deque, Counter
from document import oscar_wilde_thoughts

# Change sequence_length:
sequence_length = 1

class MarkovChain:
  def __init__(self):
    self.lookup_dict = defaultdict(list)
    self.most_common = []
    self._seeded = False
    self.__seed_me()

  def __seed_me(self, rand_seed=None):
    if self._seeded is not True:
      try:
        if rand_seed is not None:
          random.seed(rand_seed)
        else:
          random.seed()
        self._seeded = True
      except NotImplementedError:
        self._seeded = False
    
  def add_document(self, str):
    preprocessed_list = self._preprocess(str)
    self.most_common = Counter(preprocessed_list).most_common(20)
    pairs = self.__generate_tuple_keys(preprocessed_list)
    for pair in pairs:
      self.lookup_dict[pair[0]].append(pair[1])
  
  def _preprocess(self, str):
    cleaned = re.sub(r'\W+', ' ', str).lower()
    tokenized = word_tokenize(cleaned)
    return tokenized

  def __generate_tuple_keys(self, data):
    if len(data) < sequence_length:
      return

    for i in range(len(data) - 1):
      yield [ data[i], data[i + 1] ]
      
  def generate_text(self, max_length=50):
    context = deque()
    output = []
    if len(self.lookup_dict) > 0:
      self.__seed_me(rand_seed=len(self.lookup_dict))
      chain_head = [list(self.lookup_dict)[0]]
      context.extend(chain_head)
      if sequence_length > 1:
        while len(output) < (max_length - 1):
          next_choices = self.lookup_dict[context[-1]]
          if len(next_choices) > 0:
            next_word = random.choice(next_choices)
            context.append(next_word)
            output.append(context.popleft())
          else:
            break
        output.extend(list(context))
      else:
        while len(output) < (max_length - 1):
          next_choices = [word[0] for word in self.most_common]
          next_word = random.choice(next_choices)
          output.append(next_word)
    return " ".join(output)

my_markov = MarkovChain()
my_markov.add_document(oscar_wilde_thoughts)
random_oscar_wilde = my_markov.generate_text()
print(random_oscar_wilde)

#---------- result -------------------------

in london whereon we build greater hopes for in the country that in cups an honest workman and sees the fancy of your country has his work the art schools of wood carving on one could not see the song of indifference is something which they come only well and

'''
BAG-OF-WORDS LANGUAGE MODEL
Review of Bag-of-Words
You made it! And you’ve learned plenty about the bag-of-words language model along the way:

Bag-of-words (BoW) — also referred to as the unigram model — is a statistical language model based on word count.
There are loads of real-world applications for BoW.
BoW can be implemented as a Python dictionary with each key set to a word and each value set to the number of times that word appears in a text.
For BoW, training data is the text that is used to build a BoW model.
BoW test data is the new text that is converted to a BoW vector using a trained features dictionary.
A feature vector is a numeric depiction of an item’s salient features.
Feature extraction (or vectorization) is the process of turning text into a BoW vector.
A features dictionary is a mapping of each unique word in the training data to a unique index. This is used to build out BoW vectors.
BoW has less data sparsity than other statistical models. It also suffers less from overfitting.
BoW has higher perplexity than other models, making it less ideal for language prediction.
One solution to overfitting is language smoothing, in which a bit of probability is taken from known words and allotted to unknown words.
The spam data for this lesson were taken from the UCI Machine Learning Repository.

Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
'''

from spam_data import training_spam_docs, training_doc_tokens, training_labels, training_docs
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

test_text = """
Play around with the spam classifier!
"""

bow_vectorizer = CountVectorizer()

training_vectors = bow_vectorizer.fit_transform(training_docs)
test_vectors = bow_vectorizer.transform([test_text])

spam_classifier = MultinomialNB()
spam_classifier.fit(training_vectors, training_labels)

predictions = spam_classifier.predict(test_vectors)

print("Looks like a normal email!" if predictions[0] == 0 else "You've got spam!")
