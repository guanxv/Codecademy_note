



#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#- BAYES' THEOREM #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-	
'''
BAYES' THEOREM
Introduction to Bayes' Theorem
In this lesson, we’ll learn about Bayes’ Theorem. Bayes’ Theorem is the basis of a branch of statistics called Bayesian Statistics, where we take prior knowledge into account before calculating new probabilities.

This allows us to find narrow solutions from a huge universe of possibilities. British mathematician Alan Turing used it to crack the German Enigma code during WWII. And now it is used in:

Machine Learning
Statistical Modeling
A/B Testing
Robotics
By the end of this lesson, you’ll be able to solve simple problems involving prior knowledge.'''

'''
BAYES' THEOREM
Independent Events
The ability to determine whether two events are independent is an important skill for statistics.

If two events are independent, then the occurrence of one event does not affect the probability of the other event. Here are some examples of independent events:

I wear a blue shirt; my coworker wears a blue shirt
I take the subway to work; I eat sushi for lunch
The NY Giants win their football game; the NY Rangers win their hockey game
If two events are dependent, then when one event occurs, the probability of the other event occurring changes in a predictable way.

Here are some examples of dependent events:

It rains on Tuesday; I carry an umbrella on Tuesday
I eat spaghetti; I have a red stain on my shirt
I wear sunglasses; I go to the beach'''

'''
A certain family plans to have three children. Is the event that the couple’s third child is a girl independent of the event that the couple’s first two children are girls?

Save your answer ('independent' or 'not independent') to the variable third_child.


The third child is a girl is independent of the event that the first two children are girls. So:'''

third_child = 'independent'

'''
BAYES' THEOREM
Conditional Probability
Conditional probability is the probability that two events happen. It’s easiest to calculate conditional probability when the two events are independent.

Note: For the rest of this lesson, we’ll be using the statistical convention that the probability of an event is written as P(event).

If the probability of event A is P(A) and the probability of event B is P(B) and the two events are independent, then the probability of both events occurring is the product of the probabilities:

P(A ∩ B) = P(A) \times P(B)P(A∩B)=P(A)×P(B)
The symbol ∩ just means “and”, so P(A ∩ B) means the probability that both A and B happen.

For instance, suppose we are rolling a pair of dice, and want to know the probability of rolling two sixes.

BoxcarsMidnight

Each die has six sides, so the probability of rolling a six is 1/6. Each die is independent (i.e., rolling one six does not increase or decrease our chance of rolling a second six), so:

P(6 \cap 6) = P(6) \times P(6) = \frac{1}{6} \times \frac{1}{6} = \frac{1}{36}P(6∩6)=P(6)×P(6)= 
6
1
​	 × 
6
1
​	 = 
36
1



1.
This week, there is a 30% probability that it will rain on any given day. At a certain high school, gym class is held on three days out of the five day school week.

On a school day, what is the probability that it is raining and the students have gym class?

Save your answer to the variable p_rain_and_gym.'''

import numpy as np

p_rain = 0.3
p_gym = 3.0/5.0

p_rain_and_gym = p_rain * p_gym

'''
BAYES' THEOREM
Testing for a Rare Disease
Suppose you are a doctor and you need to test if a patient has a certain rare disease. The test is very accurate: it’s correct 99% of the time. The disease is very rare: only 1 in 100,000 patients have it.

You administer the test and it comes back positive, so your patient must have the disease, right?

Not necessarily. If we just consider the test, there is only a 1% chance that it is wrong, but we actually have more information: we know how rare the disease is.

Given that the test came back positive, there are two possibilities:

The patient had the disease, and the test correctly diagnosed the disease.
The patient didn’t have the disease and the test incorrectly diagnosed that they had the disease.'''
'''

Instructions
1.
What is the probability that the patient had the disease and the test correctly diagnosed the disease?

Save your answer to the variable p_disease_and_correct.


The disease is rare, so the probability that the patient had the disease is 1 out of 100,000:

P(disease) = \frac{1}{100000}P(disease)= 
100000
1
​	 
The test is only wrong 1% of the time, so it is correct 99% of the time:

P(test\ is\ correct) = 0.99P(test is correct)=0.99
So the answer should look like:

p_disease_and_correct = (1.0 / 100000) * 0.99
2.
What is the probability that the patient does not have the disease and the test incorrectly diagnosed the disease?

Save your answer to the variable p_no_disease_and_incorrect.


The disease is rare, so the probability that the patient does not have the disease the disease is 99,999 out of 100,000:

P(disease) = \frac{99999}{100000}P(disease)= 
100000
99999
​	 
The test is only wrong 1% of the time:

P(test\ is\ correct) = 0.01P(test is correct)=0.01
So the answer should look like:

p_no_disease_and_incorrect = (99999.0 / 100000) * 0.01
3.
Print both p_disease_and_correct and p_no_disease_and_incorrect.


print p_disease_and_correct
print p_no_disease_and_incorrect
'''

import numpy as np

p_correct = 0.99
p_disease = 1.0/100000

p_disease_and_correct = p_correct * p_disease
print(p_disease_and_correct)

p_no_disease_and_incorrect = (1-p_disease) * (1-p_correct)
print(p_no_disease_and_incorrect)

9.9e-06
0.0099999

'''
BAYES' THEOREM
Bayes' Theorem
In the previous exercise, we determined two probabilities:

The patient had the disease, and the test correctly diagnosed the disease ≈ 0.00001
The patient didn’t have the disease and the test incorrectly diagnosed that they had the disease ≈ 0.01
Both events are rare, but we can see that it was about 1,000 times more likely that the test was incorrect than that the patient had this rare disease.

We’re able to come to this conclusion because we had more information than just the accuracy of the test; we also knew the prevalence of this disease. That extra information about how we expect the world to work is called a prior.

When we only use the first piece of information (the result of the test), it’s called a Frequentist Approach to statistics. When we incorporate our prior, it’s called a Bayesian Approach.

In statistics, if we have two events (A and B), we write the probability that event A will happen, given that event B already happened as P(A|B). In our example, we want to find P(rare disease | positive result). In other words, we want to find the probability that the patient has the disease given the test came back positive.

We can calculate P(A|B) using Bayes’ Theorem, which states:

P(A|B) = P(B|A)*P(A)/ P(B)

​	 
So in this case, we’d say:

P(rare disease | positive result) = P(positive result | rare disease) * P(rare disease) / P(positive result)
​	 
It is important to note that on the right side of the equation, we have the term P(B|A). This is the probability that event B will happen given that event A has already happened. This is very different from P(A|B), which is the probability we are trying to solve for. The order matters!'''


'''
Instructions
1.
Calculate P(positive result | rare disease), or the probability of a positive test result, given that a patient really has this rare disease.

Save your answer (as a decimal) to p_positive_given_disease.


The test is 99% accurate; given the fact that the patient has the disease, we know that there is a 99% probability that the test will return a positive result.

This is exactly P(positive result | rare disease).

So the answer should look like:

p_positive_given_disease = 0.99
2.
What is P(rare disease), the probability that a randomly selected patient has the rare disease?

Save your answer to p_disease.


The disease is very rare. Only 1 in 100,000 people have it.

3.
We now need to compute the denominator; we need to find P(positive result).

As we discussed previously, there are two ways to get a positive result:

The patient had the disease, and the test correctly diagnosed the disease.
The patient didn’t have the disease and the test incorrectly diagnosed that they had the disease.
Using these two probabilities, calculate the total probability that a randomly selected patient receives a positive test result, P(positive result).

Save your answer to the variable p_positive.


The probability that the patient had the disease, and the test correctly diagnosed the disease is:

1.0 / 100000.0 * 0.99
The probability that the patient didn’t have the disease and the test incorrectly diagnosed that they had the disease is:

99999.0 / 100000.0 * 0.01
The probability of either event A or event B happening is given by:

P(A\ or\ B) = P(A) + P(B)P(A or B)=P(A)+P(B)
4.
Substitute all three of these values into Bayes’ Theorem and calculate P(rare disease | positive result).

Save your result as p_disease_given_positive.


The numerator should be (p_positive_given_disease) * (p_disease).

The denominator should be p_positive.

5.
Print p_disease_given_positive.

Is it likely that your patient has this disease?


print p_disease_given_positive
The result should look something like:

0.000989010989011
Not very likely!'''

import numpy as np

p_positive_given_disease = 0.99

p_disease = 1.0/100000

#patient had Disease and correct
p1 = p_disease * 0.99

#patient doesn't have Disease and in-correct
p2 = (1-p_disease)*(1-0.99)

p_positive = p1 + p2

p_disease_given_positive = p_positive_given_disease *  p_disease / p_positive

print(p_disease_given_positive)

'''
BAYES' THEOREM
Spam Filters
Let’s explore a different example. Email spam filters use Bayes’ Theorem to determine if certain words indicate that an email is spam.

Let’s a take word that often appears in spam: “enhancement”.

With just 3 facts, we can make some preliminary steps towards a good spam filter:

“enhancement” appears in just 0.1% of non-spam emails
“enhancement” appears in 5% of spam emails
Spam emails make up about 20% of total emails
Given that an email contains “enhancement”, what is the probability that the email is spam?

Instructions
1.
In this example, we are dealing with two probabilities:

P(enhancement) - the probability that the word “enhancement” appears in an email.
P(spam) - the probability that an email is spam.
Using Bayes’ Theorem to answer our question means that we want to calculate P(A|B).

But what are A and B referring to in this case?

Save the string 'enhancement' or 'spam' to the variable a.

Save the string 'enhancement' or 'spam' to the variable b.


Recall that P(A|B) means “The probability of A given that B is true”.

The correct answer is:
'''
a = 'spam'
b = 'enhancement'

'''
2.
What is P(spam)?

Save your answer to p_spam.


We’ve given you this exact probability above. What percentage of total emails are spam?

3.
What is P(enhancement | spam)?

Save your answer to p_enhancement_given_spam.


If the email is spam, what is the probability that “enhancement” is in it?

4.
We want to know the overall probability that any email (spam or non-spam) contains “enhancement”.

Because we know the probability of “enhancement” occurring in both spam (0.05) and non-spam (0.001) emails, we can use a weighted average to calculate the probability of “enhancement” occurring in an email:

P(enhancement) = P(enhancement|spam) * P(spam) + P(enhacement| not spam) * P(not spam)

Save your answer to p_enhancement.


5.
Now that we know:

P(spam)
P(enhancement | spam)
P(enhancement)
We can plug this into Bayes’ Theorem:

P(A|B) = P(B|A) * P(A) / P(B)
​	 
Save your answer as p_spam_enhancement.


P(spam | enhancement) = P(enhancement | spam) * P(spam) / P(enhancement)
​	 
6.
Print p_spam_enhancement. This is the probability that an email is spam given that it contains the word “enhancement”.

Should we block all emails that contain “enhancement”?

How much non-spam email would we block?


print p_spam_enhancement
Woah, that’s ≈ 92.59%'''

import numpy as np

a = 'spam'

b = 'enhancement'

P_ench_non_spam = 0.001

p_ench_spam = 0.05

p_spam = 0.2

p_enhancement_given_spam = p_ench_spam

p_enhancement = P_ench_non_spam * (1 - p_spam)  + p_ench_spam * p_spam

p_spam_enhancement = p_enhancement_given_spam * p_spam / p_enhancement

print(p_spam_enhancement)

'''
BAYES' THEOREM
Review
In this course, we learned several new definitions:

Two events are independent if the occurrence of one event does not affect the probability of the second event
If two events are independent then:
P(A ∩ B) = P(A) \times P(B)P(A∩B)=P(A)×P(B)
A prior is an additional piece of information that tells us how likely an event is
A frequentist approach to statistics does not incorporate a prior
A Bayesian approach to statistics incorporates prior knowledge
Bayes’ Theorem is the following:
P(A|B) = P(B|A) * P(A) / P(B) '''


'''
MACHINE LEARNING: SUPERVISED LEARNING 🤖
Dr. Dirac's Statistics Midterm
Grading a multiple choice exam is easy. But how much do multiple choice exams tell us about what a student really knows? Dr. Dirac is administering a statistics midterm exam and wants to use Bayes’ Theorem to help him understand the following:

Given that a student answered a question correctly, what is the probability that she really knows the material?
Dr. Dirac knows the following probabilities based on many years of teaching:

There is a question on the exam that 60% of students know the correct answer to.
Given that a student knows the correct answer, there is still a 15% chance that the student picked the wrong answer.
Given that a student does not know the answer, there is still a 20% chance that the student picks the correct answer by guessing.
Using these probabilities, we can answer the question.

If you get stuck during this project or would like to see an experienced developer work through it, click “Get Help“ to see a project walkthrough video.

Tasks
6/6Complete
Mark the tasks as complete by checking them off
Bayes' Theorem on a Statistics Exam
1.
In order to use Bayes Theorem, we need to phrase our question as P(A|B).

What is A and B in this case?


P(knows the material | answers correctly)

2.
What is the probability that the student knows the material?


P(knows the material) = 0.60

3.
Given that the student knows the material, what is the probability that she answers correctly?


P(answers correctly | knows material) = 1 - 0.15

4.
What is the probability of any student answering correctly?


The probability of answering correctly is equal to the weighted average of P(answers correctly | knows material) and P(answers correctly| does not know material).

You were given both of these probabilities (1 - 0.15 and 0.20, respectively).

How can you calculate the weights?

5.
Using the three probabilities and Bayes’ Theorem, calculate P(knows material | answers correctly).


Your final answer should be around 86%.'''


import numpy as np

p_know = 0.6

p_wrong_know = 0.15

p_correct_unknow = 0.2

'''
P(A|B)
A REALLY KNOW THE MATERIAL
B GIVEN ANSWER CORRECTLY
'''

p_correct_given_know = 1 - 0.15

'''
p_correct = P_know * P_correct_given_know + P_unknow * P_correct_unkonw
'''
p_correct = p_know * p_correct_given_know + (1-p_know) * p_correct_unknow

p_konw_given_correct = p_correct_given_know * p_know / p_correct

print(p_konw_given_correct)

'''
NAIVE BAYES CLASSIFIER
The Naive Bayes Classifier
A Naive Bayes classifier is a supervised machine learning algorithm that leverages Bayes’ Theorem to make predictions and classifications. Recall Bayes’ Theorem:

P(A | B) = P(B | A) * P(A) / P(B)
​	 
This equation is finding the probability of A given B. This can be turned into a classifier if we replace B with a data point and A with a class. For example, let’s say we’re trying to classify an email as either spam or not spam. We could calculate P(spam | email) and P(not spam | email). Whichever probability is higher will be the classifier’s prediction. Naive Bayes classifiers are often used for text classification.

So why is this a supervised machine learning algorithm? In order to compute the probabilities used in Bayes’ theorem, we need previous data points. For example, in the spam example, we’ll need to compute P(spam). This can be found by looking at a tagged dataset of emails and finding the ratio of spam to non-spam emails.'''

'''
NAIVE BAYES CLASSIFIER
The Naive Bayes Classifier
A Naive Bayes classifier is a supervised machine learning algorithm that leverages Bayes’ Theorem to make predictions and classifications. Recall Bayes’ Theorem:

P(A |  B) = P(B | A) * P(A) / P(B)

This equation is finding the probability of A given B. This can be turned into a classifier if we replace B with a data point and A with a class. For example, let’s say we’re trying to classify an email as either spam or not spam. We could calculate P(spam | email) and P(not spam | email). Whichever probability is higher will be the classifier’s prediction. Naive Bayes classifiers are often used for text classification.

So why is this a supervised machine learning algorithm? In order to compute the probabilities used in Bayes’ theorem, we need previous data points. For example, in the spam example, we’ll need to compute P(spam). This can be found by looking at a tagged dataset of emails and finding the ratio of spam to non-spam emails.'''

'''
NAIVE BAYES CLASSIFIER
Investigate the Data
In this lesson, we are going to create a Naive Bayes classifier that can predict whether a review for a product is positive or negative. This type of classifier could be extremely helpful for a company that is curious about the public reaction to a new product. Rather than reading thousands of reviews or tweets about the product, you could feed those documents into the Naive Bayes classifier and instantly find out how many are positive and how many are negative.

The dataset we will be using for this lesson contains Amazon product reviews for baby products. The original dataset contained many different features including the reviewer’s name, the date the review was made, and the overall score. We’ve removed many of those features; the only features that we’re interested in are the text of the review and whether the review was “positive” or “negative”. We labeled all reviews with a score less than 4 as a negative review.

Note that in the next two lessons, we’ve only imported a small percentage of the data to help the code run faster. We’ll import the full dataset later when we put everything together!

NAIVE BAYES CLASSIFIER
Bayes Theorem I
For the rest of this lesson, we’re going to write a classifier that can predict whether the review “This crib was amazing” is a positive or negative review. We want to compute both P(positive | review) and P(negative | review) and find which probability is larger. To do this, we’ll be using Bayes’ Theorem. Let’s look at Bayes’ Theorem for P(positive | review).

P(\text{positive}\ | \ \text{review}) = \frac{P(\text{review\ |\ positive}) \cdot P(\text{positive})}{P(\text{review})}P(positive ∣ review)= 
P(review)
P(review | positive)⋅P(positive)
​	 
The first part of Bayes’ Theorem that we are going to tackle is P(positive). This is the probability that any review is positive. To find this, we need to look at all of our reviews in our dataset - both positive and negative - and find the percentage of reviews that are positive.

We’ve bolded the part of Bayes’ Theorem we’re working on.

P(\text{positive}\ | \ \text{review}) = \frac{P(\text{review\ |\ positive}) \cdot \textbf{P(positive})}{P(\text{review})}P(positive ∣ review)= 
P(review)
P(review | positive)⋅P(positive)'''

'''
NAIVE BAYES CLASSIFIER
Bayes Theorem I
For the rest of this lesson, we’re going to write a classifier that can predict whether the review “This crib was amazing” is a positive or negative review. We want to compute both P(positive | review) and P(negative | review) and find which probability is larger. To do this, we’ll be using Bayes’ Theorem. Let’s look at Bayes’ Theorem for P(positive | review).

P(positive | review) = P(review | positive) * P(positive) / P(review) 

The first part of Bayes’ Theorem that we are going to tackle is P(positive). This is the probability that any review is positive. To find this, we need to look at all of our reviews in our dataset - both positive and negative - and find the percentage of reviews that are positive.

We’ve bolded the part of Bayes’ Theorem we’re working on.

P(\text{positive}\ | \ \text{review}) = \frac{P(\text{review\ |\ positive}) \cdot \textbf{P(positive})}{P(\text{review})}P(positive ∣ review)= 
P(review)
P(review | positive)⋅P(positive)'''

'''
NAIVE BAYES CLASSIFIER
Bayes Theorem II
Let’s continue to try to classify the review “This crib was amazing”.

The second part of Bayes’ Theorem is a bit more extensive. We now want to compute P(review | positive).

P( positive | review ) = P(review | positive) * P(positive) / P(review)
​	 
In other words, if we assume that the review is positive, what is the probability that the words “This”, “crib”, “was”, and “amazing” are the only words in the review?

To find this, we have to assume that each word is conditionally independent. This means that one word appearing doesn’t affect the probability of another word from showing up. This is a pretty big assumption!

We now have this equation. You can scroll to the right to see the full equation.

P("This crib was amazing" | positive) = P("This" | positive) * P("crib" |positive) * P("was" | positive) * P("amazing" | positive) 

Let’s break this down even further by looking at one of these terms. P("crib"|positive) is the probability that the word “crib” appears in a positive review. To find this, we need to count up the total number of times “crib” appeared in our dataset of positive reviews. If we take that number and divide it by the total number of words in our positive review dataset, we will end up with the probability of “crib” appearing in a positive review.

P("crib" | positive ) = # of "crib" in positive / # of words in positive

​	 
If we do this for every word in our review and multiply the results together, we have P(review | positive).

Instructions
1.
Let’s first find the total number of words in all positive reviews and store that number in a variable named total_pos.

To do this, we can use the built-in Python sum() function. sum() takes a list as a parameter. The list that you want to sum is the values of the dictionary pos_counter, which you can get by using pos_counter.values().

Do the same for total_neg.


Use pos_counter.values() as the only parameter of sum():

total_pos = sum(pos_counter.values())
2.
Create two variables named pos_probability and neg_probability. Each of these variables should start at 1. These are the variables we are going to use to keep track of the probabilities.


pos_probability = 1
neg_probability = 1
3.
Create a list of the words in review and store it in a variable named review_words. You can do this by using Python’s .split() function.

For example if the string test contained "Hello there", then test.split() would return ["Hello", "there"].


review_words = review.split()
4.
Loop through every word in review_words. Find the number of times word appears in pos_counter and neg_counter. Store those values in variables named word_in_pos and word_in_neg.

In the next steps, we’ll use this variable inside the for loop to do a series of multiplications.


The beginning of your for loop should look like this:

for word in review_words:
The number of times word appears in pos_counter can be found like this:

word_in_pos = pos_counter[word]
5.
Inside the for loop, set pos_probability to be pos_probability multiplied by word_in_pos / total_pos.

This step is finding each term to be multiplied together. For example, when word is "crib", you’re calculating the following:

P( "crib" | positive) = # of "crib" in positive / # of words in positive​	 

You can use *= to do the multiplication:

pos_probability *= word_in_pos / total_pos
Another way to do this is

pos_probability = pos_probability * word_in_pos / total_pos
6.
Do the same multiplication for neg_probability.

Outside the for loop, print both pos_probability and neg_probability. Those values are P(“This crib was amazing”|positive) and P(“This crib was amazing”|negative).


neg_probability should look very similar:

py neg_probability *= word_in_neg / total_neg

These should both be pretty small numbers. In fact, they will most likely be written in scientific notation. For example, 1.5e-5 is the same as 0.000015.'''

from reviews import neg_counter, pos_counter

review = "This crib was amazing"

percent_pos = 0.5
percent_neg = 0.5

total_pos = sum(pos_counter.values())
total_neg = sum(neg_counter.values())

print(total_pos)
print(total_neg)

pos_probability = 1
neg_probability = 1 

review_words = review.split()

word_in_pos = []
word_in_neg = []


for word in review_words:
  word_in_pos = pos_counter[word]
  word_in_neg = neg_counter[word]
  pos_probability *= word_in_pos / total_pos
  neg_probability *= word_in_neg / total_neg
  
  
print(pos_probability)
print(pos_probability)
 
'''
NAIVE BAYES CLASSIFIER
Smoothing
In the last exercise, one of the probabilities that we computed was the following:

P("crib" | positive) = # of ``crib" in positive / # of words in positive

But what happens if “crib” was never in any of the positive reviews in our dataset? This fraction would then be 0, and since everything is multiplied together, the entire probability P(review | positive) would become 0.

This is especially problematic if there are typos in the review we are trying to classify. If the unclassified review has a typo in it, it is very unlikely that that same exact typo will be in the dataset, and the entire probability will be 0. To solve this problem, we will use a technique called smoothing.

In this case, we smooth by adding 1 to the numerator of each probability and N to the denominator of each probability. N is the number of unique words in our review dataset.

For example, P("crib" | positive) goes from this:

P("crib" | positive) = # of ``crib" in positive / # of words in positive
​	 
To this:

P("crib" | positive) = # of ``crib" in positive + 1 / # of words in positive + N
​	

Instructions
1.
Let’s demonstrate how these probabilities break if there’s a word that never appears in the given datasets.

Change review to "This cribb was amazing". Notice the second b in cribb.

2.
Inside your for loop, when you multiply pos_probability and neg_probability by a fraction, add 1 to the numerator.

Make sure to include parentheses around the numerator!


The fraction that you multiply pos_probability by should now look like this:

(word_in_pos + 1) / total_pos
3.
In the denominator of those fractions, add the number of unique words in the appropriate dataset.

For the positive probability, this should be the length of pos_counter which can be found using len().

Again, make sure to put parentheses around your denominator so the division happens after the addition!

Did smoothing fix the problem?


pos_probability should now be multiplied by:

(word_in_pos + 1) / (total_pos + len(pos_counter))
Make sure to do the same for neg_probability.'''

from reviews import neg_counter, pos_counter

review = "This cribb was amazing"

percent_pos = 0.5
percent_neg = 0.5

total_pos = sum(pos_counter.values())
total_neg = sum(neg_counter.values())

pos_probability = 1
neg_probability = 1

review_words = review.split()

for word in review_words:
  word_in_pos = pos_counter[word]
  word_in_neg = neg_counter[word]
  
  pos_probability *= (word_in_pos + 1) / (total_pos +len(pos_counter))
  neg_probability *= (word_in_neg + 1) / (total_neg + len(neg_counter))
  
print(pos_probability)
print(neg_probability)

'''
NAIVE BAYES CLASSIFIER
Classify
If we look back to Bayes’ Theorem, we’ve now completed both parts of the numerator. We now need to multiply them together.

P(positive | review) = P(review | positive) * P(positive) / P(review)

Let’s now consider the denominator P(review). In our small example, this is the probability that “This”, “crib”, “was”, and “amazing” are the only words in the review. Notice that this is extremely similar to P(review | positive). The only difference is that we don’t assume that the review is positive.

However, before we start to compute the denominator, let’s think about what our ultimate question is. We want to predict whether the review “This crib was amazing” is a positive or negative review. In other words, we’re asking whether P(positive | review) is greater than P(negative | review). If we expand those two probabilities, we end up with the following equations.

P( positive | review ) = P( review | positive) * P(positive) / P(review) 

​	 
P( negative | review ) = P(review | negative) * P(negative) / P(negative )

​	 
Notice that P(review) is in the denominator of each. That value will be the same in both cases! Since we’re only interested in comparing these two probabilities, there’s no reason why we need to divide them by the same value. We can completely ignore the denominator!

Let’s see if our review was more likely to be positive or negative!

Instructions
1.
After the for loop, multiply pos_probability by percent_pos and neg_probability by percent_neg. Store the two values in final_pos and final_neg and print both.


For the final positive probability, do the following:

final_pos = pos_probability * percent_pos
2.
Compare final_pos to final_neg:

If final_pos was greater than final_neg, print "The review is positive".
Otherwise print "The review is negative".
Did our Naive Bayes Classifier get it right for the review "This crib was amazing"?


Use an if statement like this:

if final_pos > final_neg:
3.
Replace the review "This crib was amazing" with one that you think should be classified as negative. Run your program again.

Did your classifier correctly classify the new review?'''

from reviews import neg_counter, pos_counter

#review = "This crib was amazing"
#review = "bad silly regret crap waste nonsense"
review = "great wounderful nice perfect beautiful"

percent_pos = 0.5
percent_neg = 0.5

total_pos = sum(pos_counter.values())
total_neg = sum(neg_counter.values())

pos_probability = 1
neg_probability = 1

review_words = review.split()

for word in review_words:
  word_in_pos = pos_counter[word]
  word_in_neg = neg_counter[word]
  
  pos_probability *= (word_in_pos + 1) / (total_pos + len(pos_counter))
  neg_probability *= (word_in_neg + 1) / (total_neg + len(neg_counter))

final_pos = pos_probability * percent_pos 
final_neg = neg_probability * percent_neg

if final_pos > final_neg:
  print("The review is positive")
else:
  print('The review is negative')

'''NAIVE BAYES CLASSIFIER
Formatting the Data for scikit-learn
Congratulations! You’ve made your own Naive Bayes text classifier. If you have a dataset of text that has been tagged with different classes, you can give your classifier a brand new document and it will predict what class it belongs to.

We’re now going to look at how Python’s scikit-learn library can do all of that work for us!

In order to use scikit-learn’s Naive Bayes classifier, we need to first transform our data into a format that scikit-learn can use. To do so, we’re going to use scikit-learn’s CountVectorizer object.

To begin, we need to create a CountVectorizer and teach it the vocabulary of the training set. This is done by calling the .fit() method.

For example, in the code below, we’ve created a CountVectorizer that has been trained on the vocabulary "Training", "review", "one", and "Second".

vectorizer = CountVectorizer()

vectorizer.fit(["Training review one", "Second review"])
After fitting the vectorizer, we can now call its .transform() method. The .transform() method takes a list of strings and will transform those strings into counts of the trained words. Take a look at the code below.

counts = vectorizer.transform(["one review two review"])
counts now stores the array [2, 1, 0, 0]. The word "review" appeared twice, the word "one" appeared once, and neither "Training" nor "Second" appeared at all.

But how did we know that the 2 corresponded to review? You can print vectorizer.vocabulary_ to see the index that each word corresponds to. It might look something like this:

{'one': 1, 'Training': 2, 'review': 0, 'Second': 3}
Finally, notice that even though the word "two" was in our new review, there wasn’t an index for it in the vocabulary. This is because "two" wasn’t in any of the strings used in the .fit() method.

We can now usecounts as input to our Naive Bayes Classifier.

Note that in the code in the editor, we’ve imported only a small percentage of our review dataset to make load times faster. We’ll import the full dataset later when we put all of the pieces together!

Instructions
1.
Create a CountVectorizer and name it counter.


counter = CountVectorizer()
2.
Call counter‘s .fit() method. .fit() takes a list of strings and it will learn the vocabulary of those strings. We want our counter to learn the vocabulary from both neg_list and pos_list.

Call .fit() using neg_list + pos_list as a parameter.


Fill in the appropriate parameter in the code below:

counter.fit(____)
3.
Print counter.vocabulary_. This is the vocabulary that your counter just learned. The numbers associated with each word are the indices of each word when you transform a review.


Make sure to include the underscore in the print statement!

4.
Let’s transform our brand new review. Create a variable named review_counts and set it equal to counter‘s .transform() function. Remember, .transform() takes a list of strings to transform. So call .transform() using [review] as a parameter.

Print review_counts.toarray(). If you don’t include the toarray(), review_counts won’t print in a readable format.

It looks like this is an array of all 0s, but the indices that correspond to the words "this", "crib", "was", and "amazing" should all be 1.


Fill in the correct parameter. Remember, it must be a list containing review.

review_counts = counter.transform(____)
5.
We’ll use review_counts as the test point for our Naive Bayes Classifier, but we also need to transform our training set.

Our training set is neg_list + pos_list. Call .transform() using that as a parameter. Store the results in a variable named training_counts. We’ll use these variables in the next exercise.


neg_list + pos_list is already a list, so unlike last time, we don’t need to wrap a list around it.'''

from reviews import neg_list, pos_list
from sklearn.feature_extraction.text import CountVectorizer

review = "This crib crib was a amazing"

counter = CountVectorizer()

counter.fit(neg_list + pos_list)

print(counter.vocabulary_)

review_counts = counter.transform([review])

print(review_counts)

training_counts = counter.transform(neg_list + pos_list)


{'wanted': 1521, 'to': 1429, 'love': 805, 'this': 1408, 'but': 182, 'it': 712, 'was': 1525, 'pretty': 1056, 'expensive': 467, 'for': 525, 'only': 951, 'few': 495, 'months': 871, 'worth': 1584, 'of': 937, 'calendar': 187, 'pages': 981, 'ended': 434, 'up': 1486, 'buying': 185, 'regular': 1130, 'weekly': 1541, 'planner': 1024, '55': 11, 'off': 938, 'the': 1393, 'that': 1392, 'is': 709, '11': 2, 'and': 63, 'has': 618, 'all': 47, 'seven': 1219, 'days': 339, 'on': 947, 'right': 1163, 'page': 980, 'left': 765, 'room': 1166, 'write': 1588, 'do': 380, 'list': 785, 'goals': 577, 'found': 539, 'be': 120, 'more': 873, 'helpful': 633, 'because': 123, 'could': 306, 'mark': 823, 'each': 409, 'day': 337, 'eating': 417, 'sleeping': 1252, 'blocks': 149, 'then': 1397, 'also': 55, 'see': 1207, 'them': 1395, 'side': 1235, 'by': 186, 'her': 636, 'patterns': 993, 'easily': 413, 'with': 1568, 'view': 1511, 'cute': 328, 'just': 724, 'not': 919, 'what': 1550, 'like': 778, 'log': 792, 'think': 1405, 'would': 1585, 'work': 1576, 'better': 137, 'clearer': 243, 'am': 59, 'pm': 1034, 'sections': 1205, '12': 3, 'hours': 661, 'so': 1270, 'you': 1598, 'really': 1113, 'need': 903, 'two': 1474, 'if': 673, 'your': 1600, 'baby': 104, 'feeds': 490, 'or': 959, 'wets': 1549, 'lot': 803, 'in': 681, 'early': 411, 'morning': 874, 'between': 138, 'midnight': 852, '7am': 14, 'we': 1539, 're': 1104, 'cramming': 315, 'those': 1409, 'blank': 146, 'spaces': 1289, 'above': 19, 'now': 926, 'my': 888, 'wife': 1561, 'have': 623, 'six': 1243, 'month': 870, 'old': 945, 'boy': 164, 'around': 81, 'decided': 342, 'she': 1224, 'return': 1155, 'instead': 696, 'being': 132, 'stay': 1314, 'at': 92, 'home': 652, 'mom': 866, 'hired': 644, 'an': 61, 'nanny': 891, 'care': 194, 'our': 966, 'little': 787, 'arrangement': 82, 'worked': 1577, 'quite': 1097, 'well': 1544, 'ever': 452, 'since': 1240, 'shortly': 1230, 'after': 40, 'starting': 1308, 'realized': 1111, 'some': 1276, 'sort': 1285, 'journal': 723, 'track': 1446, 'activities': 28, 'while': 1556, 'he': 627, 'were': 1546, 'working': 1579, 'used': 1492, 'plain': 1022, 'notebook': 921, 'period': 1005, 'weeks': 1542, 'until': 1485, 'stumbled': 1336, 'tracker': 1447, 'daily': 332, 'childcare': 227, 'layout': 755, 'use': 1491, 'excellent': 460, 'idea': 671, 'are': 78, 'clearly': 244, 'divided': 379, 'into': 703, 'columns': 253, 'tracking': 1449, 'feedings': 489, 'nap': 892, 'time': 1425, 'diaper': 360, 'changes': 217, 'play': 1026, 'as': 86, 'general': 559, 'areas': 80, 'notes': 922, 'milestones': 855, 'legibility': 766, 'huge': 667, 'improvement': 679, 'over': 971, 'standard': 1303, 'entries': 445, 'becoming': 125, 'small': 1260, 'paragraphs': 987, 'short': 1229, 'moments': 867, 'can': 191, 'summarize': 1347, 'data': 335, 'totals': 1442, 'section': 1204, 'determine': 355, 'key': 730, 'information': 693, 'how': 665, 'much': 885, 'did': 364, 'eat': 415, 'bowel': 161, 'movement': 883, 'sleep': 1250, 'they': 1401, 'get': 564, 'etc': 449, 'there': 1399, 'however': 666, 'frustrating': 549, 'limitations': 781, 'first': 510, 'entire': 443, 'about': 18, 'half': 602, 'sheet': 1225, 'down': 392, 'middle': 851, 'portrait': 1044, 'constrains': 285, 'very': 1505, 'column': 252, 'row': 1172, 'okay': 944, 'summarized': 1348, 'ounces': 965, '34': 9, 'once': 948, 'becomes': 124, 'active': 27, 'know': 737, 'than': 1390, 'tummy': 1469, 'under': 1479, 'things': 1404, 'start': 1306, 'tight': 1423, 'another': 65, 'problem': 1067, 'covers': 313, 'out': 968, 'fine': 505, 'intention': 699, 'child': 225, 'providers': 1079, 'which': 1555, 'often': 941, 'starts': 1309, 'earlier': 410, '6am': 13, 'using': 1498, 'require': 1142, 'second': 1201, 'good': 581, 'easy': 414, 'read': 1106, 'instantly': 695, 'gather': 557, 'matter': 832, 'high': 639, 'quality': 1092, 'paper': 986, 'consistent': 282, 'don': 388, 'gets': 565, 'most': 875, 'less': 768, 'one': 949, 'babies': 103, 'cover': 312, 'thick': 1402, 'hardback': 616, 'bends': 135, 'bag': 110, 'should': 1231, 'bigger': 141, 'understand': 1480, 'portability': 1043, 'concern': 271, 'current': 326, 'size': 1244, 'entirely': 444, 'too': 1434, 'conclusion': 273, 'making': 817, 'own': 975, 'format': 533, 'spreadsheet': 1296, 'includes': 685, '24': 7, 'hour': 660, 'space': 1288, 'comments': 259, 'along': 53, 'other': 964, 'had': 601, 'bound': 160, 'cheaply': 222, 'local': 791, 'shop': 1228, 'adequate': 34, 'thought': 1411, 'keeping': 727, 'simple': 1238, 'handwritten': 608, 'nice': 913, 'haven': 624, 'thing': 1403, 'here': 637, 'why': 1560, 'when': 1551, 'breastfeeding': 171, 'phone': 1009, 'close': 245, 'keep': 726, 'yourself': 1601, 'entertained': 442, 'able': 17, 'grab': 583, 'both': 155, 'pen': 999, 'consistently': 283, 'skilled': 1247, 'me': 836, 'nurse': 928, 'same': 1179, 'place': 1019, 'every': 453, 'deprived': 351, 'least': 762, 'forget': 528, 'look': 796, 'started': 1307, 'perfect': 1002, 'app': 73, 'either': 425, 'mindlessly': 858, 'hit': 647, 'button': 183, 'connect': 276, 'gives': 571, 'example': 459, 'tell': 1385, 'long': 794, 'average': 98, 'been': 127, 'taking': 1370, 'training': 1452, 'nursed': 929, 'him': 641, '177': 5, 'times': 1427, 'last': 749, 'granted': 588, 'serves': 1215, 'no': 917, 'useful': 1493, 'purpose': 1086, 'feeling': 492, 'perverse': 1008, 'satisfaction': 1181, 'adorable': 36, 'book': 151, 'pieces': 1016, 'attached': 95, 'activity': 29, 'several': 1220, 'though': 1410, 'sew': 1221, 'make': 815, 'directed': 369, 'will': 1563, 'realize': 1110, 'priced': 1061, 'hard': 615, 'age': 42, 'group': 595, 'playing': 1027, 'teether': 1381, 'ridiculous': 1162, 'clamp': 237, 'daughter': 336, 'going': 579, 'vibrating': 1507, 'mouth': 878, 'big': 140, 'awkward': 101, 'push': 1088, 'money': 869, 'opinion': 957, 'product': 1070, 'case': 202, 'made': 812, 'bite': 144, 'vibration': 1508, 'does': 385, 'toy': 1444, 'bottom': 158, '5mo': 12, 'who': 1557, 'loves': 807, 'tap': 1373, 'toys': 1445, 'his': 645, 'yes': 1595, 'sounds': 1287, 'weird': 1543, 'husband': 668, 'drummer': 401, 'got': 582, 'obsessed': 933, 'tapping': 1376, 'even': 450, 'drinking': 397, 'bottle': 156, 'try': 1467, 'hands': 607, 'feel': 491, 'mini': 860, 'dodge': 384, 'ball': 114, 'sticks': 1320, 'face': 476, 'happy': 614, 'bought': 159, 'infant': 690, 'gum': 599, 'massager': 829, 'electric': 427, 'toothbrush': 1437, 'always': 58, 'interested': 700, 'brush': 179, 'teeth': 1380, 'supervised': 1351, 'doesn': 386, 'choke': 231, 'skinny': 1248, 'objects': 932, 'guard': 597, 'block': 148, 'prevent': 1057, 'any': 67, 'type': 1475, 'choking': 232, 'enough': 439, 'turn': 1470, 'again': 41, 'hold': 648, 'trying': 1468, 'find': 503, 'vibrations': 1509, 'himself': 642, 'sucks': 1339, 'chew': 223, 'teething': 1384, 'bites': 145, 'never': 909, 'dumb': 405, 'vibrate': 1506, 'disappointing': 372, 'liked': 779, 'massaging': 830, 'action': 26, 'took': 1435, 'finally': 502, 'battery': 119, 'gave': 558, 'nearly': 897, 'spurts': 1298, 'rest': 1149, 'wouldn': 1586, 'waste': 1531, 'great': 590, 'freaks': 541, 'wants': 1523, 'nothing': 923, 'might': 853, 'wonderful': 1574, 'appears': 75, 'hated': 621, 'cannot': 192, 'comment': 258, 'effectiveness': 421, 'bit': 143, 'seemed': 1208, 'cool': 299, 'passed': 991, 'friend': 544, 'recommended': 1118, 'throwing': 1417, 'floor': 520, 'hopefully': 656, 'squeeze': 1299, 'help': 631, 'heavy': 629, 'seems': 1209, 'enjoy': 438, 'way': 1537, 'feels': 493, 'actually': 31, 'tired': 1428, 'minute': 861, 'maybe': 835, 'older': 946, 'teethers': 1382, 'twin': 1472, 'boys': 165, 'years': 1594, 'ago': 43, 'absolutely': 21, 'loved': 806, 'new': 910, 'sought': 1286, 'surprised': 1357, 'arrived': 84, 'packaging': 978, 'stated': 1311, 'safety': 1176, 'tested': 1389, 'bpa': 166, 'lead': 757, 'phthalates': 1011, 'state': 1310, 'anywhere': 71, 'package': 977, 'free': 542, 'former': 534, 'regulator': 1131, 'medical': 841, 'devices': 359, 'sensitive': 1212, 'company': 264, 'labeling': 741, 'statements': 1312, 'perhaps': 1004, 'rather': 1102, 'err': 446, 'caution': 208, 'where': 1553, 'concerned': 272, 'lack': 742, 'clear': 242, 'pause': 994, 'specifically': 1292, 'test': 1388, 'acceptable': 22, 'amount': 60, 'allowed': 50, 'simply': 1239, 'label': 740, 'phthalate': 1010, 'eight': 424, 'sons': 1282, 'heard': 628, 'put': 1089, 'manner': 819, 'plastic': 1025, 'their': 1394, 'mouths': 879, 'sure': 1356, 'riddled': 1161, 'strongly': 1332, 'suspected': 1358, 'dangerous': 333, 'conscience': 279, 'allow': 49, 'anything': 70, 'may': 834, 'contain': 288, 'someone': 1277, 'site': 1241, 'indicated': 686, 'representative': 1141, 'from': 547, 'told': 1433, 'still': 1321, 'skeptical': 1246, 'such': 1337, 'companies': 263, 'proudly': 1077, 'products': 1071, 'language': 746, 'confused': 275, 'son': 1281, 'suck': 1338, 'fingers': 506, 'messaging': 848, 'corn': 301, 'squeezed': 1300, 'order': 960, 'young': 1599, 'already': 54, 'isn': 710, 'reason': 1114, 'pick': 1012, 'car': 193, 'stuffed': 1335, 'different': 367, 'complaint': 266, 'star': 1304, 'bead': 121, 'end': 433, 'pointy': 1038, 'catches': 205, 'sore': 1284, 'coming': 257, 'wrong': 1591, 'cry': 324, 'colors': 251, 'definitely': 345, 'catch': 204, 'eye': 473, 'overall': 972, 'ok': 943, 'price': 1060, 'necklace': 902, 'weren': 1547, 'reviewer': 1157, 'pointed': 1037, 'chews': 224, 'edge': 420, 'painful': 984, 'thrown': 1418, 'having': 626, 'pull': 1080, 'road': 1164, 'week': 1540, 'screamed': 1194, 'suddenly': 1340, 'back': 106, 'crying': 325, 'hysterically': 669, 'sling': 1255, 'seen': 1210, 'mothers': 877, 'these': 1400, 'before': 128, 'childbirth': 226, 'class': 238, 'raved': 1103, 'excited': 461, 'received': 1115, 'gift': 567, 'cradle': 314, 'position': 1045, 'afraid': 39, 'sufficate': 1341, 'fabric': 474, 'hand': 603, 'next': 912, 'easier': 412, 'grocery': 594, 'store': 1325, 'pulled': 1081, 'shoulder': 1232, 'crawling': 316, 'pulling': 1082, 'restrained': 1151, 've': 1503, 'tried': 1462, 'kangaroo': 725, 'scared': 1188, 'fall': 479, 'wiggle': 1562, 'worm': 1582, 'strap': 1328, 'neck': 901, 'interfere': 701, 'circulation': 236, 'glad': 573, 'rethink': 1153, 'kind': 733, 'carrier': 198, 'slings': 1256, 'practiced': 1050, 'cat': 203, 'comfortable': 256, 'must': 887, 'admit': 35, 'front': 548, 'packs': 979, 'cut': 327, 'across': 25, 'go': 576, 'carry': 199, 'screams': 1195, 'hardly': 617, 'cries': 320, 'say': 1187, 'mean': 838, 'turns': 1471, 'red': 1123, 'soon': 1283, 'take': 1366, 'pack': 976, 'honestly': 653, 'lay': 754, 'cooperates': 300, 'carried': 197, 'brought': 178, 'hospital': 659, 'walking': 1519, 'plan': 1023, 'hip': 643, 'watched': 1534, 'video': 1510, 'support': 1354, 'hope': 654, 'stick': 1318, 'traditional': 1451, 'larger': 748, 'women': 1572, 'endowed': 435, 'purchased': 1085, 'total': 1441, 'thinking': 1406, 'needed': 904, 'change': 215, 'bedding': 126, 'constantly': 284, 'due': 404, 'spit': 1295, 'instructions': 697, 'said': 1177, 'hang': 610, 'dry': 402, 'through': 1414, 'toss': 1440, 'dryer': 403, 'mattress': 833, 'known': 739, 'didn': 365, 'extra': 471, 'went': 1545, 'cosleeper': 304, 'material': 831, 'special': 1290, 'holds': 650, 'lots': 804, 'washes': 1527, 'uses': 1497, 'shipping': 1227, 'quick': 1095, 'sheets': 1226, 'themselves': 1396, 'soft': 1271, 'course': 311, 'inexpensive': 689, 'expected': 464, 'five': 513, 'co': 246, 'sleeper': 1251, 'assembled': 90, 'yet': 1596, 'box': 163, 'torn': 1439, 'dented': 348, 'looking': 798, 'contents': 292, 'looks': 799, 'law': 753, 'puts': 1090, 'together': 1432, 'stroller': 1331, 'won': 1573, 'fit': 511, 'graco': 585, 'straps': 1329, 'tie': 1421, 'trays': 1457, 'came': 190, 'un': 1478, 'sewed': 1222, 'buy': 184, 'happened': 612, 'unforutnately': 1481, 'travel': 1456, 'system': 1361, 'carseat': 201, 'grace': 584, 'marathon': 822, 'useless': 1495, 'quattro': 1093, 'metrolite': 850, 'complete': 268, 'primarily': 1062, 'takes': 1369, 'trash': 1455, 'bags': 111, 'supposed': 1355, 'couple': 309, 'quickly': 1096, 'discovered': 374, 'pail': 982, 'full': 550, 'dropping': 399, 'mechanism': 840, 'stuck': 1334, 'forcing': 527, 'clean': 240, 'whenever': 1552, 'wipe': 1564, 'messy': 849, 'load': 790, 'top': 1438, 'routinely': 1170, 'flip': 518, 'dump': 406, 'dirty': 370, 'lift': 774, 'whole': 1559, 'lid': 770, 'allows': 51, 'tabs': 1364, 'getting': 566, 'trap': 1454, 'smell': 1262, 'negates': 906, 'makes': 816, 'something': 1278, 'cheaper': 221, 'bad': 108, 'proof': 1074, 'fact': 477, 'empty': 432, 'leaks': 758, 'pleasant': 1029, 'truly': 1466, 'believe': 133, 'best': 136, 'market': 824, 'garbage': 555, 'fantastic': 481, 'job': 719, 'containing': 290, 'odors': 936, 'couldn': 307, 'biggest': 142, 'kitchen': 734, 'sized': 1245, 'champ': 214, 'requires': 1144, 'us': 1490, 'remove': 1138, 'wasteful': 1532, 'solution': 1275, 'smaller': 1261, 'accomodate': 23, 'common': 260, 'household': 663, 'tall': 1372, 'efficiently': 422, 'point': 1036, 'separate': 1213, 'disguise': 376, 'doubt': 391, 'expend': 466, 'effort': 423, 'works': 1580, 'piston': 1018, 'drop': 398, 'joint': 720, 'meet': 844, 'thus': 1419, 'caught': 207, 'its': 716, 'replacement': 1140, 'moved': 882, 'drum': 400, 'foot': 524, 'securing': 1206, 'handed': 604, 'operation': 956, 'states': 1313, 'third': 1407, 'trip': 1463, 'ask': 87, 'china': 230, 'contains': 291, 'category': 206, 'throw': 1416, 'soiled': 1272, 'ours': 967, 'tends': 1387, 'lazy': 756, 'fill': 499, 'yuck': 1602, 'save': 1182, 'diapers': 362, 'reading': 1107, 'reviews': 1159, 'line': 782, 'chose': 234, 'register': 1127, 'disappointed': 371, 'sense': 1211, 'super': 1350, 'hero': 638, 'smelly': 1265, 'enter': 440, 'corner': 302, 'guess': 598, 'grateful': 589, 'house': 662, 'poopie': 1040, 'wet': 1548, 'scented': 1189, 'away': 99, 'garabage': 554, 'taken': 1367, 'opening': 954, 'difficult': 368, 'blue': 150, 'round': 1168, 'part': 990, 'broke': 176, 'masking': 827, 'tape': 1374, 'thrilled': 1413, 'specific': 1291, 'cheap': 220, 'smells': 1264, 'large': 747, 'far': 482, 'gab': 553, 'nature': 895, 'awesome': 100, 'stinks': 1322, 'neat': 898, 'disposal': 377, 'shower': 1233, 'handle': 606, 'purchase': 1084, 'liners': 784, 'mistake': 864, 'stars': 1305, 'parents': 989, 'children': 228, '14': 4, 'come': 254, 'odor': 934, 'literally': 786, 'smelled': 1263, 'poop': 1039, 'stand': 1302, 'changed': 216, 'saving': 1184, 'plus': 1033, 'infants': 691, 'want': 1520, 'consider': 281, 'else': 428, 'decor': 343, 'gifts': 568, 'favorite': 483, 'positive': 1046, 'open': 952, 'without': 1570, 'breaking': 168, 'nail': 890, 'changing': 218, 'depending': 349, 'waited': 1515, 'door': 390, 'knot': 736, 'brim': 173, 'dispose': 378, 'squirmy': 1301, 'table': 1363, 'registered': 1128, 'item': 714, 'impressed': 677, 'jammed': 718, 'leaves': 764, 'faint': 478, 'live': 788, 'hassle': 620, 'beware': 139, 'value': 1501, 'manicures': 818, 'refills': 1126, 'pleased': 1031, 'diet': 366, 'bowl': 162, 'movements': 884, 'longer': 795, 'contained': 289, 'tightly': 1424, 'wrap': 1587, 'washing': 1528, 'frequently': 543, 'stench': 1317, 'comes': 255, 'maintain': 813, 'serious': 1214, 'design': 352, 'flaw': 516, 'tapered': 1375, 'noticeably': 925, 'probably': 1066, 'balancing': 113, 'purposes': 1087, 'pros': 1075, 'alternatives': 57, 'normal': 918, 'bagscons': 112, 'glowing': 575, 'receiving': 1116, 'replace': 1139, 'broken': 177, 'genie': 560, 'disappointment': 373, 'complaints': 267, 'unsanitary': 1483, 'putting': 1091, 'wipes': 1565, 'slot': 1257, 'dumping': 407, 'bucket': 180, 'germy': 563, 'mess': 847, 'unrealistic': 1482, 'fear': 484, 'toddler': 1431, 'near': 896, 'germs': 562, 'let': 769, 'frightening': 546, 'experience': 468, 'yikes': 1597, 'prevention': 1058, 'prepared': 1055, 'opened': 953, 'seals': 1198, 'individual': 688, 'sea': 1196, 'ones': 950, 'certainly': 211, 'arrives': 85, 'year': 1593, 'll': 789, 'cleaner': 241, 'involved': 707, 'version': 1504, 'operate': 955, 'model': 865, 'cross': 321, 'stays': 1316, 'friends': 545, 'houses': 664, 'fought': 538, 'genies': 561, 'warm': 1524, 'appeared': 74, 'lysol': 811, 'various': 1502, 'tricks': 1460, 'mask': 826, 'emanating': 429, 'luck': 809, 'breast': 169, 'fed': 486, 'imagine': 674, 'solids': 1274, 'ick': 670, 'wondering': 1575, 'fighting': 496, 'option': 958, 'air': 45, 'conditioning': 274, 'summer': 1349, 'reviewers': 1158, 'complained': 265, 'completely': 269, 'convenient': 298, 'drag': 395, 'extremely': 472, 'horrible': 658, 'please': 1030, 'figured': 498, 'protection': 1076, 'desired': 353, 'badly': 109, 'multiple': 886, 'nasty': 894, 'spew': 1294, 'forth': 536, 'sealed': 1197, 'somewhat': 1280, 'correctly': 303, 'concept': 270, 'fabulous': 475, 'cons': 278, 'area': 79, 'addition': 33, 'odorless': 935, 'notice': 924, 'worse': 1583, 'cost': 305, 'savings': 1185, 'began': 129, 'cylinder': 329, 'continually': 293, 'fun': 551, 'apart': 72, '00': 0, 'placed': 1020, 'hole': 651, 'sometimes': 1279, 'dealing': 341, 'flipped': 519, 'fell': 494, 'task': 1377, 'retrieve': 1154, 'figure': 497, 'fix': 514, '1st': 6, 'hauled': 622, 'pickup': 1014, 'lasted': 750, 'recommend': 1117, 'newborn': 911, 'within': 1569, 'liner': 783, 'assume': 91, 'stickies': 1319, 'anymore': 68, 'groceries': 593, 'perfectly': 1003, 'deal': 340, 'babi': 102, 'italia': 713, 'pinehurst': 1017, 'classic': 239, 'crib': 318, 'ultimate': 1476, 'backup': 107, 'wash': 1526, 'problems': 1068, 'fitting': 512, 'rails': 1099, 'forced': 526, 'attach': 94, 'snaps': 1269, 'stretch': 1330, 'flat': 515, 'today': 1430, 'marks': 825, 'paint': 985, 'ah': 44, 'improvised': 680, 'rail': 1098, 'length': 767, 'actual': 30, 'usually': 1499, 'teethes': 1383, 'based': 116, 'bargains': 115, 'expectations': 463, 'night': 914, 'unsnap': 1484, '10': 1, 'places': 1021, 'engage': 437, 'bumpers': 181, 'elastic': 426, 'slats': 1249, 'attractive': 96, 'miracle': 863, 'hoped': 655, 'trouble': 1464, 'lifting': 775, 'lightweight': 777, 'foam': 521, 'result': 1152, 'seconds': 1203, 'snapping': 1268, 'ties': 1422, 'bending': 134, 'solid': 1273, 'minutes': 862, 'manufacturer': 820, 'recommends': 1119, 'means': 839, 'snap': 1267, 'suggest': 1343, 'coil': 249, 'center': 209, 'feeding': 488, 'schedule': 1190, 'life': 771, 'doctor': 382, 'questions': 1094, 'habits': 600, 'saver': 1183, 'trends': 1459, 'answer': 66, 'pediatrician': 996, 'communicate': 261, 'everyone': 455, 'required': 1143, 'leave': 763, 'finish': 507, 'haves': 625, 'helps': 635, 'exactly': 458, 'gone': 580, 'mother': 876, 'watching': 1535, 'happier': 613, 'routine': 1169, 'sitter': 1242, 'helped': 632, 'prepare': 1054, 'evening': 451, 'likely': 780, 'sick': 1234, 'many': 821, 'producing': 1069, 'dehydrated': 346, 'note': 920, 'writes': 1589, 'whether': 1554, 'lunch': 810, 'playtime': 1028, 'included': 684, 'walk': 1517, 'moms': 868, 'wanting': 1522, 'kids': 731, 'dads': 331, 'lol': 793, 'alternative': 56, 'printing': 1064, 'searching': 1199, 'crumpled': 323, 'piece': 1015, 'previous': 1059, 'preferred': 1053, 'held': 630, 'basics': 117, 'wish': 1566, 'struggle': 1333, 'caretaker': 196, 'wrote': 1592, 'spend': 1293, 'neighbor': 907, 'loosely': 801, 'developing': 356, 'milk': 856, 'cohesion': 248, 'visits': 1514, 'brand': 167, 'books': 152, 'absolute': 20, 'trackers': 1448, 'available': 97, 'naps': 893, 'tracks': 1450, 'nights': 915, 'important': 676, 'during': 408, 'caregiver': 195, 'postpartum': 1047, 'nurses': 930, 'urination': 1489, 'remind': 1137, 'overwhelmed': 973, 'remember': 1135, 'cried': 319, 'major': 814, 'contact': 287, 'call': 188, 'trend': 1458, 'indication': 687, 'suggested': 1344, 'pumping': 1083, 'supplement': 1352, 'formula': 535, 'sufficient': 1342, 'water': 1536, 'drink': 396, 'supply': 1353, 'breastmilk': 172, 'wasn': 1530, 'giving': 572, 'gas': 556, 'ate': 93, 'food': 522, 'allergy': 48, 'peanut': 995, 'family': 480, 'smile': 1266, 'visit': 1512, 'ect': 419, 'memory': 846, 'babysitter': 105, 'grandma': 586, 'goes': 578, 'recorded': 1121, 'diary': 363, 'certain': 210, 'suit': 1345, 'rough': 1167, 'refer': 1124, 'forgot': 529, 'woke': 1571, 'emergency': 431, 'consent': 280, 'form': 531, 'needs': 905, 'immunizations': 675, 'info': 692, 'glance': 574, 'developmental': 358, 'organized': 963, 'create': 317, 'spreadsheets': 1297, 'people': 1000, 'practical': 1049, 'pees': 998, 'poops': 1041, 'breastfeed': 170, 'especially': 448, 'adults': 37, 'sharing': 1223, 'responsibilities': 1148, 'wake': 1516, 'eaten': 416, 'written': 1590, 'record': 1120, 'dr': 394, 'appts': 77, 'analyze': 62, 'asks': 89, 'urinating': 1488, 'realizing': 1112, 'oh': 942, 'hasn': 619, 'slept': 1254, 'number': 927, 'done': 389, 'looked': 797, 'pattern': 992, 'routines': 1171, 'jot': 721, 'knew': 735, 'beginning': 130, 'transfer': 1453, 'everything': 456, 'funny': 552, 'forgotten': 530, 'born': 154, 'pee': 997, 'poopy': 1042, 'asking': 88, 'per': 1001, 'bring': 174, 'color': 250, 'code': 247, 'pottied': 1048, 'ordering': 961, 'ummmm': 1477, 'pain': 983, 'killers': 732, 'mass': 828, 'hormones': 657, 'slowly': 1259, 'lst': 808, 'minds': 859, 'handy': 609, 'eats': 418, 'discuss': 375, 'progress': 1072, 'grandparents': 587, 'watch': 1533, 'doing': 387, 'highly': 640, 'continue': 294, 'delivery': 347, 'saw': 1186, 'ends': 436, 'swear': 1360, 'throughout': 1415, 'mile': 854, 'stone': 1323, '8230': 16, 'outings': 969, 'temperature': 1386, 'readings': 1108, 'calls': 189, 'stopped': 1324, 'swaddling': 1359, 'resource': 1146, 'later': 751, 'formally': 532, 'album': 46, 'valuable': 1500, 'tidbits': 1420, 'memories': 845, 'rummage': 1173, 'sale': 1178, 'helping': 634, 'ex': 457, 'visiting': 1513, 'almost': 52, 'filled': 500, 'development': 357, 'story': 1327, 'dad': 330, 'connected': 277, 'everyday': 454, 'medicine': 842, 'give': 569, 'rundown': 1175, 'three': 1412, 'issues': 711, 'necessary': 899, 'explain': 469, 'behavior': 131, 'lactation': 743, 'consultants': 286, 'contributed': 296, 'deprivation': 350, 'dark': 334, 'respond': 1147, 'referring': 1125, 'tasks': 1378, 'include': 683, 'finished': 508, 'knowing': 738, 'meal': 837, 'predicting': 1051, 'communication': 262, 'childs': 229, 'learn': 760, 'growth': 596, 'laid': 745, 'set': 1217, 'nitpick': 916, 'user': 1496, 'identical': 672, 'holding': 649, 'lean': 759, 'forward': 537, 'happend': 611, 'prior': 1065, 'remembering': 1136, 'wasi': 1529, 'kept': 729, 'moods': 872, 'became': 122, 'tool': 1436, 'documenting': 383, 'recording': 1122, 'diapering': 361, 'session': 1216, 'wayside': 1538, 'finding': 504, 'flexible': 517, 'lifestyle': 773, 'nursing': 931, 'move': 881, 'peruse': 1007, 'secondary': 1202, 'account': 24, 'talking': 1371, 'personas': 1006, 'christmas': 235, 'february': 485, 'restless': 1150, 'blending': 147, '8217': 15, 'religiously': 1132, 'stored': 1326, 'defiantly': 344, 'tricky': 1461, 'print': 1063, 'light': 776, 'grey': 592, 'bottles': 157, 'meds': 843, 'challenge': 213, 'detailed': 354, 'slots': 1258, 'prefer': 1052, 'chooses': 233, 'sleeps': 1253, 'bath': 118, 'doc': 381, 'appt': 76, 'thank': 1391, 'stayed': 1315, 'suitcase': 1346, 'loose': 800, 'output': 970, 'crucial': 322, 'iphone': 708, 'proved': 1078, 'annoying': 64, 'walked': 1518, 'scrap': 1193, 'inaccurate': 682, 'update': 1487, 'continued': 295, 'control': 297, 'add': 32, 'rely': 1133, '3mo': 10, 'chair': 212, 'jotting': 722, 'scheduling': 1192, 'twins': 1473, 'four': 540, 'givers': 570, 'inside': 694, 'feed': 487, 'intervals': 702, '30': 8, 'ribbon': 1160, 'movable': 880, 'tab': 1362, 'advertised': 38, 'keeps': 728, 'sane': 1180, 'esp': 447, 'realistic': 1109, 'filling': 501, 'download': 393, 'embrace': 430, 'world': 1581, 'entering': 441, 'tabulate': 1365, 'charts': 219, 'review': 1156, 'myself': 889, 'daycare': 338, 'workers': 1578, 'offended': 939, 'explaining': 470, 'usefulness': 1494, 'greatest': 591, 'inventions': 706, 'relying': 1134, 'learning': 761, 'overwhelming': 974, 'history': 646, 'reactions': 1105, 'coupled': 310, 'itzbeen': 717, 'pocket': 1035, 'timer': 1426, 'necessity': 900, 'lost': 802, 'mind': 857, 'takers': 1368, 'picking': 1013, 'run': 1174, 'anyone': 69, 'expecting': 465, 'items': 715, 'intake': 698, 'organize': 962, 'lifesaver': 772, 'exhausted': 462, 'sides': 1236, 'plenty': 1032, 'handing': 605, 'parent': 988, 'whoever': 1558, 'ran': 1100, 'wishing': 1567, 'researching': 1145, 'settled': 1218, 'regret': 1129, 'laugh': 752, 'progression': 1073, 'toward': 1443, 'schedules': 1191, 'count': 308, 'introduced': 705, 'foods': 523, 'offered': 940, 'signs': 1237, 'intolerance': 704, 'arrival': 83, 'seasoned': 1200, 'therapist': 1398, 'brings': 175, 'ladder': 744, 'fire': 509, 'truck': 1465, 'bored': 153, 'roll': 1165, 'nesting': 908, 'improved': 678, 'cars': 200, 'random': 1101, 'teach': 1379}
  (0, 318)	2
  (0, 1408)	1
  (0, 1525)	1
  
'''
  
NAIVE BAYES CLASSIFIER
Using scikit-learn
Now that we’ve formatted our data correctly, we can use it using scikit-learn’s MultinomialNB classifier.

This classifier can be trained using the .fit() method. .fit() takes two parameters: The array of data points (which we just made) and an array of labels corresponding to each data point.

Finally, once the model has been trained, we can use the .predict() method to predict the labels of new points. .predict() takes a list of points that you want to classify and it returns the predicted labels of those points.

Finally, .predict_proba() will return the probability of each label given a point. Instead of just returning whether the review was good or bad, it will return the likelihood of a good or bad review.

Note that in the code editor, we’ve imported some of the variables you created last time. Specifically, we’ve imported the counter object, training_counts and then make review_counts. This means the program won’t have to re-create those variables and should help the runtime of your program.

Instructions
1.
Begin by making a MultinomialNB object called classifier.


This should be similar to how you created the CountVectorizer in the previous exercise.

2.
We now want to fit the classifier. We have the transformed points (found in training_counts), but we don’t have the labels associated with those points.

We made the training points by combining neg_list and pos_list. So the first half of the labels should be 0 (for negative) and the second half should be 1 (for positive).

Create a list named training_labels that has 1000 0s followed by 1000 1s.

Note that there are 1000 negative and 1000 positive reviews. Normally you could find this out by asking for the length of your dataset — in this example, we haven’t included the dataset because it takes so long to load!


You can use the * operator to quickly make this list. We’ve shown you how to add the 0s. Add the 1s.

training_labels = [0] * 1000 + ____
3.
Call classifier‘s .fit() function. Fit takes two parameters: the training set and the training labels.


Your training set is found in the variable training_counts and the training labels are found in training_labels.

4.
Call classifier‘s .predict() method and print the results. This method takes a list of the points that you want to test.

Was your review classified as a positive or negative review?


Your test points are found in the variable review_counts.

From the way we constructed our labels, 0 is bad and 1 is good.

5.
After printing predict, print a call to the predict_proba method. The parameter to predict_proba should be the same as predict.

The first number printed is the probability that the review was a 0 (bad) and the second number is the probability the review was a 1 (good).

6.
Change the text review to see the probabilities change.

Can you create a review that the algorithm is really confident about being positive?

The review "This crib was great amazing and wonderful" had the following probabilities:

[[ 0.04977729 0.95022271]]

Can you create a review that is even more positive?

Another interesting challenge is to create a clearly negative review that our classifier thinks is positive.


Think of words that would only be in positive or negative reviews and and them to review.'''

from reviews import counter, training_counts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#review = "This crib was amazing"
review = 'This crib was great amazing and wonderful'
review_counts = counter.transform([review])

classifier = MultinomialNB()
neg_label = [0 for i in range(1000)]
pos_label = [1 for i in range(1000)]
training_labels = neg_label + pos_label

classifier.fit(training_counts, training_labels)

print(classifier.predict(review_counts))

print(classifier.predict_proba(review_counts))


'''
NAIVE BAYES CLASSIFIER
Review
In this lesson, you’ve learned how to leverage Bayes’ Theorem to create a supervised machine learning algorithm. Here are some of the major takeaways from the lesson:

A tagged dataset is necessary to calculate the probabilities used in Bayes’ Theorem.
In this example, the features of our dataset are the words used in a product review. In order to apply Bayes’ Theorem, we assume that these features are independent.
Using Bayes’ Theorem, we can find P(class|data point) for every possible class. In this example, there were two classes — positive and negative. The class with the highest probability will be the algorithm’s prediction.
Even though our algorithm is running smoothly, there’s always more that we can add to try to improve performance. The following techniques are focused on ways in which we process data before feeding it into the Naive Bayes classifier:

Remove punctuation from the training set. Right now in our dataset, there are 702 instances of "great!" and 2322 instances of "great.". We should probably combine those into 3024 instances of "great".
Lowercase every word in the training set. We do this for the same reason why we remove punctuation. We want "Great" and "great" to be the same.
Use a bigram or trigram model. Right now, the features of a review are individual words. For example, the features of the point “This crib is great” are “This”, “crib”, “is”, and “great”. If we used a bigram model, the features would be “This crib”, “crib is”, and “is great”. Using a bigram model makes the assumption of independence more reasonable.
These three improvements would all be considered part of the field Natural Language Processing.

You can find the baby product review dataset, along with many others, on Dr. Julian McAuley’s website.
http://jmcauley.ucsd.edu/data/amazon/
'''


from reviews import baby_counter, baby_training, instant_video_counter, instant_video_training, video_game_counter, video_game_training
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

review = "this game was violent"

baby_review_counts = baby_counter.transform([review])
instant_video_review_counts = instant_video_counter.transform([review])
video_game_review_counts = video_game_counter.transform([review])

baby_classifier = MultinomialNB()
instant_video_classifier = MultinomialNB()
video_game_classifier = MultinomialNB()

baby_labels = [0] * 1000 + [1] * 1000
instant_video_labels = [0] * 1000 + [1] * 1000
video_game_labels = [0] * 1000 + [1] * 1000


baby_classifier.fit(baby_training, baby_labels)
instant_video_classifier.fit(instant_video_training, instant_video_labels)
video_game_classifier.fit(video_game_training, video_game_labels)

print("Baby training set: " +str(baby_classifier.predict_proba(baby_review_counts)))
print("Amazon Instant Video training set: " + str(instant_video_classifier.predict_proba(instant_video_review_counts)))
print("Video Games training set: " + str(video_game_classifier.predict_proba(video_game_review_counts)))

'''
MACHINE LEARNING: SUPERVISED LEARNING 🤖
Email Similarity
In this project, you will use scikit-learn’s Naive Bayes implementation on several different datasets. By reporting the accuracy of the classifier, we can find which datasets are harder to distinguish. For example, how difficult do you think it is to distinguish the difference between emails about hockey and emails about soccer? How hard is it to tell the difference between emails about hockey and emails about tech? In this project, we’ll find out exactly how difficult those two tasks are.

If you get stuck during this project or would like to see an experienced developer work through it, click “Get Help“ to see a project walkthrough video.

Tasks
15/15Complete
Mark the tasks as complete by checking them off
Exploring the Data
1.
We’ve imported a dataset of emails from scikit-learn’s datasets. All of these emails are tagged based on their content.

Print emails.target_names to see the different categories.

2.
We’re interested in seeing how effective our Naive Bayes classifier is at telling the difference between a baseball email and a hockey email. We can select the categories of articles we want from fetch_20newsgroups by adding the parameter categories.

In the function call, set categories equal to the list ['rec.sport.baseball', 'rec.sport.hockey']


The function should now look like this:
'''
emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'])
'''
3.
Let’s take a look at one of these emails.

All of the emails are stored in a list called emails.data. Print the email at index 5 in the list.

'''
print(emails.data[5])
'''
4.
All of the labels can be found in the list emails.target. Print the label of the email at index 5.

The labels themselves are numbers, but those numbers correspond to the label names found at emails.target_names.

Is this a baseball email or a hockey email?


Instead of printing index 5 from emails.data, print it from emails.target.

Print emails.target_names to see what that number corresponds to.

The target of email 5 is 1, which corresponds to rec.sport.hockey.

Making the Training and Test Sets
5.
We now want to split our data into training and test sets. Change the name of your variable from emails to train_emails. Add these three parameters to the function call:

subset='train'
shuffle = True
random_state = 108
Adding the random_state parameter will make sure that every time you run the code, your dataset is split in the same way.


Adding the subset parameter will make your function call look like this:
'''
train_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], subset = 'train')
'''
Add the other two parameters in addition to subset.

6.
Create another variable named test_emails and set it equal to fetch_20newsgroups. The parameters of the function should be the same as before except subset should now be 'test'.

Counting Words
7.
We want to transform these emails into lists of word counts. The CountVectorizer class makes this easy for us.

Create a CountVectorizer object and name it counter.

'''
counter = CountVectorizer()
'''
8.
We need to tell counter what possible words can exist in our emails. counter has a .fit() a function that takes a list of all your data.

Call .fit() with test_emails.data + train_emails.data as a parameter.


Fill test_emails.data + train_emails.data into the blank
'''
counter.fit(_____)
'''
9.
We can now make a list of the counts of our words in our training set.

Create a variable named train_counts. Set it equal to counter‘s transform function using train_emails.data as a parameter.


Use the apporpriate parameter in the function below:
'''
train_counts = counter.transform(_____)
'''
10.
Let’s also make a variable named test_counts. This should be the same function call as before, but use test_emails.data as the parameter of transform.

Making a Naive Bayes Classifier
11.
Let’s now make a Naive Bayes classifier that we can train and test on. Create a MultinomialNB object named classifier.


This is similar to how you created counter. Instead of making a CountVectorizer, create a MultinomialNB

12.
Call classifier‘s .fit() function. .fit() takes two parameters. The first should be our training set, which for us is train_counts. The second should be the labels associated with the training emails. Those are found in train_emails.target.


Fill train_counts and train_emails.target into the two blanks:
'''
classifier.fit(_____, _____)
'''
13.
Test the Naive Bayes Classifier by printing classifier‘s .score() function. .score() takes the test set and the test labels as parameters.

.score() returns the accuracy of the classifier on the test data. Accuracy measures the percentage of classifications a classifier correctly made.


The two parameters to .score() should be test_counts and test_emails.target.

Make sure to print this function call:
'''
print(classifier.score(____, ____))
'''
.score() will classify all the emails in the test set and compare the classification of each email to its actual label. After completing these comparisons, it will calculate and return the accuracy.

Testing Other Datasets
14.
Our classifier does a pretty good job distinguishing between soccer emails and hockey emails. But let’s see how it does with emails about really different topics.

Find where you create train_emails and test_emails. Change the categories to be ['comp.sys.ibm.pc.hardware','rec.sport.hockey'].

Did your classifier do a better or worse job on these two datasets?


The classifier was 99% accurate when trying to classify hockey and tech emails.

This is better than when it was trying to classify hockey and soccer emails. This makes sense — emails about sports probably share more words in common.

15.
Play around with different sets of data. Can you find a set that’s incredibly accurate or incredibly inaccurate?

The possible categories are listed below.

'alt.atheism'
'comp.graphics'
'comp.os.ms-windows.misc'
'comp.sys.ibm.pc.hardware'
'comp.sys.mac.hardware'
'comp.windows.x'
'misc.forsale'
'rec.autos'
'rec.motorcycles'
'rec.sport.baseball'
'rec.sport.hockey'
'sci.crypt'
'sci.electronics'
'sci.med'
'sci.space'
'soc.religion.christian'
'talk.politics.guns'
'talk.politics.mideast'
'talk.politics.misc'
'talk.religion.misc'

Your classifier can work even when there are more than two labels. Try setting categories equal to a list of three or four of the categories.'''

from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'])

#print(emails.target_names)
#print(type(emails))
#print(emails)

#print(len(emails.data))
#print(emails.data[10])

#print(emails.target_names)

#test between baseball and hockey
#train_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], subset = 'train', shuffle = True, random_state = 108)
#test_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], subset = 'test', shuffle = True, random_state = 108)

#test between Technology and hockey
#train_emails = fetch_20newsgroups(categories = ['comp.sys.ibm.pc.hardware', 'rec.sport.hockey'], subset = 'train', shuffle = True, random_state = 108)
#test_emails = fetch_20newsgroups(categories = ['comp.sys.ibm.pc.hardware', 'rec.sport.hockey'], subset = 'test', shuffle = True, random_state = 108)

#play around with different topic
train_emails = fetch_20newsgroups(categories = ['comp.graphics', 'talk.politics.mideast'], subset = 'train', shuffle = True, random_state = 108)
test_emails = fetch_20newsgroups(categories = ['comp.graphics', 'talk.politics.mideast'], subset = 'test', shuffle = True, random_state = 108)


counter = CountVectorizer()
counter.fit(train_emails.data + test_emails.data)
train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)

classifier = MultinomialNB()
classifier.fit(train_counts, train_emails.target)
print(classifier.score(test_counts, test_emails.target))




#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#- MACHINE LEARN NAIVE BAYES MACHINE LEARN NAIVE BAYES #-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#- PROJECT PROJECT PROJECT PROJECT #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

'''
Off-Platform Project: Classifying Tweets

In this off-platform project, you will use a Naive Bayes Classifier to find patterns in real tweets. We've given you three files: new_york.json, london.json, and paris.json. These three files contain tweets that we gathered from those locations.

The goal is to create a classification algorithm that can classify any tweet (or sentence) and predict whether that sentence came from New York, London, or Paris.



In this off-platform project, you will use a Naive Bayes Classifier to find patterns in real tweets. We've given you three fileInvestigate the Data¶

To begin, let's take a look at the data. We've imported new_york.json and printed the following information:
•The number of tweets.
•The columns, or features, of a tweet.
•The text of the 12th tweet in the New York dataset.

Run the cell below to see the results
'''


import pandas as pd

new_york_tweets = pd.read_json("new_york.json", lines=True)
print(len(new_york_tweets))
print(new_york_tweets.columns)
print(new_york_tweets.loc[12]["text"])


'''RESUTL
4723

Index(['created_at', 'id', 'id_str', 'text', 'display_text_range', 'source',
       'truncated', 'in_reply_to_status_id', 'in_reply_to_status_id_str',
       'in_reply_to_user_id', 'in_reply_to_user_id_str',
       'in_reply_to_screen_name', 'user', 'geo', 'coordinates', 'place',
       'contributors', 'is_quote_status', 'quote_count', 'reply_count',
       'retweet_count', 'favorite_count', 'entities', 'favorited', 'retweeted',
       'filter_level', 'lang', 'timestamp_ms', 'extended_tweet',
       'possibly_sensitive', 'quoted_status_id', 'quoted_status_id_str',
       'quoted_status', 'quoted_status_permalink', 'extended_entities',
       'withheld_in_countries'],
      dtype='object')
Be best #ThursdayThoughts'''

'''
In the code block below, load the London and Paris tweets into DataFrames named london_tweets and paris_tweets.

How many London tweets are there? How many Paris ones are there?'''

​london_tweets = pd.read_json('london.json', lines = True)
paris_tweets = pd.read_json('paris.json', lines = True)

print(len(london_tweets))

print(london_tweets.columns)

print(len(paris_tweets))

print(paris_tweets.columns)

'''RESUTL
5341
Index(['created_at', 'id', 'id_str', 'text', 'display_text_range', 'source',
       'truncated', 'in_reply_to_status_id', 'in_reply_to_status_id_str',
       'in_reply_to_user_id', 'in_reply_to_user_id_str',
       'in_reply_to_screen_name', 'user', 'geo', 'coordinates', 'place',
       'contributors', 'is_quote_status', 'extended_tweet', 'quote_count',
       'reply_count', 'retweet_count', 'favorite_count', 'entities',
       'favorited', 'retweeted', 'filter_level', 'lang', 'timestamp_ms',
       'possibly_sensitive', 'quoted_status_id', 'quoted_status_id_str',
       'quoted_status', 'quoted_status_permalink', 'extended_entities'],
      dtype='object')
2510
Index(['created_at', 'id', 'id_str', 'text', 'source', 'truncated',
       'in_reply_to_status_id', 'in_reply_to_status_id_str',
       'in_reply_to_user_id', 'in_reply_to_user_id_str',
       'in_reply_to_screen_name', 'user', 'geo', 'coordinates', 'place',
       'contributors', 'is_quote_status', 'quote_count', 'reply_count',
       'retweet_count', 'favorite_count', 'entities', 'favorited', 'retweeted',
       'filter_level', 'lang', 'timestamp_ms', 'display_text_range',
       'extended_entities', 'possibly_sensitive', 'quoted_status_id',
       'quoted_status_id_str', 'quoted_status', 'quoted_status_permalink',
       'extended_tweet'],
      dtype='object')
      '''
      
'''
Classifying using language: Naive Bayes Classifier

We're going to create a Naive Bayes Classifier! Let's begin by looking at the way language is used differently in these three locations. Let's grab the text of all of the tweets and make it one big list. In the code block below, we've created a list of all the New York tweets. Do the same for london_tweets and paris_tweets.

Then combine all three into a variable named all_tweets by using the + operator. For example, all_tweets = new_york_text + london_text + ...

Let's also make the labels associated with those tweets. 0 represents a New York tweet, 1 represents a London tweet, and 2 represents a Paris tweet. Finish the definition of labels.'''

new_york_text = new_york_tweets["text"].tolist()
london_text = london_tweets['text'].tolist()
paris_text = paris_tweets['text'].tolist()

all_tweets = new_york_text + london_text + paris_text
labels = [0] * len(new_york_text) + [1]*len(london_text) + [2]*len(paris_text)

#print(new_york_text)
#print(labels)

'''Making a Training and Test Set

We can now break our data into a training set and a test set. We'll use scikit-learn's train_test_split function to do this split. This function takes two required parameters: It takes the data, followed by the labels. Set the optional parameter test_size to be 0.2. Finally, set the optional parameter random_state to 1. This will make it so your data is split in the same way as the data in our solution code. 

Remember, this function returns 4 items in this order:
1.The training data
2.The testing data
3.The training labels
4.The testing labels

Store the results in variables named train_data, test_data, train_labels, and test_labels.

Print the length of train_data and the length of test_data.'''


from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(all_tweets, labels, train_size = 0.8, test_size = 0.2 , random_state = 1)

print(len(train_data))

print(len(test_data))

#10059
#2515

'''
Making the Count Vectors

To use a Naive Bayes Classifier, we need to transform our lists of words into count vectors. Recall that this changes the sentence "I love New York, New York" into a list that contains:
•Two 1s because the words "I" and "love" each appear once.
•Two 2s because the words "New" and "York" each appear twice.
•Many 0s because every other word in the training set didn't appear at all.

To start, create a CountVectorizer named counter.

Next, call the .fit() method using train_data as a parameter. This teaches the counter our vocabulary.

Finally, let's transform train_data and test_data into Count Vectors. Call counter's .transform() method using train_data as a parameter and store the result in train_counts. Do the same for test_data and store the result in test_counts.

Print train_data[3] and train_counts[3] to see what a tweet looks like as a Count Vector.'''

from sklearn.feature_extraction.text import CountVectorizer

counter = CountVectorizer()

counter.fit(train_data)

train_counts = counter.transform(train_data)
test_counts = counter.transform(test_data)

print(train_data[3])
print(train_counts[3])

'''result
saying bye is hard. Especially when youre saying bye to comfort.
  (0, 5022)	2
  (0, 6371)	1
  (0, 9552)	1
  (0, 12314)	1
  (0, 13903)	1
  (0, 23994)	2
  (0, 27146)	1
  (0, 29397)	1
  (0, 30274)	1'''
  
'''Train and Test the Naive Bayes Classifier

We now have the inputs to our classifier. Let's use the CountVectors to train and test the Naive Bayes Classifier!

First, make a MultinomialNB named classifier.

Next, call classifier's .fit() method. This method takes two parameters — the training data and the training labels. train_counts contains the training data and train_labels containts the labels for that data.

Calling .fit() calculates all of the probabilities used in Bayes Theorem. The model is now ready to quickly predict the location of a new tweet. 

Finally, let's test our model. classifier's .predict() method using test_counts as a parameter. Store the results in a variable named predictions.
'''

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()

classifier.fit(train_counts, train_labels)
predictions = classifier.predict(test_counts)

'''Evaluating Your Model

Now that the classifier has made its predictions, let's see how well it did. Let's look at two different ways to do this. First, call scikit-learn's accuracy_score function. This function should take two parameters — the test_labels and your predictions. Print the results. This prints the percentage of tweets in the test set that the classifier correctly classified.
'''

from sklearn.metrics import accuracy_score

print(accuracy_score(test_labels, predictions))

#0.6779324055666004

'''The other way you can evaluate your model is by looking at the confusion matrix. A confusion matrix is a table that describes how your classifier made its predictions. For example, if there were two labels, A and B, a confusion matrix might look like this:
9 1
3 5

In this example, the first row shows how the classifier classified the true A's. It guessed that 9 of them were A's and 1 of them was a B. The second row shows how the classifier did on the true B's. It guessed that 3 of them were A's and 5 of them were B's.

For our project using tweets, there were three classes — 0 for New York, 1 for London, and 2 for Paris. You can see the confustion matrix by printing the result of the confusion_matrix function using test_labels and predictions as parameters.
'''

from sklearn.metrics import confusion_matrix

print(confusion_matrix(test_labels, predictions))

'''
[[541 404  28]
 [203 824  34]
 [ 38 103 340]]
'''
'''
Test Your Own Tweet¶

Nice work! The confusion matrix should line up with your intuition. The classifier predicts tweets that were actually from New York as either New York tweets or London tweets, but almost never Paris tweets. Similarly, the classifier rarely misclassifies the tweets that were actually from Paris. Tweets coming from two English speaking countries are harder to distinguish than tweets in different languages.

Now it's your chance to write a tweet and see how the classifier works! Create a string and store it in a variable named tweet. 

Call counter's .transform() method using [tweet] as a parameter. Save the result as tweet_counts. Notice that your variable has to be in an array — .transform() can't take just a string, it must be a list. 

Finally, pass tweet_counts as parameter to classifier's .predict() method. Print the result. This should give you the prediction for the tweet. Remember a 0 represents New York, a 1 represents London, and a 2 represents Paris. Can you write different tweets that the classifier predicts as being from New York, London, and Paris?'''

tweet = "The Statue of Liberty is beautiful"
tweet_counts = counter.transform([tweet])
print(classifier.predict(tweet_counts))

#[0]
