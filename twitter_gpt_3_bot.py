# make all necesscary imports
import openai
import tweepy
from transformers import GPT2TokenizerFast
import numpy as np
import configparser

# set global parameters to be used later

initial_data = ["""Bring me your pain, love. Spread it out like fine rugs, silk sashes,warm eggs, cinnamon and 
        cloves in burlap sacks. Show me the detail, the intricate embroidery on the collar, tiny shell buttons, 
        the hem stitched the way you were taught," pricking just a thread, almost invisible. Unclasp it like jewels, 
        the gold still hot from your body. Empty your basket of figs. Spill your wine. That hard nugget of pain, 
        I would suck it, cradling it on my tongue like the slick seed of pomegranate. I would lift it tenderly, 
        as a great animal might carry a small one in the private cave of the mouth.""", """At the touch of you, 
                    As if you were an archer with your swift hand at the bow,
                    The arrows of delight shot through my body.
                    You were spring,
                    And I the edge of a cliff,
                    And a shining waterfall rushed over me.""", """I just wanted to let you know how much I 
                    appreciate having you in my life. For helping me through the bad times and being there to 
                    help me celebrate the good times, I cherish all of the moments that we share together. 
                    There aren’t enough words in the dictionary for me to tell you how glad I am to have you in my life. 
                    I am so lucky to have you by my side. Everything you do for me never goes unnoticed. 
                    I don’t know what I did to deserve someone as wonderful as you, 
                    but I am eternally grateful to have your love, support, and affection. 
                    Thank you for being you, and for having me by your side."""]

# make config object

config = configparser.ConfigParser(interpolation=None)
config.read('config.ini')

# read config file

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

open_ai_key = config['openai']['open_ai_key']
# create all required relevant tweepy objects


client = tweepy.Client(consumer_key=api_key, consumer_secret=api_key_secret, access_token=access_token,
                       access_token_secret=access_token_secret, wait_on_rate_limit=True)

# create relevant openai obects

openai.api_key = open_ai_key

# Load the tokenizer.

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Make sure the labels are formatted correctly.

labels = ["Positive", "Negative"]
labels = [label.strip().lower().capitalize() for label in labels]

# Encode the labels with extra white space prepended.

labels_tokens = {label: tokenizer.encode(" " + label) for label in labels}


# main proogram starts

# function to get searched tweets text and store it in a array

def get_tweets(query, leng):
    x = []
    for i in range(0, leng):
        x.append(client.search_recent_tweets(query=query, max_results=leng, user_auth=True)[0][i]['text'])
    return x


# function to search tweets from a specific account
def get_tweet_of(identity, leng):
    x = []
    for i in range(0, leng):
        x.append(client.get_users_tweets(id=identity, max_results=leng, user_auth=True)[0][i]['text'])
    return x


# function to write tweet to bot account while saving in list

def write_tweet(text):
    client.create_tweet(text=text, user_auth=True)


#  Create an examples file for classification model
"""
A = []
for i in get_tweets("Happiness", 15):
        A.append([i, 'Positive'])
for i in get_tweets("I wish to die", 15):
        A.append([i, 'Negative'])
"""


# function to write a tweet using gpt-3 model

def write(X, question):
    result = openai.Answer.create(
        search_model="davinci",
        model="davinci",
        documents=X,
        question=question,
        examples_context="Love is the purpose of existence.",
        examples=[["Where do lovers meet?", "Lovers don’t finally meet somewhere. They’re in each other all along."],
                  ["What is the first condition to make oneself a pilgrim of  love?",
                   "The first condition is that you make yourself humble as dust and ashes."],
                  ["What should one do with love?",
                   """Let yourself be silently drawn by the strange pull of what you really love. It will not lead you 
                   astray."""]],
        max_rerank=10,
        max_tokens=1000,
        temperature=1,
        stop=["\n", "<|endoftext|>"]
    )
    return result


# function to classify and do sentiment analysis on recored tweets

def classification(examp, tweet):
    result = openai.Classification.create(
        examples=examp,
        query=tweet,
        search_model="davinci",
        model="davinci",
        max_examples=10,
        labels=labels,
        logprobs=3,  # Here we set it to be len(labels) + 1, but it can be larger.
        expand=["completion"],
    )

    # Take the starting tokens for probability estimation.
    # Labels should have distinct starting tokens.
    # Here tokens are case-sensitive.

    first_token_to_label = {tokens[0]: label for label, tokens in labels_tokens.items()}

    top_logprobs = result["completion"]["choices"][0]["logprobs"]["top_logprobs"][0]
    token_probs = {
        tokenizer.encode(token)[0]: np.exp(logp)
        for token, logp in top_logprobs.items()
    }
    label_probs = {
        first_token_to_label[token]: prob
        for token, prob in token_probs.items()
        if token in first_token_to_label
    }

    # Fill in the probability for the special "Unknown" label.

    if sum(label_probs.values()) < 1.0:
        label_probs["Unknown"] = 1.0 - sum(label_probs.values())
    return label_probs


# function to classify a tweet as negative,positive or indeterminate

def classified_logits(tweet):
    if classification(A, tweet)['Negative'] > 0.5:
        return -1
    elif classification(A, tweet)['Positive'] > 0.5:
        return 1
    else:
        return 0


# function to classify an entire twitter account as negative,positive, or indeterminate

def classify_account(account_id):
    x = 0
    for i in get_tweet_of(account_id, 15):
        x = x + classified_logits(i)
    return x / 15


ques="Is life cruel?"
write_tweet("Question asked: "+ques+"\n\n"+write(initial_data,ques)['answers'][0])