from snorkel.labeling import labeling_function
import re

POSITIVE = 1
NEGATIVE = 0
ABSTAIN = -1

@labeling_function()
def has_love(x):
    return POSITIVE if "love" in x.text.lower() else ABSTAIN

@labeling_function()
def has_hate(x):
    return NEGATIVE if "hate" in x.text.lower() else ABSTAIN

@labeling_function()
def sentiment_good(x):
    return POSITIVE if any(word in x.text.lower() for word in ["great", "amazing", "fantastic", "brilliant", "awesome"]) else ABSTAIN

@labeling_function()
def sentiment_bad(x):
    return NEGATIVE if any(word in x.text.lower() for word in ["boring", "bad", "awful", "terrible", "worst"]) else ABSTAIN

@labeling_function()
def has_excellent(x):
    return POSITIVE if "excellent" in x.text.lower() else ABSTAIN

@labeling_function()
def has_terrible(x):
    return NEGATIVE if "terrible" in x.text.lower() else ABSTAIN

@labeling_function()
def not_bad(x):
    return POSITIVE if "not bad" in x.text.lower() else ABSTAIN

@labeling_function()
def very_disappointing(x):
    return NEGATIVE if "very disappointing" in x.text.lower() else ABSTAIN

@labeling_function()
def outstanding(x):
    return POSITIVE if "outstanding" in x.text.lower() else ABSTAIN

@labeling_function()
def dull_or_slow(x):
    return NEGATIVE if re.search(r"\bdull\b|\bslow\b", x.text.lower()) else ABSTAIN

@labeling_function()
def engaging(x):
    return POSITIVE if "engaging" in x.text.lower() else ABSTAIN

@labeling_function()
def waste_time(x):
    return NEGATIVE if "waste of time" in x.text.lower() else ABSTAIN

@labeling_function()
def must_watch(x):
    return POSITIVE if "must watch" in x.text.lower() else ABSTAIN

# Group all LFs into a list
lfs = [
    has_love,
    has_hate,
    sentiment_good,
    sentiment_bad,
    has_excellent,
    has_terrible,
    not_bad,
    very_disappointing,
    outstanding,
    dull_or_slow,
    engaging,
    waste_time,
    must_watch,
]
