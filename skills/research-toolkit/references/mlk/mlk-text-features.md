---
name: mlk-text-features
description: Text features for machine learning in pure Python (sklearn.feature_extraction.text + numpy/scipy, no nltk) - strip whitespace and punctuation, extract plain text from HTML, tokenise words and sentences, drop stop words, reduced Porter-style suffix stemming plus vocabulary-guarded lemmatisation, bag-of-words counts with n-grams and restricted vocabulary, TF-IDF weighting with the smoothed idf = log((1+nd)/(1+df))+1 and L2 row normalisation rebuilt from the formula and asserted on a six-document corpus, cosine document similarity, LSA/SVD latent topic features and a negation-aware lexicon sentiment score. Use when turning raw text, tweets or reviews into model features, when building CountVectorizer/TfidfVectorizer pipelines, when stemming or stop-word lists are asked for, or when nltk/gensim/textblob/bs4 are unavailable and the numerics must be rebuilt inline.
---

# Handling Text: Cleaning, Bag-of-Words and TF-IDF (Python)

## Purpose
Turn unstructured text into numeric ML features: clean and tokenise it, normalise words (stem or lemma), encode
each observation as bag-of-words counts or TF-IDF weights, then derive similarity, topic and sentiment features.

## Inputs
- `texts`: list/1-D array of raw strings, one per observation (or already-tokenised word lists).
- `STOP`, `RULES`, `VOCAB`, `POS/NEG`: small inline tables, nothing downloaded; every demo corpus is a literal
  string list or `numpy.random.default_rng(42)` output - files are never read.
- `K = 2` latent topics; sklearn defaults kept (`smooth_idf=True`, `norm="l2"`, sparse output).

## Algorithms (all pure numpy/scipy)
### Clean text, strip punctuation, extract HTML text (Albon ch.6 p110-113)
Core string work - `strip`/`replace`/`split` plus a custom `capitalizer`; `str.translate` over every Unicode `P*` code point is the fast punctuation strip, though punctuation itself carries signal (p113), and the tag regex stands in for BeautifulSoup's `.text` (p112).
```python
import re, sys, unicodedata
raw = ["   Interrobang. By Aishwarya Henriette     ", "Parking And Going. By Karl Gautier",
       "    Today Is The night. By Jarek Prakash   "]                              # Albon ch.6 p110
clean = [s.strip().replace(".", "").upper() for s in raw]         # strip, remove periods, capitalizer
assert clean[0] == "INTERROBANG BY AISHWARYA HENRIETTE" and clean[2] == "TODAY IS THE NIGHT BY JAREK PRAKASH"
assert [re.sub(r"[a-zA-Z]", "X", s) for s in clean][1] == "XXXXXXX XXX XXXXX XX XXXX XXXXXXX"  # p111
punct = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P"))
stripped = [s.translate(punct) for s in ["Hi!!!! I. Love. This. Song....", "10000% Agree!!!! #LoveIT", "Right?!?!"]]
assert stripped == ["Hi I Love This Song", "10000 Agree LoveIT", "Right"]                    # Albon ch.6 p113
text = " ".join(re.sub(r"<[^>]+>", " ", "<div class='full_name'><span style='font-weight:bold'>Masego</span> Azra</div>").split())
assert text == "Masego Azra" and "<" not in text and "full_name" not in text                  # Albon ch.6 p112
print(f"RESULT clean0={clean[0]!r} stripped={stripped} html_text={text!r}")
```
### Tokenize and remove stop words (Albon ch.6 p113-115)
A word token is letters plus an intra-word apostrophe, a sentence ends at `.!?` + space, and stop-word lists assume lowercased tokens; sklearn's built-in `english` list is larger than the inline one (p115).
```python
import re; from sklearn.feature_extraction.text import CountVectorizer
tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", "The science of today is the technology of tomorrow")
assert tokens == ["The", "science", "of", "today", "is", "the", "technology", "of", "tomorrow"]   # p114
sents = re.split(r"(?<=[.!?])\s+", "The science of today is the technology of tomorrow. Tomorrow is today.")
assert sents == ["The science of today is the technology of tomorrow.", "Tomorrow is today."]
STOP = set("a am an and are as at be but by for from has have he her him his i in into is it its me my"
           " not of on or our she so that the their them then there these they this to us we were".split())
kept = [w for w in ["i", "am", "going", "to", "go", "to", "the", "store", "and", "park"] if w not in STOP]
assert kept == ["going", "go", "store", "park"]                                 # Albon ch.6 p115 output
sk = CountVectorizer(stop_words="english").build_analyzer()("i am going to go to the store and park")
print(f"RESULT n_tokens={len(tokens)} n_sents={len(sents)} kept={kept} sklearn_english={sk}")
```
### Stem and lemmatise by rule (Albon ch.6 p115-116)
Stemming removes affixes so "tradition"/"traditional" share the stem "tradit" - less readable, more comparable (p116).
`stem` is a reduced Porter-style longest-suffix stripper (one pass plus the `-bl/-at/-iz`, final-`e` and doubled-consonant repairs), not bit-exact Porter; `lemmatise` only accepts a de-inflected form present in a base-form vocabulary.
```python
RULES = (("sses", "", 4), ("ies", "y", 3), ("ingly", "", 4), ("edly", "", 4), ("ings", "", 4), ("ing", "", 4),
         ("ed", "", 3), ("ions", "", 4), ("ion", "", 4), ("ness", "", 4), ("less", "", 4), ("ous", "", 4),
         ("ive", "", 4), ("ional", "", 4), ("al", "", 3), ("ic", "", 3), ("ly", "", 3), ("es", "", 3), ("s", "", 3))
def stem(word):
    for suf, repl, keep in RULES:
        if word.endswith(suf) and len(word) - len(suf) >= keep:
            word = word[: len(word) - len(suf)] + repl; break
    if word.endswith(("bl", "at", "iz")): word += "e"
    if len(word) >= 5 and word[-1] == "e" and word[-2] not in "aeiou": word = word[:-1]
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeioulsz": word = word[:-1]
    return word
toks = ["i", "am", "humbled", "by", "this", "traditional", "meeting"]                     # Albon ch.6 p115
assert [stem(t) for t in toks] == ["i", "am", "humbl", "by", "thi", "tradit", "meet"]     # book output p116
VOCAB = {"meeting", "city", "humble", "go", "store", "park"}; IRREG = {"went": "go", "better": "good"}
FORMS = (("ies", "y"), ("es", ""), ("s", ""), ("ed", "e"), ("ed", ""), ("ing", "e"), ("ing", ""))
def lemmatise(w): return next((w[: -len(s)] + r for s, r in FORMS if w.endswith(s) and w[: -len(s)] + r in VOCAB), IRREG.get(w, w))
assert [lemmatise(x) for x in ["went", "cities", "humbling"]] == ["go", "city", "humble"]
print(f"RESULT stemmed={[stem(t) for t in toks]} lemmas={[lemmatise(w) for w in ['meetings','cities','humbling','went','stores','park']]}")
```
### Bag of words, n-grams, restricted vocabulary (Albon ch.6 p119-121)
One feature per unique word holding its count - "I love Brazil. Brazil!" gives 2 in the "brazil" column (p119); the
matrix is sparse because most words are absent from most observations (p120); `ngram_range` makes features word pairs,
`stop_words` drops filler and `vocabulary` keeps only listed terms (Albon ch.6 p120-121).
```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
text_data = np.array(["I love Brazil. Brazil!", "Sweden is best", "Germany beats both"])
count = CountVectorizer(); bag = count.fit_transform(text_data)
assert bag.shape == (3, 8) and bag.nnz == 8 and type(bag).__name__ == "csr_matrix"      # Albon ch.6 p119
assert bag.toarray().tolist() == [[0, 0, 0, 2, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 0, 0]]
names = [str(c) for c in count.get_feature_names_out()]        # book's get_feature_names() was removed
assert names == ["beats", "best", "both", "brazil", "germany", "is", "love", "sweden"]  # Albon ch.6 p120
g2 = CountVectorizer(ngram_range=(1, 2), stop_words="english", vocabulary=["brazil"])
assert g2.fit_transform(text_data).toarray().tolist() == [[2], [0], [0]] and g2.vocabulary_ == {"brazil": 0}
print(f"RESULT matrix={bag.shape} nnz={bag.nnz} sparsity={1 - bag.nnz / np.prod(bag.shape):.3f} brazil_col={bag.toarray()[:, names.index('brazil')].tolist()} vocab={names}")
```
### TF-IDF, cosine similarity and LSA topics on a six-document corpus (Albon ch.6 p121-123)
`tf-idf(t,d) = tf(t,d) x idf(t)` with sklearn's tf = raw count, `idf(t) = log((1+nd)/(1+df(d,t)))+1` and L2 row normalisation, so a bigger value means a more important word (p122-123). The 3-document matrix reproduces the book's printed values; the formula is then rebuilt from counts on six documents, asserted equal to `TfidfVectorizer`, and SVD-factorised into `K = 2` dense topic features (Albon ch.6 p119-123).
```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
three = np.array(["I love Brazil. Brazil!", "Sweden is best", "Germany beats both"])
tv3 = TfidfVectorizer(); F = tv3.fit_transform(three).toarray(); fn = [str(c) for c in tv3.get_feature_names_out()]
assert abs(F[0, fn.index("brazil")] - 0.89442719) < 1e-8 and abs(F[0, fn.index("love")] - 0.4472136) < 1e-7
assert np.allclose(F[1], [0, 0.57735027, 0, 0, 0, 0.57735027, 0, 0.57735027], atol=1e-8)   # Albon ch.6 p122
docs = ["the stock market rose sharply today", "the bond market fell sharply today", "interest rates rose while the stock market stalled",
        "the football team won the match with a last minute goal", "the team scored a late goal and won the match",
        "a great goalkeeper saved the football match in extra time"]
tf = CountVectorizer().fit_transform(docs).toarray().astype(float)                 # tf = raw term counts
tv = TfidfVectorizer(); X = tv.fit_transform(docs).toarray(); K = 2                # sklearn reference matrix
tn = np.array([str(c) for c in tv.get_feature_names_out()]); nd = len(docs); df = (tf > 0).sum(axis=0)
idf = np.log((1 + nd) / (1 + df)) + 1.0; W = tf * idf; W = W / np.linalg.norm(W, axis=1, keepdims=True)
assert W.shape == (6, 29) and np.allclose(W, X, atol=1e-12)                        # the rebuild matches
assert df[tn == "the"][0] == nd and abs(idf[tn == "the"][0] - 1.0) < 1e-12         # in every doc -> idf 1
assert abs(W[0, tn == "market"][0] - 0.37734855) < 1e-8                            # df(market) = 3, idf 1.559616
cos = W @ W.T; assert cos[0, 1] > cos[0, 3] and abs(cos[0, 1] - 0.549377) < 1e-6   # unit rows: dot IS cosine
U, S, Vt = np.linalg.svd(X, full_matrices=False); T, loadings, evr = U[:, :K] * S[:K], Vt[:K], S[:K] ** 2 / (S ** 2).sum()
assert T.shape == (6, K) and np.allclose(U * S @ Vt, X, atol=1e-12) and np.all(np.sign(T[:3, 1]) == np.sign(T[0, 1])) and np.all(np.sign(T[3:, 1]) != np.sign(T[0, 1]))
print(f"RESULT docs={nd} vocab={W.shape[1]} idf_the={idf[tn == 'the'][0]:.6f} tfidf_market_d0={W[0, tn == 'market'][0]:.8f} cos_d0d1={cos[0, 1]:.6f} cos_d0d3={cos[0, 3]:.6f} evr={np.round(evr, 4).tolist()} topic_terms={tn[np.argsort(-loadings[1])][:4].tolist()} d0_topic1={T[0, 1]:.4f} d3_topic1={T[3, 1]:.4f}")
```
### Negation-aware lexicon sentiment score (stand-in)
No sentiment package is installed: score a lexicon - +1 per positive term, -1 per negative term, flipped by a preceding `not`/`no`, plus an exclamation bonus, divided by token count.
```python
import re
POS = {"love", "great", "best", "won", "amazing", "saved"}; NEG = {"hate", "bad", "worst", "lost", "awful", "fell", "stalled"}; FLIP = {"not", "no"}
def polarity(text, exclamation=0.25):
    words = re.findall(r"[a-z']+", text.lower()); score, flip = 0.0, 1.0
    for w in words:
        if w in FLIP: flip = -1.0; continue
        score += flip * (w in POS) - flip * (w in NEG)
    return round(score / max(len(words), 1) + exclamation * text.count("!"), 4)
assert polarity("I love this great song!") > 0 > polarity("I hate this awful song, it is not great at all")
print(f"RESULT pos={polarity('I love this great song!')} neg={polarity('I hate this awful song, it is not great at all')} neutral={polarity('the report has forty pages')}")
```

## Output contract
Each block prints `RESULT name=value`. Delivered: cleaned strings, token lists, stems/lemmas; a sparse CSR
`[n_docs, n_terms]` int count matrix (the book's 3x8: `nnz=8`, sparsity 0.667, Albon ch.6 p119-120) whose
`vocabulary_`/`get_feature_names_out()` are the column contract; a float TF-IDF matrix with L2-unit rows
(6 docs x 29 terms) whose cosine similarity is `W @ W.T` (same-topic 0.549377 vs cross-topic 0.081437); a dense
`[n_docs, K]` topic-score matrix; one polarity float per text. `transform` with the fitted vectorizer, never refit.

## Validity conditions
- Fit vectorizers on training text only, then `transform` test text: fitting everywhere leaks vocabulary and idf
  statistics into evaluation (Albon ch.6 p120).
- Stop-word and lexicon matching assumes lowercase (p115); punctuation stripping discards real signal (p113);
  bag-of-words ignores word order and costs a column per unique word, so keep sparse output and set
  `stop_words`/`min_df`/`max_df`/`vocabulary` before memory blows up (p120).
- Stems are not keywords (p116) and TF-IDF magnitudes only compare under one idf smoothing + L2 normalisation (p123).

## Conventions
- nltk, gensim, spacy, textblob and bs4 are NOT installed: tokenising, sentence splitting, the stop-word list, the
  reduced Porter stemmer, vocabulary-guarded lemmatisation, HTML tag stripping and the idf/tf-idf/cosine/SVD/lexicon
  numerics are re-implemented on stdlib `re`/`unicodedata` + numpy + sklearn.
- sklearn 1.9 API: `get_feature_names_out()` replaces the book's removed `get_feature_names` (p119); `TfidfVectorizer`
  defaults are `smooth_idf=True, norm="l2"`, `idf = log((1+nd)/(1+df))+1` (p123); scipy sparse `.size` counts stored
  elements, so take sparsity from `.shape`.
- Demo corpora are literal string lists (or `default_rng(42)`); compute and print every number, never hardcode one.

## Out of scope
POS tagging and its one-hot POS-tag features (Albon ch.6 p116-118) need nltk's trained tagger, and class-scoped HTML
selection via `soup.find("div", {"class": ...}).text` (p112) needs bs4; word2vec/transformer embeddings (gensim,
spacy), generative LDA and real sentiment engines (TextBlob, VADER) are references only, so the LSA block and the
lexicon above are declared stand-ins. Scaling the resulting matrix is a sibling skill (Albon ch.6 p110-123).
