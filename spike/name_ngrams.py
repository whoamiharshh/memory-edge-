"""Spike: does a character n-gram model of "what a given name looks like" catch UNSEEN names in ALL CAPS / lower case
notes without flagging maintenance words? Trained on real names (Wikidata, 80 %) vs real maintenance/technical words
(domain vocabulary built without the held-out notes); tested on the 20 % held-out names and on held-out notes.
Run: .venv\\Scripts\\python.exe -m spike.name_ngrams
"""
import json
import pathlib
import random
import re

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

from data import build_vocab

ROOT = pathlib.Path(__file__).resolve().parents[1]
rng = random.Random(7)
notes = build_vocab.logbook_notes()
idx = list(range(len(notes)))
rng.shuffle(idx)
test = idx[:1000]
vocab = sorted(build_vocab.build(exclude=set(test)))
names = sorted(json.loads((ROOT / "knowledge" / "given_names.json").read_text(encoding="utf-8"))["names"])
rng.shuffle(names)
cut = int(0.8 * len(names))
train_names, test_names = [n for n in names[:cut] if n not in vocab], [n for n in names[cut:] if n not in vocab]
vec = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=2 ** 16, alternate_sign=False, norm="l2")
X = vec.transform(train_names + vocab)
y = np.array([1] * len(train_names) + [0] * len(vocab))
clf = LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced").fit(X, y)
words_in_test_notes = sorted({w.lower() for i in test[500:] for w in re.findall(r"[A-Za-z]{3,20}", notes[i])} - set(vocab))
for thr in (0.5, 0.7, 0.8, 0.9, 0.95):
    rec = float(np.mean(clf.predict_proba(vec.transform(test_names))[:, 1] >= thr))
    p = clf.predict_proba(vec.transform(words_in_test_notes))[:, 1] if words_in_test_notes else np.array([])
    fp_words = [w for w, q in zip(words_in_test_notes, p) if q >= thr]
    notes_flagged = sum(any(w.lower() in fp_words for w in re.findall(r"[A-Za-z]{3,20}", notes[i])) for i in test[500:])
    print(f"thr {thr}: unseen-name recall {rec:.3f} | clean held-out notes flagged {notes_flagged}/500 "
          f"| out-of-vocab words {len(words_in_test_notes)}, flagged {len(fp_words)} e.g. {fp_words[:8]}")
