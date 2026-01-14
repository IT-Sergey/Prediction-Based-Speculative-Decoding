# Prediction-Based Speculative-Decoding
We propose a prediction-based speculative decoding method for parallelizing the decoding of 
prefix codes, using Huffman coding as a case study. By initiating concurrent decoding attempts at 
predicted bit positions, the method aims to overcome the sequential dependency inherent in decoding prefix codes. 
The approach is tested on text and image datasets, showing the 
potential of the proposed speculative decoding technique.

This repository contains Python model for prediction-based speculative decoder.

# Structure
 - 📁 `datasets` specially designed place to put datasets it. Due to potential problems with size, 
the content is not uploaded to repository,
 - 📁 `source` contains main part of model code (in Python),
 - 📁 `source/scripts` contains Python scripts for running experiments.

# Scripts

Prior to running scripts please install all required packages listed in `source/requirements.txt`.

There are **two** scripts: for _context-free_ and _context-aware_ prediction-based speculative
decoders.

Script `source/scripts/context_free.py` has the following main configurable parameters. They should be adjusted, if needed,
in the beginning of `__main__`.

```python
files = ["../../datasets/books/anna_karenina.txt"] # Files to form the dataset.
speculation_width = 2  # number of speculation units (additional decoders).
use_real_probabilities = True  # use real probabilities (or estimate via codeword lengths).
use_baseline = False  # `True` if utilize baseline (position-restricted) approach.
```

Execution with sample parameters results in the following output:
```console
==================================================
READING INPUT FILES
==================================================
100%|██████████| 2067960/2067960 [00:00<00:00, 2104988.35it/s]
--------------------------------------------------
1 files are processed
use real probabilities  = True
use baseline approach   = True
--------------------------------------------------
selected L = (3, 4)
expected total rate     = 1.5801
expected total rate [k] = 1.5801

compression ratio       = 1.7228
Decoding: 100%|██████████| 9602877/9602877 [00:09<00:00, 961299.56it/s]
decoded correctly       = True
real decoding rate      = 1.6044
```

Script `source/scripts/context_aware.py` has the following main configurable parameters. They should be adjusted, if needed,
in the beginning of `__main__`.

```python
files = ["../../datasets/books/anna_karenina.txt"] # Files to form the dataset.
speculation_width = 2 # number of speculation units (additional decoders).
model_order = 3 # Markov order (k-order).
```

Execution with sample parameters results in the following output:
```console
==================================================
READING INPUT FILES
==================================================
100%|██████████| 2067960/2067960 [00:01<00:00, 2021235.45it/s]
100%|██████████| 1198/1198 [00:00<00:00, 566938.53it/s]
--------------------------------------------------
1 files are processed
w = 2
expected total rate     = 1.6907
==================================================
SIMULATION STARTED
==================================================

original size           = 2067960
compressed size         = 1200360 (bytes)	9602877 (bits)
compression ratio       = 1.72278
Decoding: 100%|██████████| 9602877/9602877 [00:09<00:00, 974638.74it/s]
decoded correctly       = True
real decoding rate      = 1.70338
```

# Books dataset
For dataset consisting of books in English we used the following ones from Project Gutenberg:

| Author | Book Title | Gutenberg eBook # | Direct Link |
| :--- | :--- | :--- | :--- |
| **Leo Tolstoy** | Anna Karenina | 1399 | [https://www.gutenberg.org/ebooks/1399](https://www.gutenberg.org/ebooks/1399) |
| **Leo Tolstoy** | Resurrection | 1938 | [https://www.gutenberg.org/ebooks/1938](https://www.gutenberg.org/ebooks/1938) |
| **Leo Tolstoy** | War and Peace | 2600 | [https://www.gutenberg.org/ebooks/2600](https://www.gutenberg.org/ebooks/2600) |
| **Fyodor Dostoyevsky** | The Brothers Karamazov | 28054 | [https://www.gutenberg.org/ebooks/28054](https://www.gutenberg.org/ebooks/28054) |
| **Fyodor Dostoyevsky** | Crime and Punishment | 2554 | [https://www.gutenberg.org/ebooks/2554](https://www.gutenberg.org/ebooks/2554) |
| **Fyodor Dostoyevsky** | The Gambler | 2197 | [https://www.gutenberg.org/ebooks/2197](https://www.gutenberg.org/ebooks/2197) |
| **Fyodor Dostoyevsky** | The Idiot | 2638 | [https://www.gutenberg.org/ebooks/2638](https://www.gutenberg.org/ebooks/2638) |
| **Charles Dickens** | Great Expectations | 1400 | [https://www.gutenberg.org/ebooks/1400](https://www.gutenberg.org/ebooks/1400) |
| **Charles Dickens** | Oliver Twist | 730 | [https://www.gutenberg.org/ebooks/730](https://www.gutenberg.org/ebooks/730) |
| **Alexandre Dumas** | The Man in the Iron Mask | 2759 | [https://www.gutenberg.org/ebooks/2759](https://www.gutenberg.org/ebooks/2759) |
| **Alexandre Dumas** | The Count of Monte Cristo | 1184 | [https://www.gutenberg.org/ebooks/1184](https://www.gutenberg.org/ebooks/1184) |
| **Alexandre Dumas** | The Three Musketeers | 1257 | [https://www.gutenberg.org/ebooks/1257](https://www.gutenberg.org/ebooks/1257) |
| **Alexandre Dumas** | Twenty Years After | 1259 | [https://www.gutenberg.org/ebooks/1259](https://www.gutenberg.org/ebooks/1259) |