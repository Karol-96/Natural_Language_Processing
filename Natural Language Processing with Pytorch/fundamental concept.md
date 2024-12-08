1. backpropagation: The process of iteratively updating the
parameters is called backpropagation. Each step (aka epoch) of
backpropagation consists of a forward pass and a backward pass. The
forward pass evaluates the inputs with the current values of the
parameters and computes the loss function. The backward pass updates
the parameters using the gradient of the loss.

2. One-Hot Representation : The one-hot representation, as the name suggests, starts with a zero vector,
and sets as 1 the corresponding entry in the vector if the word is present in
the sentence or document. Consider the following two sentences:
![alt text](image.png)

3. TF Representation (Term Frequency): The TF representation of a phrase, sentence, or document is simply the sum
of the one-hot representations of its constituent words., using the aforementioned one-hot encoding, the sentence
“Fruit flies like time flies a fruit” has the following TF representation: [1,
2, 2, 1, 1, 0, 0, 0]. Notice that each entry is a count of the number of
times the corresponding word appears in the sentence (corpus). We denote
the TF of a word w by TF(w).
![alt text](image-1.png)

4. TFIDF Representation (Term Frequency Inverse Document Frequency):Consider a collection of patent documents. You would expect most of them
to contain words like claim, system, method, procedure, and so on, often
repeated multiple times. The TF representation weights words
proportionally to their frequency. However, common words such as “claim”
do not add anything to our understanding of a specific patent. Conversely, if
a rare word (such as “tetrafluoroethylene”) occurs less frequently but is
quite likely to be indicative of the nature of the patent document, we would
want to give it a larger weight in our representation. The InverseDocument-Frequency (IDF) is a heuristic to do exactly that.
The IDF representation penalizes common tokens and rewards rare tokens
in the vector representation. he TF-IDF score is simply the product TF(w)
* IDF(w).
![alt text](image-2.png)




