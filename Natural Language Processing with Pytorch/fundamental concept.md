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

5. Any PyTorch method with
an underscore (_) refers to an in-place operation; that is, it modifies the
content in place without creating a new object, 
![alt text](image-4.png)

6. Tensor Types and Sizes: Each tensor has an associated type and size. The default tensor type when
you use the torch.Tensor constructor is torch.FloatTensor. However,
you can convert a tensor to a different type (float, long, double, etc.) by
specifying it at initialization or later using one of the typecasting methods.
There are two ways to specify the initialization type: either by directly
calling the constructor of a specific tensor type, such as FloatTensor or
LongTensor, or using a special method, torch.tensor(), and providing
the dtype.
![alt text](image-5.png)

7. Tensor Operations
![alt text](image-6.png)

8. Dimension based Tensors
![alt text](image-6.png)

9. Concatenating Tensors
![alt text](image-6.png)

10. Linear Algebra: Multiplication
![alt text](image-6.png)

11. Tensors and Computational Graphs
Creating tensors for gradient bookkeeping
![alt text](image-7.png)

12. Using Cuda Sensors(Compute Unified Device Architecture)
![alt text](image-8.png)
![alt text](image-12.png)

13. Exercise 
![alt text](image-9.png)
![alt text](image-10.png)

14. Visualizing Computation Graphs
You can visualize the computation graph using a tool like tensorboard or directly within PyTorch using the following methods:

![alt text](image-11.png)




