# Q2Q
Question to Question Similarity 

PLease follow: https://ieeexplore.ieee.org/abstract/document/9035248

Deep Learning (DL) is a part of the significant family Machine Learning (ML), where learning can be, supervised, unsupervised and semi-supervised. It's essential to know that deep learning techniques have been innovated by the idea of information passing and processing within the biological nervous system.

Choosing the DL technique is very important since neither traditional methods nor ML baseline systems can maintain a vast amount of data. It’s essential to know what makes us consists of upgrading our model to be on the DL level, that it dispenses with using the features extraction.

To start working with DL and use it as an efficient tool, we need to know that it uses several nonlinear processing layers. So how deep the model is referred to the number of layers the data are transformed through. One of the most variables we need to put in mind is the Credit Assignment Path (CAP) which is the depth of the length data walked through one layer to the next layer. 

Since DL can be implemented through several techniques, in our approach we’ve found Recurrent Neural Network (RNN) used as a trusted architecture in the field of Natural Language Processing (NLP) tasks is proper to handle input of different lengths in order to its structure, since it serves us well-finding each pair of questions similarities in Arabic language.

Where Long Short-Term Memory Networks (LSTMs), can compute a complete sequence of data.

Deep Learning Model to be implemented 
In our model we will use:
Bidirectional Gated recurrent unit (GRU).

Deep learning models:
First: SemEval-2016 Task 1: Semantic Textual Similarity, Monolingual, and Cross-Lingual Evaluation

Task: Semantic Textual Similarity (STS) seeks to measure the degree of semantic equivalence between two snippets of text. 



Rank 1:
🡪(Samsung Poland NLP team) 
They used bi-directional Gated Recurrent Neural Network

Rank 2:
🡪(UWB) 
They used Long Short-Term Memory (LSTM) is a type of Recurrent Neural Network (RNN)

“MayoNLP at SemEval-2016 Task 1: Semantic Textual Similarity based on Lexical Semantic Net and Deep Learning Semantic Model.” 


Deep Structured Semantic Model (DSSM) which based on deep neural network, (DNN), which used to the model semantic similarity between a pair of strings. In simple terms, the semantic similarity of two sentences is the similarity based on their meaning.

The architecture of the model:
Used to transform the sentences into term vector.
Perform word hashing to reduce the dimensionality of vectors and generate feature vectors.
Through the hidden layers feature vectors projected and formed as semantic feature vectors in the top layer for each paired snippets of text. Cosine similarity is utilized to measure the semantic similarity between the pair.



In the paper, they used one hidden layer with 1000 hidden nodes in the neural networks.

The performance of DSSM was the highest for a question-question dataset.
And better slightly than convolutional DSSM (C-DSSM).
Performance as Pearson correlation = 0.73035.

Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks

In this paper, the deep learning model based on a convolutional neural network (ConvNets).  

From our reading, we want our model to be implemented through the deep learning approaches and what we found after several reading some research papers about measure Semantic Textual Similarity (STS) between two sentences, like the following:

Sanbornet al. [1], proposed some models such as SVM, Logistic Regression, Decision tree, Random forest, Nearest neighbors, recurrent and recursive neural networks with stochastic gradient descent and word vector representations. The best result achieved by RNN with 100-dimensional word vectors and 20% dropout.

Taking into consideration the importance of the semantic meaning of sentences to learn the model how to compute the STS, as in Prijatelj et al. [2], rely on word embedding vectors to represent the sentences. Then, they implement and compare three models based on the same structure: LSTM, Stacked LSTMs and L2-LSTM which simplified from MV-LSTM. The best Pearson Correlation Coefficient value is 0.8608 achieved by a stack of 2 LSTMs (which consists of two stacks of 2 LSTMs) one of them represent the embedding of two sentences separately then it compared with another one represent the embedding of concatenating two sentences. 
To find the essential information in a given sentence in terms of semantics. Zhou et al. [3], proposed deep learning model called Attention-Based Bidirectional Long Short-Term Memory Networks (Att-BLSTM). It attains a decent result of F1 score around 0.84% on a dataset composed of 10,717 labeled sentences distributed into 8,000 as training and 2,717 as testing. Att-BLSTM consists of five layers summed up as follows: the first one is the input layer, then the embedding layer which converts the words in the sentence into a vector representation. After that they applied bidirectional LSTM to cover the forward and backward states, the output of the LSTM layer goes to attention layer, and finally, the output layer represents the semantic features of the sentences.


References 

[1]: Sanborn, A., & Skryzalin, J. (2015). Deep learning for semantic similarity. CS224d: Deep Learning for Natural Language Processing Stanford, CA, USA: Stanford University.‏

[2]: Prijatelj, D., Ventura, J., & Kalita, J. (2017, December). Neural networks for semantic textual similarity. In International Conference on Natural Language Processing.‏

[3]: Zhou, Peng, et al. "Attention-based bidirectional long short-term memory networks for relation classification." Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). Vol. 2. 2016.
