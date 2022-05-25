#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
df = pd.read_csv("IMDB_Dataset.csv")


# In[14]:


df = df[["review","sentiment"]]


# In[15]:


df


# In[16]:


df["sentiment"].value_counts()


# In[17]:


sentiment_label = df.sentiment.factorize()


# In[18]:


text = df.review.values


# In[20]:


# text


# In[21]:


from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(text)


# In[22]:


encoded_docs = tokenizer.texts_to_sequences(text)


# In[23]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_sequence = pad_sequences(encoded_docs, maxlen=200)


# In[25]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding

vocab_size = len(tokenizer.word_index) + 1

embedding_vector_length = 32
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[32]:


history = model.fit(padded_sequence,sentiment_label[0],validation_split=0.2, epochs=40, batch_size=32)


# In[43]:


import pickle

# save the model to disk
pickle.dump(model, open('imdb_model.sav', 'wb'))
pickle.dump(tokenizer, open('imdb_model_tokenizer.sav', 'wb'))


# In[46]:


loaded_model = pickle.load(open('imdb_model.sav', 'rb'))
loaded_tokenizer = pickle.load(open('imdb_model_tokenizer.sav', 'rb'))

text = ["This is a Good news"]

tw = loaded_tokenizer.texts_to_sequences(text)
tw = pad_sequences(tw,maxlen=200)
prediction = int(loaded_model.predict(tw).round().item())
print(prediction)
print("Predicted label: ", sentiment_label[1][prediction])


# In[ ]:




