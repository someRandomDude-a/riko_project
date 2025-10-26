# Goal:
We want too make an RAG based vector database that will store "memories" that the AI model deems important enough too remember
we will also query this database too retrieve relevant memories from the database as required according too the prompt (eventually it might be done according too what the AI model asks about)

## Features :

* Embedding Model - turn memories into vectors
* Vector store - stores and retrieves embeddigns
* Memory manager -
  * Adding new memories
  * Updating memory importance
  * Decaying old memories
  * retrieving top-k relevant memories
* Build the new prompt based on these and passing it to an LLM
