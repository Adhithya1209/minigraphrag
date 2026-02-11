# minigraphrag
Graphrag implemented with hallucination detection framework

The committed framework:
1) So far the user interface is created using framework and available in the form of ChatInterface() framework
2) GraphRag() framework has functions that can create the network of graphs from the preprocessed data/chunks
3) The PrepareRetrieval() frameworks allows preprocess of the document-limited to pdf files for now. It also inherits features from GraphRag() to help with building graphs
4) The PrepareRetrieval() allows additional functionality such as metadata retrieval for enriched data and subsequent graphs
5) GraphRag() framework allows both local and neo4j database storage of the generated graphs.

Important information
Due to the limited usage in free tier of Groq api, the tests are carried out using local llm, i.e, llama3.2 but will be provided with an option to connect to api servers in the future. The community detection framework, summarization and hallucination detection frameworks are all in testing phase and are done locally. They will go live by Feb 18th.
Due to the complexity of this project and the requirement of decent hardware has limited the deployment of the project to HuggingFace spaces (free tier) and only local usage will be provided with an updated readme to properly install and use the frameworks.
