def query_data(query):
    docs = vector_search.similarity_search(query, k=1)
    as_output=docs[0].page_content
    llm= model
    retriever =vector_search.as_retriever()
    qa=RetrievalQA.from_chain_type(llm,chain_type="stuff", retriever=retriever)
    retriever_output = qa.run(query)

    return as_output, retriever_output
