import localgpt

localgpt_object = localgpt.DocumentProcessor()

# localgpt_object.ingest()

# if PERSIST_DIRECTORY is the same in config.py but you want to add more data to the DB
localgpt_object.ingest()

localgpt_object.load_model()

localgpt_object.load_QA()

while True:
    query = input("\nEnter a query: ")
    if query == "exit":
        break
    elif query == "reingest":
        localgpt_object.ingest()
        continue

    # Get the answer from the chain
    res = localgpt_object.QA(query)
    answer, docs = res["result"], res["source_documents"]

    # Print the result
    print("\n\n> Question:")
    print(query)
    print("\n> Answer:")
    print(answer)

    if localgpt.SHOW_SOURCES:
        # # Print the relevant sources used for the answer
        print("----------------------------------SOURCE DOCUMENTS---------------------------")
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)
        print("----------------------------------SOURCE DOCUMENTS---------------------------")
