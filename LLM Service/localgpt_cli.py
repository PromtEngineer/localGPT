import localgpt

localgpt_object = localgpt.TextQAEngine()

# localgpt_object.ingest()

# if PERSIST_DIRECTORY is the same in config.py but you want to add more data to the DB
localgpt_object.ingest(db_name="CNN")
localgpt_object.load_model()
# use_history wont work here since we generate conversational memory of db's from a different section of load_QA()
localgpt_object.load_QA(promptTemplate_type="ChatML", db_name="CNN") 

while True:
    user_prompt = input("\nEnter a query: ")
    if user_prompt == "exit":
        break
    elif user_prompt == "reingest":
        localgpt_object.ingest()
        continue

    # Get the answer from the chain
    res = localgpt_object.QA(user_prompt)
    print(res)
    answer, docs = res["result"], res["source_documents"]

    # Print the result
    print("\n\n> Question:")
    print(user_prompt)
    print("\n> Answer:")
    print(answer)

    if True:
        # # Print the relevant sources used for the answer
        print("----------------------------------SOURCE DOCUMENTS---------------------------")
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)
        print("----------------------------------SOURCE DOCUMENTS---------------------------")
