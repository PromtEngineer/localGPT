import ingest
import run_localGPT
import json
from bs4 import BeautifulSoup
import requests
import hashlib
import gradio as gr
import base64
import os
import time
import pandas as pd

from variables import *

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from googlesearch import search

# Define the URL for the CoinMarketCap page
CoinMarketCap_DEX_page_URL = "https://coinmarketcap.com/rankings/exchanges/dex/"

def scrape_coinmarketcap_dex_page():
    """
    Scrape the CoinMarketCap DEX page to get the list of DEXs.
    """

    # Parse the HTML content
    html_content = requests.get(CoinMarketCap_DEX_page_URL).content
    soup = BeautifulSoup(html_content, "html.parser")

    # Find the table and table rows containing the DEX information
    table = soup.find("table", {"class": "sc-66133f36-3 dOrjhR cmc-table"})
    # table's body
    table_body = table.find("tbody")
    table_rows = table_body.find_all("tr")

    # Prepare the data for the table
    table_data = []
    for row in table_rows:
        columns = row.find_all("td")
        if len(columns) >= 2:
            dex_name_elem = columns[1]
            dex_website_elem = columns[1].find("a", {"class": "cmc-link"})
            if dex_name_elem and dex_website_elem:
                dex_name = dex_name_elem.text
                table_data.append(dex_name)

    # Write the data to a xlsx file
    df = pd.DataFrame(table_data, columns=["Dex Name"])

    # Preporcessing dex names :
    # - Uniswap v3 (Ethereum)2	-> Uniswap v3
    # - if the dex id from the top 10 remove the last char if it is a digit (classement)

    df["Dex Name"][:10] = df["Dex Name"][:10].apply(lambda x: x[:-1] if x[-1].isdigit() else x)
    df["Dex Name"] = df["Dex Name"].apply(lambda x: x.split("(")[0].strip())

    # delete duplicates based on name and show the percentage of duplicates
    #print("Percentage of duplicates : ", 100 - len(df.drop_duplicates(subset=['Dex Name'], keep='first'))/len(df)*100, "%")
    df = df.drop_duplicates(subset=['Dex Name'], keep='first')

    return df


def save_url_as_html(url, save_path):
    """
    Save the content of a URL as an HTML file.
    """
    try:
        # Send a GET request to the URL to fetch the content
        response = requests.get(url, timeout=10)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Save the content as an HTML file
            with open(save_path, 'w', encoding='utf-8') as html_file:
                html_file.write(response.text)
            print(f"HTML content saved as {save_path}")
        else:
            print(f"Failed to fetch the URL. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def get_documents(dex_name, save_path=SOURCE_DIRECTORY):
    """
    Scrape the documents for a DEX.
    """

    URL_MAP = json.load(open("url_map.json", "r"))

    # Extract links for liquidity model using googlesearch (only html files)
    query = f'{dex_name} liquidity model'
    search_results = search(query, num_results=1)
    liquidity_model_link = list(search_results)

    # create a folder for the dex liquidity model
    os.makedirs(f'{save_path}/{dex_name}/liquidity_model', exist_ok=True)

    # If no link is found, create a txt file with the message
    if len(liquidity_model_link) == 0:
        with open(f'{save_path}/{dex_name}/liquidity_model/no_liquidity_model.txt', 'w') as f:
            f.write("No liquidity model found for this DEX.")

    else:
        # save liquidity model pages as html
        for i, link in enumerate(liquidity_model_link):
            try:
                filename = f'{hashlib.md5(link.encode()).hexdigest()}_{i+1}.html'
                save_url_as_html(link, f'{save_path}/{dex_name}/liquidity_model/{filename}')
                URL_MAP[filename] = {'url source': link}
            except:
                print("Could not save the page.")

    # create a folder for the dex if it doesn't exist
    os.makedirs(f'{save_path}/{dex_name}/license', exist_ok=True)
    # Flag to track if a license has been found for this DEX
    license_found = False

    # Make a GitHub API repository search request based on the DEX name
    search_url = f'https://api.github.com/search/repositories?q={dex_name}&per_page=10'
    headers = {'Authorization': f'token {GITHUB_API_KEY}'}
    response = requests.get(search_url, headers=headers)

    if response.status_code == 200:
        search_results = response.json()['items']

        for repo in search_results:
            # Check if a license file exists and retrieve the license text
            license_url = f'https://api.github.com/repos/{repo["owner"]["login"]}/{repo["name"]}/license'
            response = requests.get(license_url, headers=headers)

            if response.status_code == 200:
                license_data = response.json()
                if 'content' in license_data:
                    license_text = base64.b64decode(license_data['content']).decode('utf-8')
                    # Save license text to a file in the dex folder in the license folder
                    filename = f'{hashlib.md5(license_url.encode()).hexdigest()}.txt'
                    with open(f'{save_path}/{dex_name}/license/{filename}', 'w') as f:
                        f.write(license_text)
                    URL_MAP[filename] = {'url source': license_url, 'repo': repo["full_name"]}
                    # Set the flag to True to indicate that a license has been found
                    license_found = True
                    break  # Stop searching for licenses in other repositories for this DEX
            else:
                print(f'Failed to fetch license for {repo["full_name"]}: {response.status_code}')

        # If no official license is found, create a txt file with the message
        if not license_found:
            with open(f'{save_path}/{dex_name}/license/no_license.txt', 'w') as f:
                f.write("No license found for this DEX.")
    else:
        print(f'Failed to search for repositories related to {dex_name}: {response.status_code}')
        with open(f'{save_path}/{dex_name}/license/no_license.txt', 'w') as f:
            f.write("No license found for this DEX.")

    # save the URL_MAP to a json file
    with open("C:/Users/mmahmoud/localGPT/app/url_map.json", "w") as f:
        json.dump(URL_MAP, f)

    # return path to the dex folder
    return f'{save_path}/{dex_name}'



def user_interaction(dex_name, k, co, cs, progress=gr.Progress()):
    results = {}
    # Define features to process
    features = ["liquidity_model", "license"]

    # Check if all parameters are provided
    if dex_name and k is not None and co is not None and cs is not None:
        progress(0.0, desc="Scraping documents...")
        time.sleep(1)
        # if the dex folder doesn't exist, use get_documents to scrape the documents
        dex_folder = f"{SOURCE_DIRECTORY}/{dex_name}"
        if not os.path.exists(dex_folder):
            dex_folder = get_documents(dex_name)

        progress(0.1, desc="Loading embedding model...")
        time.sleep(1)
        embedding_model = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_NAME, model_kwargs={"device": DEVICE_EMBEDDING})
        #embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, openai_organization=OPENAI_ORGANIZATION)

        progress(0.2, desc="Loading LLM model...")
        time.sleep(1)
        llm = run_localGPT.load_model(device_type=DEVICE_MODEL, model_id=MODEL_ID, model_basename=MODEL_BASENAME)
        #llm = OpenAI(openai_api_key=OPENAI_API_KEY, openai_organization=OPENAI_ORGANIZATION, temperature=0.0)

        for i, feature in enumerate(features):
            source_directory = f"{dex_folder}/{feature}"
            progress(0.4 + 0.6*(i)/len(features), desc=f"Processing {feature}..")
            time.sleep(1)

            save_path = f"{source_directory}/{embedding_model.model_name}"
            #save_path = f"{source_directory}/openaiembeddings"
            save_path = f"{PERSIST_DIRECTORY}/{dex_name}/{feature}/{embedding_model.model_name.replace('/', '_')}"
            #save_path = f"{PERSIST_DIRECTORY}/{dex_name}/{feature}/openaiembeddings"

            # Convert chunk_size and chunk_overlap to integers
            cs = int(cs)
            co = int(co)
            ingest.main(embedding_model=embedding_model, chunk_size=cs, chunk_overlap=co,
                        source_directory=source_directory, save_path=save_path)

            persist_directory = os.path.join(save_path, f'cs_{cs}_co_{co}')

            # Getting the query from queries/feature.txt
            with open(f"queries/{feature}.txt", "r") as f:
                query = f.read()
                query = query.replace("the DEX", dex_name)

            # Convert k to an integer
            k = int(k)

            # Running localGPT
            answer, docs = run_localGPT.main(llm, embedding_model, k, persist_directory, query, promptTemplate_type=None)

            # Store the results
            results[feature] = {"answer": answer, "sources": [document for document in docs]}

            # save the results to a json file
            """results_path = f"json/{dex_name}.json"
            os.makedirs("json", exist_ok=True)
            with open(results_path, "w") as f:
                json.dump(results, f)"""


        progress(1, desc="Done !")
        time.sleep(1)

        # Assuming you have obtained features from the results
        features = list(results.keys())
        new_row = {"Dex name": dex_name}

        for feature in features:
            format_answer = results[feature]['answer'].replace('\n', '<br>')
            sources = []
            for source in results[feature]['sources']:
                page_content = source.page_content.replace('\n', ' ')
                metadata = source.metadata
                format_metadata = "<br>".join([f"&nbsp;{' ' * 4}<span style='color: Orange;'><strong>{key}:</strong></span> {value}" for key, value in metadata.items() if key!="source"])
                sources.append(f"{page_content}<br>{format_metadata}")
            # enumerate the sources
            format_sources = "<br>".join([f"<span style='color: Red;'><strong>{i+1}.</strong></span> {source}" for i, source in enumerate(sources)])
            new_row[feature] = f"<span style='color: green;'><strong>Answer:</strong></span> {format_answer}\
                <br> <span style='color: Red;'><strong>Sources:</strong></span><br> {format_sources}"


        # Append the new row to the DataFrame
        df = pd.read_excel("dataframes/table.xlsx")
        # Concatenate the new row with the DataFrame if the row doesn't already exist
        # Otherwise, update the row
        if dex_name in df["Dex name"].values:
            for feature in features:
                df.loc[df["Dex name"] == dex_name, feature] = new_row[feature]
        else:
            df = pd.concat([df, pd.DataFrame(new_row, index=[0])])
        df.to_excel("dataframes/table.xlsx", index=False)

        table = gr.Dataframe(
            headers=["Dex name", "liquidity_model", "license"],
            datatype=["str", "markdown", "markdown"],
            value=pd.read_excel("dataframes/table.xlsx"),
            wrap=True
            )
        # Unload the model and free up resources
        #del llm
        dex_to_search_from_table = gr.Dropdown(
                                    label="DEX Name",
                                    choices=pd.read_excel("dataframes/table.xlsx")["Dex name"].tolist(),
                                    info="Choose the DEX to search from the table."
                                    )
    return table, results, dex_to_search_from_table


def refresh_dex_list():
    df_dex_list = pd.read_excel("dataframes/dex_list.xlsx")
    df = scrape_coinmarketcap_dex_page()
    # add def to df_dex_list if not already present
    #df_dex_list = df_dex_list.append(df[~df["Dex Name"].isin(df_dex_list["Dex Name"])])
    # Assuming df and df_dex_list are your DataFrames
    df_dex_list = pd.concat([df_dex_list, df[~df["Dex Name"].isin(df_dex_list["Dex Name"])]])
    # Reset the index of the resulting DataFrame
    df_dex_list.reset_index(drop=True, inplace=True)

    df_dex_list.to_excel("dataframes/dex_list.xlsx", index=False)
    dropdown = gr.Dropdown(
                label="DEX Name",
                choices=pd.read_excel("dataframes/dex_list.xlsx")["Dex Name"].tolist(),
                value="Uniswap v3",
                allow_custom_value=True,
                filterable=True)
    return dropdown


def delete_table(confirm_checkbox):
    if confirm_checkbox:
        df = pd.DataFrame({"Dex name": [], "liquidity_model": [], "license": []})
        df.to_excel("dataframes/table.xlsx", index=False)
    else:
        gr.Info("Please confirm that you want to delete the main table.")
    table = gr.Dataframe(
            headers=["DEX name", "liquidity_model", "license"],
            datatype=["str", "markdown", "markdown"],
            value=pd.read_excel("dataframes/table.xlsx"),
            wrap=True
            )
    dropdown = gr.Dropdown(
                label="DEX Name",
                choices=pd.read_excel("dataframes/table.xlsx")["Dex name"].tolist())
    dex_to_search_from_table = gr.Dropdown(
                                    label="DEX Name",
                                    choices=pd.read_excel("dataframes/table.xlsx")["Dex name"].tolist(),
                                    info="Choose the DEX to search from the table."
                                    )
    return dex_to_search_from_table, dropdown, table

def delete_dex_from_table(dex_to_delete_from_table, confirm_delete_dex_from_table):
    if not dex_to_delete_from_table:
        gr.Info("Please select a DEX to delete from the main table.")
    else:
        if not confirm_delete_dex_from_table:
            gr.Info("Please confirm that you want to delete the DEX from the main table.")
        else:
            df = pd.read_excel("dataframes/table.xlsx")
            df = df[df["Dex name"] != dex_to_delete_from_table]
            df.to_excel("dataframes/table.xlsx", index=False)
    table = gr.Dataframe(
            headers=["DEX name", "liquidity_model", "license"],
            datatype=["str", "markdown", "markdown"],
            value = pd.read_excel("dataframes/table.xlsx"),
            wrap=True,
            height=1000
        )
    dropdown = gr.Dropdown(
                label="DEX Name",
                choices=pd.read_excel("dataframes/table.xlsx")["Dex name"].tolist())

    dex_to_search_from_table = gr.Dropdown(
                                    label="DEX Name",
                                    choices=pd.read_excel("dataframes/table.xlsx")["Dex name"].tolist(),
                                    info="Choose the DEX to search from the table."
                                    )
    return dex_to_search_from_table, dropdown, table

def delete_dex_from_dropdown(dex_to_delete_from_dropdown, confirm_delete_dex_from_dropdown):
    if not dex_to_delete_from_dropdown:
        gr.Info("Please select a DEX to delete from the CoinMarketCap list.")
    else:
        if not confirm_delete_dex_from_dropdown:
            gr.Info("Please confirm that you want to delete the DEX from the CoinMarketCap list.")
        else:
            df = pd.read_excel("dataframes/dex_list.xlsx")
            df = df[df["Dex Name"] != dex_to_delete_from_dropdown]
            df.to_excel("dataframes/dex_list.xlsx", index=False)
    dropdown = gr.Dropdown(
                label="DEX Name",
                choices=pd.read_excel("dataframes/dex_list.xlsx")["Dex Name"].tolist(),
                value="Uniswap v3",
                allow_custom_value=True,
                filterable=True)
    dropdown1 = gr.Dropdown(
                label="DEX Name",
                choices=pd.read_excel("dataframes/dex_list.xlsx")["Dex Name"].tolist()
                )
    return dropdown, dropdown1


def delete_dropdown_list(confirm_delete_dropdown_list):
    if not confirm_delete_dropdown_list:
        gr.Info("Please confirm that you want to delete the CoinMarketCap list.")
    else:
        df = pd.DataFrame({"Dex Name": []})
        df.to_excel("dataframes/dex_list.xlsx", index=False)
    dropdown = gr.Dropdown(
                label="DEX Name",
                choices=pd.read_excel("dataframes/dex_list.xlsx")["Dex Name"].tolist(),
                value="Uniswap v3",
                allow_custom_value=True,
                filterable=True)
    dropdown1 = gr.Dropdown(
                label="DEX Name",
                choices=pd.read_excel("dataframes/dex_list.xlsx")["Dex Name"].tolist()
                )
    return dropdown, dropdown1


def update_and_extract_all(k, co, cs, progress=gr.Progress()):
    results = gr.JSON(label="Results")
    old_list=pd.read_excel("dataframes/dex_list.xlsx")["Dex Name"].tolist()
    #dropdown = refresh_dex_list()
    #updated_list=pd.read_excel("dataframes/dex_list.xlsx")["Dex Name"].tolist()
    #new_list = list(set(updated_list) - set(old_list))
    #if len(new_list) == 0:
        #gr.Info("No new DEXs added to the dropdown.")
    # Show the new DEXs in the dropdown
    #else:
        #gr.Info(f"New DEXs added to the dropdown: {', '.join(new_list)}")
    for dex_name in progress.tqdm(old_list):
        _, results, _ = user_interaction(dex_name, k, co, cs)
    dropdown = gr.Dropdown(
                label="DEX Name",
                choices=pd.read_excel("dataframes/dex_list.xlsx")["Dex Name"].tolist(),
                value="Uniswap v3",
                allow_custom_value=True,
                filterable=True)
    dropdown1 = gr.Dropdown(
                label="DEX Name",
                choices=pd.read_excel("dataframes/dex_list.xlsx")["Dex Name"].tolist()
                )
    dropdown2 = gr.Dropdown(
                label="DEX Name",
                choices=pd.read_excel("dataframes/table.xlsx")["Dex name"].tolist())
    dropdown3 = gr.Dropdown(
                label="DEX Name",
                choices=pd.read_excel("dataframes/table.xlsx")["Dex name"].tolist())

    table = gr.Dataframe(
            headers=["DEX name", "liquidity_model", "license"],
            datatype=["str", "markdown", "markdown"],
            value = pd.read_excel("dataframes/table.xlsx"),
            wrap=True,
            height=1000
        )
    return dropdown, dropdown1, dropdown2, dropdown3, results, table

def search_fn(dex_name):
    dex_row = pd.DataFrame({"Dex name": [], "liquidity_model": [], "license": []})
    visible = False
    if not dex_name:
        gr.Info("Please select a DEX to search for.")
    else:
        visible = True
        dex_row = pd.read_excel("dataframes/table.xlsx")[pd.read_excel("dataframes/table.xlsx")["Dex name"] == dex_name]
    return gr.Column(visible=visible), gr.DataFrame(
                    headers=["DEX name", "liquidity_model", "license"],
                    datatype=["str", "markdown", "markdown"],
                    value=dex_row,
                    wrap=True,
                    height=1000
                    )

