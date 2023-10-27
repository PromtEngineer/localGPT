from utils import *
import os
import pandas as pd


# Create table.xlsx if it doesn't exist
if not os.path.exists("dataframes/table.xlsx"):
    df = pd.DataFrame({"Dex name": [], "liquidity_model": [], "license": []})
    df.to_excel("dataframes/table.xlsx", index=False)

solution1 = "images/chatwdoc (1).png"
solution2 = "images/chatwdoc (2).png"

with gr.Blocks(theme=gr.themes.Default(primary_hue='indigo', secondary_hue='orange')) as demo:
    # title
    gr.Markdown("<h1 style='color: #6C63FF; font-size: 40px; text-align: center;'>DEX explorer</h1>")
    gr.Markdown("<p style='font-size: 20px; text-align: center; margin-top: 10px;'>An app that helps you find answers to questions about a DEX.</p>")

    with gr.Tab("Table"):
        table = gr.Dataframe(
            headers=["DEX name", "liquidity_model", "license"],
            datatype=["str", "markdown", "markdown"],
            value = pd.read_excel("dataframes/table.xlsx"),
            wrap=True,
            height=1000
        )
        #refresh_button = gr.Button("Refresh Table")
        #refresh_button.click(refresh_table, outputs=[table], show_progress=True)
    with gr.Tab("Search"):
        with gr.Row():
            dex_to_search_from_table = gr.Dropdown(
                                    label="DEX Name",
                                    choices=pd.read_excel("dataframes/table.xlsx")["Dex name"].tolist(),
                                    info="Choose the DEX to search from the table."
                                    )
            search_button = gr.Button("Search", variant='primary')
        with gr.Column(visible=False) as search_column:
            search_result = gr.DataFrame(
                headers=["DEX name", "liquidity_model", "license"],
                datatype=["str", "markdown", "markdown"],
            )
        search_button.click(search_fn, inputs=[dex_to_search_from_table], outputs=[search_column, search_result], show_progress=True)

    with gr.Tab("Interact with the app"):
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        dex = gr.Dropdown(
                            label="DEX Name",
                            choices=pd.read_excel("dataframes/dex_list.xlsx")["Dex Name"].tolist(),
                            value="Uniswap v3",
                            allow_custom_value=True,
                            filterable=True
                            )
                        update_dex_list_button = gr.Button("Update DEX List", variant='primary')
                    gr.Slider(minimum=0, maximum=1, value=0, label="Temperature", info="Choose between 0 and 1")
                    gr.Slider(minimum=0, maximum=1, value=0, label="Top P", info="Choose between 0 and 1")
                with gr.Column("Ingesting Parameters"):
                    cs = gr.Number(label="Chunk Size", value=500)
                    co = gr.Number(label="Chunk Overlap", value=100)
                    k = gr.Number(label="Number of Chunks", minimum=1, maximum=5, value=3, info="Choose between 1 and 5")
            with gr.Column():
                results = gr.JSON(label="Results")
        with gr.Row():
            extract_button = gr.Button("Extract", variant='primary')
            update_and_extract_all_button = gr.Button("Update and Extract All", variant='primary')

        extract_button.click(user_interaction, inputs=[dex, k, co, cs], outputs=[table, results, dex_to_search_from_table])
        update_dex_list_button.click(refresh_dex_list, outputs=[dex], show_progress=True)

    # Add a new advanced features tab
    # Theses features are : deleting a dex name from the dropdown list, deleting a dex_name from the table
    # deleting the entire table, deleting the entire dropdown list
    with gr.Tab("Delete"):
        with gr.Row():
            dex_to_delete_from_dropdown = gr.Dropdown(
                                        label="DEX Name",
                                        choices=pd.read_excel("dataframes/dex_list.xlsx")["Dex Name"].tolist(),
                                        info="Choose the DEX to delete from the dropdown list."
                                        )
            with gr.Column():
                confirm_delete_dex_from_dropdown = gr.Checkbox(
                                                    label="Confirm",
                                                    info="This will delete the DEX from the dropdown \
                                                        list and cannot be undone. By checking this box, \
                                                        you confirm that you want to delete the DEX from the dropdown list."
                                                    )
                delete_dex_from_dropdown_button = gr.Button("Delete DEX from dropdown list", variant='primary')
        with gr.Row():
            dex_to_delete_from_table = gr.Dropdown(
                                    label="DEX Name",
                                    choices=pd.read_excel("dataframes/table.xlsx")["Dex name"].tolist(),
                                    info="Choose the DEX to delete from the table."
                                    )
            with gr.Column():
                confirm_delete_dex_from_table = gr.Checkbox(
                                                label="Confirm",
                                                info="This will delete the DEX from the table and \
                                                    cannot be undone. By checking this box, you \
                                                    confirm that you want to delete the DEX from the table."
                                                    )
                delete_dex_from_table_button = gr.Button("Delete DEX from table", variant='primary')
        with gr.Row():
            with gr.Column():
                confirm_delete_table = gr.Checkbox(label="Confirm", info="This will delete the table and cannot be undone. By checking this box, you confirm that you want to delete the table.")
                delete_table_button = gr.Button("Delete Table", variant='primary')
            with gr.Column():
                confirm_delete_dropdown_list = gr.Checkbox(label="Confirm", info="This will delete the dropdown list and cannot be undone. By checking this box, you confirm that you want to delete the dropdown list.")
                delete_dropdown_list_button = gr.Button("Delete Dropdown List", variant='primary')

        delete_dex_from_table_button.click(delete_dex_from_table, inputs=[dex_to_delete_from_table, confirm_delete_dex_from_table], outputs=[dex_to_search_from_table, dex_to_delete_from_table, table], show_progress=True)
        delete_dex_from_dropdown_button.click(delete_dex_from_dropdown, inputs=[dex_to_delete_from_dropdown, confirm_delete_dex_from_dropdown], outputs=[dex, dex_to_delete_from_dropdown], show_progress=True)
        delete_dropdown_list_button.click(delete_dropdown_list, inputs=[confirm_delete_dropdown_list], outputs=[dex, dex_to_delete_from_dropdown], show_progress=True)
        delete_table_button.click(delete_table, inputs=[confirm_delete_table], outputs=[dex_to_search_from_table, dex_to_delete_from_table, table], show_progress=True)

    update_and_extract_all_button.click(
            update_and_extract_all,
            inputs=[k, co, cs],
            outputs=[dex, dex_to_delete_from_dropdown, dex_to_delete_from_table, dex_to_search_from_table, results, table],
            show_progress=True
            )

    with gr.Tab("How does it work?"):
        gr.Gallery(label="Solution", value=[solution1, solution2], columns=2, rows=1, object_fit="scale-down")


if __name__ == "__main__":
    demo.queue(concurrency_count=20).launch()
