from shutil import copy
# import chunking
from dotenv import load_dotenv
import pandas as pd
import gradio as gr
# import indexing
import os
# import preprocess
import pickle
import requests
import shutil
import ingest_functions

# Load environment variables from .env file
load_dotenv()

# Environment variables for Elasticsearch connection
es_user = os.getenv("ES_USERNAME")
es_password = os.getenv("ES_PASSWORD")
index_name = os.getenv("ES_INDEX_NAME")
es_url_delete = f"{os.getenv('ES_URL')}/{index_name}/_delete_by_query"
es_url_list = f"{os.getenv('ES_URL')}/{index_name}/_search"
es_url_refresh = f"{os.getenv('ES_URL')}/{index_name}/_refresh"

def get_source_list():
    """
    Fetches a list of unique sources ('fuente') from the Elasticsearch index.
    
    It sends a query to Elasticsearch to retrieve distinct source names and returns 
    them as a list.
    
    Returns:
        list: A list of sources ('fuente') available in the Elasticsearch index.
    """
    # Refresh Elasticsearch index
    requests.post(es_url_refresh, verify=False)
    
    # Query to fetch distinct sources ('fuente')
    response = requests.get(es_url_list, json={
        "size":0,
        "aggs": {
            "fuentes": {
                "terms": {"field": "fuente"}
            }
        }
    }, verify=False)
    
    # Extract source names from the response
    new_source_list = [fuente["key"] for fuente in response.json()["aggregations"]["fuentes"]["buckets"]]
    return new_source_list

def file_processing(file_list, source_list):
    """
    Processes files by reading their content, chunking, preprocessing, and indexing.
    
    Args:
        file_list (DataFrame): A dataframe containing the file paths and titles.
        source_list (list): A list of selected sources to associate with the files.
    
    Returns:
        tuple: Gradio components for files, file list, and source list after processing.
    """
    titles = []
    
    # Process each file in the list
    for row in file_list.itertuples():
        with open(row.ruta, "r") as file:
            lines = file.readlines()
            title = row.título
            text = "\n".join(lines[1:]).strip()  # Skip the first line, typically headers
            
            # Preprocess text (e.g., clean, normalize, etc.)
            text = preprocess.run(text)
            
            # Chunk the text and run indexing
            chunks = chunking.run(text, title)
            indexing.run(chunks)
        
        titles.append(title)  # Add the title for tracking

    # Gradio components to update UI after file processing
    files_to_upload = gr.Files(value=None, visible=True)
    file_list = gr.DataFrame(value=None, visible=False)
   
    return files_to_upload, file_list

def update_source_list(source_list):
     new_source_list = gr.CheckboxGroup(choices=get_source_list(), value=[])  # Update source list
     return new_source_list

def load_file_titles(files):
    """
    Loads file titles into the Gradio interface.
    
    Args:
        files (list): A list of uploaded file paths.
    
    Returns:
        tuple: Gradio components for file titles and the data frame with file information.
    """
    if files:
        file_list = []
        
        # Extract file names and prepare a list of tuples (file path, file name, title)
        for file in files:
            file_name = file.split("/")[-1]
            file_list.append((file, file_name, file_name))
        
        # Return the file list and a data frame with file information
        return gr.Files(visible=False), gr.DataFrame(
            row_count=[len(files), "fixed"],
            value=file_list,
            visible=True
        )
    else:
        # If no files are provided, return empty Gradio components
        return gr.Files(), gr.DataFrame()

def delete_sources(sources):
    """
    Deletes documents from Elasticsearch based on selected sources.
    
    Args:
        sources (list): List of sources to be deleted from the index.
    
    Returns:
        CheckboxGroup: The updated source list after deletion.
    """
    # Send a delete request to Elasticsearch to remove documents with the selected sources
    response_delete = requests.post(es_url_delete, json={
        "query": {
            "terms": {
                "fuente": sources
            }
        }
    }, verify=False, auth=(es_user, es_password))

def move_to_doc_rep(file):
    """Copia un fichero, a un repositorio indicado

    Args:
        file (_type_): _description_
    """
    # Ruta fija en tu repositorio
    rep_doc = os.getenv("DOC_REP")
    # ruta_fija = "C:/Users/1068429/OneDrive - quest-global.com/Documentos/02 VSCODE/Front_ceset_test/PRUEBA"
    shutil.copy(file.name, os.path.join(rep_doc, os.path.basename(file.name)))

def move_to_pdf_rep(file):
    """Copia un fichero, a un repositorio indicado

    Args:
        file (_type_): _description_
    """
    # Ruta fija en tu repositorio
    pdf_doc = os.getenv("PDF_REP")
    # ruta_fija = "C:/Users/1068429/OneDrive - quest-global.com/Documentos/02 VSCODE/Front_ceset_test/PRUEBA"
    shutil.copy(file.name, os.path.join(pdf_doc, os.path.basename(file.name)))

def move_to_md_rep(file):
    """Copia un fichero, a un repositorio indicado

    Args:
        file (_type_): _description_
    """
    # Ruta fija en tu repositorio
    md_doc = os.getenv("MD_REP")
        # ruta_fija = "C:/Users/1068429/OneDrive - quest-global.com/Documentos/02 VSCODE/Front_ceset_test/PRUEBA"
    shutil.copy(file.name, os.path.join(md_doc, os.path.basename(file.name)))

def procesar(files):
    """Coge los ficheros subidos , mira si están en la base de datos , 
    y los que no los clasifica devolviendo un df
    """

    df_classification = pd.DataFrame(columns=['filename','report','operation','country','from','to'])
    doc_list = []
    # 1. Llamada a ES para recuperar los informes que ya hay indexados
    informes_es = get_source_list() # TIENE QUE ESTAR EN MINUSCULA

    # 2. Recorro los ficheros subidos por el usuario
    for file in files:
        if file.lower() not in informes_es:
            # 2.1. Copiar el archivo al repo de PDFs
            move_to_pdf_rep(file)

            # 2.2. Paso .pdf a .md y lo dejo en repository_md
            ingest_functions.pdf_to_md(file)

            # 2.3. Clasifico y devuelvo un Document() y añado una fila a data
            documento = ingest_functions.classify_document(file)
            data = {
                'filename': documento.path,
                'report': documento.doc_type,
                'operation': documento.operation,
                'country': documento.geo_zone,
                'from': documento.date_from,
                'to': documento.date_to
            }
            df_classification = pd.concat([df_classification, pd.DataFrame([data])], ignore_index=True)
            doc_list.append(documento)
            ## TENGO QUE GUARDAR ESTA LISTA EN UN REPO

            print(f"{file} procesado correctamente")

            return gr.update(value=df_classification, visible=True), gr.update(visible=True)
        
def save_files(df, pkl_path):
    """1. Leo el df editado y voy recorriendo fila a fila
       2. Por cada fila actualizo el Document()
       3. Realizo chunking e indexado    
    """
    # Load the list of Document objects from the .pkl file
    with open(pkl_path, 'rb') as f:
        documents = pickle.load(f)
    
    # Create a dictionary for quick lookup of Document objects by their path
    doc_dict = {doc.path: doc for doc in documents}
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        filename = row['filename']
        
        # Check if the filename exists in the dictionary
        if filename in doc_dict:
            # Update the Document object with new values from the DataFrame
            doc_dict[filename].doc_type = row['report']
            doc_dict[filename].operation = row['operation']
            doc_dict[filename].geo_zone = row['country']
            doc_dict[filename].date_from = row['from']
            doc_dict[filename].date_to = row['to']
    
    # Convert the dictionary back to a list of Document objects
    updated_documents = list(doc_dict.values())
    
    for document in updated_documents:
        chunks = ingest_functions.chunk_document(document)
        # Indexo el documento en el indice de Elastic
        ingest_functions.index_in_ES(chunks)

    return gr.Files(value=None), gr.update(visible=False), gr.update(visible=False)


# Define the Gradio interface and components
with gr.Blocks(title="Gonzalo - IA generativa del ET") as frontend:
    
    # Header block with custom styling
    big_block = gr.HTML("""
    <div style="height: 100px; width: 100%; background-image: url('/gradio_api/file=C:/Users/1068429/OneDrive - quest-global.com/Documentos/06 CASO USO CESET/GZLO_CESET/static/header.jpg'); background-repeat: no-repeat; background-size: cover"></div>
    """)
    
    # Main content layout with two columns
    with gr.Row():
        with gr.Column():
            # File upload section
            files_to_upload = gr.Files(label="Archivos a procesar")
            submit_button = gr.Button("Procesar")
            
        with gr.Column():
            # Source list section
            source_list = gr.CheckboxGroup(
                label="Fuentes disponibles"
                # choices=update_source_list(),
            )
            with gr.Row():
                # Delete button for removing selected sources
                delete_button = gr.Button("Borrar")


    with gr.Row():
        df_clas = gr.DataFrame(
                interactive=True,
                label="Clasificación de informes",
                show_search="search",
                static_columns=[0],
                # headers=["ruta", "fichero", "título"],
                # col_count=[3, "fixed"],
                column_widths=[0, None, None],
                visible=False,
            )
    with gr.Row():
        save_button = gr.Button("Guardar", visible=False)
        
    # Define the interactions and callbacks between Gradio components

    # frontend.load(update_source_list, inputs=None, outputs=source_list)
    # files_to_upload.change(load_file_titles, inputs=[files_to_upload], outputs=[files_to_upload, file_list])
    submit_button.click(procesar, inputs=[files_to_upload], outputs=[df_clas, save_button])
    save_button.click(save_files, inputs=None, outputs=[files_to_upload, df_clas,save_button])
    # .then(update_source_list, inputs=[source_list], outputs=[source_list])
    # delete_button.click(delete_sources, inputs=[source_list], outputs=[source_list]).then(update_source_list, inputs=[source_list], outputs=[source_list])

# Close all Gradio components
gr.close_all()

# Launch the Gradio interface with specific configurations
frontend.queue().launch(
    # server_port=4895, server_name="0.0.0.0",
    inline=False, debug=True,
    allowed_paths=["C:/Users/1068429/OneDrive - quest-global.com/Documentos/06 CASO USO CESET/GZLO_CESET/static/header.jpg"],
    favicon_path="C:/Users/1068429/OneDrive - quest-global.com/Documentos/06 CASO USO CESET/GZLO_CESET/static/favicon-32x32.png",
)
