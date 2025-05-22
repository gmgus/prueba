import re
import pickle

from text_chunk import Chunk
import time
import os
import pickle
from docling.document_converter import DocumentConverter
from document import Document
from datetime import date
import nltk as nl
from pickle import load
import numpy as np
import datetime
from classification.classify import predict_operation, predict_geo_zone, predict_interval_llm, predict_type_doc
from embeddings import get_embeddings
import os
import pickle
import json
import requests
from datetime import datetime
from es_create_index import create_index
from es_eliminate_index import delete_index
import pandas as pd
import requests



# ElasticSearch
index_name = "pruebas_ceset"
url_ES = 'http://elasticsearch:9200'
es_url = f"{url_ES}/{index_name}/_doc/"

# Ollama
url_OL = 'http://ollama:11434' 


pdf_repository = os.getenv("PDF_REP")
md_repository = os.getenv("MD_REP")
pkl_repository = os.getenv("PKL_REP")



def get_distinct_docs_from_index():
    url = "http://elasticsearch:9200/pruebas_ceset/_search"
    headers = {'Content-Type': 'application/json'}
    
    query = {
        "size": 0,
        "aggs": {
            "distinct_fuentes": {
                "terms": {
                    "field": "fuente",
                    "size": 100  # Ajusta el tamaño según tus necesidades
                }
            }
        }
    }
    
    response = requests.post(url, json=query, headers=headers)
    
    if response.status_code == 200:
        resultados = response.json()
        fuentes = [bucket["key"] for bucket in resultados["aggregations"]["distinct_fuentes"]["buckets"]]
        fuentes_lower = {fuente.lower() for fuente in fuentes}
        return fuentes_lower
    else:
        print(f"Error en la consulta: {response.status_code}, {response.text}")
        return []

def delete_tables(text_markdown):
    """
    Elimina todas las tablas en formato Markdown de un texto.

    Args:
        text_markdown (str): Texto en formato Markdown.

    Returns:
        str: Texto sin tablas.
    """
    pattern = r"(\|.+?\|[\r\n]+(\|[-:]+[-|:]*\|[\r\n]+)(\|.+?\|[\r\n]+)*)"
    text_without_tables = re.sub(pattern, "", text_markdown)
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', text_without_tables).strip()

    return cleaned_text 

def pdf_to_md(filename):
    """
        Coge un documento del repositorio de pdfs, lo convierte a md y lo deja en el repositorio de md
    """

    converter = DocumentConverter()

    if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_repository, filename)
            md_path = os.path.join(md_repository, filename.replace(".pdf", ".md"))
            if filename.endswith(".PDF"):
                md_path = os.path.join(md_repository, filename.replace(".PDF", ".md"))
            print(f"Procesando {filename}...")
            start_time = time.time()
            result = converter.convert(pdf_path)

            # Convert to Markdown
            md_content = result.document.export_to_markdown()
            md_content = delete_tables(md_content)
            # Eliminar las etiquetas de imágenes <!-- image -->
            md_content = re.sub(r'<!-- image -->', '', md_content)
            
            with open(md_path, 'w', encoding='utf-8') as md_file:
                md_file.write(md_content)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Procesado {filename} en {elapsed_time:.2f} segundos.")

def classify_document(filename_pdf):
    """
        Coge un documento del repositorio md y lo clasifica según sus metadatos, obteniendo así un objeto de tipo Document()
    """
    if filename_pdf.endswith(".PDF"):
        filename_md = os.path.join(md_repository, filename_pdf.replace(".PDF", ".md"))
    else:
        filename_md = os.path.join(md_repository, filename_pdf.replace(".pdf", ".md"))
    with open(filename_md, "r", encoding="utf-8") as file:
        text = "".join(file.readlines())

    ## Clasificar el documento 
    documento = Document()
    documento.content = text
    documento.operation = predict_operation(text)  
    documento.doc_type = predict_type_doc(text)
    documento.geo_zone = predict_geo_zone(text)
    intervalo = predict_interval_llm(text[:700])
    documento.path = os.path.basename(filename_md).replace('.md', '.pdf')

    if isinstance(intervalo, dict):
        if intervalo['start']:
            documento.date_from = intervalo['start']
        else:
            documento.date_from = None
            print(f"Intervalo inicial del informe {documento.path} no reconocido")

        if intervalo['end']:
            documento.date_to = intervalo['end']
        else:
            documento.date_to = None
            print(f"Intervalo final del informe {documento.path} no reconocido")
        print(f"INFORME: {documento.doc_type}: {file}")
        print(f"OPERACION: {documento.operation} con fecha de {documento.date_from} a {documento.date_to}")

        return documento
    
    else:
        print(f"Algo ha ido  en el intervalo: {intervalo}")   

        print("El intervalo proporcionado no es un diccionario válido. Por favor, introduce las fechas manualmente.")
        try:
            start_date = input("Introduce la fecha de inicio (formato YYYY-MM-DD): ")
            documento.date_from = start_date if start_date else None

            end_date = input("Introduce la fecha de fin (formato YYYY-MM-DD): ")
            documento.date_to = end_date if end_date else None
        except Exception as e:
            print(f"Error al introducir las fechas: {e}")

            print(f"INFORME: {documento.doc_type}: {file}")
            print(f"OPERACION: {documento.operation}")
        return documento 

def chunk_document(documento):
    """
        Coge un objeto de tipo Document(), hace el chunking de este y lo guarda en el repositorio de pkl
    """
    # Solo se divide cuando hay un título principal, no cuando hay subtítulos como '1.1', '2.2', etc.
    componentes = re.split(r'(?=^##\s\d+\.\s)', documento.content, flags=re.MULTILINE)
    chunks = []
    
    i=1

    # Procesar cada componente dividido
    for componente in componentes:
        
        chunk = Chunk()
        chunk.texto = componente
        chunk.operation = documento.operation
        chunk.doc_type = documento.doc_type
        chunk.geo_zone = documento.geo_zone
        chunk.date_from = documento.date_from
        chunk.date_to = documento.date_to
        chunk.fuente = documento.path
        chunks.append(chunk)

    # # Guardar los chunks en un archivo .pkl
    # if documento.path.endswith('.PDF'):
    #     filename_pkl = os.path.join(pkl_repository, documento.path.replace(".PDF", ".pkl"))
    # else:
    #     filename_pkl = os.path.join(pkl_repository, documento.path.replace(".pdf", ".pkl"))

    # with open(filename_pkl, "wb") as file:
    #     pickle.dump(chunks, file)

    # print(f"{len(chunks)} Chunks guardados en \"{filename_pkl}\"")


    return chunks

# Función para procesar los archivos .pkl dentro de un directorio
def index_in_ES(chunks):
    # if filename.endswith(".PDF"):
    #     filename_pkl = os.path.join(pkl_repository, filename.replace(".PDF", ".pkl"))
    # else:
    #     filename_pkl = os.path.join(pkl_repository, filename.replace(".pdf", ".pkl"))
    
    # # Cargar el archivo .pkl
    # with open(filename_pkl, "rb") as file:
    #     chunks = pickle.load(file)

    # print(f"Loaded {len(chunks)} chunks from {filename_pkl}.")
    
    # Procesar cada chunk
    for i, chunk in enumerate(chunks):

        # formatted_date_from = format_date(chunk.date_from) if chunk.date_from else None
        # formatted_date_to = format_date(chunk.date_to) if chunk.date_to else None
        embedding = get_embeddings(chunk.texto)
        if chunk.date_from == None or chunk.date_to == None:
            # Indexar el chunk en ElasticSearch
            response = requests.post(es_url, json = {
                "content": chunk.texto,
                "embedding": embedding,
                "fuente": chunk.fuente,
                "geo_zone": chunk.geo_zone,
                "operation": chunk.operation,
                "doc_type": chunk.doc_type
            }, verify=False)
        else:
            # Indexar el chunk en ElasticSearch
            response = requests.post(es_url, json = {
                "content": chunk.texto,
                "embedding": embedding,
                "fuente": chunk.fuente,
                "geo_zone": chunk.geo_zone,
                "operation": chunk.operation,
                "doc_type": chunk.doc_type,
                "date_from": chunk.date_from,
                "date_to": chunk.date_to,
            }, verify=False)
        
        # try:
        #     response_data = response.json()
        #     if response.status_code == 201:  # Verifica que la respuesta sea de tipo 'Created'
        #         print(f"Indexed Chunk {i + 1} from {filename_pkl}: {response_data.get('result', 'No result key')}")
        #     else:
        #         # Si la respuesta no es 201, imprime el error para diagnóstico
        #         print(f"Error indexing Chunk {i + 1} from {filename_pkl}: {response_data}")
        # except json.JSONDecodeError:
        #     print(f"Error decoding response for Chunk {i + 1} from {filename_pkl}: {response.text}")




def __main__():

    # 1. Saco una lista de los documentos ya indexados, si filename_pdf = a alguno de la lista, paso al siguiente
    informes_indexados = get_distinct_docs_from_index()
    i = 0
    # print(informes_indexados)
    df_test = pd.DataFrame(columns=['filename','report','operation','country','from','to'])
    
    
    # SI HACE FALTA REGENERO INDICE
    # delete_index(index_name, url_ES)
    # create_index(index_name, url_ES)

    # 2. Recorro todos los docs del repositorio de informes
    for filename_pdf in os.listdir(pdf_repository):
        if filename_pdf.lower() in informes_indexados:
            continue
        else:
            # 2.1. Paso .pdf a .md y lo dejo en repository_md
            pdf_to_md(filename_pdf)

            # 2.2. Clasifico los metadatos del documento y guardo las clasificaciones en un csv
            documento = classify_document(filename_pdf)
            data = {
                'filename': documento.path,
                'report': documento.doc_type,
                'operation': documento.operation,
                'country': documento.geo_zone,
                'from': documento.date_from,
                'to': documento.date_to
            }
            df_test = pd.concat([df_test, pd.DataFrame([data])], ignore_index=True)
            print(f"{filename_pdf} procesado correctamente")
            print()

            # 2.3. Realizo el particionamiento de los documentos
            chunk_document(documento)
            print()

            # 2.4. Indexo el documento en el indice de Elastic
            index_in_ES(filename_pdf)
            i+=1


    name_csv = f"classification_test_{datetime.today()}.csv"
    df_test.to_csv(name_csv,sep=",")
    print(f"CARGA FINALIZADA, {len(os.listdir(pdf_repository))} INFORMES INDEXADOS")
    print(f"{name_csv} guardado")


