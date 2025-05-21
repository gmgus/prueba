import pdfplumber

source = r"docs\51000001-E-25-028607 Re CONSULTA NECESIDAD CAMBIO NORMATIVO PARA EMPLEO MAS.PDF"

import PyPDF2

def extraer_texto_pypdf2(ruta_pdf):
    texto_completo = ""
    with open(ruta_pdf, 'rb') as archivo:
        lector = PyPDF2.PdfReader(archivo)
        for i, pagina in enumerate(lector.pages):
            texto = pagina.extract_text()
            texto_completo += f"Página {i + 1}:\n{texto}\n{'-'*40}\n"
    return texto_completo

txt = extraer_texto_pypdf2(source)

import regex as re

dict = {
    "FECHA": "",
    "ASUNTO": "",
    "DESTINATARIO": ""
}

fecha = re.search(r"FECHA\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})", txt)

if fecha:
    print("Fecha encontrada:", fecha.group(1))
else:
    print("No se encontró la fecha.")

import re

asunto = re.search(r"ASUNTO\s*[:\-]?\s*(.*)", txt)
destinatario = re.search(r"DESTINATARIO\s*[:\-]?\s*(.*)", txt)

# Para obtener solo hasta el primer salto de línea:
asunto_texto = asunto.group(1).split('\n')[0].strip() if asunto else None
destinatario_texto = destinatario.group(1).split('\n')[0].strip() if destinatario else None


print(asunto_texto)
print(destinatario_texto)
