from flask import Flask, request
from llama_index.llms.ollama import Ollama
from werkzeug.utils import secure_filename
from document_ingestion import ingest_data
from document_info_extractor import document_assistant
import os
import json

app = Flask(__name__)

@app.route("/validate_resume", methods=['POST'])
def validate_resume():
    job_desc = request.form.get("Job Description")
    file = request.files["Resume"]

    if not os.path.exists("temp"):
      os.makedirs("temp")
      print("Directory created successfully!")
    else:
        print("Directory already exists!")

    filename = secure_filename(file.filename)
    filepath = os.path.join("temp", filename)
    file.save(filepath)

    response = ingest_data(filepath, job_desc)
    print(response)
    return response

@app.route("/answer_query", methods=["POST"])
def answer_query():
    query = request.form.get("query")
    files = request.files.getlist("files")
    print(files)

    if not os.path.exists("files"):
      os.makedirs("files")
      print("Directory created successfully!")
    else:
        print("Directory already exists!")

    folderpath = "./files"
    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join("files", filename)
        file.save(filepath)

    response = document_assistant(folderpath, query)
    print(response)
    return response


if __name__ == "__main__":
    app.run(debug=True)