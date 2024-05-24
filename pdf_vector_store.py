from text_vector_store import *
import PyPDF2
import ollama

def create_pdf_vector_store(file_name,model_name,batch_size=5):
    pdffile=open(file_name,'rb')
    pdfReader=PyPDF2.PdfReader(pdffile)

    document=list()
    for page in pdfReader.pages:
        for chunk in page.extract_text().split('\n'):
            if chunk!='\n':
                document.append(chunk)
    store=dict()
    re_index=list()    
    document.append(chunk)
    store=dict()
    re_index=list()
    for i in range(0,len(document)-1,batch_size):
        chunk=' '.join(document[i:(i+batch_size)])
        re_index.append(chunk)
        vector=ollama.embeddings(model=model_name,prompt=chunk)
        store.update({i:vector['embedding']})
    return store,re_index

if __name__ == "__main__":
    model_name='all-minilm'
    vector_store,document=create_pdf_vector_store('data/transformer_time_series.pdf',model_name)
    print(len(vector_store))
    while True:
        query = input('Enter your query: ')
        if query=='exit':
            break 
        else:
            response=query_engine(vector_store,document,model_name,query)
            print(response)