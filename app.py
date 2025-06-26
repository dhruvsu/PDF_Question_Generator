import streamlit as st
import read_pdf as rpdf
import question_gen as qgen
import rag_indexer as rindex

st.title("PDF Q&A:")

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

user_input = st.text_input("Enter a question:")

if(uploaded_file is not None):
    extracted_text = rpdf.extract_text_from_pdf(uploaded_file)
    chunks = rpdf.chunk_text(extracted_text)
    questions = qgen.generate_questions(extracted_text)
    rindex.index_chunks(chunks)
    
    if user_input:
        answer = qgen.run_rag_pipeline(user_input)
        st.write("### Answer")
        st.write(answer)
    
    st.markdown("### Generated Questions:")
    st.text_area("Questions", questions, height=600)
    
    st.markdown("### Extracted Text:")
    st.text_area("Text", extracted_text, height=300)