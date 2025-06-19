import streamlit as st
import read_pdf as rpdf
import question_gen as qgen

st.title("PDF Question Generator:")

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

if(uploaded_file is not None):
    extracted_text = rpdf.extract_text_from_pdf(uploaded_file)
    questions = qgen.generate_questions(extracted_text)
    
    st.markdown("### Generated Questions:")
    st.text_area("Questions", questions, height=600)
    
    st.markdown("### Extracted Text:")
    st.text_area("Text", extracted_text, height=300)