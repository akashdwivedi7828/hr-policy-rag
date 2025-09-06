import streamlit as st
from document_processor import DocumentProcessor
from hr_rag import HRRAGSystem

st.set_page_config(page_title="HR Knowledge Assistant", page_icon="🏢", layout="wide")
st.title("🏢 HR Knowledge Assistant")

# Initialize RAG and Document Processor
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = HRRAGSystem()
    st.session_state.doc_processor = DocumentProcessor()
    st.session_state.query_history = []

# Create UI with tabs
tab1, tab2 = st.tabs(["📄 Upload Documents", "💬 Chat with HR Assistant"])

# Tab 1: Upload Documents
with tab1:
    st.header("Upload HR Documents")
    uploaded_files = st.file_uploader(
        "Choose files (PDF or DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Process and Store"):
        with st.spinner("Processing documents..."):
            total_chunks = 0
            for file in uploaded_files:
                chunks = st.session_state.doc_processor.process_uploaded_file(file)
                if chunks:
                    stored = st.session_state.rag_system.store_documents(chunks, file.name)
                    total_chunks += stored
                    st.success(f"{file.name}: {stored} chunks stored.")
            st.success(f"✅ Total: {total_chunks} chunks stored successfully.")

# Tab 2: Chat Interface
with tab2:
    st.header("Ask HR Policy Questions")

    sample_qs = [
        "How many vacation days do new employees get?",
        "What's the remote work policy?",
        "How do I apply for parental leave?",
        "What are the health insurance options?"
    ]
    selected_sample = st.selectbox("Pick a sample question or type your own:", [""] + sample_qs)
    user_query = st.text_input("Your question:", value=selected_sample)

    if user_query and st.button("Ask"):
        with st.spinner("Generating answer..."):
            result = st.session_state.rag_system.generate_answer(user_query)
            st.session_state.query_history.append(user_query)

            st.subheader("📋 Answer")
            st.write(result["answer"])

            if result["sources"]:
                st.subheader("📚 Sources")
                for src in result["sources"]:
                    st.markdown(f"- {src}")

    if st.session_state.query_history:
        st.divider()
        st.subheader("🕓 Recent Questions")
        for q in reversed(st.session_state.query_history[-5:]):
            st.markdown(f"- {q}")