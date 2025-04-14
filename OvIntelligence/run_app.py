# --- START OF FILE run_app.py ---
import streamlit as st
import app
import operator_mode

# Store the original function from app.py
original_process_query = app.process_query_with_progress

def patched_process_query_with_progress(query, rag_system, selected_package, user="unknown"):
    # 1) Check if user wants Operator Mode
    if query.strip() == "OmicVerseOperator":
        st.markdown("## Operator Mode Activated!")
        operator_mode.show_tutorial()
        # Stop so we do not continue with the normal RAG pipeline
        st.stop()

    # 2) Otherwise, just run the original function
    return original_process_query(query, rag_system, selected_package, user)

# Monkey-patch the original process_query_with_progress in the imported app
app.process_query_with_progress = patched_process_query_with_progress

def main():
    # Ensure session_state is properly set up BEFORE we run the main app
    app.initialize_session_state()
    # Now call the main function in app.py
    app.main()

if __name__ == "__main__":
    main()
# --- END OF FILE run_app.py ---
