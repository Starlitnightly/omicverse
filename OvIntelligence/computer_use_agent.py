import streamlit as st
import subprocess
from logger_setup import logger


def execute_command(command: str) -> str:
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"Executed command: {command}")
        return result.stdout or "(no output)"
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {command}", exc_info=True)
        return e.stdout + e.stderr


def computer_use_interface() -> None:
    """Simple Streamlit interface for executing shell commands."""
    st.markdown("## Computer Use Agent")
    st.markdown("Enter a shell command to run on this machine.")
    command = st.text_input("Command", key="computer_use_command")
    if st.button("Run Command"):
        if command.strip():
            with st.spinner("Running command..."):
                output = execute_command(command)
            st.markdown("### Command Output")
            st.code(output)
        else:
            st.error("Please enter a command.")
