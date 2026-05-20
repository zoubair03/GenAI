import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
import numexpr as ne
from langchain_classic.agents import Tool, AgentExecutor, create_react_agent
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_classic.prompts import PromptTemplate

# --- Processing Functions (From main.py) ---
def process_hardware_pdfs(pdf_directory):
    st.info(f"Scanning {pdf_directory} for datasheets...")
    all_pages = []

    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)
        st.warning(f"Created {pdf_directory}. Please add PDFs and rebuild.")
        return []

    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_directory, filename)
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            all_pages.extend(pages)

    if not all_pages:
        st.warning("No PDFs found. Add some to build the knowledge base.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    chunks = text_splitter.split_documents(all_pages)
    st.success(f"Split documents into {len(chunks)} searchable chunks.")
    return chunks

def build_chroma_database(chunks, persist_directory):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    st.success("Knowledge base built and saved successfully!")
    return vector_db


# --- Caching expensive agent operations ---
@st.cache_resource
def init_agent():
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Groq API Key is not set!")
        return None

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_directory = "./chroma_db"

    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)

    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 30})

    datasheet_tool = create_retriever_tool(
        retriever=retriever,
        name="Datasheet_Search",
        description="Crucial: Use this FIRST to find exact pinouts, operating voltages, max current draw, memory/cache sizes,thermal specs and general informations from the manufacturer PDFs."
    )

    def calculate_math(query: str) -> str:
        try:
            result = ne.evaluate(query)
            return f"Calculation Result: {result}"
        except Exception as e:
            return f"Error: Action Input MUST be a pure arithmetic expression. You provided: '{query}'. Please re-format it as pure math (e.g., '3.3 / 330 * 1000')."

    def generate_firmware_code(query: str) -> str:
        code_prompt = PromptTemplate.from_template(
            "You are an expert firmware engineer (micropython/C++). Write clean, well-commented code for the following request. "
            "IMPORTANT RULES: "
            "1. Always use the EXACT names, passwords, pins, and values provided in the request. Do not use placeholders like 'your_wifi_password'. "
            "2. Format the code beautifully using Markdown code blocks (e.g. ```python ... ```). "
            "Request: {request}"
        )
        chain = code_prompt | llm
        response = chain.invoke({"request": query})

        return response.content if hasattr(response, 'content') else str(response)

    tools = [
        datasheet_tool,
        Tool(
            name="Calculator",
            func=lambda q: calculate_math(q),
            description="Use for Ohm's law, calculating required resistor values, battery life, or stepper motor step rates."
        ),
        Tool(
            name="Firmware_Writer",
            func=lambda q: generate_firmware_code(q),
            description="Use this tool to generate Arduino C++ or MicroPython code snippets for initializing sensors or driving motors."
        )
    ]

    iot_template = """You are a strict, highly logical Senior Embedded Systems Architect. 
    Your job is to assist engineers using ESP32, ESP8266, and Arduino modules.

    STRICT RULES:
    1. Safety First: Always warn the user about voltage mismatches.
    2. Evidence-Based: Base your technical specs on the Datasheet_Search tool. Do not guess.
    3. Missing Data Protocol: If you use Datasheet_Search and the exact answer is NOT in the returned text, DO NOT guess and DO NOT search again. Immediately output the Final Answer stating the information is missing.
    4. NO CHIT-CHAT: Never use conversational filler like "I see what's going on", "I'd be happy to help", or "Let's get started". 

    You have access to the following tools:
    {tools}

    FORMAT INSTRUCTIONS - YOU MUST STRICTLY FOLLOW ONE OF THESE TWO FORMATS:

    Format Option 1 (To use a tool):
    Thought: I need to use a tool to answer the question.
    Action: The tool to use, MUST be one of [{tool_names}]
    Action Input: The input for the tool.
        - For Datasheet_Search, this should be specific keywords from the user's question (e.g., "ESP32-S3 operating voltage").
        - For Firmware_Writer, this should be the user's full request.
        - For Calculator, this MUST be ONLY the mathematical expression (e.g., if the user asks "what is the current for 3.3V and 330 ohms in milliamps?", the Action Input MUST be "3.3 / 330 * 1000").
    Observation: the result of the action
    
    (After Observation, you MUST use Format Option 2)

    Format Option 2 (To answer the user):
    Thought: I now have the final answer.
    Final Answer: The final answer to the original input question.

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""

    custom_prompt = PromptTemplate.from_template(iot_template)

    agent = create_react_agent(llm, tools, custom_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors="Check your output and make sure it begins with 'Thought:' followed by 'Final Answer:' or 'Action:'. If you used a tool, make sure you used 'Action Input:' properly.",
        max_iterations=5
    )

    return agent_executor


# --- Streamlit UI Configurations ---
st.set_page_config(
    page_title="Hardware Architect AI",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .chat-bubble {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .welcome-text {
        text-align: center;
        color: #666;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.title("Settings")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    with st.expander("Knowledge Base Management", expanded=True):
        st.markdown("Upload new datasheets directly to your local library.")

        # New Uploader Feature
        uploaded_files = st.file_uploader("Upload PDF Datasheets", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            if st.button("Save & Rebuild Database"):
                with st.spinner("Saving files and processing DB..."):
                    # Ensure directory exists
                    os.makedirs("./iot_devices_datasheets", exist_ok=True)

                    # Save uploaded files to disk
                    for uploaded_file in uploaded_files:
                        with open(os.path.join("./iot_devices_datasheets", uploaded_file.name), "wb") as f:
                            f.write(uploaded_file.getbuffer())

                    # Process and build DB
                    chunks = process_hardware_pdfs("./iot_devices_datasheets")
                    if chunks:
                        build_chroma_database(chunks, "./chroma_db")
                        st.cache_resource.clear()
                        st.success(f"Successfully processed {len(uploaded_files)} files!")

        st.divider()
        if st.button("Force Rebuild (Existing Files)"):
            with st.spinner("Processing existing PDFs..."):
                chunks = process_hardware_pdfs("./iot_devices_datasheets")
                if chunks:
                    build_chroma_database(chunks, "./chroma_db")
                    st.cache_resource.clear()
                    st.success("Database rebuilt successfully!")

# --- Main Chat Area ---
st.title("Embedded Systems AI Assistant")
st.markdown("Your expert hardware architect for ESP32, ESP8266, and Arduino projects.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the agent
agent_executor = init_agent()

# 3. Welcome Screen & Quick Prompts
if not st.session_state.messages:
    st.markdown('<div class="welcome-text">', unsafe_allow_html=True)
    st.markdown("### 👋 Welcome! How can I help you build today?")
    st.markdown("I can search manufacturer datasheets, perform Ohm's law calculations, and write microcontroller firmware.")
    st.markdown('</div>', unsafe_allow_html=True)
# Check if a preset prompt was clicked
user_prompt = st.chat_input("E.g., What is the operating voltage of the ESP32?")
if getattr(st.session_state, 'preset_prompt', None):
    user_prompt = st.session_state.preset_prompt
    st.session_state.preset_prompt = None

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_prompt:
    if agent_executor is None:
        st.error("Agent could not be initialized. Please check your API key.")
    else:
        # Display user message
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = agent_executor.invoke({"input": user_prompt})
                    result_text = response["output"]
                except Exception as e:
                    result_text = f"An error occurred: {str(e)}"

                # If the response contains literal \n characters, evaluate them correctly
                if r'\n' in result_text:
                    result_text = result_text.replace(r'\n', '\n')

                st.markdown(result_text)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": result_text})