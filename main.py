import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_classic.chains import RetrievalQA
import numexpr as ne
from langchain_classic.agents import Tool, AgentExecutor, create_react_agent
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_classic.prompts import PromptTemplate


def process_hardware_pdfs(pdf_directory):
    print(f"Scanning {pdf_directory} for datasheets...")
    all_pages = []

    # 1. Load all PDFs from the folder (The "Bronze" Layer)
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_directory, filename)
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            all_pages.extend(pages)
            print(f"Loaded {filename}: {len(pages)} pages.")

    # 2. Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    chunks = text_splitter.split_documents(all_pages)
    print(f"\nSuccess! Split the documents into {len(chunks)} searchable chunks.")
    return chunks


def build_chroma_database(chunks, persist_directory):
    print("Loading the Hugging Face embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"Embedding {len(chunks)} chunks and saving to '{persist_directory}'...")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    print("Success! Your knowledge base is built and saved.")
    return vector_db





def build_modern_hardware_agent(vector_db):
    print("Initializing local LLM via Ollama...")
    llm = ChatOllama(model="llama3", temperature=0.1) # Added low temperature for technical accuracy

    print("Creating Chains...")

    print("Equipping the Agent with IoT Tools...")
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 30})

    datasheet_tool = create_retriever_tool(
        retriever=retriever,
        name="Datasheet_Search",
        description="Crucial: Use this FIRST to find exact pinouts, operating voltages, max current draw, memory/cache sizes,thermal specs and general informations from the manufacturer PDFs."
    )

    tools = [
        datasheet_tool,
        Tool(
            name="Calculator",
            func=lambda q: calculate_math(q),
            description="Use for Ohm's law, calculating required resistor values, battery life, or stepper motor step rates."
        ),
        Tool(
            name="Firmware_Writer",
            func=lambda q: generate_firmware_code(q, llm),
            description="Use this tool to generate Arduino C++ or MicroPython code snippets for initializing sensors or driving motors."
        )
    ]

    print("Injecting Custom IoT Prompt Engineering...")
    # This is the secret sauce. We define the persona, rules, and the strict ReAct format.
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
    Thought: I need to use a tool to find this.
    Action: the action to take, MUST be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this can repeat N times)

    Format Option 2 (To answer the user):
    Thought: I now know the final answer, OR the information is missing.
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""

    custom_prompt = PromptTemplate.from_template(iot_template)

    # Create the modern agent logic using our custom prompt
    agent = create_react_agent(llm, tools, custom_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    return agent_executor

#tools
def calculate_math(query: str) -> str:
    try:
        # Evaluate the math directly using Python
        result = ne.evaluate(query)
        return f"Calculation Result: {result}"
    except Exception as e:
        # If the LLM passes "V = 3.3", we bounce it back and teach it to format correctly
        return "Error: Action Input MUST be a pure arithmetic expression with numbers and operators only (e.g., '3.3 / 330 * 1000'). Do NOT use variables, letters, or equal signs."
def generate_firmware_code(query: str, llm) -> str:
    code_prompt = PromptTemplate.from_template(
        "You are an expert firmware engineer. Write clean, well-commented code for the following request. "
        "Prefer Arduino C++ or MicroPython. Only output the code and brief wiring instructions. Request: {request}"
    )
    # Using the modern pipe syntax for LangChain
    chain = code_prompt | llm
    return chain.invoke({"request": query})

if __name__ == "__main__":
    # Ensure your target directory actually exists to prevent FileNotFoundError
    if not os.path.exists("./iot_devices_datasheets"):
        print("Error: The directory './iot_devices_datasheets' does not exist.")
        print("Please create it and add your PDFs.")
    else:
        # 1. Process the PDFs
        my_chunks = process_hardware_pdfs("./iot_devices_datasheets")

        # 2. Build the database
        db = build_chroma_database(my_chunks, "./chroma_db")

        # 3. Build the agent
        my_agent = build_modern_hardware_agent(db)

        # 4. Query the agent using the modern .invoke() syntax
        print("\n--- Running Agent Query ---")

        querys = ["What is the cache size of the ESP8266EX","What is the operating voltage range of the ESP32-S3, and what are its strapping pins?","If I have a 3.3V circuit and a 330-ohm resistor, calculate the current in milliamps.","Write a simple MicroPython script to connect an ESP8266 to a Wi-Fi network named 'TestNet' with password '12345'."]
        for query in querys:
            response = my_agent.invoke({"input": query})

            print("\n--- Final Output ---")
            print(response["output"])