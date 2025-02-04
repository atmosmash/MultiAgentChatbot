import streamlit as st
import asyncio
from semantic_kernel import Kernel
from services import Service
from service_settings import ServiceSettings
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import KernelArguments

# Configure Kernel
kernel = Kernel()

# Service settings
service_settings = ServiceSettings.create()
selectedService = (
    Service.OpenAI if service_settings.global_llm_service is None
    else Service(service_settings.global_llm_service.lower())
)
kernel.remove_all_services()

# Add OpenAI service
service_id = "default"
kernel.add_service(
    OpenAIChatCompletion(
        service_id=service_id,
        ai_model_id="gpt-3.5-turbo"
    ),
)

# Load plugins
plugin_directory = r"C:/Users/ujjjain/OneDrive - Publicis Groupe/projects/FileQuest/Semantic Kernel/python/New/MA/prompt_template_samples"

fun_plugin = kernel.add_plugin(parent_directory=plugin_directory, plugin_name="FunPlugin")
chat_plugin = kernel.add_plugin(parent_directory=plugin_directory, plugin_name="ChatPlugin")
coding_plugin = kernel.add_plugin(parent_directory=plugin_directory, plugin_name="CodingPlugin")
summarize_plugin = kernel.add_plugin(parent_directory=plugin_directory, plugin_name="SummarizePlugin")
intent_plugin = kernel.add_plugin(parent_directory=plugin_directory, plugin_name="IntentDetectionPlugin")

# Async wrapper to run in Streamlit
def run_async(func, *args):
    return asyncio.run(func(*args))

# Function to determine intent and route queries
async def handle_query(user_input):
    intent_function = intent_plugin["AssistantIntent"]
    detected_intent = await kernel.invoke(intent_function, KernelArguments(input=user_input))

    if isinstance(detected_intent, dict):
        detected_intent = detected_intent.get("intent", "").strip()
    else:
        detected_intent = str(detected_intent).strip()

    coding_intents = {"FindContentAboutX"}
    summarization_intents = {"AutoSummarize", "OnDemandSummary", "OnDemandNotes"}

    if detected_intent in coding_intents:
        return await kernel.invoke(coding_plugin["Code"], KernelArguments(input=user_input))
    elif detected_intent in summarization_intents:
        return await kernel.invoke(summarize_plugin["Summarize"], KernelArguments(input=user_input))
    else:
        return await kernel.invoke(chat_plugin["Chat"], KernelArguments(input=user_input))

# Streamlit UI
st.title("💬 Multi-Agent Chatbot using Microsoft Semantic Kernel")

# Display chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    response = run_async(handle_query, user_input)

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
