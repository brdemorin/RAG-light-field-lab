# the index_names_light_field_lab.json" file is in brian/chat which is the root folder for the flask app. This will load the light-field-lab pinecone index by default if JSON can't be accessed
# streamlit run chat.py; to close, ctrl+C in terminal first then close browser
# >3K files chunked into 2K chunks; 8 chunks retrieved
# With translation. Choice of similarity search or .as_retriever (which seem identical except I can choose to retrieve more than 4 chunks if I want with similarity search) and finally able to get retrieved docs while still using the chain.invoke sequence 

import os
#from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
#from langchain.prompts import ChatPromptTemplate # deprecated
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from google.cloud import translate_v2 as translate
from langdetect import detect
import streamlit as st
import tiktoken
# Set verbosity using the new approach
from langchain import globals as langchain_globals
langchain_globals.set_verbose(True)
from langchain_groq import ChatGroq
import json
from google.oauth2 import service_account
import requests
from langchain_community.vectorstores.chroma import Chroma
from google.cloud import texttospeech
import base64


# consider passing "--theme.base dark" in the streamlit run command unless I want to put here .streamlit/config.toml

# set_page_config documentation: https://docs.streamlit.io/develop/api-reference/configuration/st.set_page_config
st.set_page_config(layout="wide")


# if using secrets in .env in root folder:
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
# # I had to take the .json I downloaded from google cloud service account and convert into a single-line string then update .env with it
# GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# if using secrets in secrets.toml (which is what streamlit.io cloud supports), must create .streamlit subfolder and put secrets.toml in it then also load as environment variables below
OPENAI_API_KEY = st.secrets["secrets"]["OPENAI_API_KEY"]
GROQ_API_KEY = st.secrets["secrets"]["GROQ_API_KEY"]
PINECONE_API_KEY = st.secrets["secrets"]["PINECONE_API_KEY"]
PINECONE_API_ENV = st.secrets["secrets"]["PINECONE_API_ENV"]

# google service cloud API for translation fixed by gemini 1.5 pro
GOOGLE_APPLICATION_CREDENTIALS = st.secrets["secrets"]["GOOGLE_APPLICATION_CREDENTIALS"]
credentials_info = json.loads(GOOGLE_APPLICATION_CREDENTIALS) 
credentials = service_account.Credentials.from_service_account_info(credentials_info)
translate_client = translate.Client(credentials=credentials)
tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
# Now set SOME of the secrets from secrets.toml as environment variables. It doesn't appear openAI needs this but I did it anyhow. If I try to do this for google, google will stop working
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
#os.environ["PINECONE_API_ENV"] = PINECONE_API_ENV
os.environ["PINECONE_ENVIRONMENT"] = PINECONE_API_ENV 


# I get faster and shorter responses API'ing into GROQ and using the llama3 70b model vs openAI GPT 3.5 Turbo and price is about same ... but sacrifice a decent amount of context window. I change to GPT3.5 when retrieval is more than Groq can handle for a given question
#model_openai = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
#model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o") # or gpt-4o-2024-05-13
# response = model.invoke("This is a test. Simply respond with, 'GPT initialized'")
# print(response)



# pip install langchain-groq (I added to requirements.txt for future) # langchain documentation for groq: https://python.langchain.com/v0.1/docs/integrations/chat/groq/
#model_llama3 = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192") # 8192 token limit on groq but actually can't even reach that since the API limit on groq with this model is 6,000 tokens per minute
# response = model2.invoke("This is a test. Simply respond with, 'Llama3 70B on Groq initialized'")
# print(response)

# Don't run this. Groq has a bigger context window with mistral (Context Window: 32,768 tokens) but can't use it since groq limits API to 5,000 tokens per min with this model which is less than the 6K it allows for Llama3 70b
#model = ChatGroq(temperature=0.7, groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")

def is_english(raw_query):
    try:
        lang = detect(raw_query)
        if lang == 'en':
            print(f'English detected')
            return True
        else:
            print(f'Not English detected.')
            return False
    except:
        return False
def get_translation(text, detected_language = None):
    #translate_client = translate.Client() # commented out since new script defines in the beginning and makes global
    result = translate_client.detect_language(text)
    # this is path for incoming question from human to detect language and translate to English if not English
    if not detected_language:
        detected_language = result['language']
        print("Detected language:", detected_language)
        if detected_language == 'en':  # If language is English, do nothing
            translated_text = text
            print(f'Question is in "{detected_language}", do nothing.')
            return translated_text, detected_language
        else: # if not English, translate to English so we can do the semantic search in English
            translated_text = translate_client.translate(text, target_language='en')['translatedText']
            print("Translated text for semantic search:", translated_text)
            return translated_text, detected_language

    # this is path to translate the response if user does not speak English
    if detected_language:
        translated_text = translate_client.translate(text, target_language=detected_language)['translatedText']
        #print("Translated text:", translated_text)
        return translated_text

def count_tokens(input_string: str) -> int:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(input_string)
    return len(tokens)

# def calculate_input_cost(original_prompt):
#     # Convert the "prompt" to a string for token counting
#     input_string = str(original_prompt)
#     num_tokens = count_tokens(input_string)
#     # GPT-4o cost per M input tokens
#     cost_per_million_tokens: float = 5
#     total_cost = (num_tokens / 1_000_000) * cost_per_million_tokens
#     print(f"The total cost for using gpt-4o is: ${total_cost:.6f}")

def get_response(query, model_selection, detected_language, retrieval_selection, index_name, voice_playback):
    # I get faster and shorter responses API'ing into GROQ and using the llama3 70b model vs openAI GPT 3.5 Turbo and price is about same ... but sacrifice a decent amount of context window. I change to GPT3.5 when retrieval is more than Groq can handle for a given question
    #model_openai = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    model_gpt_3_5 = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    model_gpt_4o = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o") # or gpt-4o-2024-05-13
    model_llama3 = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192") # 8192 token limit on groq but actually can't even reach that since the API limit on groq with this model is 6,000 tokens per minute
    # this would not work with pinecone. I had to use the OpenAIEmbeddings() since that is what was used when we created the pinecone index. Even though I specified the small model in pinecone, OpenAIEmbeddings() is what was in the script so that must have been "ada"?? No I ought to thoroughly test OpenAIEmbeddings(model="text-embedding-3-small") for chroma since that is what I specify to create the embedding and retrieval
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    parser = StrOutputParser()
    

    if retrieval_selection == "Auto":
        k=8
    elif retrieval_selection == 8:
        k=8
    elif retrieval_selection == 12:
        k=12
    elif retrieval_selection == 16:
        k=16
    elif retrieval_selection == 20:
        k=20
    elif retrieval_selection == 24:
        k=24
    
    # if pinecone vector store
    print(f'index_name before conditional statements = {index_name}')
    if index_name == "bakersfield":
        index_name = "bakersfield-small"
    if index_name == "cognigy":
        index_name = "cognigy-small"
    if index_name == "shari" or index_name == "bakersfield-small" or index_name == "cognigy-small" or index_name == "light-field-lab":
        print(f'index_name being sent to pinecone = {index_name}')
        embeddings = OpenAIEmbeddings()
        pinecone = PineconeVectorStore.from_existing_index(index_name, embeddings)
        retrieved_docs = pinecone.similarity_search(query, k=k)  # Adjust k as needed which not something I can change when using retriever.invoke(query). I can put in k=5 but I will only get 4 which is the default


    # for chroma vector store -> old before adding class
    # def send_to_flask(query, index_name, k):
    #     webhook_url = "https://up-poodle-resolved.ngrok-free.app/brian-retrieval"
    #     payload = {'query': query, 'index_name': index_name, 'k': k}
    #     headers = {'Content-Type': 'application/json'}
    #     response = requests.post(webhook_url, json=payload, headers=headers)
    #     if response.status_code == 200:
    #         retrieved_docs_json = response.json()
    #         retrieved_docs = [
    #             Document(page_content=doc["page_content"], metadata=doc["metadata"])
    #             for doc in retrieved_docs_json
    #         ]
    #         return retrieved_docs
    #     else:
    #         raise Exception(f"Error in Flask API: {response.text}")


    # for chroma vector store
    else:
        class Document:
            def __init__(self, page_content, metadata):
                self.page_content = page_content
                self.metadata = metadata

        def send_to_flask(query, index_name, k):
            webhook_url = "https://up-poodle-resolved.ngrok-free.app/brian-retrieval"
            payload = {'query': query, 'index_name': index_name, 'k': k}
            headers = {'Content-Type': 'application/json'}

            response = requests.post(webhook_url, json=payload, headers=headers)
            if response.status_code == 200:
                retrieved_docs_json = response.json()
                retrieved_docs = [
                    Document(page_content=doc["page_content"], metadata=doc["metadata"])
                    for doc in retrieved_docs_json
                ]
                return retrieved_docs
            else:
                raise Exception(f"Error in Flask API: {response.text}")
        
        print(f'index_name being sent to flask app = {index_name}')
        retrieved_docs = send_to_flask(query, index_name, k)





    # Join the retrieved documents' page_content into a single string
    context = "\n".join(doc.page_content for doc in retrieved_docs)
    #print(context)

    def get_prompt(context, question):
        template = f"""
        Provide a deep-dive explanation to the below question based upon the "context" section. If you can't answer the question based on the context, then say so.

        Context: {context}

        Question: Give me as much info as you can on this subject: {question}
        """
        return template

    prompt_template = get_prompt(context, query)
    #print(f'Prompt: {prompt_template}')
    tokens = count_tokens(prompt_template)
    print(f'Number of tokens: {tokens}')
    # my token counter said it was 5335 Groq error said I requested 6652. When I pasted the context in a token counter, I got 6,000.

    if model_selection == "GPT 3.5 Turbo":
        model = model_gpt_3_5
    elif model_selection == "Llama3 70b":
        model = model_llama3
    elif model_selection == "GPT 4o":
        model = model_gpt_4o
    # groq has a 6K tokien limit per minute. When my tokenizer says I'm sending 5327 tokens, groq errors and says I sent 6641 tokens for 1300 token desparity so I lowered the condition limit to 4700 tokens according to my tokenizer
    elif 4700 < tokens <= 16385:
        model = model_gpt_3_5
        model_selection = "GPT 3.5 Turbo"
    elif tokens > 16385:
        model = model_gpt_4o
        model_selection = "GPT 4o"
    else:
        model = model_llama3
        model_selection = "Llama3 70b"
    print(f'Selected model: {model_selection}')
    # Define the prompt as a ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(prompt_template)
    #print(f'Prompt: {prompt}')

    if detected_language == "en" and voice_playback == False:
        try:
            def stream(prompt, model, parser):
                chain = prompt | model | parser
                return chain.stream({
                    "context": context,
                    "question": query
                })
            result = st.write_stream(stream(prompt, model, parser))

        except Exception as e:
            print(f'Error: {e}\n"Changing model to GPT 3.5 Turbo and trying again."')
            model_selection = "GPT 3.5 Turbo"
            model = model_gpt_3_5
            def stream(prompt, model, parser):
                chain = prompt | model | parser
                return chain.stream({
                    "context": context,
                    "question": query
                })
            result = st.write_stream(stream(prompt, model, parser))


    # the non-streaming path for non-English and voice playback
    else:
        try:
            chain = prompt | model | parser
            result = chain.invoke({
                "context": context,
                "question": query
            })
            # note - result wil be printed back in the main function in the non-English path
        except Exception as e:
            print(f'Error: {e}\n"Changing model to GPT 3.5 Turbo and trying again."')
            model_selection = "GPT 3.5 Turbo"
            model = model_gpt_3_5
            chain = prompt | model | parser
            result = chain.invoke({
                "context": context,
                "question": query
            })


    if voice_playback:
        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=result)

        # Build the voice request
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Journey-F"
        )
        
        # Select the type of audio file you want returned
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        # Perform the text-to-speech request
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # The response's audio_content is binary
        with open("output.mp3", "wb") as out:
            out.write(response.audio_content)

        # Read the audio content
        audio_bytes = open("output.mp3", "rb").read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        # Generate the HTML with autoplay
        audio_html = f"""
        <audio id="audio-player" autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        """

        # Inject the HTML into Streamlit app
        st.markdown(audio_html, unsafe_allow_html=True)

        # Create a button to stop the audio playback
        #if st.button("⬛"):
        if st.button("⏹️", key="stop_audio_button"):    
            # Inject JavaScript code to pause the audio and reset the current time
            st.markdown(
                """
                <script>
                    var audioPlayer = document.getElementById('audio-player');
                    audioPlayer.pause();
                    audioPlayer.currentTime = 0;
                </script>
                """,
                unsafe_allow_html=True
            )

    if detected_language == "en" and voice_playback == True:
        st.write(result)
    ##-------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # these alternatives below follow the streamlit documentation on using a wrapper for models not compatible with streamlit.write_stream
    # # v1b
    # def stream_response():
    #     for chunk in chain.stream({
    #         "context": context,
    #         "question": query
    #     }):
    #         yield chunk

    # if detected_language != "en":
    #     result = st.write_stream(stream_response())
    #     translation = get_translation(result, detected_language)
    #     result = translation
    # else:
    #     result = st.write_stream(stream_response())

    # # v2 
    # def stream_response():
    #     for chunk in chain.stream({
    #         "context": context,
    #         "question": query
    #     }):
    #         if isinstance(chunk, str):
    #             yield chunk
    #         else:
    #             yield str(chunk)

    # response_gen = stream_response()

    # if detected_language != "en":
    #     result = st.write_stream(response_gen)
    #     translation = get_translation(result, detected_language)
    #     result = translation
    # else:
    #     result = st.write_stream(response_gen)


    # # v3 attempt
    # def stream_response():
    #     for chunk in chain.stream({
    #         "context": context,
    #         "question": query
    #     }):
    #         yield chunk

    # if detected_language != "en":
    #     response_gen = stream_response()
    #     result = st.write_stream(response_gen)
    #     translation = get_translation(result, detected_language)
    #     result = translation
    # else:
    #     response_gen = stream_response()
    #     result = st.write_stream(response_gen)








    # if model == model_openai:
    #     # Get the selected model
    #     selected_model = "model"
    #     # Print or log the selected model
    #     print(f'The selected model was: {selected_model}')
    # else:
    #     selected_model = "model2"
    #     # Print or log the selected model
    #     print(f'The selected model was: {selected_model}')
    if model_selection == "GPT 3.5 Turbo":
        # Token count + cost
        num_input_tokens = tokens
        input_cost = (num_input_tokens / 1_000_000) * .5 # GPT-3.5 Turbo is $0.5 per M token (input)
        num_output_tokens = count_tokens(str(result))
        print(f'Number of output tokens: {num_output_tokens}')
        output_cost = (num_output_tokens / 1_000_000) * 1.5 # GPT-3.5 Turbo is $1.5 per M token (output)
        total_cost = input_cost + output_cost
    elif model_selection == "Llama3 70b":
        # Token count + cost
        num_input_tokens = tokens
        input_cost = (num_input_tokens / 1_000_000) * .59 # Llama3 70b on Groq is $0.59 per M token (input)
        num_output_tokens = count_tokens(str(result))
        print(f'Number of output tokens: {num_output_tokens}')
        output_cost = (num_output_tokens / 1_000_000) * .79 # Llama3 70b on Groq is $.79 per M token (output)
        total_cost = input_cost + output_cost
    elif model_selection == "GPT 4o":
        num_input_tokens = tokens
        input_cost = (num_input_tokens / 1_000_000) * 5 # GPT-4o is $5 per M token (input)
        num_output_tokens = count_tokens(str(result))
        print(f'Number of output tokens: {num_output_tokens}')
        output_cost = (num_output_tokens / 1_000_000) * 15 # GPT-4o is $15 per M tokens (output)
        total_cost = input_cost + output_cost
    if detected_language != "en":
        translation = get_translation(result, detected_language)
        result = translation
        return result, retrieved_docs, num_input_tokens, input_cost, num_output_tokens, output_cost, total_cost, model_selection
    else: 
        return result, retrieved_docs, num_input_tokens, input_cost, num_output_tokens, output_cost, total_cost, model_selection

def sources_to_print(retrieved_docs):
    """Prints unique URLs and titles from retrieved documents."""
    unique_urls = set()
    unique_titles = set()
    unique_sources = set()
    for doc in retrieved_docs:

        # print(f"Document {i}:")
        # print("Content:")
        # print(doc.page_content)
        # print("Metadata:")
        # print(doc.metadata)




        title = doc.metadata.get("title")
        if title:
            unique_titles.add(title)
        if not title:
            source = doc.metadata.get("source")
            if source:
                unique_sources.add(source)
        url = doc.metadata.get("url")
        if url:
            unique_urls.add(url)

    return unique_urls, unique_titles, unique_sources
        #print(context)

# # Custom CSS to make the sidebar width more dynamic to fit the content it contains
# st.markdown("""
# <style>
#     .css-1d391kg {
#         width: auto !important; /* Allow the width to adjust automatically */
#         padding-right: 10px; /* Add padding to the right */
#         padding-left: 10px; /* Add padding to the left */
#     }
#     .css-1d391kg .css-1n76uvr {
#         width: auto !important; /* Ensure inner elements also adjust */
#     }
#     .css-1d391kg .css-1n76uvr div {
#         max-width: 250px; /* Set a maximum width for the content */
#     }
# </style>
# """, unsafe_allow_html=True)

st.markdown("""
<style>
    .css-1d391kg {
        width: auto !important; /* Allow the width to adjust automatically */
        padding-right: 10px; /* Add padding to the right */
        padding-left: 10px; /* Add padding to the left */
    }
    .css-1d391kg .css-1n76uvr {
        width: auto !important; /* Ensure inner elements also adjust */
    }
    .css-1d391kg .css-1n76uvr div {
        max-width: 250px; /* Set a maximum width for the content */
    }
    
    /* Move main content up by adjusting the negative value */
    .main .block-container {
        margin-top: -4rem;
    }
            
</style>
""", unsafe_allow_html=True)
                     
def main(): 

    # Load index options from Flask app
    def load_index_options():
        webhook_url = "https://up-poodle-resolved.ngrok-free.app/light-field-lab-index-names"
        response = requests.get(webhook_url)
        if response.status_code == 200:
            return response.json()
        # else, show the pinecone vector stores that don't need my backend flask to be online
        else:
            fixed_first_position = "Backend server offline. Only able to load pinecone vector stores."
            data = ["light-field-lab"]
            return [fixed_first_position] + sorted(data)

    # Save index options to Flask app
    def save_index_options(options):
        webhook_url = "https://up-poodle-resolved.ngrok-free.app/light-field-lab-index-names"
        payload = {"index_names": options}
        response = requests.post(webhook_url, json=payload)
        if response.status_code == 200:
            st.experimental_rerun() # this refreshes index drop-down in both the sidebar and main page without refreshing the entire page
            # not writing anything back here due to refresh above 
        else:
            st.sidebar.error("Backend server offline. Failed to save index names.")

    # Load index options
    index_options = load_index_options()


    st.title("Light Field Lab Bot 🤖")
    st.write("**by Brian Morin** | **Model: Mixture of GPT 3.5 | GPT 4o | Llama3 70b** 🧠") 
    st.write("This genAI model has been trained on public domain information from lightfieldlab.com and related news.")
    #st.write("**by digitalcxpartners.com**\n\nModel: GPT3.5 Turbo")

    # Create columns for a single line
    col1, col2, col3 = st.columns([1, 1, 1])  # this is equal width; if I do something like 1, 2, 1 then the middle column will be twice as wide as the other two
    with col1:
        model_selection = st.selectbox("Select model:", ["Auto", "GPT 3.5 Turbo", "Llama3 70b", "GPT 4o"], index=0)
    with col2:
        retrieval_selection = st.selectbox("Select retrieval size:", ["Auto", 8, 12, 16, 20, 24], index=0)
    with col3:
        st.selectbox("Select index:", index_options, index=0, key="main_index_name")

    #query = st.text_area("Enter your prompt: :pencil2:")
    query = st.text_area(":pencil2: Enter your prompt:")

    # For onscreen variables that need ability to change with each run. Check if session variables are already initialized
    if 'session_cost' not in st.session_state:
        st.session_state.session_cost = 0.0
    if 'question_count' not in st.session_state:
        st.session_state.question_count = 0
    # Initialize a new session state variable to store the previous session cost:
    if 'previous_session_cost' not in st.session_state:
        st.session_state.previous_session_cost = 0.0
    # if 'selected_model' not in st.session_state:
    #     st.session_state.selected_model = "default"



    url_to_index = st.sidebar.text_input("Paste URL to web page, youtube video, or PDF to scrape/parse/transcribe then vectorize:", key="url_to_index_sidebar")
    st.sidebar.empty()
    if url_to_index:
        # Sidebar input for index name
        index_name = st.sidebar.selectbox("Select index to append:", index_options, index=0)
        # Add radio button for email summary option
        special_instructions = st.sidebar.text_input(":memo: Editing Instructions (e.g. what to retain/remove)", key="special_instructions")
        #email_summary = st.sidebar.radio("Email summary:", ("No", "Yes"))
        email_summary = st.sidebar.checkbox("Email summary")
        st.sidebar.empty()  # Placeholder for the selection box
        if email_summary:
            st.sidebar.selectbox("Select model:", ["Auto", "GPT 3.5 Turbo", "Llama3 70b", "GPT 4o"], index=0, key="sidebar_model_selection")
            model_selection_for_summary = st.session_state.sidebar_model_selection
            print(f'model_selection_for_summary = {model_selection_for_summary}')
            
        if not email_summary:
            model_selection_for_summary = 0
        if st.sidebar.button("Run 🚀", key="scrape_url_button"):
            if url_to_index and index_name:
                with st.spinner("Sending URL to be scraped then indexed to vector store... :hourglass_flowing_sand:"):
                    def send_to_flask(url_to_index, index_name, email_summary, model_selection_for_summary):
                        webhook_url = "https://up-poodle-resolved.ngrok-free.app/brian-indexing"
                        payload = {'url': url_to_index, 'index_name': index_name, 'email_summary': email_summary, 'model_selection_for_summary': model_selection_for_summary, 'special_instructions': special_instructions}
                        headers = {'Content-Type': 'application/json'}
                        response = requests.post(webhook_url, json=payload, headers=headers)

                        if response.status_code == 200:
                            response_json = response.json()
                            message = response_json.get("message")
                            st.sidebar.markdown(f'**Response from backend flask server:**\n"{message}"')
                        else:
                            st.sidebar.markdown(f'{response.text}')
                    send_to_flask(url_to_index, index_name, email_summary, model_selection_for_summary)
            else:
                st.sidebar.markdown("Please enter both a URL and an index name.")

    add_text_to_vectorize = st.sidebar.text_area("Add plain text to vectorize:", key="text_to_vectorize")
    if add_text_to_vectorize:
    # do I need empty() here?
        if add_text_to_vectorize:
            # Sidebar input for index name
            index_name = st.sidebar.selectbox("Select index to append:", index_options, index=0, key="index_name_add_text")
            # Add radio button for email summary option
            special_instructions = st.sidebar.text_input(":memo: Editing Instructions (e.g. what to retain/remove)", key="special_instructions_add_text")
            #email_summary = st.sidebar.radio("Email summary:", ("No", "Yes"))
            email_summary_checkbox = st.sidebar.checkbox("Email summary", key="email_summary_checkbox")
            st.sidebar.empty()  # Placeholder for the selection box
            if email_summary_checkbox:
                st.sidebar.selectbox("Select model:", ["Auto", "GPT 3.5 Turbo", "Llama3 70b", "GPT 4o"], index=0, key="sidebar_model_selection_add_text")
                model_selection_for_summary_add_text = st.session_state.sidebar_model_selection_add_text
                print(f'model_selection_for_summary = {model_selection_for_summary_add_text}')
                
            if not email_summary_checkbox:
                model_selection_for_summary_add_text = 0
            if st.sidebar.button("Vectorize 🚀", key="add_text_to_vectorize_button"):
                if add_text_to_vectorize and index_name:
                    with st.spinner("Processing text to add to vector store... :hourglass_flowing_sand:"):
                        def send_to_flask(add_text_to_vectorize, index_name, email_summary, model_selection_for_summary_add_text):
                            webhook_url = "https://up-poodle-resolved.ngrok-free.app/brian-indexing"
                            print(f'model_selection_for_summary before sending to flask: {model_selection_for_summary_add_text}')
                            payload = {'text': add_text_to_vectorize, 'index_name': index_name, 'email_summary': email_summary_checkbox, 'model_selection_for_summary': model_selection_for_summary_add_text, 'special_instructions': special_instructions}
                            headers = {'Content-Type': 'application/json'}
                            response = requests.post(webhook_url, json=payload, headers=headers)

                            if response.status_code == 200:
                                response_json = response.json()
                                message = response_json.get("message")
                                st.sidebar.markdown(f'**Response from backend flask server:**\n"{message}"')
                            else:
                                st.sidebar.markdown(f'{response.text}')
                        send_to_flask(add_text_to_vectorize, index_name, email_summary_checkbox, model_selection_for_summary_add_text)
                else:
                    st.sidebar.markdown("Please enter text and an index name.")


    #st.sidebar.markdown(f"**Add to vector store**")
    uploaded_file = st.sidebar.file_uploader("Upload a document to vectorize", type=["txt", "md", "pdf"])
    if uploaded_file is not None:
        def format_file_size(size):
            # Convert size to kilobytes and megabytes
            size_kb = size / 1024
            size_mb = size / (1024 * 1024)
            
            # Return size in the appropriate format
            if size_mb >= 1:
                return f"{size_mb:.2f} MB"
            else:
                return f"{size_kb:.2f} KB"
        file_size_formatted = format_file_size(uploaded_file.size)
        file_details = {
            "filename": uploaded_file.name,
            "filetype": uploaded_file.type,
            "filesize": file_size_formatted
        }
        #st.sidebar.write(file_details)
        #st.sidebar.caption(f'Downloaded. {file_details}')
        st.sidebar.markdown(f'**File details**:\n{file_details}')

        def send_to_flask(uploaded_file):
            webhook_url = "https://up-poodle-resolved.ngrok-free.app/brian-indexing"
            files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post(webhook_url, files=files)
            #st.write(response.text)
            #st.sidebar.write(response.text)

            # Parse the JSON response instead of writing the entire message like the above did
            response_json = response.json()
            
            # Extract and display the message
            message = response_json.get("message", "No message received")
            st.sidebar.markdown(f'**Response from server:**\n"{message}"')

        send_to_flask(uploaded_file)

    # if st.button("Add 🚀", key="text_to_vectorize_button"):
    #     if add_text_to_vectorize:
    #         with st.spinner("Sending text to be vectorized... :hourglass_flowing_sand:"):
    #             def send_to_flask(add_text_to_vectorize):
    #                 webhook_url = "https://up-poodle-resolved.ngrok-free.app/brian-indexing"
    #                 payload = {'text': add_text_to_vectorize}
    #                 headers = {'Content-Type': 'application/json'}
    #                 response = requests.post(webhook_url, json=payload, headers=headers)
    #                 response_json = response.json()
    #                 message = response_json.get("message")
    #                 st.markdown(f'**Response from backend flask server:**\n"{message}"')
    #             send_to_flask(add_text_to_vectorize)
    #     else:
    #         st.warning("Please enter some text to vectorize.")
    st.sidebar.divider()

    # Sidebar input for adding a new index name
    index_name_add = st.sidebar.text_input(":memo: Create new index")

    if st.sidebar.button("Create new index", key="add_index_button"):
        if index_name_add:
            index_options.append(index_name_add)
            save_index_options(index_options)
            #st.sidebar.success(f"Index '{index_name_add}' added successfully!")
        else:
            st.sidebar.warning("Please enter an index name to add.")

    # if uploaded_file is not None:
    #     file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "filesize": uploaded_file.size}
    #     st.sidebar.write(file_details)
        
    #     if uploaded_file.type == "csv":
    #         # Process CSV file
    #         import pandas as pd
    #         df = pd.read_csv(uploaded_file)
    #         st.write("CSV file uploaded successfully:")
    #         st.write(df)
        
    #     elif uploaded_file.type == "text":
    #         # Process TXT file
    #         text = uploaded_file.read().decode("utf-8")
    #         st.write("TXT file uploaded successfully:")
    #         st.write(text)
        
    #     elif uploaded_file.type == "pdf":
    #         # Process PDF file
    #         import PyPDF2
    #         pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
    #         st.write("PDF file uploaded successfully:")
    #         for page_num in range(pdf_reader.getNumPages()):
    #             page = pdf_reader.getPage(page_num)
    #             st.write(page.extract_text())
    #     else:
    #         st.write("Unsupported file type.")

    voice_playback = st.checkbox("Voice playback (Slow due to post processing. Reminder to fix by streaming tokens thru TTS)", value=False, key="voice_playback_checkbox") # by default "False" means not checked. If checked, it will be "True"
    # Generate output
    if st.button("Run 🚀"):
        if query:
            query = query.replace("{", "[").replace("}", "]") # curly braces will screw up the prompt template
            st.write(":thought_balloon:", query)
            with st.spinner("Generating response... :hourglass_flowing_sand:"):     

                # Add your logic here to use the selected model
                if model_selection == "Auto":
                    # Logic for auto model selection
                    pass
                elif model_selection == "GPT 3.5 Turbo":
                    # Logic for GPT 3.5 Turbo
                    pass
                elif model_selection == "Llama3 70b":
                    # Logic for Llama3 70b
                    pass
                elif model_selection == "GPT 4o":
                    # Logic for Llama3 70b
                    pass
                # Replace the pass statements with the actual model handling logic
                #st.success("Response generated successfully!")


                # I introduced is_english function to run a local True/False on English to eliminate unecessary API call to Google translate. If I remove is_english, the script will still work perfectly but just means burning a lot of API calls for English detection
                language_check = is_english(query)
                if language_check: # English path
                    #print('English path taken')
                    detected_language = "en"
                    #st.markdown("Generating response...wait just a few seconds")
                    index_name = st.session_state.main_index_name
                    # if index_name == "__pick__":
                    #     st.sidebar.markdown("Please select an index name.")
                    result, retrieved_docs, num_input_tokens, input_cost, num_output_tokens, output_cost, total_cost, model_selection = get_response(query, model_selection, detected_language, retrieval_selection, index_name, voice_playback)
                else: # non-English path
                    #print('non-English path taken')
                    query2, detected_language = get_translation(query)
                    index_name = st.session_state.main_index_name
                    result, retrieved_docs, num_input_tokens, input_cost, num_output_tokens, output_cost, total_cost, model_selection = get_response(query2, model_selection, detected_language, retrieval_selection, index_name, voice_playback)
                #print(result)
                unique_urls, unique_titles, unique_sources = sources_to_print(retrieved_docs)
                #st.write(result)

                url_list = "\n".join(f"{url}," for url in unique_urls)
                title_list = "\n".join(f"- {title}" for title in unique_titles)  # Using bullet points for clarity
                source_list = "\n".join(f"- {source}" for source in unique_sources)  # Using bullet points for clarity

                if detected_language == "en":
                    if title_list:
                        st.markdown(f"**Check out these links for more:**\n\n{url_list}\n\n**Titles sourced:**\n\n{title_list}")
                    if not title_list:
                        st.markdown(f"**Check out these links for more:**\n\n{url_list}\n\n**Sources:**\n\n{source_list}")
                    #with st.expander("📈 Expand to see cost for query"):
                    with st.expander("🔽 Expand for query cost"):
                        st.markdown(f'\n\n**Cost:**\n\n- Input tokens = {num_input_tokens}\n\n- Cost for question: ${input_cost:.6f}')
                        st.markdown(f'- Number of output tokens = {num_output_tokens}\n\n- Cost for response: ${output_cost:.6f}')
                        st.markdown(f'- Total cost for question + answer = ${total_cost:.6f}')
                        #st.write(f"{result}\nCheck out these links for more:\n{unique_urls}\n{unique_titles}")

                else:
                    st.markdown(f"{result}\n\n{url_list}")
                    st.markdown(f'\n\n**Cost:**\n\n- Input tokens = {num_input_tokens}\n\n- Cost for question: ${input_cost:.6f}')
                    st.markdown(f'- Number of output tokens = {num_output_tokens}\n\n- Cost for response: ${output_cost:.6f}')
                    st.markdown(f'- Total cost for question + answer = ${total_cost:.6f}')

                # using "session state" to change onscreen variables
                st.session_state.session_cost += total_cost
                st.session_state.question_count += 1
                st.session_state.selected_model = model_selection


                session_cost_delta = st.session_state.session_cost - st.session_state.previous_session_cost
                st.session_state.previous_session_cost = st.session_state.session_cost

                # # Display the session variables in the sidebar
                # #st.sidebar.markdown(f"**Session Cost:** ${st.session_state.session_cost:.6f}")
                st.sidebar.markdown(f"**Session Cost:**")
                st.sidebar.metric(label="USD", value=f"${st.session_state.session_cost:.6f}", delta=f"${session_cost_delta:.6f}")
                st.sidebar.markdown(f"** **")
                st.sidebar.markdown(f"**Question Count:** {st.session_state.question_count} 💬")
                st.sidebar.markdown(f"** **")
                if model_selection == "GPT 3.5 Turbo":
                    st.sidebar.caption(f"**Last model invoked:**\n\nGPT-3.5 🧠")
                if model_selection == "Llama3 70b":
                    st.sidebar.caption(f"**Last model invoked:**\n\nLlama3 70b 🧠")
                if model_selection == "GPT 4o":
                    st.sidebar.caption(f"**Last model invoked:**\n\nGPT-4o 🧠")

                # def send_to_flask(query, result):
                #     webhook_url = "https://up-poodle-resolved.ngrok-free.app/brian"
                #     data = {"query": query, "result": result}
                #     response = requests.post(webhook_url, json=data)
                #     print(response.text)
                # send_to_flask(query, result)
        else:
            st.warning("Please enter a question.")


if __name__ == "__main__":
    main()
    
