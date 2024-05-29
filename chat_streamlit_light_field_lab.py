# query = I heard this summary statement from the CEO about the technology and need each part of it explained in extensive detail as I'm doing research, "Karafin explained that the nanomaterial layer acts as a spatial light modulator to produce the light wavefront amplitudes, while a waveguide layer introduces the phase delays that make the light converge to form the holographic object."



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

def get_response(query, model_selection, detected_language, retrieval_selection):
    # I get faster and shorter responses API'ing into GROQ and using the llama3 70b model vs openAI GPT 3.5 Turbo and price is about same ... but sacrifice a decent amount of context window. I change to GPT3.5 when retrieval is more than Groq can handle for a given question
    #model_openai = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    model_gpt_3_5 = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    model_gpt_4o = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o") # or gpt-4o-2024-05-13
    model_llama3 = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192") # 8192 token limit on groq but actually can't even reach that since the API limit on groq with this model is 6,000 tokens per minute
    index_name = "light-field-lab"
    embeddings = OpenAIEmbeddings()
    pinecone = PineconeVectorStore.from_existing_index(index_name, embeddings)
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
    retrieved_docs = pinecone.similarity_search(query, k=k)  # Adjust k as needed which not something I can change when using retriever.invoke(query). I can put in k=5 but I will only get 4 which is the default

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

    if model_selection == "GPT 3.5 Turbo":
        model = model_gpt_3_5
    elif model_selection == "Llama3 70b":
        model = model_llama3
    elif model_selection == "GPT 4o":
        model = model_gpt_4o
    elif 6000 < tokens <= 30000:
        model = model_gpt_3_5
        model_selection = "GPT 3.5 Turbo"
    elif tokens > 30000:
        model = model_gpt_4o
        model_selection = "GPT 4o"
    else:
        model = model_llama3
        model_selection = "Llama3 70b"
    # Define the prompt as a ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(prompt_template)
    #print(f'Prompt: {prompt}')

    if detected_language == "en":
        def stream(prompt, model, parser):
            # def copy_prompt(prompt):
            #     global original_prompt
            #     original_prompt = prompt
            #     #calculate_input_cost(original_prompt)
            #     count_tokens(str(original_prompt))
            #     print(f"Prompt: {prompt}")
            #     return prompt

            #chain = prompt | copy_prompt | model | parser
            chain = prompt | model | parser


            return chain.stream({
                "context": context,
                "question": query
            })
        
        result = st.write_stream(stream(prompt, model, parser))
        
        if result:
            pass
    else:
        chain = prompt | model | parser
        result = chain.invoke({
            "context": context,
            "question": query
        })
        if result:
            pass

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
    for doc in retrieved_docs:

        # print(f"Document {i}:")
        # print("Content:")
        # print(doc.page_content)
        # print("Metadata:")
        # print(doc.metadata)

        title = doc.metadata.get("title")
        if title:
            unique_titles.add(title)
        url = doc.metadata.get("url")
        if url:
            unique_urls.add(url)

    return unique_urls, unique_titles
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
    st.title("Light Field Lab Bot 🤖")
    st.write("**by Brian Morin** | **Model: Mixture of GPT3.5 Turbo and Llama3 70b** 🧠") 
    st.write("This genAI model has been trained on public domain information from lightfieldlab.com and related news.")
    #st.write("**by digitalcxpartners.com**\n\nModel: GPT3.5 Turbo")

    # Create columns for a single line
    col1, col2 = st.columns([1, 3])  # Adjust the column width ratios as needed
    with col1:
        model_selection = st.selectbox("Select foundation model:", ["Auto", "GPT 3.5 Turbo", "Llama3 70b", "GPT 4o"], index=0)
    with col2:
        retrieval_selection = st.selectbox("Select retrieval size:", ["Auto", 8, 12, 16, 20, 24], index=0)

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


    # Generate output
    if st.button("Run 🚀"):
        if query:
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
                    result, retrieved_docs, num_input_tokens, input_cost, num_output_tokens, output_cost, total_cost, model_selection = get_response(query, model_selection, detected_language, retrieval_selection)
                else: # non-English path
                    #print('non-English path taken')
                    query2, detected_language = get_translation(query)
                    result, retrieved_docs, num_input_tokens, input_cost, num_output_tokens, output_cost, total_cost, model_selection = get_response(query2, model_selection, detected_language, retrieval_selection)
                #print(result)
                unique_urls, unique_titles = sources_to_print(retrieved_docs)
                #st.write(result)

                url_list = "\n".join(f"{url}," for url in unique_urls)
                title_list = "\n".join(f"- {title}" for title in unique_titles)  # Using bullet points for clarity

                if detected_language == "en":
                    st.markdown(f"**Check out these links for more:**\n\n{url_list}\n\n**Titles sourced:**\n\n{title_list}")
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
                if model_selection == "GPt 3.5 Turbo":
                    st.sidebar.caption(f"**Last model invoked:**\n\nGPT-3.5 🧠")
                if model_selection == "Llama3 70b":
                    st.sidebar.caption(f"**Last model invoked:**\n\nLlama3 70b 🧠")
                if model_selection == "GPt 4o":
                    st.sidebar.caption(f"**Last model invoked:**\n\nGPT 4o 🧠")

                def send_to_flask(query, result):
                    webhook_url = "https://up-poodle-resolved.ngrok-free.app/light-field-lab"
                    data = {"query": query, "result": result}
                    response = requests.post(webhook_url, json=data)
                    print(response.text)
                send_to_flask(query, result)


                # st.sidebar.write("**Session Cost:**")
                # st.sidebar.write(f"${st.session_state.session_cost:.6f}")
                # st.sidebar.write(f"Delta: ${session_cost_delta:.6f}")
                # st.sidebar.write(f"**Question Count:** {st.session_state.question_count} 💬")



        else:
            st.warning("Please enter a question.")


if __name__ == "__main__":
    main()
