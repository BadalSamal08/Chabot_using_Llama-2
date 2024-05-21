import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

## Function to get response from LLama 2 model
def getLLamaresponse(user_query):
    llm = CTransformers(model='llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})
    
    ## Prompt template
    prompt = PromptTemplate(input_variables=["user_query"],
                            template="Answer the following question: {user_query}")

    ## Generate the response from the llama 2 model
    response = llm(prompt.format(user_query=user_query))
    return response

st.set_page_config(page_title="Chat with Llama 2",
                   layout='centered',
                   initial_sidebar_state='collapsed')
st.header("Chat with Llama 2")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Input for user query
user_query = st.text_input("Ask a question", key="user_query_input")

# Button to send the question
if st.button("Send"):
    if user_query:
        response = getLLamaresponse(user_query)
        st.session_state.chat_history.append({"question": user_query, "answer": response})
        # Trigger JavaScript to clear the input field
        st.write('<script>document.getElementById("user_query_input").value = "";</script>', unsafe_allow_html=True)

# Display chat history
if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        st.write(f"**You:** {chat['question']}")
        st.write(f"**Llama 2:** {chat['answer']}")
