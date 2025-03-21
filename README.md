# FoRAGE
FoRAGE (FOra Retrieval-Augmented GEneration)

## Preliminaries:

- Install the required python dependencies:
  - ``pip3 install -r requirements.txt``
- You will need to install the data sets you wish to explore under the "indexes" subdirectory. Contact the authors for this data.
- You will need an OpenAI API key in order to produce LLM-generated analyses of the data. (If you don't have one, you can still use the tool to do basic embedding-based retrieval.) Set the ``OPENAI_API_KEY`` environment variable accordingly in the shell from which you run this application.


## To run:

- Start the Streamlit app from your terminal:
  - ``streamlit run forage.py``
