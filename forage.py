"""
Forage streamlit app.

streamlit run forage.py
"""

import json
import streamlit as st
import faiss
import gpt_lib
import sys
import vectorize
import yaml
from yaml.loader import SafeLoader

import util

if util.USE_AUTHENTICATION:
    import streamlit_authenticator as stauth


MODELS = [
    ("gpt-3", "gpt-3.5-turbo"),
    ("gpt-4", "gpt-4-0125-preview"),
]

INPUT_SCOPES = [
    ("No analysis (just do search)", "search_only"),
    ("Relevant speaker turns (top 100)", "top_100"),
    ("Relevant speaker turns (top 50 with context)", "top_100_with_context_1"),
    ("Relevant speaker turns with speaker bios (top 50)", "top_50_with_bio"),
    (
        "[NOT YET IMPLEMENTED] All speaker turns of speakers of 10 most-relvant turns",
        "allspeaker_10",
    ),
]


@st.cache_resource
def load_data(use_local_embeddings=True):
    return util.load_all_data(use_local_embeddings=True)


@st.cache_data(persist=True)
def run_rag_query(user_input, results, model, input_scope):
    return util.run_rag_query(user_input, results, model, input_scope)


def get_help_text(res):
    if "speaker_intro" in res and res["speaker_intro"]:
        return res["speaker_intro"]
    elif "prev_speaker_name" in res and res["prev_speaker_name"]:
        s = "Previous speaker: %s\n" % (res["prev_speaker_name"])
        s += "Previous speaker turn: %s\n" % (res["prev_content"])
        s += "Next speaker: %s\n" % (res["next_speaker_name"])
        s += "Next speaker turn: %s\n" % (res["next_content"])
        return s
    return None


def render_results(
    results,
    results_container,
    data,
    reranked=False,
    show_download_button=False,
    corpus=None,
):
    with results_container:
        with st.container():
            if show_download_button:
                st.download_button(
                    "Download search results as a CSV file",
                    util.convert_results_to_csv(results),
                    "search_results.csv",
                    "text/csv",
                    key="download-csv",
                )
            if reranked:
                st.caption(
                    "Speaker turns ranked by how often they are cited in the analysis"
                )
            else:
                st.caption(
                    "Speaker turns ranked by how closely they match the search query"
                )

            for i, res in enumerate(results):
                st.write(
                    '<a name="cite-%s"></a>\n\n\n' % (res["res_idx"]),
                    unsafe_allow_html=True,
                )
                # Spacer is needed to prevent the overlay strip on top of streamlit from hiding citations when they're clicked.
                st.write("")
                st.markdown(render_result(res, data), help=get_help_text(res))


def main():
    st.set_page_config(layout="wide", page_title="Forage")
    params = st.query_params
    objective_default = params.get("objective", "")
    subject_default = params.get("subject", "")
    corpus_default = params.get("corpus", "")
    analysis_default = params.get("analysis_input", "")

    with open("assets/style.css") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

    hide_streamlit_style = """
    <style>
    #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
    </style>
    """
    st.markdown(
        """
    <style>
    .reportview-container .main .block-container{
        padding-top: 75px;
    background: blue;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    corpus_to_data = load_data()
    encode = vectorize.get_encoder(use_local_embeddings=True)
    allowed_corpora = set(corpus_to_data.keys())
    display_name = "user"
    username = "none"

    # Authentication
    if util.USE_AUTHENTICATION:
        with open("config.yml") as file:
            auth_config = yaml.load(file, Loader=SafeLoader)
        authenticator = stauth.Authenticate(
            auth_config["credentials"],
            auth_config["cookie"]["name"],
            auth_config["cookie"]["key"],
            auth_config["cookie"]["expiry_days"],
            auth_config["preauthorized"],
        )
        name, authentication_status, username = authenticator.login()
        if authentication_status and (
            "corpora" in auth_config["credentials"]["usernames"][username]
        ):
            user_info = auth_config["credentials"]["usernames"][username]
            allowed_corpora = set(user_info["corpora"])
            display_name = user_info.get("name", username)
            col1, col2 = st.columns([0.25, 1])
            with col1:
                st.write("Now logged in: **%s**" % (display_name))
            with col2:
                authenticator.logout()
    else:
        authentication_status = True

    if authentication_status:
        st.image("assets/title.gif", width=1200)
        st.title("Forage (FOra Retrieval-Augmented GEneration)")
        with st.expander("How this works"):
            st.markdown(open("assets/help.md").read())

        with st.expander("Configuration options"):
            col1, col2, col3 = st.columns(3)
            with col1:
                corpus_keys = [c for c in util.CORPORA.keys() if c in allowed_corpora]
                if corpus_default in corpus_keys:
                    default_index = corpus_keys.index(corpus_default)
                else:
                    default_index = 0
                corpus = st.selectbox(
                    "Data set",
                    corpus_keys,
                    format_func=lambda x: util.CORPORA[x]["name"],
                    index=default_index,
                )
            with col2:
                analysis_input = st.selectbox(
                    "Which info goes into the analysis",
                    [x[0] for x in INPUT_SCOPES],
                    index=1,
                )
            with col3:
                input_scope = dict(INPUT_SCOPES)[analysis_input]
                if input_scope != "search_only":
                    analysis_model = st.selectbox(
                        "Which AI model to use for analysis",
                        [x[0] for x in MODELS],
                        index=1,
                    )
                    model = dict(MODELS)[analysis_model]

        col1, col2 = st.columns(2)
        with col1:
            objective_keys = list(util.OBJECTIVES.keys())
            objective = st.selectbox(
                "What is your objective?",
                objective_keys,
                format_func=lambda x: util.OBJECTIVES[x]["display"],
                help="Select what you're trying to accomplish, broadly speaking.  You will pair this with a specific subject in the next input.",
                index=(objective_default in objective_keys)
                and objective_keys.index(objective_default)
                or 0,
            )
        with col2:
            if objective == "other":
                subject = st.text_input(
                    "Enter your custom objective and subject area here.",
                    help='Please enter a full sentence describing what you want, being sure to include a concrete subject area.  For example, "Generate story feature ideas about community events with kids and adults."',
                    value=subject_default,
                )
            else:
                subject = st.text_input(
                    "What is your subject?",
                    help='Enter a short description of the topic of the content you\'re interested in.  Examples: "immigration", "cultural divides due to religion", "challenges of bureaucracy in Wyandotte County"',
                    value=subject_default,
                )

        user_input = util.OBJECTIVES[objective]["prompt_template"] % (subject)

        submit1 = st.button(
            "Click to search " + util.CORPORA[corpus]["name"], type="primary"
        )
        if submit1:
            search_results = analysis = None
            data = corpus_to_data[corpus]
            log_line = {
                "corpus": corpus,
                "user": username,
                "subject": subject,
                "objective": objective,
                "user_input": user_input,
                "analysis_input": analysis_input,
                "input_scope": input_scope,
            }
            with open("log.jsonl", "a") as fs_log:
                print(json.dumps(log_line), file=fs_log)
            with st.spinner("Fetching results..."):
                results = util.run_query(subject, data, encode, input_scope, corpus)
            if input_scope == "search_only":
                raw_results_col = st.container()
            else:
                raw_results_col, analysis_col = st.columns(2)
            with raw_results_col:
                st.write('<a name="results"></a>', unsafe_allow_html=True)
                st.header("Search results")
                results_container = st.empty()
                render_results(
                    results,
                    results_container,
                    data,
                    show_download_button=(input_scope == "search_only"),
                    corpus=corpus,
                )
            if input_scope != "search_only":
                with analysis_col:
                    st.write('<a name="analysis"></a>', unsafe_allow_html=True)
                    st.header("Analysis")
                    st.write('<a name="spacer"></a>', unsafe_allow_html=True)
                    with st.spinner():
                        analysis = run_rag_query(
                            user_input, results, model, input_scope
                        )
                        analysis_markdown, citation_counts = util.analyze_citations(
                            analysis, results, corpus=corpus
                        )
                        st.markdown(analysis_markdown, unsafe_allow_html=True)
                        results.sort(
                            key=lambda x: citation_counts[x["res_idx"]], reverse=True
                        )
                        render_results(
                            results,
                            results_container,
                            data,
                            reranked=True,
                            show_download_button=True,
                        )
            with open("full_log.jsonl", "a") as fs_log:
                log_line["search_results"] = results
                log_line["analysis_output"] = analysis
                print(json.dumps(log_line), file=fs_log)
            st.query_params.corpus = corpus
            st.query_params.objective = objective
            st.query_params.subject = subject

    elif authentication_status == False:
        st.error("Username/password is incorrect")
    elif authentication_status == None:
        st.warning("Please enter your username and password")

    st.write(
        "[Terms](https://docs.google.com/document/d/1A7ZjyVL39JTE0c3HbW6xixT76I79kqh-G6PsQeI6JAs/edit?usp=sharing), [Feedback](https://docs.google.com/forms/d/e/1FAIpQLScWLVZNiE-KNwR-3fFHGHpsmZIvHWwCQhvXba20HwaclJz6qQ/viewform)"
    )


def render_result(res, data):
    md = "%d. **:blue[%s]**: %s" % (
        res["res_idx"],
        util.display_speaker_name(res["speaker_name"]),
        res["content"],
    )
    link = util.playback_link(
        res["conversation_id"], res.get("audio_start_offset", 0.0)
    )
    conv_info = data["conversations"][res["conversation_id"]]
    md += "\n*(From [%s](%s), %s)*" % (
        conv_info["title"],
        link,
        conv_info["start_time"][:10],
    )
    md += " [&uarr;](#analysis) "  # up arrow to go back to analysis

    return md


main()
