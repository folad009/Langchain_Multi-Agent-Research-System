
import streamlit as st

from src.pipeline.pipline import run_research_pipeline


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)


# ==========================================
# PAGE HEADER
# ==========================================

st.title("🔬 Multi-Agent Research Assistant")

st.markdown(
    "Research powered by LangChain, OpenAI and Tavily."
)

st.divider()


# ==========================================
# RESEARCH INPUT
# ==========================================

topic = st.text_input(
    "Enter your research topic",
    placeholder="e.g. Impact of Artificial Intelligence on FMCG businesses"
)

run_button = st.button(
    "🚀 Start Research",
    type="primary",
    disabled=not topic.strip()
)


# ==========================================
# RUN RESEARCH PIPELINE
# ==========================================

if run_button:

    with st.spinner("AI Research Agents are working..."):

        try:

            result = run_research_pipeline(topic)

            st.session_state["research_result"] = result

            st.success("Research completed successfully!")

        except Exception as error:

            st.error(f"Research failed: {str(error)}")


# ==========================================
# DISPLAY FINAL RESEARCH
# ==========================================

if "research_result" in st.session_state:

    result = st.session_state["research_result"]

    st.divider()

    st.header("📄 Final Research Report")

    # Research score
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Critic Score",
        f"{result.get('score', 0)}/10"
    )

    col2.metric(
        "Revision Attempts",
        result.get("revision_attempts", 0)
    )

    col3.metric(
        "Research Status",
        result.get("status", "Unknown")
    )

    st.divider()

    # Final report
    st.subheader("Research Report")

    st.markdown(result.get("report", "No report generated."))

    # Download report
    st.download_button(
        label="⬇️ Download Research Report",
        data=result.get("report", ""),
        file_name="research_report.md",
        mime="text/markdown"
    )

    st.divider()

    # Critic feedback
    with st.expander("🔍 Critic Review and Feedback"):

        st.markdown(
            result.get("feedback", "No feedback available.")
        )

    # Search results
    with st.expander("🌐 Search Results"):

        st.markdown(
            result.get("search_results", "No search results.")
        )

    # Scraped content
    with st.expander("📚 Scraped Research Content"):

        st.markdown(
            result.get("scraped_content", "No scraped content.")
        )
