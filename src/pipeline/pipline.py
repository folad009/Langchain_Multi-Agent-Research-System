
import logging
import re

from langchain_core.messages import AIMessage

from src.agents.agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain
)


# ==========================================
# LOGGING CONFIGURATION
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================
# CONFIGURATION
# ==========================================

MAX_REVISION_ATTEMPTS = 1
MINIMUM_REPORT_SCORE = 8


# ==========================================
# HELPER: EXTRACT AGENT RESPONSE
# ==========================================

def extract_agent_response(result: dict) -> str:
    """
    Extract the final AI response from a LangChain agent.
    """

    messages = result.get("messages", [])

    for message in reversed(messages):

        if isinstance(message, AIMessage):

            content = message.content

            if isinstance(content, str) and content.strip():
                return content

            if isinstance(content, list):
                text_parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict)
                    and block.get("type") == "text"
                ]

                text = "\n".join(text_parts)

                if text.strip():
                    return text

    raise ValueError(
        "Agent did not return a valid AI response."
    )


# ==========================================
# HELPER: EXTRACT SCORE
# ==========================================

def extract_score(feedback: str) -> int:
    """
    Extract the critic score from its response.
    """

    match = re.search(
        r"Score:\s*(\d{1,2})\s*/\s*10",
        feedback,
        re.IGNORECASE
    )

    if not match:
        logger.warning("Could not extract critic score.")
        return 0

    score = int(match.group(1))

    if score < 0 or score > 10:
        return 0

    return score


# ==========================================
# MAIN RESEARCH PIPELINE
# ==========================================

def run_research_pipeline(topic: str) -> dict:

    # ======================================
    # INPUT VALIDATION
    # ======================================

    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("Research topic cannot be empty.")

    topic = topic.strip()

    state = {
        "topic": topic,
        "search_results": "",
        "scraped_content": "",
        "report": "",
        "feedback": "",
        "score": 0,
        "revision_attempts": 0,
        "status": "processing"
    }

    # ======================================
    # STEP 1: SEARCH AGENT
    # ======================================

    try:

        logger.info("STEP 1: Search Agent started.")

        search_agent = build_search_agent()

        search_result = search_agent.invoke({
            "messages": [
                (
                    "user",
                    f"""
                    Find recent, reliable and detailed
                    information about: {topic}

                    Return at least five relevant sources
                    where available.

                    Include:
                    - Title
                    - URL
                    - Relevant information
                    """
                )
            ]
        })

        state["search_results"] = extract_agent_response(
            search_result
        )

        logger.info("Search Agent completed successfully.")

    except Exception as error:

        logger.exception("Search Agent failed.")

        raise RuntimeError(
            f"Search Agent failed: {error}"
        ) from error

    # ======================================
    # STEP 2: READER AGENT
    # ======================================

    try:

        logger.info("STEP 2: Reader Agent started.")

        reader_agent = build_reader_agent()

        reader_result = reader_agent.invoke({
            "messages": [
                (
                    "user",
                    f"""
                    Conduct deeper research on:

                    {topic}

                    Review these search results:

                    {state['search_results']}

                    Instructions:

                    1. Identify up to three relevant URLs.
                    2. Use the scraping tool to retrieve their content.
                    3. Extract relevant facts, statistics and findings.
                    4. Preserve source URLs.
                    5. Identify conflicting information.
                    6. Do not invent information if a source
                       cannot be accessed.

                    Return a structured research summary.
                    """
                )
            ]
        })

        state["scraped_content"] = extract_agent_response(
            reader_result
        )

        logger.info("Reader Agent completed successfully.")

    except Exception as error:

        logger.exception("Reader Agent failed.")

        raise RuntimeError(
            f"Reader Agent failed: {error}"
        ) from error

    # ======================================
    # COMBINE RESEARCH
    # ======================================

    research_combined = f"""
    RESEARCH TOPIC:
    {topic}

    SEARCH RESULTS:
    {state['search_results']}

    SCRAPED RESEARCH:
    {state['scraped_content']}
    """

    # ======================================
    # STEP 3: WRITER CHAIN
    # ======================================

    try:

        logger.info("STEP 3: Writer started.")

        state["report"] = writer_chain.invoke({
            "topic": topic,
            "research": research_combined
        })

        logger.info("Initial report generated.")

    except Exception as error:

        logger.exception("Writer failed.")

        raise RuntimeError(
            f"Writer failed: {error}"
        ) from error

    # ======================================
    # STEP 4: CRITIC AND REVISION LOOP
    # ======================================

    try:

        logger.info("STEP 4: Critic started.")

        for attempt in range(MAX_REVISION_ATTEMPTS + 1):

            state["feedback"] = critic_chain.invoke({
                "report": state["report"]
            })

            state["score"] = extract_score(
                state["feedback"]
            )

            logger.info(
                "Critic score: %s/10",
                state["score"]
            )

            # APPROVE REPORT

            if state["score"] >= MINIMUM_REPORT_SCORE:

                state["status"] = "approved"

                logger.info("Report approved.")

                break

            # STOP IF MAXIMUM ATTEMPTS REACHED

            if attempt == MAX_REVISION_ATTEMPTS:

                state["status"] = "completed_needs_review"

                logger.warning(
                    "Maximum revision attempts reached."
                )

                break

            # REVISE REPORT

            logger.info(
                "Report needs revision. Attempt %s",
                attempt + 1
            )

            revised_research = f"""
            ORIGINAL RESEARCH:
            {research_combined}

            CRITIC FEEDBACK:
            {state['feedback']}

            PREVIOUS REPORT:
            {state['report']}

            Instructions:
            Revise the report using the critic feedback.
            Correct factual weaknesses.
            Improve evidence and clarity.
            Preserve source attribution.
            Do not invent facts or references.
            """

            state["report"] = writer_chain.invoke({
                "topic": topic,
                "research": revised_research
            })

            state["revision_attempts"] += 1

        logger.info("Critic review completed.")

    except Exception as error:

        logger.exception("Critic or revision failed.")

        raise RuntimeError(
            f"Critic or revision failed: {error}"
        ) from error

    # ======================================
    # FINAL OUTPUT
    # ======================================

    logger.info("Research pipeline completed.")

    return state
