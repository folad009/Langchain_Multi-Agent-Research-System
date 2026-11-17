
import os

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.tools.tools import web_search, scrape_url


# ==========================================
# ENVIRONMENT CONFIGURATION
# ==========================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from .env")


# ==========================================
# MODEL INITIALIZATION
# ==========================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY
)


# ==========================================
# 1. SEARCH AGENT
# ==========================================

def build_search_agent():

    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt="""
        You are an expert web research agent.

        Your responsibility is to:
        1. Search the internet for relevant information.
        2. Find recent and reliable sources.
        3. Identify important facts and findings.
        4. Include source URLs in your findings.
        5. Avoid unsupported claims.

        Return comprehensive research findings.
        """
    )


# ==========================================
# 2. READER AGENT
# ==========================================

def build_reader_agent():

    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="""
        You are an expert research analyst and web reader.

        Your responsibility is to:
        1. Read and analyse supplied web pages.
        2. Extract relevant information.
        3. Identify important facts and statistics.
        4. Capture evidence supporting research findings.
        5. Identify the source URL.
        6. Ignore irrelevant website content.

        Return a detailed and structured analysis.
        """
    )


# ==========================================
# 3. WRITER CHAIN
# ==========================================

writer_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
        You are an expert research writer.

        Write clear, structured, insightful,
        factual and professional research reports.

        Do not invent statistics, facts or sources.
        Distinguish verified information from uncertainty.
        """
    ),

    (
        "human",
        """
        Write a detailed research report on the topic.

        Topic:
        {topic}

        Research Gathered:
        {research}

        Structure the report as:

        1. Introduction

        2. Key Findings
           - Minimum of 3 well-explained findings.
           - Include relevant evidence and statistics.

        3. Analysis and Insights

        4. Conclusion

        5. Sources
           - List all URLs found in the research.

        Be detailed, factual and professional.
        """
    )

])

writer_chain = (
    writer_prompt
    | llm
    | StrOutputParser()
)


# ==========================================
# 4. CRITIC CHAIN
# ==========================================

critic_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
        You are a sharp and constructive research critic.

        Be honest, objective and specific.

        Evaluate research reports based on:
        - Accuracy
        - Evidence quality
        - Research depth
        - Logical consistency
        - Completeness
        - Clarity
        - Source credibility
        """
    ),

    (
        "human",
        """
        Review the research report below and evaluate it strictly.

        Report:
        {report}

        Respond in this exact format:

        Score: X/10

        Strengths:
        - ...
        - ...

        Areas to improve:
        - ...
        - ...

        One-line verdict:
        ...

        Be specific and avoid generic feedback.
        """
    )

])

critic_chain = (
    critic_prompt
    | llm
    | StrOutputParser()
)
