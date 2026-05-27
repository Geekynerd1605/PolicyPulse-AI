import logging
import os
import json
import re
from typing import Any, Dict, List

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# import state schema
from backend.src.graph.state import VideoAuditState, ComplianceIssue
from backend.src.services.video_indexer import VideoIndexerService

logger = logging.getLogger("policypulse-ai")
logging.basicConfig(level=logging.INFO)

def index_video_node(state: VideoAuditState) -> Dict[str, Any]:
    '''
    Downloads the youtube video from the URL
    & uploads the same to the Azure Video Indexer &
    extracts the insights
    '''
    video_url=state.get("video_url")
    video_id_input=state.get("video_id")

    logger.info(f"[Node:Indexer] Processing: {video_url}")

    local_filename="temp_audit_video.mp4"

    try:
        vi_service=VideoIndexerService()
        # download
        if "youtube.com" in video_url or "youtu.be" in video_url:
            local_path = vi_service.download_youtube_video(video_url, output_path=local_filename)
        else:
            raise Exception("Please provide a valid YouTube URL.")
        # upload to Azure Video Indexer
        azure_video_id = vi_service.upload_video(local_path, video_name=video_id_input)
        logger.info(f"Uploaded to Azure Video Indexer: {azure_video_id}")
        # Cleanup
        if os.path.exists(local_path):
            os.remove(local_path)
        # wait
        raw_insights = vi_service.wait_for_processing(azure_video_id)
        # extract insights
        clean_data = vi_service.extract_data(raw_insights)
        logger.info(f"[Node:Indexer] Extraction Completed")
        return clean_data

    except Exception as e:
        logger.error(f"Video Indexer Failed: {e}")
        return {
            "errors": [str(e)],
            "final_status": "FAIL",
            "transcript": "",
            "ocr_text": [],
        }

# Node 2: Compliance Auditor
def audio_content_node(state: VideoAuditState) -> Dict[str, Any]:
    '''
    Performs Retrieval Augmented Generation (RAG) to audit the content - brand video
    '''
    logger.info(f"[Node:Auditor] querying the knowledge base and LLM")
    transcript = state.get("transcript", "")
    if not transcript:
        logger.warning("No transcript available. Skipping audit.")
        return {
            "final_status": "FAIL",
            "final_report": "Audit skipped because video processing failed (No transcript available).",
            "errors": ["No transcript available. Skipping audit."],
        }
    
    # initialize Azure clients
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.0
    )

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-large"
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    # initialize vector store
    vector_store = AzureSearch(
        azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
        embedding_function = embeddings.embed_query,
    )
    # RAG Retrieval
    ocr_text = state.get("ocr_text", [])
    query_text = f"{transcript} {''.join(ocr_text)}"
    docs=vector_store.similarity_search(query_text, k=3)
    retrived_rules="\n\n".join([doc.page_content for doc in docs]) 
    system_prompt = f"""
        You are a senior compliance auditor.
        OFFICIAL REGULATORY RULES:
        {retrived_rules}
        INSTRUCTIONS:
        1. Analyze the transcript and OCR text below.and
        2. Identify any violations of the regulatory rules.
        3. Return strictly JSON in the following format:
            {{
                "compliance_results": [
                    {{
                        "category": "Claim validation",
                        "description": "Explaination of the violation...",
                        "severity": "CRITICAL 
                    }}
                ],
                "status": "FAIL",
                "final_report": "Summary of findings...."
            }}

            If no violations are found, set "status" to "PASS" and "compliance_results" should be an [].
            """

    user_message = f"""
    VIDEO_METADATA: {state.get('video_metadata', {})}
    TRANSCRIPT: {transcript}
    ON-SCREEN TEXT (OCR): {ocr_text}
    """

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ])
        content=response.content
        if "```" in content:
            content=re.search(r"```(?:json)?(.?)```", content, re.DOTALL).group(1)
        audit_data=json.loads(content.strip())
        return{
            "compliance_results": audit_data.get("compliance_results", []),
            "final_status": audit_data.get("status", "FAIL"),
            "final_report": audit_data.get("final_report", "No report generated"),
        }
    except Exception as e:
        logger.error(f"System Error in Auditor Node: {str(e)}")
        logger.error(f"Raw LLM Response: {response.content if response in locals() else 'None'}")

        return{
            "errors": [str(e)],
            "final_status": "FAIL",            
        }