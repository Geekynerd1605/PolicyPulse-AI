import operator
from typing import Annotated, List, Optional, Any, TypedDict, Dict

# Define the schema for a single compliance Result
class ComplianceIssue(TypedDict):
    category: str 
    description: str # specific detail of violation
    severity: str # CRITICAL | WARNING 
    timestamp: Optional[str]

# Define the global graph state
# this defines the state that gets passed around in the agentic workflow
class VideoAuditState(TypedDict):
    '''
    Defines the data schema for LangGraph execution state.
    Main container: holds all the information about the audit
    right from the initial URL to the final report
    '''
    # input parameters
    video_url: str
    video_id: str

    # ingestion and extraction data
    local_file_path: Optional[str]
    video_metadata: Dict[str, Any] # {"duration":15, "resolution":"1080p", "fps":30}
    transcript: Optional[str] # fully extracted speech-to-text  
    ocr_text: List[str]

    # analysis output
    # stores the list of all violations found during the audit by AI 
    compliance_results: Annotated[List[ComplianceIssue], operator.add]

    # final deliverables
    final_status: str # PASS | FAIL
    final_report: str # markdown format

    # system observability
    # Errors: API tieout, system errors, etc.
    # stores list of system level crashes
    errors: Annotated[List[str], operator.add]
