"""
Report Agent Service
Uses LangChain + Zep to implement ReACT-pattern simulation report generation

Features:
1. Generate reports based on simulation requirements and Zep graph info
2. Plan directory structure first, then generate section by section
3. Each section uses ReACT multi-round thinking and reflection
4. Support conversation with users, autonomously calling retrieval tools
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from .zep_tools import (
    ZepToolsService, 
    SearchResult, 
    InsightForgeResult, 
    PanoramaResult,
    InterviewResult
)

logger = get_logger('mirofish.report_agent')


class ReportLogger:
    """
    Report Agent 详细Logger
    
    在ReportFile夹中Generate agent_log.jsonl File, Record每一步详细动作.
    每行是一个完整的 JSON object containing timestamp, actionType, 详细Content等.
    """
    
    def __init__(self, report_id: str):
        """
        InitializeLogger
        
        Args:
            report_id: Report ID, 用于确定Log filePath
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'agent_log.jsonl'
        )
        self.start_time = datetime.now()
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Ensure log file directory exists"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _get_elapsed_time(self) -> float:
        """Get从Startelapsed time to now (seconds)"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def log(
        self, 
        action: str, 
        stage: str,
        details: Dict[str, Any],
        section_title: str = None,
        section_index: int = None
    ):
        """
        Record一条日志
        
        Args:
            action: 动作Type, 如 'start', 'tool_call', 'llm_response', 'section_complete' 等
            stage: 当前Phase, 如 'planning', 'generating', 'completed'
            details: 详细ContentDict, 不截断
            section_title: 当前Section标题(Optional)
            section_index: 当前SectionIndex(Optional)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details
        }
        
        # Append to JSONL file
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        """RecordReportGenerateStart"""
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": "Report generation started"
            }
        )
    
    def log_planning_start(self):
        """RecordOutline规划Start"""
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": "Starting outline planning"}
        )
    
    def log_planning_context(self, context: Dict[str, Any]):
        """Record规划时Get的ContextInfo"""
        self.log(
            action="planning_context",
            stage="planning",
            details={
                "message": "Fetching simulation context",
                "context": context
            }
        )
    
    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        """RecordOutline规划Complete"""
        self.log(
            action="planning_complete",
            stage="planning",
            details={
                "message": "Outline planning completed",
                "outline": outline_dict
            }
        )
    
    def log_section_start(self, section_title: str, section_index: int):
        """RecordSectionGenerateStart"""
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": f"StartGenerateSection: {section_title}"}
        )
    
    def log_react_thought(self, section_title: str, section_index: int, iteration: int, thought: str):
        """Record ReACT 思考过程"""
        self.log(
            action="react_thought",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "thought": thought,
                "message": f"ReACT 第{iteration}轮思考"
            }
        )
    
    def log_tool_call(
        self, 
        section_title: str, 
        section_index: int,
        tool_name: str, 
        parameters: Dict[str, Any],
        iteration: int
    ):
        """RecordTool调用"""
        self.log(
            action="tool_call",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": f"调用Tool: {tool_name}"
            }
        )
    
    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteration: int
    ):
        """RecordTool调用Result(完整Content, 不截断)"""
        self.log(
            action="tool_result",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "result": result,  # 完整Result, 不截断
                "result_length": len(result),
                "message": f"Tool {tool_name} ReturnResult"
            }
        )
    
    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteration: int,
        has_tool_calls: bool,
        has_final_answer: bool
    ):
        """Record LLM Response(完整Content, 不截断)"""
        self.log(
            action="llm_response",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "response": response,  # 完整Response, 不截断
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": f"LLM Response (Tool调用: {has_tool_calls}, 最终答案: {has_final_answer})"
            }
        )
    
    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int
    ):
        """RecordSectionContentGenerateComplete(仅RecordContent, 不代表整个SectionComplete)"""
        self.log(
            action="section_content",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,  # 完整Content, 不截断
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": f"Section {section_title} ContentGenerateComplete"
            }
        )
    
    def log_section_full_complete(
        self,
        section_title: str,
        section_index: int,
        full_content: str
    ):
        """
        RecordSectionGenerateComplete

        Frontend should listen to this log toJudge一个Section是否真正Complete, 并Get完整Content
        """
        self.log(
            action="section_complete",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": f"Section {section_title} GenerateComplete"
            }
        )
    
    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        """RecordReport generation complete"""
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": "Report generation completed"
            }
        )
    
    def log_error(self, error_message: str, stage: str, section_title: str = None):
        """RecordError"""
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={
                "error": error_message,
                "message": f"Error: {error_message}"
            }
        )


class ReportConsoleLogger:
    """
    Report Agent 控制台Logger
    
    Console-style log (INFO, WARNING等)WriteReportFile夹中的 console_log.txt File.
    这些日志与 agent_log.jsonl 不同, 是纯TextFormat的控制台Output.
    """
    
    def __init__(self, report_id: str):
        """
        Initialize控制台Logger
        
        Args:
            report_id: Report ID, 用于确定Log filePath
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'console_log.txt'
        )
        self._ensure_log_file()
        self._file_handler = None
        self._setup_file_handler()
    
    def _ensure_log_file(self):
        """Ensure log file directory exists"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _setup_file_handler(self):
        """SetFileProcessor, 将日志同时WriteFile"""
        import logging
        
        # CreateFileProcessor
        self._file_handler = logging.FileHandler(
            self.log_file_path,
            mode='a',
            encoding='utf-8'
        )
        self._file_handler.setLevel(logging.INFO)
        
        # Use the same concise style as consoleFormat
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self._file_handler.setFormatter(formatter)
        
        # Add到 report_agent 相关的 logger
        loggers_to_attach = [
            'mirofish.report_agent',
            'mirofish.zep_tools',
        ]
        
        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)
            # 避免重复Add
            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)
    
    def close(self):
        """CloseFileProcessor并从 logger 中Remove"""
        import logging
        
        if self._file_handler:
            loggers_to_detach = [
                'mirofish.report_agent',
                'mirofish.zep_tools',
            ]
            
            for logger_name in loggers_to_detach:
                target_logger = logging.getLogger(logger_name)
                if self._file_handler in target_logger.handlers:
                    target_logger.removeHandler(self._file_handler)
            
            self._file_handler.close()
            self._file_handler = None
    
    def __del__(self):
        """析构时确保CloseFileProcessor"""
        self.close()


class ReportStatus(str, Enum):
    """Report status"""
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """Report section"""
    title: str
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content
        }

    def to_markdown(self, level: int = 2) -> str:
        """Convert为MarkdownFormat"""
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    """Report outline"""
    title: str
    summary: str
    sections: List[ReportSection]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections]
        }
    
    def to_markdown(self) -> str:
        """Convert为MarkdownFormat"""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """Complete report"""
    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


# ═══════════════════════════════════════════════════════════════
# Prompt Template Constants
# ═══════════════════════════════════════════════════════════════

# ── Tool Descriptions ──

TOOL_DESC_INSIGHT_FORGE = """\
[Deep Insight Retrieval - Powerful retrieval tool]
This is our powerful retrieval function designed for deep analysis. It will:
1. Automatically decompose your question into multiple sub-questions
2. Retrieve information from the simulation graph across multiple dimensions
3. Integrate results from semantic search, entity analysis, and relationship chain tracking
4. Return the most comprehensive and in-depth retrieval content

[Use Cases]
- Need deep analysis of a specific topic
- Need to understand multiple aspects of an event
- Need rich material to support report sections

[Returns]
- Related factual quotes (can be directly cited)
- Core entity insights
- Relationship chain analysis"""

TOOL_DESC_PANORAMA_SEARCH = """\
[Panoramic Search - Get full overview]
This tool retrieves the complete overview of simulation results, ideal for understanding event evolution. It will:
1. Get all related nodes and relationships
2. Distinguish between currently valid facts and historical/expired facts
3. Help you understand how public opinion evolved

[Use Cases]
- Need to understand the complete development trajectory of events
- Need to compare opinion changes across different stages
- Need comprehensive entity and relationship information

[Returns]
- Currently valid facts (latest simulation results)
- Historical/expired facts (evolution records)
- All involved entities"""

TOOL_DESC_QUICK_SEARCH = """\
[Quick Search - Fast retrieval]
A lightweight fast retrieval tool suitable for simple, direct information queries.

[Use Cases]
- Need to quickly look up specific information
- Need to verify a fact
- Simple information retrieval

[Returns]
- List of facts most relevant to the query"""

TOOL_DESC_INTERVIEW_AGENTS = """\
[Deep Interview - Real Agent Interview (Dual Platform)]
Calls the OASIS simulation environment's interview API to conduct real interviews with running simulation Agents!
This is not LLM simulation - it calls real interview endpoints to get simulation Agents' original responses.
By default interviews on both Twitter and Reddit platforms for more comprehensive perspectives.

Workflow:
1. Automatically reads persona files to understand all simulation Agents
2. Intelligently selects Agents most relevant to the interview topic (e.g., students, media, officials)
3. Automatically generates interview questions
4. Calls /api/simulation/interview/batch endpoint for real dual-platform interviews
5. Integrates all interview results for multi-perspective analysis

[Use Cases]
- Need different role perspectives on events (What do students think? Media? Officials?)
- Need to collect opinions and stances from multiple parties
- Need authentic simulation Agent responses (from OASIS simulation environment)
- Want to make reports more vivid with "interview transcripts"

[Returns]
- Identity information of interviewed Agents
- Each Agent's interview responses on both Twitter and Reddit platforms
- Key quotes (can be directly cited)
- Interview summary and viewpoint comparison

[IMPORTANT] Requires OASIS simulation environment to be running!"""

# ── Outline Planning Prompt ──

PLAN_SYSTEM_PROMPT = """\
You are an expert writer of "Future Prediction Reports" with a "God's Eye View" of the simulation world — you can observe every Agent's behavior, statements, and interactions.

[Core Concept]
We built a simulation world and injected specific "simulation requirements" as variables. The simulation's evolution results are predictions of what might happen in the future. You are observing not "experimental data" but a "rehearsal of the future."

[Your Task]
Write a "Future Prediction Report" answering:
1. Under our set conditions, what happened in the future?
2. How did various Agents (populations) react and act?
3. What noteworthy future trends and risks does this simulation reveal?

[Report Positioning]
- ✅ This is a simulation-based future prediction report revealing "if this happens, what will the future look like"
- ✅ Focus on prediction results: event trajectory, group reactions, emergent phenomena, potential risks
- ✅ Agent behavior in the simulation world is a prediction of future human behavior
- ❌ This is NOT an analysis of current real-world conditions
- ❌ This is NOT a generic public opinion summary

[Section Limits]
- Minimum 2 sections, maximum 5 sections
- No sub-sections needed, each section directly contains complete content
- Content should be concise, focused on core prediction findings
- Section structure should be designed based on prediction results

IMPORTANT: ALL output text MUST be in English.

Please output the report outline in JSON format:
{
    "title": "Report title (in English)",
    "summary": "Report summary (one sentence summarizing core prediction findings, in English)",
    "sections": [
        {
            "title": "Section title (in English)",
            "description": "Section content description (in English)"
        }
    ]
}

Note: The sections array must have minimum 2, maximum 5 elements!"""

PLAN_USER_PROMPT_TEMPLATE = """\
[Prediction Scenario Setup]
Variable injected into the simulation world (simulation requirement): {simulation_requirement}

[Simulation World Scale]
- Number of entities in simulation: {total_nodes}
- Number of relationships between entities: {total_edges}
- Entity type distribution: {entity_types}
- Number of active Agents: {total_entities}

[Sample Future Facts Predicted by Simulation]
{related_facts_json}

Review this future rehearsal from a "God's Eye View":
1. Under our set conditions, what state did the future present?
2. How did various populations (Agents) react and act?
3. What noteworthy future trends does this simulation reveal?

Design the most appropriate report section structure based on prediction results.

IMPORTANT: ALL output MUST be in English.

[Reminder] Report sections: minimum 2, maximum 5, content should be concise and focused on core prediction findings."""

# ── Section Generation Prompt ──

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert writer of "Future Prediction Reports", currently writing one section of the report.

Report title: {report_title}
Report summary: {report_summary}
Prediction scenario (simulation requirement): {simulation_requirement}

Current section to write: {section_title}

═══════════════════════════════════════════════════════════════
[Core Concept]
═══════════════════════════════════════════════════════════════

The simulation world is a rehearsal of the future. We injected specific conditions (simulation requirements),
and Agent behavior and interactions represent predictions of future human behavior.

Your task is:
- Reveal what happened in the future under the set conditions
- Predict how various populations (Agents) reacted and acted
- Discover noteworthy future trends, risks and opportunities

❌ Do NOT write it as an analysis of current real-world conditions
✅ Focus on "what will the future be like" — simulation results ARE the predicted future

═══════════════════════════════════════════════════════════════
[Most Important Rules - Must Follow]
═══════════════════════════════════════════════════════════════

1. [Must call tools to observe the simulation world]
   - You are observing the future rehearsal from a "God's Eye View"
   - All content must come from events and Agent actions in the simulation world
   - Do NOT use your own knowledge to write report content
   - Each section must call tools at least 3 times (max 5) to observe the simulation world

2. [Must quote Agents' original statements and actions]
   - Agent statements and behavior are predictions of future human behavior
   - Use quote format to display these predictions in the report, e.g.:
     > "A certain group would say: original content..."
   - These quotes are core evidence of simulation predictions

3. [Language consistency - ALL content must be in English]
   - Tool results may contain mixed language content
   - The report MUST be written entirely in English
   - When quoting tool results, translate any non-English content into fluent English
   - Maintain original meaning while ensuring natural expression
   - This rule applies to both body text and quote blocks (> format)

4. [Faithfully present prediction results]
   - Report content must reflect simulation results representing the future
   - Do not add information that doesn't exist in the simulation
   - If information is insufficient on some aspect, state this honestly

═══════════════════════════════════════════════════════════════
[⚠️ Format Rules - Extremely Important!]
═══════════════════════════════════════════════════════════════

[One section = smallest content unit]
- Each section is the smallest unit of the report
- ❌ Do NOT use any Markdown headings (#, ##, ###, #### etc.) within sections
- ❌ Do NOT add the section title at the beginning of content
- ✅ Section titles are automatically added by the system, you only write body text
- ✅ Use **bold text**, paragraph breaks, quotes, and lists to organize content

[Correct Example]
```
This section analyzes the public opinion spread. Through deep analysis of simulation data, we found...

**Initial Outbreak Phase**

Twitter served as the primary platform for information dissemination:

> "Twitter contributed 68% of initial voice volume..."

**Emotion Amplification Phase**

Short-video platforms further amplified the event's impact:

- Strong visual impact
- High emotional resonance
```

[Wrong Example]
```
## Executive Summary          ← Wrong! Don't add any headings
### Phase One               ← Wrong! Don't use ### for subsections
#### 1.1 Detailed Analysis  ← Wrong! Don't use #### for subdivisions

This section analyzes...
```

═══════════════════════════════════════════════════════════════
[Available Retrieval Tools] (call 3-5 times per section)
═══════════════════════════════════════════════════════════════

{tools_description}

[Tool Usage Tips - Mix different tools, don't use just one]
- insight_forge: Deep insight analysis, auto-decomposes questions for multi-dimensional fact/relationship retrieval
- panorama_search: Wide-angle panoramic search, understand full picture, timeline, and evolution
- quick_search: Quickly verify a specific piece of information
- interview_agents: Interview simulation Agents, get first-person perspectives and real reactions from different roles

═══════════════════════════════════════════════════════════════
[Workflow]
═══════════════════════════════════════════════════════════════

Each reply you can only do ONE of the following two things (not both):

Option A - Call a tool:
Output your thinking, then call a tool using this format:
<tool_call>
{{"name": "tool_name", "parameters": {{"param_name": "param_value"}}}}
</tool_call>
The system will execute the tool and return results. You cannot write tool results yourself.

Option B - Output final content:
When you have enough information from tools, output section content starting with "Final Answer:"

⚠️ Strictly prohibited:
- Do NOT include both a tool call and Final Answer in one reply
- Do NOT fabricate tool results (Observations) — all tool results are injected by the system
- Maximum one tool call per reply

═══════════════════════════════════════════════════════════════
[Section Content Requirements]
═══════════════════════════════════════════════════════════════

1. Content must be based on simulation data retrieved by tools
2. Extensively quote original text to demonstrate simulation results
3. Use Markdown format (but NO headings):
   - Use **bold text** for emphasis (instead of subheadings)
   - Use lists (- or 1.2.3.) to organize key points
   - Use blank lines to separate paragraphs
   - ❌ Do NOT use #, ##, ###, #### or any heading syntax
4. [Quote format - must be standalone paragraphs]
   Quotes must be standalone paragraphs with blank lines before and after:

   ✅ Correct:
   ```
   The response was considered lacking in substance.

   > "The response pattern appeared rigid and slow in the rapidly changing social media environment."

   This evaluation reflects widespread public dissatisfaction.
   ```

   ❌ Wrong:
   ```
   The response was considered lacking.> "The response pattern..." This reflects...
   ```
5. Maintain logical coherence with other sections
6. [Avoid repetition] Carefully read completed sections below, don't repeat the same information
7. [Emphasis] Do NOT add any headings! Use **bold** instead of subheadings
8. ALL output MUST be in English"""

SECTION_USER_PROMPT_TEMPLATE = """\
Completed sections (read carefully to avoid repetition):
{previous_content}

═══════════════════════════════════════════════════════════════
[Current Task] Write section: {section_title}
═══════════════════════════════════════════════════════════════

[Important Reminders]
1. Carefully read the completed sections above to avoid repeating content!
2. You MUST call tools to get simulation data before writing
3. Mix different tools, don't use just one type
4. Report content must come from retrieval results, not your own knowledge

[⚠️ Format Warning - Must Follow]
- ❌ Do NOT write any headings (#, ##, ###, #### are all prohibited)
- ❌ Do NOT write "{section_title}" as the opening
- ✅ Section title is automatically added by the system
- ✅ Write body text directly, use **bold** instead of subheadings

ALL output MUST be in English.

Begin:
1. First think (Thought) about what information this section needs
2. Then call tools (Action) to get simulation data
3. After collecting enough info, output Final Answer (body text only, no headings)"""

# ── ReACT Loop Message Templates ──

REACT_OBSERVATION_TEMPLATE = """\
Observation (retrieval results):

═══ Tool {tool_name} returned ═══
{result}

═══════════════════════════════════════════════════════════════
Tools called: {tool_calls_count}/{max_tool_calls} (used: {used_tools_str}){unused_hint}
- If information is sufficient: output section content starting with "Final Answer:" (must quote original text above)
- If more information needed: call a tool to continue retrieving
═══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "[Notice] You only called tools {tool_calls_count} times, minimum {min_tool_calls} required. "
    "Please call more tools to get simulation data before outputting Final Answer. {unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "Only {tool_calls_count} tool calls so far, minimum {min_tool_calls} required. "
    "Please call tools to get simulation data. {unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "Tool call limit reached ({tool_calls_count}/{max_tool_calls}), no more tool calls allowed. "
    'Please immediately output section content starting with "Final Answer:" based on information already gathered.'
)

REACT_UNUSED_TOOLS_HINT = "\n💡 You haven't used: {unused_list}. Try different tools for multi-perspective information."

REACT_FORCE_FINAL_MSG = "Tool call limit reached. Please output Final Answer: and generate section content now."

# ── Chat Prompt ──

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
You are a concise and efficient simulation prediction assistant.

[Background]
Prediction scenario: {simulation_requirement}

[Generated Analysis Report]
{report_content}

[Rules]
1. Prioritize answering based on the report content above
2. Answer directly, avoid lengthy reasoning
3. Only call tools when report content is insufficient to answer
4. Answers should be concise, clear, and organized

[Available Tools] (use only when needed, max 1-2 calls)
{tools_description}

[Tool Call Format]
<tool_call>
{{"name": "tool_name", "parameters": {{"param_name": "param_value"}}}}
</tool_call>

[Response Style]
- Concise and direct, no lengthy essays
- Use > format to quote key content
- Give conclusions first, then explain reasoning
- ALL responses MUST be in English"""

CHAT_OBSERVATION_SUFFIX = "\n\nPlease answer the question concisely. ALL output MUST be in English."


# ═══════════════════════════════════════════════════════════════
# ReportAgent Main Class
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """
    Report Agent - SimulationReportGenerateAgent

    采用ReACT(Reasoning + Acting)Pattern: 
    1. 规划Phase: AnalysisSimulation需求, 规划ReportDirectory结构
    2. GeneratePhase: 逐SectionGenerateContent, 每Section可多次调用ToolGetInfo
    3. 反思Phase: 检查Content完整性和准确性
    """
    
    # Max tool calls per section
    MAX_TOOL_CALLS_PER_SECTION = 5
    
    # Max reflection rounds
    MAX_REFLECTION_ROUNDS = 3
    
    # Max tool calls per chat
    MAX_TOOL_CALLS_PER_CHAT = 2
    
    def __init__(
        self, 
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        zep_tools: Optional[ZepToolsService] = None
    ):
        """
        InitializeReport Agent
        
        Args:
            graph_id: Graph ID
            simulation_id: SimulationID
            simulation_requirement: Simulation需求Description
            llm_client: LLM客户端(Optional)
            zep_tools: ZepToolService(Optional)
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement
        
        self.llm = llm_client or LLMClient()
        self.zep_tools = zep_tools or ZepToolsService()
        
        # Tool definitions
        self.tools = self._define_tools()
        
        # Logger(在 generate_report 中Initialize)
        self.report_logger: Optional[ReportLogger] = None
        # Console logger(在 generate_report 中Initialize)
        self.console_logger: Optional[ReportConsoleLogger] = None
        
        logger.info(f"ReportAgent initialized: graph_id={graph_id}, simulation_id={simulation_id}")
    
    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """Define available tools"""
        return {
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "The question or topic you want to analyze deeply",
                    "report_context": "Context of the current report section (Optional, helps generate more precise sub-questions)"
                }
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "Search query used for relevance sorting",
                    "include_expired": "Whether to include expired/historical content (Default: True)"
                }
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "Search query string",
                    "limit": "Number of results to return (Optional, Default: 10)"
                }
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": "Interview topic or requirement description (e.g., 'understand consumers\' complaints about pricing')",
                    "max_agents": "Maximum number of agents to interview (Optional, Default: 5, Max: 10)"
                }
            }
        }
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any], report_context: str = "") -> str:
        """
        ExecuteTool调用
        
        Args:
            tool_name: ToolName
            parameters: Toolparameter
            report_context: ReportContext(用于InsightForge)
            
        Returns:
            ToolExecuteResult(TextFormat)
        """
        logger.info(f"Executing tool: {tool_name}, params: {parameters}")
        
        try:
            if tool_name == "insight_forge":
                query = parameters.get("query", "")
                ctx = parameters.get("report_context", "") or report_context
                result = self.zep_tools.insight_forge(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx
                )
                return result.to_text()
            
            elif tool_name == "panorama_search":
                # Panoramic search - get full view
                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ['true', '1', 'yes']
                result = self.zep_tools.panorama_search(
                    graph_id=self.graph_id,
                    query=query,
                    include_expired=include_expired
                )
                return result.to_text()
            
            elif tool_name == "quick_search":
                # Simple search - fast retrieval
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.zep_tools.quick_search(
                    graph_id=self.graph_id,
                    query=query,
                    limit=limit
                )
                return result.to_text()
            
            elif tool_name == "interview_agents":
                # Deep interview - call real OASIS interview API for Agent responses (dual platform)
                interview_topic = parameters.get("interview_topic", parameters.get("query", ""))
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.zep_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents
                )
                return result.to_text()
            
            # ========== Legacy tools (internally redirect to new tools) ==========
            
            elif tool_name == "search_graph":
                # Redirect to quick_search
                logger.info("search_graph redirected to quick_search")
                return self._execute_tool("quick_search", parameters, report_context)
            
            elif tool_name == "get_graph_statistics":
                result = self.zep_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.zep_tools.get_entity_summary(
                    graph_id=self.graph_id,
                    entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_simulation_context":
                # Redirect to insight_forge (more powerful)
                logger.info("get_simulation_context redirected to insight_forge")
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool("insight_forge", {"query": query}, report_context)
            
            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.zep_tools.get_entities_by_type(
                    graph_id=self.graph_id,
                    entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            else:
                return f"Unknown tool: {tool_name}. Use one of: insight_forge, panorama_search, quick_search"
                
        except Exception as e:
            logger.error(f"Tool failed: {tool_name}, error: {str(e)}")
            return f"Tool failed: {str(e)}"
    
    # Valid tool names set, 用于裸 JSON 兜底Parse时校验
    VALID_TOOL_NAMES = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        从LLMResponse中ParseTool调用

        Support的Format(按优先级): 
        1. <tool_call>{"name": "tool_name", "parameters": {...}}</tool_call>
        2. 裸 JSON(ResponseThe whole or single line is aTool调用 JSON)
        """
        tool_calls = []

        # Format1: XML风格(StandardFormat)
        xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # Format2: 兜底 - LLM 直接Output裸 JSON(没包 <tool_call> 标签)
        # 只在Format1未Match时尝试, 避免误Match正文中的 JSON
        stripped = response.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # Response可能包含思考文字 + 裸 JSON, try to extract the last JSON 对象
        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                call_data = json.loads(match.group(1))
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:
        """Validate if parsed JSON is a valid tool call"""
        # Support {"name": ..., "parameters": ...} 和 {"tool": ..., "params": ...} 两种键名
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.VALID_TOOL_NAMES:
            # 统一键名为 name / parameters
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False
    
    def _get_tools_description(self) -> str:
        """Generate tool description text"""
        desc_parts = ["Available tools:"]
        for name, tool in self.tools.items():
            params_desc = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  Parameters: {params_desc}")
        return "\n".join(desc_parts)
    
    def plan_outline(
        self, 
        progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        规划ReportOutline
        
        使用LLMAnalysisSimulation需求, 规划Report的Directory结构
        
        Args:
            progress_callback: ProgressCallbackFunction
            
        Returns:
            ReportOutline: ReportOutline
        """
        logger.info("Planning report outline...")
        
        if progress_callback:
            progress_callback("planning", 0, "Analyzing simulation requirements...")
        
        # 首先GetSimulationContext
        context = self.zep_tools.get_simulation_context(
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement
        )
        
        if progress_callback:
            progress_callback("planning", 30, "Generating report outline...")
        
        system_prompt = PLAN_SYSTEM_PROMPT
        user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            total_nodes=context.get('graph_statistics', {}).get('total_nodes', 0),
            total_edges=context.get('graph_statistics', {}).get('total_edges', 0),
            entity_types=list(context.get('graph_statistics', {}).get('entity_types', {}).keys()),
            total_entities=context.get('total_entities', 0),
            related_facts_json=json.dumps(context.get('related_facts', [])[:10], ensure_ascii=False, indent=2),
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            if progress_callback:
                progress_callback("planning", 80, "Parsing outline structure...")
            
            # ParseOutline
            sections = []
            for section_data in response.get("sections", []):
                sections.append(ReportSection(
                    title=section_data.get("title", ""),
                    content=""
                ))
            
            outline = ReportOutline(
                title=response.get("title", "Simulation Analysis Report"),
                summary=response.get("summary", ""),
                sections=sections
            )
            
            if progress_callback:
                progress_callback("planning", 100, "Outline planning completed")
            
            logger.info(f"Outline planning complete: {len(sections)} sections")
            return outline
            
        except Exception as e:
            logger.error(f"Outline planning failed: {str(e)}")
            # ReturnDefaultOutline(3个Section, 作为fallback)
            return ReportOutline(
                title="Future Prediction Report",
                summary="Future trends and risk analysis based on simulation predictions",
                sections=[
                    ReportSection(title="Prediction Scenarios and Core Findings"),
                    ReportSection(title="Crowd Behavior Prediction Analysis"),
                    ReportSection(title="Trend Outlook and Risk Alerts")
                ]
            )
    
    def _generate_section_react(
        self, 
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0
    ) -> str:
        """
        使用ReACTPatternGenerate单个SectionContent
        
        ReACTLoop: 
        1. Thought(思考)- Analysis需要什么Info
        2. Action(行动)- 调用ToolGetInfo
        3. Observation(观察)- AnalysisToolReturnResult
        4. 重复直到Infosufficient or maximum count reached
        5. Final Answer(最终回答)- GenerateSectionContent
        
        Args:
            section: 要Generate的Section
            outline: 完整Outline
            previous_sections: 之前Section的Content(for maintaining coherence)
            progress_callback: ProgressCallback
            section_index: SectionIndex(用于日志Record)
            
        Returns:
            SectionContent(MarkdownFormat)
        """
        logger.info(f"ReACT generating section: {section.title}")
        
        # Log section start
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)
        
        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            tools_description=self._get_tools_description(),
        )

        # Build user prompt - pass max 4000 chars per completed section
        if previous_sections:
            previous_parts = []
            for sec in previous_sections:
                # Max 4000 chars per section
                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = "(This is the first section)"
        
        user_prompt = SECTION_USER_PROMPT_TEMPLATE.format(
            previous_content=previous_content,
            section_title=section.title,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # ReACT loop
        tool_calls_count = 0
        max_iterations = 5  # Max iteration rounds
        min_tool_calls = 3  # Min tool calls
        conflict_retries = 0  # Consecutive conflicts with both tool call and Final Answer
        used_tools = set()  # Track used tool names
        all_tools = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

        # Report context for InsightForge sub-question generation
        report_context = f"Section title: {section.title}\nSimulation requirement: {self.simulation_requirement}"
        
        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating", 
                    int((iteration / max_iterations) * 100),
                    f"Deep retrieval and writing ({tool_calls_count}/{self.MAX_TOOL_CALLS_PER_SECTION})"
                )
            
            # Call LLM
            response = self.llm.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=4096
            )

            # Check if LLM returned None (API error or empty content)
            if response is None:
                logger.warning(f"Section {section.title} iteration {iteration + 1}: LLM returned None")
                # If iterations remain, add message and retry
                if iteration < max_iterations - 1:
                    messages.append({"role": "assistant", "content": "(Empty response)"})
                    messages.append({"role": "user", "content": "Please continue generating content."})
                    continue
                # Last iteration also None, break to force finalize
                break

            logger.debug(f"LLMResponse: {response[:200]}...")

            # Parse once, reuse results
            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response

            # ── Conflict: LLM output both tool call and Final Answer ──
            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    f"Section {section.title} round {iteration+1}: "
                    f"LLM output both tool call and Final Answer (conflict #{conflict_retries})"
                )

                if conflict_retries <= 2:
                    # First two times: discard response, ask LLM to retry
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "[FORMAT ERROR] You included both a tool call and Final Answer in one reply, which is not allowed.\n"
                            "Each reply must do exactly one of these two things:\n"
                            "- Call a tool (output a <tool_call> block, do NOT write Final Answer)\n"
                            "- Output final content (start with 'Final Answer:', do NOT include <tool_call>)\n"
                            "Please reply again, doing only one of these."
                        ),
                    })
                    continue
                else:
                    # Third time: degrade, truncate to first tool call, force execute
                    logger.warning(
                        f"Section {section.title}: {conflict_retries} consecutive conflicts, "
                        "degrading to truncate and execute first tool call"
                    )
                    first_tool_end = response.find('</tool_call>')
                    if first_tool_end != -1:
                        response = response[:first_tool_end + len('</tool_call>')]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            # Log LLM response
            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer
                )

            # ── Case 1: LLM output Final Answer ──
            if has_final_answer:
                # Insufficient tool calls, reject and ask to continue
                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": response})
                    unused_tools = all_tools - used_tools
                    unused_hint = f"(These tools haven't been used yet, try them: {', '.join(unused_tools)})" if unused_tools else ""
                    messages.append({
                        "role": "user",
                        "content": REACT_INSUFFICIENT_TOOLS_MSG.format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    })
                    continue

                # Normal completion
                final_answer = response.split("Final Answer:")[-1].strip()
                logger.info(f"Section {section.title} completed (tool calls: {tool_calls_count})")

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count
                    )
                return final_answer

            # ── Case 2: LLM tried to call a tool ──
            if has_tool_calls:
                # Tool quota exhausted → tell LLM to output Final Answer
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": REACT_TOOL_LIMIT_MSG.format(
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        ),
                    })
                    continue

                # Only execute the first tool call
                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(f"LLM tried to call {len(tool_calls)} tools, executing only the first: {call['name']}")

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1
                    )

                result = self._execute_tool(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context
                )

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=result,
                        iteration=iteration + 1
                    )

                tool_calls_count += 1
                used_tools.add(call['name'])

                # Build unused tools hint
                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.format(unused_list=", ".join(unused_tools))

                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": REACT_OBSERVATION_TEMPLATE.format(
                        tool_name=call["name"],
                        result=result,
                        tool_calls_count=tool_calls_count,
                        max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        used_tools_str=", ".join(used_tools),
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # ── Case 3: Neither tool call nor Final Answer ──
            messages.append({"role": "assistant", "content": response})

            if tool_calls_count < min_tool_calls:
                # Insufficient tool calls, recommend unused tools
                unused_tools = all_tools - used_tools
                unused_hint = f"(These tools haven't been used yet, try them: {', '.join(unused_tools)})" if unused_tools else ""

                messages.append({
                    "role": "user",
                    "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                        tool_calls_count=tool_calls_count,
                        min_tool_calls=min_tool_calls,
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # Tool calls sufficient, LLM output content without "Final Answer:" prefix
            # Use this content directly as final answer
            logger.info(f"Section {section.title}: no 'Final Answer:' prefix detected, using LLM output directly (tool calls: {tool_calls_count})")
            final_answer = response.strip()

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count
                )
            return final_answer
        
        # Max iterations reached, force generate
        logger.warning(f"Section {section.title} reached max iterations, forcing generation")
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})
        
        response = self.llm.chat(
            messages=messages,
            temperature=0.5,
            max_tokens=4096
        )

        # Check if LLM returned None during force finalize
        if response is None:
            logger.error(f"Section {section.title} force finalize: LLM returned None, using default error")
            final_answer = f"(Section generation failed: LLM returned empty response, please retry later)"
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response
        
        # Log section content generated
        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count
            )
        
        return final_answer
    
    def generate_report(
        self, 
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None
    ) -> Report:
        """
        Generate完整Report(分Section实时Output)
        
        每个SectionGenerateComplete后立即Save到File夹, 不需要Wait整个ReportComplete.
        File结构: 
        reports/{report_id}/
            meta.json       - Report元Info
            outline.json    - ReportOutline
            progress.json   - GenerateProgress
            section_01.md   - 第1Section
            section_02.md   - 第2Section
            ...
            full_report.md  - 完整Report
        
        Args:
            progress_callback: ProgressCallbackFunction (stage, progress, message)
            report_id: Report ID(Optional, 如果不传则AutoGenerate)
            
        Returns:
            Report: 完整Report
        """
        import uuid
        
        # Auto-generate report_id if not provided
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()
        
        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        
        # List of completed section titles(用于Progress追踪)
        completed_section_titles = []
        
        try:
            # Initialize: create report folder and save initial state
            ReportManager._ensure_report_folder(report_id)
            
            # Init logger (structured log agent_log.jsonl)
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement
            )
            
            # Init console logger (console_log.txt)
            self.console_logger = ReportConsoleLogger(report_id)
            
            ReportManager.update_progress(
                report_id, "pending", 0, "Initializing report...",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            # Phase 1: Plan outline
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id, "planning", 5, "Planning report outline...",
                completed_sections=[]
            )
            
            # Log planning start
            self.report_logger.log_planning_start()
            
            if progress_callback:
                progress_callback("planning", 0, "Planning report outline...")
            
            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg: 
                    progress_callback(stage, prog // 5, msg) if progress_callback else None
            )
            report.outline = outline
            
            # Log planning completed
            self.report_logger.log_planning_complete(outline.to_dict())
            
            # Save outline to file
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id, "planning", 15, f"Outline planning complete, {len(outline.sections)} sections",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            logger.info(f"Outline saved to file: {report_id}/outline.json")
            
            # Phase 2: Generate sections (save each section)
            report.status = ReportStatus.GENERATING
            
            total_sections = len(outline.sections)
            generated_sections = []  # Save content for context
            
            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)
                
                # Update progress
                ReportManager.update_progress(
                    report_id, "generating", base_progress,
                    f"Generating section: {section.title} ({section_num}/{total_sections})",
                    current_section=section.title,
                    completed_sections=completed_section_titles
                )
                
                if progress_callback:
                    progress_callback(
                        "generating", 
                        base_progress, 
                        f"Generating section: {section.title} ({section_num}/{total_sections})"
                    )
                
                # Generate main section content
                section_content = self._generate_section_react(
                    section=section,
                    outline=outline,
                    previous_sections=generated_sections,
                    progress_callback=lambda stage, prog, msg:
                        progress_callback(
                            stage, 
                            base_progress + int(prog * 0.7 / total_sections),
                            msg
                        ) if progress_callback else None,
                    section_index=section_num
                )
                
                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")

                # Save section
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # Log section completed
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip()
                    )

                logger.info(f"Section saved: {report_id}/section_{section_num:02d}.md")
                
                # Update progress
                ReportManager.update_progress(
                    report_id, "generating", 
                    base_progress + int(70 / total_sections),
                    f"Section {section.title} completed",
                    current_section=None,
                    completed_sections=completed_section_titles
                )
            
            # Phase 3: Assemble full report
            if progress_callback:
                progress_callback("generating", 95, "Assembling full report...")
            
            ReportManager.update_progress(
                report_id, "generating", 95, "Assembling full report...",
                completed_sections=completed_section_titles
            )
            
            # Use ReportManager to assemble full report
            report.markdown_content = ReportManager.assemble_full_report(report_id, outline)
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()
            
            # Calculate total time
            total_time_seconds = (datetime.now() - start_time).total_seconds()
            
            # Log report completed
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections,
                    total_time_seconds=total_time_seconds
                )
            
            # Save final report
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id, "completed", 100, "Report generation completed",
                completed_sections=completed_section_titles
            )
            
            if progress_callback:
                progress_callback("completed", 100, "Report generation completed")
            
            logger.info(f"Report generation completed: {report_id}")
            
            # Close console logger
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            report.status = ReportStatus.FAILED
            report.error = str(e)
            
            # Log error
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")
            
            # Save failed status
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id, "failed", -1, f"Report generation failed: {str(e)}",
                    completed_sections=completed_section_titles
                )
            except Exception:
                pass  # Ignore save failure errors
            
            # Close console logger
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
    
    def chat(
        self, 
        message: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        与Report Agent对话
        
        在对话中Agent可以自主调用RetrievalTool来回答问题
        
        Args:
            message: UserMessage
            chat_history: 对话历史
            
        Returns:
            {
                "response": "AgentReply",
                "tool_calls": [调用的ToolList],
                "sources": [Info来源]
            }
        """
        logger.info(f"Report Agent chat: {message[:50]}...")
        
        chat_history = chat_history or []
        
        # Get generated report content
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                # Limit report length to avoid context overflow
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [Report content truncated] ..."
        except Exception as e:
            logger.warning(f"Failed to get report content: {e}")
        
        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else "(No report available yet)",
            tools_description=self._get_tools_description(),
        )

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add chat history
        for h in chat_history[-10:]:  # Limit历史长度
            messages.append(h)
        
        # Add user message
        messages.append({
            "role": "user", 
            "content": message
        })
        
        # ReACT loop(简化版)
        tool_calls_made = []
        max_iterations = 2  # Reduce iteration rounds
        
        for iteration in range(max_iterations):
            response = self.llm.chat(
                messages=messages,
                temperature=0.5
            )
            
            # Parse tool calls
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # No tool calls, return response directly
                clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL)
                clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
                
                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
                }
            
            # Execute tool calls (limited)
            tool_results = []
            for call in tool_calls[:1]:  # Max 1 tool call per round
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append({
                    "tool": call["name"],
                    "result": result[:1500]  # LimitResult长度
                })
                tool_calls_made.append(call)
            
            # Add results to messages
            messages.append({"role": "assistant", "content": response})
            observation = "\n".join([f"[{r['tool']} result]\n{r['result']}" for r in tool_results])
            messages.append({
                "role": "user",
                "content": observation + CHAT_OBSERVATION_SUFFIX
            })
        
        # Max iterations reached, get final response
        final_response = self.llm.chat(
            messages=messages,
            temperature=0.5
        )
        
        # Clean response
        clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', final_response, flags=re.DOTALL)
        clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
        
        return {
            "response": clean_response.strip(),
            "tool_calls": tool_calls_made,
            "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
        }


class ReportManager:
    """
    ReportManager
    
    负责Report的持久化Storage和Retrieval
    
    File结构(分SectionOutput): 
    reports/
      {report_id}/
        meta.json          - Report元Info和Status
        outline.json       - ReportOutline
        progress.json      - GenerateProgress
        section_01.md      - 第1Section
        section_02.md      - 第2Section
        ...
        full_report.md     - 完整Report
    """
    
    # Report storage directory
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, 'reports')
    
    @classmethod
    def _ensure_reports_dir(cls):
        """Ensure reports root directory exists"""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)
    
    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """Get report folder path"""
        return os.path.join(cls.REPORTS_DIR, report_id)
    
    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """Ensure report folder exists and return path"""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """Get report metadata file path"""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")
    
    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """Get full report Markdown file path"""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")
    
    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """Get outline file path"""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")
    
    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """Get progress file path"""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")
    
    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """Get section Markdown file path"""
        return os.path.join(cls._get_report_folder(report_id), f"section_{section_index:02d}.md")
    
    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """Get agent log file path"""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")
    
    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """Get console log file path"""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")
    
    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Get控制台日志Content
        
        这是ReportGenerate过程中的控制台Output日志(INFO, WARNING等), 
        与 agent_log.jsonl different from structured logs.
        
        Args:
            report_id: Report ID
            from_line: 从第几行StartRead(用于增量Get, 0 表示从头Start)
            
        Returns:
            {
                "logs": [日志行List],
                "total_lines": 总行数,
                "from_line": 起始行号,
                "has_more": 是否还有更多日志
            }
        """
        log_path = cls._get_console_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    # Keep original log line, remove trailing newline
                    logs.append(line.rstrip('\n\r'))
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # 已Read到末尾
        }
    
    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """
        GetComplete console log (one-timeGet全部)
        
        Args:
            report_id: Report ID
            
        Returns:
            日志行List
        """
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Get Agent 日志Content
        
        Args:
            report_id: Report ID
            from_line: 从第几行StartRead(用于增量Get, 0 表示从头Start)
            
        Returns:
            {
                "logs": [日志条目List],
                "total_lines": 总行数,
                "from_line": 起始行号,
                "has_more": 是否还有更多日志
            }
        """
        log_path = cls._get_agent_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # Skip lines that fail to parse
                        continue
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # 已Read到末尾
        }
    
    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Get完整的 Agent 日志(用于一次性Get全部)
        
        Args:
            report_id: Report ID
            
        Returns:
            日志条目List
        """
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        SaveReportOutline
        
        在规划PhaseComplete后立即调用
        """
        cls._ensure_report_folder(report_id)
        
        with open(cls._get_outline_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Outline saved: {report_id}")
    
    @classmethod
    def save_section(
        cls,
        report_id: str,
        section_index: int,
        section: ReportSection
    ) -> str:
        """
        Save单个Section

        在每个SectionGenerateComplete后立即调用, Implement分SectionOutput

        Args:
            report_id: Report ID
            section_index: SectionIndex(从1Start)
            section: Section对象

        Returns:
            Save的FilePath
        """
        cls._ensure_report_folder(report_id)

        # Build section Markdown content - clean possible duplicate titles
        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        # Save file
        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"Section saved: {report_id}/{file_suffix}")
        return file_path
    
    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        清理SectionContent
        
        1. RemoveContent开头与Section标题重复的Markdown标题行
        2. 将所有 ### 及以下级别的标题Convert为粗体Text
        
        Args:
            content: 原始Content
            section_title: Section标题
            
        Returns:
            清理后的Content
        """
        import re
        
        if not content:
            return content
        
        content = content.strip()
        lines = content.split('\n')
        cleaned_lines = []
        skip_next_empty = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check if line is a Markdown heading
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()
                
                # Check for duplicate section title (skip duplicates in first 5 lines)
                if i < 5:
                    if title_text == section_title or title_text.replace(' ', '') == section_title.replace(' ', ''):
                        skip_next_empty = True
                        continue
                
                # Convert all heading levels to bold
                # Section titles are added by system, content should have no headings
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")  # Add空行
                continue
            
            # If previous line was skipped title and current is empty, skip
            if skip_next_empty and stripped == '':
                skip_next_empty = False
                continue
            
            skip_next_empty = False
            cleaned_lines.append(line)
        
        # Remove leading empty lines
        while cleaned_lines and cleaned_lines[0].strip() == '':
            cleaned_lines.pop(0)
        
        # Remove leading separators
        while cleaned_lines and cleaned_lines[0].strip() in ['---', '***', '___']:
            cleaned_lines.pop(0)
            # Also remove empty lines after separator
            while cleaned_lines and cleaned_lines[0].strip() == '':
                cleaned_lines.pop(0)
        
        return '\n'.join(cleaned_lines)
    
    @classmethod
    def update_progress(
        cls, 
        report_id: str, 
        status: str, 
        progress: int, 
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None
    ) -> None:
        """
        UpdateReportGenerateProgress
        
        前端可以通过Readprogress.jsonGet实时Progress
        """
        cls._ensure_report_folder(report_id)
        
        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat()
        }
        
        with open(cls._get_progress_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """Get report generation progress"""
        path = cls._get_progress_path(report_id)
        
        if not os.path.exists(path):
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Get已Generate的SectionList
        
        Return所有已Save的SectionFileInfo
        """
        folder = cls._get_report_folder(report_id)
        
        if not os.path.exists(folder):
            return []
        
        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith('section_') and filename.endswith('.md'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse section index from filename
                parts = filename.replace('.md', '').split('_')
                section_index = int(parts[1])

                sections.append({
                    "filename": filename,
                    "section_index": section_index,
                    "content": content
                })

        return sections
    
    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        组装完整Report
        
        从已Save的SectionFile组装完整Report, 并进行标题清理
        """
        folder = cls._get_report_folder(report_id)
        
        # Build report header
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"
        
        # Read all section files in order
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]
        
        # Post-processing: clean up title issues in report
        md_content = cls._post_process_report(md_content, outline)
        
        # Save full report
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Full report assembled: {report_id}")
        return md_content
    
    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        后ProcessReportContent
        
        1. Remove重复的标题
        2. 保留Report主标题(#)和Section标题(##), Remove其他级别的标题(###, ####等)
        3. Clean up extra blank lines and separators
        
        Args:
            content: 原始ReportContent
            outline: ReportOutline
            
        Returns:
            Process后的Content
        """
        import re
        
        lines = content.split('\n')
        processed_lines = []
        prev_was_heading = False
        
        # Collect all section titles from outline
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check if heading line
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # Check for duplicate title (same title within 5 lines)
                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r'^(#{1,6})\s+(.+)$', prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    # Skip duplicate title and following empty lines
                    i += 1
                    while i < len(lines) and lines[i].strip() == '':
                        i += 1
                    continue
                
                # Heading level handling:
                # - # (level=1) keep only report main title
                # - ## (level=2) keep section titles
                # - ### and below (level>=3) convert to bold
                
                if level == 1:
                    if title == outline.title:
                        # Keep report main title
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        # Section title incorrectly used #, fix to ##
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        # Other h1 titles convert to bold
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        # Keep section title
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        # Non-section h2 titles convert to bold
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    # ### and below headings convert to bold
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False
                
                i += 1
                continue
            
            elif stripped == '---' and prev_was_heading:
                # Skip separator immediately after heading
                i += 1
                continue
            
            elif stripped == '' and prev_was_heading:
                # Keep only one blank line after title
                if processed_lines and processed_lines[-1].strip() != '':
                    processed_lines.append(line)
                prev_was_heading = False
            
            else:
                processed_lines.append(line)
                prev_was_heading = False
            
            i += 1
        
        # Clean multiple consecutive blank lines (keep at most2个)
        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    @classmethod
    def save_report(cls, report: Report) -> None:
        """SaveReport元Info和完整Report"""
        cls._ensure_report_folder(report.report_id)
        
        # Save元InfoJSON
        with open(cls._get_report_path(report.report_id), 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        # SaveOutline
        if report.outline:
            cls.save_outline(report.report_id, report.outline)
        
        # Save完整MarkdownReport
        if report.markdown_content:
            with open(cls._get_report_markdown_path(report.report_id), 'w', encoding='utf-8') as f:
                f.write(report.markdown_content)
        
        logger.info(f"Report已Save: {report.report_id}")
    
    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """GetReport"""
        path = cls._get_report_path(report_id)
        
        if not os.path.exists(path):
            # Compatible旧Format: 检查直接Storage在reportsDirectory下的File
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 重建Report对象
        outline = None
        if data.get('outline'):
            outline_data = data['outline']
            sections = []
            for s in outline_data.get('sections', []):
                sections.append(ReportSection(
                    title=s['title'],
                    content=s.get('content', '')
                ))
            outline = ReportOutline(
                title=outline_data['title'],
                summary=outline_data['summary'],
                sections=sections
            )
        
        # 如果markdown_content为空, 尝试从full_report.mdRead
        markdown_content = data.get('markdown_content', '')
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
        
        return Report(
            report_id=data['report_id'],
            simulation_id=data['simulation_id'],
            graph_id=data['graph_id'],
            simulation_requirement=data['simulation_requirement'],
            status=ReportStatus(data['status']),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get('created_at', ''),
            completed_at=data.get('completed_at', ''),
            error=data.get('error')
        )
    
    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
        """根据SimulationIDGetReport"""
        cls._ensure_reports_dir()
        
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # 新Format: File夹
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    return report
            # Compatible旧Format: JSONFile
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report and report.simulation_id == simulation_id:
                    return report
        
        return None
    
    @classmethod
    def list_reports(cls, simulation_id: Optional[str] = None, limit: int = 50) -> List[Report]:
        """列出Report"""
        cls._ensure_reports_dir()
        
        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # 新Format: File夹
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # Compatible旧Format: JSONFile
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
        
        # 按Create时间倒序
        reports.sort(key=lambda r: r.created_at, reverse=True)
        
        return reports[:limit]
    
    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """Delete report(整个File夹)"""
        import shutil
        
        folder_path = cls._get_report_folder(report_id)
        
        # 新Format: Delete整个File夹
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(f"ReportFile夹已Delete: {report_id}")
            return True
        
        # Compatible旧Format: Delete单独的File
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")
        
        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True
        
        return deleted
