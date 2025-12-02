def build_action_prompt(user_goal: str, dom_summary: str, current_url: str) -> str:
    """统一的动作协议提示词，所有模型复用。"""

    return f"""
You are a web-browsing agent.

USER TASK:
{user_goal}

CURRENT PAGE (STRUCTURED SUMMARY):
- URL: {current_url}
- DOM SUMMARY (JSON-like):
{dom_summary}

You must decide EXACTLY ONE next action.

Valid values for "action":
- "GOTO": open a new URL
- "CLICK": click an element (by CSS selector or visible text)
- "TYPE": type text into an input
- "SCROLL": scroll the page
- "DONE": task is finished and you have the final answer

Rules for "parameter":
- If action == "GOTO": parameter MUST be a full URL string (e.g. "https://www.sogou.com").
- If action == "CLICK": parameter MUST be a selector or identifier (e.g. "text=搜索", "a:has-text('最新资讯')").
- If action == "TYPE": parameter MUST be "selector:::text" (e.g. "input[name='query']:::最新 AI 新闻").
- If action == "SCROLL": parameter MUST be "down" or "up".
- If action == "DONE": parameter MUST be a short natural language summary of the FINAL answer.

SEARCH-SPECIFIC GUIDANCE:
- If the user task requires searching for information (e.g. "find latest AI news" or "search for ..."), you MUST:
  1) First open a suitable search website with GOTO (for example a Chinese-accessible engine like https://www.sogou.com or the URL explicitly given in the task).
  2) Then locate the MAIN search input box on the page. This is typically an <input> whose placeholder / aria-label / name contains words like "搜索", "search", "query".
  3) Use TYPE with parameter "selector:::query", where selector is a CSS selector for that input (for example "input[name='query']" or simply "input" if there is only one main search box), and query is the search text from the user task.
  4) After TYPE, you MUST trigger the search, either by CLICK on the search button (e.g. "text=搜索" / "button:has-text('搜索')") or by PRESSing the Enter key (if supported by the controller).
  5) Only after you have seen concrete search results and read them carefully, you may consider using DONE with a final natural-language answer.

IMPORTANT:
- You may only use action "DONE" when, on the CURRENT page, you have already found the FINAL answer to the user task, and there is no need to open further pages or refine the result.
- If the current page only shows intermediate results (e.g. a search result list or a list of links), you MUST choose GOTO/CLICK/TYPE/SCROLL instead of DONE, and set "completed": false.

OUTPUT FORMAT:
You MUST output a SINGLE JSON object ONLY, with this exact schema:

{{
  "thought": "1-3 sentences of reasoning",
  "action": "GOTO or CLICK or TYPE or SCROLL or DONE",
  "parameter": "string, following the above rules",
  "completed": true or false
}}

CONSTRAINTS:
- On the FIRST step (when there is no history yet), you MUST NOT use action "DONE". You MUST first use "GOTO" to open an appropriate website or the URL specified by the user.
- Do NOT output anything except the JSON.
- Do NOT wrap JSON in markdown.
""".strip()
