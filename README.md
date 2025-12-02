# Web Agent with VLLM

This project implements an autonomous web browsing agent powered by Vision-Language Models (VLLM). It uses Playwright for browser automation and supports various open-source VLLMs to analyze web pages and perform actions.

## Features

- **Autonomous Navigation**: The agent can navigate websites, click links/buttons, type text, and scroll based on natural language tasks.
- **Multi-Model Support**: Compatible with various VLLMs including Qwen, LLaMA, LLaVA, Mistral, Mixtral, DeepSeek, ChatGLM, and Phi.
- **Visual Understanding**: Uses screenshots and DOM summaries to understand the page context.
- **Loop Detection**: Prevents the agent from getting stuck in repetitive actions.
- **Step-by-Step Execution**: Can break down tasks into steps and execute them sequentially.

## Installation

1.  **Clone the repository** (if applicable).

2.  **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Playwright browsers**:
    ```bash
    playwright install chromium
    ```

## Usage

### Running the Demo

A standalone demo script is provided to test the agent's capabilities with a default model (Qwen2-VL).

```bash
python demo.py
```

### Running the Main Agent

You can run the agent with specific models and tasks using `main.py`.

```bash
python main.py --model qwen --task "Go to https://www.python.org/ and find the latest Python version."
```

**Arguments:**

-   `--model`: The model to use. Options: `qwen`, `llama`, `llava`, `mistral`, `mixtral`, `deepseek`, `chatglm`, `phi`. Default is `qwen`.
-   `--task`: The natural language task for the agent to perform.
-   `--max_steps`: Maximum number of steps the agent can take. Default is 8.

## Project Structure

-   `main.py`: Main entry point for the agent.
-   `demo.py`: A simplified standalone demo script.
-   `agent/`: Contains the core agent logic.
    -   `web_agent.py`: The main agent class.
    -   `planner.py`: Handles planning and action generation.
    -   `controller.py`: Manages the execution flow.
    -   `tools/`: Contains tools for browser interaction (`browser_tool.py`) and OCR (`ocr_tool.py`).
-   `models/`: Contains model implementations.
    -   `base_model.py`: Abstract base class for VLLMs.
    -   `qwen_model.py`, `llama_model.py`, etc.: Specific model implementations.
-   `pipline.md`: Documentation of the agent's workflow.

## Requirements

-   Python 3.8+
-   NVIDIA GPU (recommended for running local VLLMs)
-   See `requirements.txt` for full list of dependencies.
