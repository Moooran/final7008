```mermaid
    flowchart TD
    START((START<br>输入任务)) --> PLAN

    PLAN((PLAN<br>生成下一步动作)) -->|goto| GOTO
    PLAN -->|click| CLICK
    PLAN -->|type| TYPE
    PLAN -->|scroll| SCROLL
    PLAN -->|download| DOWNLOAD
    PLAN -->|press| PRESS

    GOTO((GOTO<br>打开页面)) --> WAIT_DOM
    CLICK((CLICK<br>点击元素)) --> WAIT_DOM
    TYPE((TYPE<br>输入文本)) --> WAIT_DOM
    SCROLL((SCROLL<br>滚动页面)) --> WAIT_DOM
    PRESS((PRESS<br>键盘操作)) --> WAIT_DOM

    WAIT_DOM((WAIT_DOM<br>等待页面稳定)) --> SCREENSHOT
    WAIT_DOM --> DOM_SUMMARY

    SCREENSHOT((SCREENSHOT<br>截图)) --> ANALYSIS
    DOM_SUMMARY((DOM_SUMMARY<br>抽取DOM摘要)) --> ANALYSIS

    ANALYSIS((ANALYSIS<br>分析页面并决定下一步)) -->|final answer| SAVE_OUTPUT
    SAVE_OUTPUT((SAVE_OUTPUT<br>保存最终输出)) --> DONE((DONE))

    ANALYSIS -->|navigate| PLAN
    ANALYSIS -->|click| CLICK
    ANALYSIS -->|scroll| SCROLL
    ANALYSIS -->|download| DOWNLOAD

    DOWNLOAD((DOWNLOAD<br>下载文件)) --> PDF_PROCESS
    PDF_PROCESS((PDF_PROCESS<br>解析PDF)) --> ANALYSIS

    ANALYSIS -->|retry| RETRY
    RETRY((RETRY<br>重试)) --> PLAN

    ANALYSIS -->|error| ERROR
    ERROR((ERROR<br>错误处理)) -->|recoverable| RETRY
    ERROR -->|fatal| SAVE_OUTPUT

    TIMEOUT((TIMEOUT)) --> ERROR
'''