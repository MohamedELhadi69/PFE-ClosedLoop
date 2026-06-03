# Closed-Loop Telecom Remediation Architecture

```mermaid
flowchart LR
    DC["DATA COLLECTION"] --> IL["INGESTION LAYER"]
    IL --> AD(("ANOMALY DETECTION<br/>(MODEL 1)"))
    AD --> G{"ANOMALY DETECTED?"}
    G -- "YES" --> AA["ANOMALY ANALYSIS"]
    G -- "NO" --> DC

    AA --> DM["DECISION MAKING<br/>(STUDENT LLM)"]
    DM --> PA["PERFORMANCE ASSESSMENT"]
    PA -. "feedback" .-> DC

    AA --> AC(("ANOMALY CLASSIFICATION<br/>(MODEL 2)"))
    AC --> DP(("DURATION PREDICTION<br/>(MODEL 3)"))
    DP --> AA

    Q["DESCRIPTION + QnA"] --> TL["TEACHER LLM"]
    C["REMEDY CATALOGUE"] --> TL
    TL --> SM["STUDENT MEMORY"]
    SM --> DM
    C --> DM
```

Suggested caption:

Overall architecture of the proposed telecom closed-loop remediation framework. The system first collects and ingests telecom data, then applies three specialized models for anomaly detection, anomaly classification, and duration prediction. A dataset-grounded memory is constructed from resolved anomaly cases and aligned with the reviewed remedy catalogue. The LLM recommender then combines the predictive model outputs, the remedy catalogue, and retrieved past cases to support decision making and generate structured remediation plans, which are subsequently assessed in a closed-loop workflow.
