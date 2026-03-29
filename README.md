# AI Prediction Engine 🚀

A powerful, multi-agent swarm intelligence engine designed to simulate complex real-world scenarios, perform strategic forecasting, and build high-fidelity digital twins of social discourse.

---

## 🌟 Overview

**AI Prediction Engine** is a next-generation simulation platform that transforms raw data (news, policies, financial signals) into a dynamic, parallel digital world. By simulating thousands of independent AI agents with distinct personalities and long-term memories, the engine allows decision-makers to "pre-play" the future in a risk-free digital sandbox.

### Key Capabilities
- **Multi-Agent Simulation**: Run large-scale simulations across parallel platforms (Twitter, Reddit).
- **Offline Knowledge Graph**: Powered by a robust SQLite-backed GraphStore (no cloud dependencies required).
- **Windows Optimized**: Hardened for high-concurrency simulation on Windows with automated file-locking mitigations.
- **Deep Strategic Insights**: Extract entities, relationships, and emerging trends to generate comprehensive AI-driven reports.

---

## 📸 System Interface

The engine features a modern, responsive UI for project management, real-time agent monitoring, and Knowledge Graph visualization.

---

## 🔄 Core Workflow

1.  **Ingestion & Graph Building**: Extract entities and relationships from seed documents to build a foundational Knowledge Graph.
2.  **Simulation Configuration**: Automatically configure agent personas, activity levels, and peak periods using advanced LLM reasoning.
3.  **Active Simulation**: Observe real-time interactions between agents as they react to the evolving scenario.
4.  **Reporting & Analysis**: Generate exportable PDF reports with deep-dive analysis and AI-driven predictions.

---

## 🚀 Getting Started (Local Setup)

### Prerequisites

| Tool | Version | Requirement |
|------|---------|-------------|
| **Node.js** | 18+ | Frontend & Task Runner |
| **Python** | 3.11 - 3.12 | Backend Engine |
| **Git** | Latest | Version Control |

### 1. Environment Configuration

Copy the example environment file and fill in your LLM API details:

```bash
cp .env.example .env
```

**Required Variables:**
- `LLM_API_KEY`: Your OpenAI-compatible API key.
- `LLM_BASE_URL`: API Endpoint (e.g.,阿里百炼 for Qwen, or OpenAI).
- `LLM_MODEL_NAME`: Targeted model (e.g., `qwen-plus`).

### 2. Installation

Install all dependencies for both frontend and backend automatically:

```bash
npm run setup:all
```

### 3. Launch the Engine

Start both the frontend (Port 3000) and backend (Port 5001) simultaneously:

```bash
npm run dev
```

---

## 🏗️ Architecture Note

This version of the engine has been refactored for **complete local autonomy**:
- **Offline Graph Memory**: Replaced Zep Cloud with a local SQLite `GraphStore`, significantly reducing latency and external dependencies.
-  **Resilient Execution**: Implemented system-level monkey-patches for SQLite connection stability and UTF-8 encoding enforcement on Windows.

---

## 📬 Contact & Support

This project is part of an advanced AI prediction research initiative. For inquiries regarding custom integrations or enterprise-grade simulations, please refer to the internal documentation.

---

## 📄 License & Credits

The simulation engine is powered by **[OASIS](https://github.com/camel-ai/oasis)**. We acknowledge the contributions of the CAMEL-AI team.

---
*Built for strategic foresight and data-driven decision making.*
