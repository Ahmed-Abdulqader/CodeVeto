# CodeVeto

> **An AI coding assistant built for developers who demand explicit control.**  
> No auto-pilot. No silent overwrites. No context bloat. You are the architect; the AI is just the tool.

---

## 💡 The Idea

Modern AI coding assistants often act like blunt instruments. They silently read sensitive files, bloat context windows with irrelevant code, and confidently overwrite working logic without asking. They are designed for passive consumption.

**CodeVeto** is built for the opposite: experienced developers who know exactly what they want to build and refuse to give up the wheel. 

The core philosophy of CodeVeto is the **Explicit Permission Model**:
1. The AI can suggest and navigate, but it must request explicit permission to **read** a file or code chunk.
2. The AI must request explicit permission to **edit** a file or run a command.
3. The developer holds the **Veto**. Every action is presented for approval before execution. 

This project is not for "vibe coders" who want to hand over control. It is for engineers who want AI to accelerate their workflow without compromising their agency, security, or codebase integrity.

---

## 🏗️ How It Works (Conceptual)

CodeVeto intentionally separates **Context** from **Execution**:

- **The Context Engine**: Handles codebase indexing, semantic search, and session memory. It is strictly read-only. It parses code into semantic chunks (functions, classes, methods) so the AI can find exactly what it needs without dumping entire files into the context window.
- **The Veto Layer**: Intercepts any tool call from the AI that attempts to modify state (`write_file`, `run_command`, etc.). It pauses the agent, presents a clear preview or diff to the developer, and waits for an explicit `y/N` approval before proceeding.

---

## 🙏 Acknowledgments

CodeVeto’s context indexing and search capabilities are built upon the brilliant foundation of the **[Code Context Engine (CCE)](https://github.com/elara-labs/code-context-engine)** by `elara-labs`. 

We are deeply grateful for their open-source work, which pioneered efficient, local-first code indexing and token-saving retrieval. CodeVeto takes this powerful foundation and wraps it in a strict, developer-controlled permission model. This project would not be possible without their innovation.

---

## 🚧 Current Status

**Under Active Development.**  
CodeVeto is currently in the early build and system design phase. The core context engine integration is underway, and the interactive "Veto" permission layer is being actively developed. 

This is the perfect time to follow along, provide feedback, or contribute to shaping the foundation of a truly developer-owned AI tooling ecosystem.

---

## 📜 License

Licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**.  
This ensures CodeVeto remains open, transparent, and developer-owned. No hidden telemetry, no proprietary lock-in.

