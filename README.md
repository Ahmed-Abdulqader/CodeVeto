# CodeVeto

> An open-source, multi-model AI coding assistant built for developers who want to stay in charge. No auto-pilot. No hidden token costs.

## 🎯 The Vision
Traditional AI coding assistants act like blunt instruments: they silently edit multiple files, bloat context windows, and waste tokens. 

**CodeVeto** is built for experienced developers. It operates on a strict **explicit permission model**: the AI must be granted explicit read access to review a file, and explicit write access to modify a specific function. You are the architect; the AI is just the tool.

## ⚡ Core Principles
- 🔒 **Granular Control:** Zero silent overwrites. Every read and write action requires explicit developer approval.
- 🔄 **Unbroken Workflow:** Native multi-model routing and fallbacks ensure your coding session never stops due to a single provider's rate limits.
- 💰 **Token Economy:** The architecture must minimize context-window bloat. We only feed the AI exactly what it needs.

## 🚧 Current Status: System Design Phase
We are currently architecting the core system. This is the perfect time to join and shape the foundation of the project. 

**Key areas we are designing right now:**
- Multi-model routing and fallback logic
- AST-based granular code parsing and sandboxing
- Secure, local-first token and permission management

---
*Licensed under the GNU Affero General Public License v3.0 (AGPLv3) to ensure this tool remains open, transparent, and developer-owned.*
