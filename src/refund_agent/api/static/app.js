/**
 * Refund Agent Frontend Logic
 */

class RefundApp {
  constructor() {
    this.threadId = this.generateThreadId();
    this.currentStage = "need_username";
    this.isSending = false;

    // DOM Elements
    this.chatMessages = document.getElementById("chatMessages");
    this.chatForm = document.getElementById("chatForm");
    this.userInput = document.getElementById("userInput");
    this.sendBtn = document.getElementById("sendBtn");
    this.activeThreadIdEl = document.getElementById("activeThreadId");
    this.currentStageBadge = document.getElementById("currentStageBadge");
    this.chipsList = document.getElementById("chipsList");
    this.resetChatBtn = document.getElementById("resetChatBtn");

    this.init();
  }

  generateThreadId() {
    return "session-" + Math.random().toString(36).substring(2, 10);
  }

  init() {
    this.activeThreadIdEl.textContent = this.threadId;
    this.setupEventListeners();
    this.renderSuggestionChips();
  }

  setupEventListeners() {
    this.chatForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const text = this.userInput.value.trim();
      if (text && !this.isSending) {
        this.sendMessage(text);
      }
    });

    this.resetChatBtn.addEventListener("click", () => {
      this.resetSession();
    });
  }

  selectCustomer(username, name, isEligible) {
    const prompt = `My username is ${username}`;
    this.sendMessage(prompt);
  }

  sendQuickPrompt(text) {
    this.sendMessage(text);
  }

  async sendMessage(text) {
    if (this.isSending) return;
    this.isSending = true;

    // Add User Message to UI
    this.appendMessage("user", text);
    this.userInput.value = "";
    this.sendBtn.disabled = true;

    // Show Typing Indicator
    const typingRow = this.showTypingIndicator();

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          thread_id: this.threadId,
        }),
      });

      typingRow.remove();

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        const errMsg = errData.detail || "Server error occurred while communicating with the agent.";
        this.appendMessage("bot", `⚠️ ${errMsg}`);
        return;
      }

      const data = await res.json();
      this.threadId = data.thread_id;
      this.activeThreadIdEl.textContent = this.threadId;

      // Append Agent Reply
      this.appendMessage("bot", data.reply);

      // Update stage badge and suggestion chips
      if (data.state) {
        this.updateStateUI(data.state);
      }
    } catch (err) {
      typingRow.remove();
      this.appendMessage("bot", "⚠️ Network connection error. Please make sure the FastAPI server is running.");
    } finally {
      this.isSending = false;
      this.sendBtn.disabled = false;
      this.userInput.focus();
    }
  }

  appendMessage(role, content) {
    const row = document.createElement("div");
    row.className = `msg-row ${role === "user" ? "user-row" : "bot-row"}`;

    const avatar = document.createElement("div");
    avatar.className = `msg-avatar ${role === "user" ? "user-avatar" : "bot-avatar"}`;
    avatar.textContent = role === "user" ? "You" : "AI";

    const bubble = document.createElement("div");
    bubble.className = `msg-bubble ${role === "user" ? "user-bubble" : "bot-bubble"}`;

    if (role === "bot") {
      const sender = document.createElement("div");
      sender.className = "msg-sender";
      sender.textContent = "Refund Assistant";
      bubble.appendChild(sender);

      // Render formatted text
      const formatted = this.formatBotMessage(content);
      const contentDiv = document.createElement("div");
      contentDiv.innerHTML = formatted;
      bubble.appendChild(contentDiv);
    } else {
      const p = document.createElement("p");
      p.textContent = content;
      bubble.appendChild(p);
    }

    row.appendChild(avatar);
    row.appendChild(bubble);
    this.chatMessages.appendChild(row);

    this.scrollToBottom();
  }

  formatBotMessage(text) {
    if (!text) return "";

    // Escape basic html
    let safe = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // Convert newlines to breaks
    safe = safe.replace(/\n/g, "<br>");

    // Highlight key labels
    safe = safe.replace(/date of purchase:/gi, "<strong>Date of Purchase:</strong>");
    safe = safe.replace(/total amount:/gi, "<strong>Total Amount:</strong>");
    safe = safe.replace(/Total refund:/gi, "<strong>Total Refund:</strong>");
    safe = safe.replace(/Refund summary:/gi, "<strong>Refund Summary:</strong>");

    return safe;
  }

  showTypingIndicator() {
    const row = document.createElement("div");
    row.className = "msg-row bot-row";
    row.id = "typingIndicator";

    const avatar = document.createElement("div");
    avatar.className = "msg-avatar bot-avatar";
    avatar.textContent = "AI";

    const bubble = document.createElement("div");
    bubble.className = "msg-bubble bot-bubble typing-bubble";
    bubble.innerHTML = `
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
    `;

    row.appendChild(avatar);
    row.appendChild(bubble);
    this.chatMessages.appendChild(row);
    this.scrollToBottom();

    return row;
  }

  updateStateUI(state) {
    this.currentStage = state.stage || "initial";

    // Update Badge
    const badge = this.currentStageBadge;
    badge.className = "stage-pill";

    switch (state.stage) {
      case "need_username":
        badge.textContent = "Awaiting Username";
        badge.classList.add("stage-neutral");
        break;
      case "order_lookup":
        badge.textContent = "Order Lookup";
        badge.classList.add("stage-blue");
        break;
      case "selecting_items":
        badge.textContent = "Select Items";
        badge.classList.add("stage-blue");
        break;
      case "awaiting_reason":
        badge.textContent = "Validating Reason";
        badge.classList.add("stage-amber");
        break;
      case "completed":
        if (state.refund_approved) {
          badge.textContent = "Refund Approved ($" + (state.to_refund || 0).toFixed(2) + ")";
          badge.classList.add("stage-green");
        } else {
          badge.textContent = "Refund Denied";
          badge.classList.add("stage-red");
        }
        break;
      default:
        badge.textContent = "In Progress";
        badge.classList.add("stage-neutral");
    }

    this.renderSuggestionChips(state);
  }

  renderSuggestionChips(state = null) {
    this.chipsList.innerHTML = "";
    const chips = [];

    if (!state || state.stage === "need_username" || !state.username) {
      chips.push({ label: "👤 alextaylor1001 (Eligible)", text: "My username is alextaylor1001" });
      chips.push({ label: "👤 jordanmiller1002 (Ineligible)", text: "My username is jordanmiller1002" });
    } else if (state.stage === "selecting_items") {
      chips.push({ label: "🥩 2 Ground Beef", text: "I want a refund for 2 Ground Beef" });
      chips.push({ label: "🥛 Greek Yogurt", text: "Greek Yogurt" });
      chips.push({ label: "🍊 1 Orange Juice", text: "1 Orange Juice" });
    } else if (state.stage === "awaiting_reason") {
      chips.push({ label: "✓ Spoiled & damaged (Approved)", text: "The item arrived spoiled and damaged." });
      chips.push({ label: "✓ Past expiry date (Approved)", text: "The product was expired." });
      chips.push({ label: "✓ Wrong product delivered (Approved)", text: "The wrong product was delivered." });
      chips.push({ label: "✗ Changed my mind (Denied)", text: "I changed my mind and no longer need it." });
    } else if (state.stage === "completed") {
      chips.push({ label: "🔄 Check Status Again", text: "What is the status of my refund?" });
      chips.push({ label: "✨ New Session", action: () => this.resetSession() });
    }

    chips.forEach((c) => {
      const btn = document.createElement("button");
      btn.className = "chip-btn";
      btn.textContent = c.label;
      btn.onclick = () => {
        if (c.action) {
          c.action();
        } else {
          this.sendQuickPrompt(c.text);
        }
      };
      this.chipsList.appendChild(btn);
    });
  }

  async resetSession() {
    if (this.threadId) {
      try {
        await fetch(`/api/threads/${this.threadId}`, { method: "DELETE" });
      } catch (e) {}
    }

    this.threadId = this.generateThreadId();
    this.activeThreadIdEl.textContent = this.threadId;
    this.currentStageBadge.textContent = "Awaiting Username";
    this.currentStageBadge.className = "stage-pill stage-neutral";

    this.chatMessages.innerHTML = `
      <div class="msg-row bot-row">
        <div class="msg-avatar bot-avatar">AI</div>
        <div class="msg-bubble bot-bubble">
          <div class="msg-sender">Refund Assistant</div>
          <p>Hello! Welcome to the online grocery refund service. I can help you verify your order and process refunds for damaged, expired, or incorrect products.</p>
          <p class="msg-hint">Could you please provide your <strong>username</strong> to look up your purchase?</p>
        </div>
      </div>
    `;

    this.renderSuggestionChips();
    this.userInput.value = "";
    this.userInput.focus();
  }

  switchEvalTab(tabName) {
    document.querySelectorAll(".eval-tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));

    const targetBtn = document.querySelector(`[data-tab="${tabName}"]`);
    const targetPane = document.getElementById(`pane-${tabName}`);

    if (targetBtn) targetBtn.classList.add("active");
    if (targetPane) targetPane.classList.add("active");
  }

  scrollToBottom() {
    this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
  }
}

// Global instance
const app = new RefundApp();
window.app = app;
