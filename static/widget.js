(function () {
  "use strict";

  // ── Read config from script tag ───────────────────────
  const script = document.currentScript || (function () {
    const scripts = document.getElementsByTagName("script");
    return scripts[scripts.length - 1];
  })();

  const WIDGET_ID   = script.getAttribute("data-widget-id") || "";
  const API_URL     = (script.getAttribute("data-api") || "").replace(/\/$/, "");
  const TITLE       = script.getAttribute("data-title")       || "AI Assistant";
  const COLOR       = script.getAttribute("data-color")       || "#4a9eff";
  const WELCOME     = script.getAttribute("data-welcome")     || "Hi! How can I help you today?";
  const LANGUAGE    = script.getAttribute("data-language")    || "English";
  const PLACEHOLDER = script.getAttribute("data-placeholder") || "Ask me anything...";
  const POSITION    = script.getAttribute("data-position")    || "right";

  if (!WIDGET_ID || !API_URL) {
    console.error("[AI Widget] Missing data-widget-id or data-api attribute.");
    return;
  }

  // ── Session ───────────────────────────────────────────
  const SESSION_ID = "aiw_" + Math.random().toString(36).substr(2, 9);

  // ── State ─────────────────────────────────────────────
  let isOpen              = false;
  let isTyping            = false;
  let pendingGeneralQuery = null;
  let messages            = [];

  // ── Helpers ───────────────────────────────────────────
  function hexToRgba(hex, alpha) {
    const num = parseInt(hex.replace("#", ""), 16);
    return `rgba(${(num >> 16) & 255},${(num >> 8) & 255},${num & 255},${alpha})`;
  }

  function escapeHtml(text) {
    const d = document.createElement("div");
    d.textContent = text;
    return d.innerHTML;
  }

  function formatMessage(text) {
    return escapeHtml(text)
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g,     "<em>$1</em>")
      .replace(/^### (.+)$/gm,  '<p class="aiw-h3">$1</p>')
      .replace(/^## (.+)$/gm,   '<p class="aiw-h2">$1</p>')
      .replace(/^# (.+)$/gm,    '<p class="aiw-h1">$1</p>')
      .replace(/^- (.+)$/gm,    "<li>$1</li>")
      .replace(/^(\d+)\. (.+)$/gm, "<li>$2</li>")
      .replace(/(<li>[\s\S]*?<\/li>)/g, '<ul class="aiw-list">$1</ul>')
      .replace(/\n\n/g, '</p><p class="aiw-para">')
      .replace(/\n/g,   "<br>");
  }

  function getTime() {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  // ── Styles ────────────────────────────────────────────
  const css = `
    #aiw-wrap * { box-sizing:border-box; margin:0; padding:0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; }

    #aiw-bubble {
      position:fixed; ${POSITION==="left"?"left:24px;":"right:24px;"} bottom:24px;
      width:56px; height:56px; border-radius:50%;
      background:${COLOR};
      box-shadow:0 4px 20px ${hexToRgba(COLOR, 0.45)};
      cursor:pointer; border:none; outline:none;
      display:flex; align-items:center; justify-content:center;
      z-index:999999; transition:transform .2s ease, box-shadow .2s ease;
    }
    #aiw-bubble:hover { transform:scale(1.08); box-shadow:0 6px 28px ${hexToRgba(COLOR, 0.55)}; }
    #aiw-bubble svg { width:26px; height:26px; fill:white; }
    #aiw-badge {
      position:absolute; top:-2px; right:-2px;
      width:14px; height:14px; background:#ff4757;
      border-radius:50%; border:2px solid white; display:none;
    }

    #aiw-panel {
      position:fixed; ${POSITION==="left"?"left:24px;":"right:24px;"} bottom:92px;
      width:370px; max-width:calc(100vw - 32px);
      height:560px; max-height:calc(100vh - 120px);
      background:#fff; border-radius:20px;
      box-shadow:0 12px 48px rgba(0,0,0,.15), 0 2px 8px rgba(0,0,0,.08);
      display:flex; flex-direction:column; overflow:hidden;
      z-index:999998; opacity:0; transform:translateY(16px) scale(.97);
      pointer-events:none; transition:opacity .25s ease, transform .25s ease;
    }
    #aiw-panel.aiw-open { opacity:1; transform:translateY(0) scale(1); pointer-events:all; }

    #aiw-header {
      background:${COLOR}; padding:16px 18px;
      display:flex; align-items:center; gap:12px; flex-shrink:0;
    }
    #aiw-hav {
      width:36px; height:36px; border-radius:50%;
      background:rgba(255,255,255,.25);
      display:flex; align-items:center; justify-content:center; flex-shrink:0;
    }
    #aiw-hav svg { width:20px; height:20px; fill:white; }
    #aiw-hinfo { flex:1; }
    #aiw-htitle { color:white; font-size:15px; font-weight:600; line-height:1.2; }
    #aiw-hstatus { color:rgba(255,255,255,.8); font-size:12px; margin-top:2px; display:flex; align-items:center; gap:4px; }
    #aiw-dot { width:7px; height:7px; border-radius:50%; background:#4ade80; display:inline-block; }
    #aiw-xbtn {
      background:rgba(255,255,255,.2); border:none; border-radius:50%;
      width:30px; height:30px; cursor:pointer;
      display:flex; align-items:center; justify-content:center;
      color:white; font-size:16px; transition:background .15s; flex-shrink:0;
    }
    #aiw-xbtn:hover { background:rgba(255,255,255,.3); }

    #aiw-msgs {
      flex:1; overflow-y:auto; overflow-x:hidden; padding:16px;
      display:flex; flex-direction:column; gap:12px; scroll-behavior:smooth;
      width:100%;
    }
    #aiw-msgs::-webkit-scrollbar { width:4px; }
    #aiw-msgs::-webkit-scrollbar-thumb { background:#e0e0e0; border-radius:4px; }

    .aiw-msg { display:flex; flex-direction:column; max-width:85%; min-width:0; animation:aiw-fi .2s ease; }
    .aiw-bot  { align-self:flex-start; }
    .aiw-user { align-self:flex-end; }

    .aiw-bub {
      padding:10px 14px; border-radius:16px;
      font-size:14px; line-height:1.55; color:#1a1a1a;
      word-wrap:break-word;
      overflow-wrap:break-word;
      word-break:break-word;
      white-space:pre-wrap;
      min-width:0;
      width:100%;
    }
    .aiw-bot  .aiw-bub { background:#f4f4f5; border-bottom-left-radius:4px; }
    .aiw-user .aiw-bub { background:${COLOR}; color:white; border-bottom-right-radius:4px; }
    .aiw-time { font-size:11px; color:#aaa; margin-top:4px; padding:0 4px; }
    .aiw-bot  .aiw-time { align-self:flex-start; }
    .aiw-user .aiw-time { align-self:flex-end; }

    .aiw-list { padding-left:18px; margin:6px 0; }
    .aiw-list li { margin-bottom:4px; }
    .aiw-para { margin-bottom:6px; }
    .aiw-h1 { font-size:15px; font-weight:700; margin-bottom:6px; }
    .aiw-h2 { font-size:14px; font-weight:700; margin-bottom:4px; }
    .aiw-h3 { font-size:13px; font-weight:600; margin-bottom:4px; }

    .aiw-typing {
      display:flex; align-items:center; gap:5px;
      padding:12px 14px; background:#f4f4f5;
      border-radius:16px; border-bottom-left-radius:4px; width:fit-content;
    }
    .aiw-d {
      width:7px; height:7px; border-radius:50%; background:#999;
      animation:aiw-b 1.2s infinite;
    }
    .aiw-d:nth-child(2) { animation-delay:.2s; }
    .aiw-d:nth-child(3) { animation-delay:.4s; }

    .aiw-cfm-btns { display:flex; gap:8px; margin-top:8px; }
    .aiw-cfm-btn {
      padding:7px 16px; border-radius:20px; border:none;
      font-size:13px; font-weight:500; cursor:pointer; transition:opacity .15s;
    }
    .aiw-cfm-btn:hover { opacity:.85; }
    .aiw-cfm-yes { background:${COLOR}; color:white; }
    .aiw-cfm-no  { background:#f0f0f0; color:#555; }
    .aiw-cfm-btn:disabled { opacity:.5; cursor:not-allowed; }

    #aiw-footer { padding:12px 14px; border-top:1px solid #f0f0f0; flex-shrink:0; }
    #aiw-irow   { display:flex; gap:8px; align-items:flex-end; }
    #aiw-input  {
      flex:1; border:1.5px solid #e8e8e8; border-radius:22px;
      padding:10px 16px; font-size:14px; outline:none; resize:none;
      max-height:100px; line-height:1.4; color:#1a1a1a; background:#fafafa;
      transition:border-color .15s; font-family:inherit;
      overflow-y:hidden;
    }
    #aiw-input::-webkit-scrollbar { display:none; }
    #aiw-input { scrollbar-width:none; -ms-overflow-style:none; }
    #aiw-input:focus { border-color:${COLOR}; background:white; }
    #aiw-input::placeholder { color:#bbb; }

    /* FIX: send button hidden by default, appears only when input has text */
    #aiw-send {
      width:40px; height:40px; border-radius:50%;
      background:${COLOR}; border:none; cursor:pointer;
      display:flex; align-items:center; justify-content:center;
      flex-shrink:0;
      opacity:0;
      transform:scale(0.6);
      pointer-events:none;
      transition:opacity .2s ease, transform .2s ease;
    }
    #aiw-send.aiw-send-visible {
      opacity:1;
      transform:scale(1);
      pointer-events:all;
    }
    #aiw-send:hover   { opacity:.88 !important; transform:scale(1.05) !important; }
    #aiw-send:disabled { opacity:.45 !important; cursor:not-allowed; transform:scale(1) !important; }
    #aiw-send svg { width:18px; height:18px; fill:white; }

    #aiw-powered { text-align:center; font-size:11px; color:#ccc; margin-top:8px; }

    @keyframes aiw-b  { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-5px)} }
    @keyframes aiw-fi { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }

    @media(max-width:420px) {
      #aiw-panel { width:calc(100vw - 32px); ${POSITION==="left"?"left:16px;":"right:16px;"} }
    }
  `;

  const styleEl = document.createElement("style");
  styleEl.textContent = css;
  document.head.appendChild(styleEl);

  // ── Build DOM ─────────────────────────────────────────
  const wrap = document.createElement("div");
  wrap.id    = "aiw-wrap";
  wrap.innerHTML = `
    <button id="aiw-bubble" aria-label="Open chat">
      <svg viewBox="0 0 24 24"><path d="M12 2C6.477 2 2 6.477 2 12c0 1.89.525 3.66 1.438 5.168L2.05 21.5l4.518-1.374A9.953 9.953 0 0012 22c5.523 0 10-4.477 10-10S17.523 2 12 2zm0 18a7.95 7.95 0 01-4.073-1.117l-.292-.174-3.024.92.938-2.94-.19-.302A7.96 7.96 0 014 12c0-4.411 3.589-8 8-8s8 3.589 8 8-3.589 8-8 8z"/></svg>
      <div id="aiw-badge"></div>
    </button>

    <div id="aiw-panel" role="dialog" aria-label="${escapeHtml(TITLE)}">
      <div id="aiw-header">
        <div id="aiw-hav">
          <svg viewBox="0 0 24 24"><path d="M12 2C6.477 2 2 6.477 2 12c0 1.89.525 3.66 1.438 5.168L2.05 21.5l4.518-1.374A9.953 9.953 0 0012 22c5.523 0 10-4.477 10-10S17.523 2 12 2z"/></svg>
        </div>
        <div id="aiw-hinfo">
          <div id="aiw-htitle">${escapeHtml(TITLE)}</div>
          <div id="aiw-hstatus"><span id="aiw-dot"></span> Online</div>
        </div>
        <button id="aiw-xbtn" aria-label="Close">✕</button>
      </div>

      <div id="aiw-msgs"></div>

      <div id="aiw-footer">
        <div id="aiw-irow">
          <textarea id="aiw-input" rows="1" placeholder="${escapeHtml(PLACEHOLDER)}" aria-label="Message"></textarea>
          <button id="aiw-send" aria-label="Send">
            <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
          </button>
        </div>
        <div id="aiw-powered">Powered by AI Widget</div>
      </div>
    </div>
  `;
  document.body.appendChild(wrap);

  // ── Refs ──────────────────────────────────────────────
  const bubble   = document.getElementById("aiw-bubble");
  const panel    = document.getElementById("aiw-panel");
  const xbtn     = document.getElementById("aiw-xbtn");
  const msgsEl   = document.getElementById("aiw-msgs");
  const inputEl  = document.getElementById("aiw-input");
  const sendBtn  = document.getElementById("aiw-send");
  const badge    = document.getElementById("aiw-badge");

  // ── Open / Close ──────────────────────────────────────
  function openPanel() {
    isOpen = true;
    panel.classList.add("aiw-open");
    badge.style.display = "none";
    inputEl.focus();
    if (messages.length === 0) addBot(WELCOME);
  }
  function closePanel() {
    isOpen = false;
    panel.classList.remove("aiw-open");
  }

  bubble.addEventListener("click", () => isOpen ? closePanel() : openPanel());
  xbtn.addEventListener("click", closePanel);

  // ── Add messages ──────────────────────────────────────
  function addBot(text, withConfirm) {
    messages.push({ role: "bot", text });
    const el = document.createElement("div");
    el.className = "aiw-msg aiw-bot";
    el.innerHTML = `
      <div class="aiw-bub">${formatMessage(text)}</div>
      ${withConfirm ? `
        <div class="aiw-cfm-btns">
          <button class="aiw-cfm-btn aiw-cfm-yes" data-ans="yes">Yes, please</button>
          <button class="aiw-cfm-btn aiw-cfm-no"  data-ans="no">No thanks</button>
        </div>` : ""}
      <div class="aiw-time">${getTime()}</div>`;

    if (withConfirm) {
      el.querySelectorAll(".aiw-cfm-btn").forEach(btn => {
        btn.addEventListener("click", function () {
          el.querySelectorAll(".aiw-cfm-btn").forEach(b => { b.disabled = true; });
          handleConfirm(this.getAttribute("data-ans"));
        });
      });
    }

    msgsEl.appendChild(el);
    scrollBottom();
    if (!isOpen) badge.style.display = "block";
  }

  function addUser(text) {
    messages.push({ role: "user", text });
    const el = document.createElement("div");
    el.className = "aiw-msg aiw-user";
    el.innerHTML = `
      <div class="aiw-bub">${escapeHtml(text)}</div>
      <div class="aiw-time">${getTime()}</div>`;
    msgsEl.appendChild(el);
    scrollBottom();
  }

  function showTyping() {
    const el = document.createElement("div");
    el.className = "aiw-msg aiw-bot"; el.id = "aiw-typing";
    el.innerHTML = `<div class="aiw-typing"><div class="aiw-d"></div><div class="aiw-d"></div><div class="aiw-d"></div></div>`;
    msgsEl.appendChild(el); scrollBottom();
  }
  function hideTyping() { const el = document.getElementById("aiw-typing"); if (el) el.remove(); }
  function scrollBottom() { msgsEl.scrollTop = msgsEl.scrollHeight; }

  // ── API call ──────────────────────────────────────────
  async function callAsk(query, useGeneral) {
    const res = await fetch(API_URL + "/ask", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        query,
        session_id:  SESSION_ID,
        widget_id:   WIDGET_ID,
        language:    LANGUAGE,
        use_general: useGeneral || false
      })
    });
    if (!res.ok) throw new Error("Server error " + res.status);
    return res.json();
  }

  // ── Handle yes/no confirm ─────────────────────────────
  async function handleConfirm(answer) {
    if (!pendingGeneralQuery) return;
    const query = pendingGeneralQuery;
    pendingGeneralQuery = null;

    if (answer === "yes") {
      showTyping(); setDisabled(true);
      try {
        const data = await callAsk(query, true);
        hideTyping();
        addBot(data.response || "I couldn't find an answer. Please try rephrasing.");
      } catch {
        hideTyping();
        addBot("Something went wrong. Please try again.");
      } finally { setDisabled(false); }
    } else {
      addBot("No problem! Feel free to ask me anything else.");
    }
  }

  // ── Send ──────────────────────────────────────────────
  async function send() {
    const text = inputEl.value.trim();
    if (!text || isTyping) return;

    addUser(text);
    inputEl.value = "";
    autoResize();
    updateSendBtn(); // hide button after clearing input

    isTyping = true; setDisabled(true); showTyping();

    try {
      const data = await callAsk(text, false);
      hideTyping();

      if (data.response === null || data.response === undefined) {
        pendingGeneralQuery = text;
        addBot(
          "I don't have specific information about that in my documents.\n\nWould you like me to answer from my general knowledge?",
          true
        );
      } else {
        addBot(data.response);
      }
    } catch {
      hideTyping();
      addBot("⚠️ Something went wrong. Please try again in a moment.");
    } finally {
      isTyping = false; setDisabled(false); inputEl.focus();
    }
  }

  // ── Input ─────────────────────────────────────────────
  function setDisabled(v) { inputEl.disabled = v; sendBtn.disabled = v; }

  function autoResize() {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 100) + "px";
  }

  // FIX: show/hide send button based on whether input has text
  function updateSendBtn() {
    const hasText = inputEl.value.trim().length > 0;
    if (hasText) {
      sendBtn.classList.add("aiw-send-visible");
    } else {
      sendBtn.classList.remove("aiw-send-visible");
    }
  }

  inputEl.addEventListener("input", () => { autoResize(); updateSendBtn(); });
  inputEl.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
  sendBtn.addEventListener("click", send);

  // ── Close on outside click ────────────────────────────
  document.addEventListener("click", e => {
    if (isOpen && !panel.contains(e.target) && !bubble.contains(e.target)) closePanel();
  });
  document.addEventListener("keydown", e => { if (e.key === "Escape" && isOpen) closePanel(); });

})();