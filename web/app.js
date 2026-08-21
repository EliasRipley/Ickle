const CHAT_KEY = "ickle_chat_v4";
const MAX_MSGS = 300;

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  const node = document.createElement("span");
  node.textContent = String(value ?? "");
  return node.innerHTML;
}

const statusDot = $("status-dot");
const statusLabel = $("status-label");
const modelSelect = $("model-select");
const memoryToggle = $("memory-toggle");
const webToggle = $("web-toggle");
const thinkingToggle = $("thinking-toggle");
const agentToggle = $("agent-toggle");
const codeExecToggle = $("code-exec-toggle");
const codeExecToggleLabel = $("code-exec-toggle-label");
const agentCapabilities = $("agent-capabilities");
const rawOutputToggle = $("raw-output-toggle");
const capabilitiesButton = $("capabilities-button");
const capabilitiesPanel = $("capabilities-panel");
const capabilitiesClose = $("capabilities-close");
const capabilitiesCount = $("capabilities-count");
const imageAttachBtn = $("image-attach-btn");
const imageAttachInput = $("image-attach-input");
const imageAttachPreview = $("image-attach-preview");
const imageAttachThumb = $("image-attach-thumb");
if (imageAttachThumb) {
  // Never show a broken-image icon if the preview genuinely can't render
  // for some reason -- fall back to filename-only, which is still useful.
  imageAttachThumb.addEventListener("error", () => {
    imageAttachThumb.hidden = true;
  });
  imageAttachThumb.addEventListener("load", () => {
    imageAttachThumb.hidden = false;
  });
}
const imageAttachName = $("image-attach-name");
const imageAttachRemove = $("image-attach-remove");
const chatArea = $("chat-area");
const promptInput = $("prompt-input");
const sendBtn = $("send-btn");
const sessionList = $("session-list");
const sessionListEmpty = $("session-list-empty");
const newSessionBtn = $("new-session-btn");
const statusButton = $("status-button");
const systemPanel = $("system-panel");
const systemModel = $("system-model");
const systemTraining = $("system-training");
const systemDetail = $("system-detail");
const sidebar = $("sidebar");
const sidebarToggle = $("sidebar-toggle");
const sidebarScrim = $("sidebar-scrim");

// `?control_port=` still works (desktop_app.py passes it as a fallback), but
// the primary path is auto-discovery via /api/control-port below -- so a
// plain `serve-web` browser tab gets the Manage panel too, not just the
// desktop app. See discoverControlPort(), called once from init().
let CONTROL_PORT = new URLSearchParams(window.location.search).get("control_port");

async function discoverControlPort() {
  if (CONTROL_PORT) return;
  try {
    const res = await fetch("/api/control-port");
    const body = await res.json().catch(() => ({}));
    if (body && body.control_port) CONTROL_PORT = String(body.control_port);
  } catch {
    // Chat still works without the control API; Manage just stays hidden.
  }
}

async function controlApi(path, opts = {}) {
  if (!CONTROL_PORT) throw new Error("Management features need the control API, which isn't running for this session.");
  const res = await fetch(`http://127.0.0.1:${CONTROL_PORT}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `Error ${res.status}`);
  return body;
}

let messages = [];
let modelList = [];
let sending = false;
let activeSessionId = null;
let activeStreamController = null;
let attachedImageBase64 = null;
let attachedPreviewId = null;

function clearAttachedImage() {
  attachedImageBase64 = null;
  if (attachedPreviewId) {
    const idToDelete = attachedPreviewId;
    fetch(`/api/attach-preview/${idToDelete}`, { method: "DELETE" }).catch(() => {});
    attachedPreviewId = null;
  }
  if (imageAttachInput) imageAttachInput.value = "";
  if (imageAttachPreview) imageAttachPreview.hidden = true;
  if (imageAttachThumb) imageAttachThumb.src = "";
  if (imageAttachName) imageAttachName.textContent = "";
}

if (imageAttachBtn && imageAttachInput) {
  imageAttachBtn.addEventListener("click", () => imageAttachInput.click());
  imageAttachInput.addEventListener("change", () => {
    const file = imageAttachInput.files && imageAttachInput.files[0];
    if (!file) return;

    if (imageAttachName) imageAttachName.textContent = file.name;
    if (imageAttachPreview) imageAttachPreview.hidden = false;

    const reader = new FileReader();
    reader.onload = async () => {
      const dataUrl = String(reader.result || "");
      const base64 = dataUrl.split(",")[1] || null;
      attachedImageBase64 = base64;
      if (!base64) return;

      // The thumbnail is served back through this same local HTTP server
      // (rather than a blob: or data: URI in the <img src>) because
      // pywebview's WebView2 backend has documented, unresolved quirks
      // rendering both of those directly (broken/blank image), even though
      // the exact same markup works fine in a standalone Chromium/Firefox
      // tab. A plain http:// image source -- how the app itself is already
      // loaded -- is the one thing WebView2 reliably renders.
      try {
        const res = await fetch("/api/attach-preview", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_base64: base64, content_type: file.type || "image/jpeg" }),
        });
        const body = await res.json().catch(() => ({}));
        if (res.ok && body.url && imageAttachThumb) {
          attachedPreviewId = body.id || null;
          imageAttachThumb.src = body.url;
        }
      } catch {
        // No thumbnail if the preview upload fails -- filename-only is still useful.
      }
    };
    reader.readAsDataURL(file);
  });
}
if (imageAttachRemove) {
  imageAttachRemove.addEventListener("click", clearAttachedImage);
}

async function api(url, opts = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `Error ${res.status}`);
  return body;
}

function loadMessagesLocal() {
  try {
    const raw = localStorage.getItem(CHAT_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveMessagesLocal(list) {
  const trimmed = (list || messages).slice(-MAX_MSGS);
  try { localStorage.setItem(CHAT_KEY, JSON.stringify(trimmed)); } catch {}
}

function timeAgo(iso) {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    const now = new Date();
    const min = Math.floor((now - d) / 60000);
    if (min < 1) return "just now";
    if (min < 60) return `${min}m ago`;
    const h = Math.floor(min / 60);
    if (h < 24) return `${h}h ago`;
    return `${Math.floor(h / 24)}d ago`;
  } catch {
    return "";
  }
}

// Small, dependency-free renderer for fenced code blocks (```lang ... ```)
// and inline code (`code`) -- not full markdown, kept minimal on purpose to
// avoid XSS risk from a broader parser surface. Builds DOM nodes explicitly;
// never assigns raw model output via innerHTML.
function renderMessageText(container, text) {
  const str = String(text || "");
  const fenceRe = /```([a-zA-Z0-9_+-]*)\n?([\s\S]*?)```/g;
  let lastIndex = 0;
  let match;
  let sawFence = false;

  const appendInline = (parent, chunk) => {
    const inlineRe = /`([^`\n]+)`/g;
    let last = 0;
    let m;
    let any = false;
    while ((m = inlineRe.exec(chunk)) !== null) {
      any = true;
      if (m.index > last) parent.appendChild(document.createTextNode(chunk.slice(last, m.index)));
      const code = document.createElement("code");
      code.className = "inline-code";
      code.textContent = m[1];
      parent.appendChild(code);
      last = inlineRe.lastIndex;
    }
    if (!any) {
      parent.appendChild(document.createTextNode(chunk));
    } else if (last < chunk.length) {
      parent.appendChild(document.createTextNode(chunk.slice(last)));
    }
  };

  while ((match = fenceRe.exec(str)) !== null) {
    sawFence = true;
    if (match.index > lastIndex) {
      const before = document.createElement("span");
      appendInline(before, str.slice(lastIndex, match.index));
      container.appendChild(before);
    }

    const lang = match[1] || "";
    const code = match[2].replace(/\n$/, "");

    const block = document.createElement("div");
    block.className = "code-block";

    const header = document.createElement("div");
    header.className = "code-block-header";

    const langLabel = document.createElement("span");
    langLabel.className = "code-block-lang";
    langLabel.textContent = lang || "code";
    header.appendChild(langLabel);

    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "code-block-copy";
    copyBtn.textContent = "Copy";
    copyBtn.onclick = () => {
      navigator.clipboard.writeText(code).then(
        () => {
          copyBtn.textContent = "Copied";
          setTimeout(() => (copyBtn.textContent = "Copy"), 1500);
        },
        () => {
          copyBtn.textContent = "Failed";
          setTimeout(() => (copyBtn.textContent = "Copy"), 1500);
        }
      );
    };
    header.appendChild(copyBtn);

    const pre = document.createElement("pre");
    const codeEl = document.createElement("code");
    codeEl.textContent = code;
    pre.appendChild(codeEl);

    block.appendChild(header);
    block.appendChild(pre);
    container.appendChild(block);

    lastIndex = fenceRe.lastIndex;
  }

  if (!sawFence) {
    appendInline(container, str);
    return;
  }

  if (lastIndex < str.length) {
    const after = document.createElement("span");
    appendInline(after, str.slice(lastIndex));
    container.appendChild(after);
  }
}

function createThinkingBlock(reasoning) {
  const block = document.createElement("div");
  block.className = "thinking-block";

  const toggle = document.createElement("button");
  toggle.className = "thinking-toggle";
  toggle.textContent = "Show Ickle's thinking";
  toggle.title = "Ickle's step-by-step reasoning before its final answer. This is exploratory and may not always be accurate.";
  toggle.onclick = () => {
    const hidden = content.hidden;
    content.hidden = !hidden;
    toggle.textContent = hidden ? "Hide Ickle's thinking" : "Show Ickle's thinking";
  };

  const content = document.createElement("div");
  content.className = "thinking-content";
  content.hidden = true;
  content.textContent = reasoning;

  block.appendChild(toggle);
  block.appendChild(content);
  return block;
}

const EPISTEMIC_STATUS_LABELS = {
  corrected: "Human correction",
  contested: "Contested",
  human_reviewed: "Human reviewed",
  peer_perspective: "Peer perspective",
  source_linked: "Related sources",
  advice: "Advice / judgement",
  open: "Open claim",
};

async function saveClaimReview(claim, relation, correctionText, sourceUrl, shared) {
  const result = await api("/api/epistemics/reviews", {
    method: "POST",
    body: JSON.stringify({
      claim_text: claim.text,
      relation,
      correction_text: correctionText || "",
      source_url: sourceUrl || "",
      shared: Boolean(shared),
    }),
  });
  if (shared && CONTROL_PORT) {
    // Saving the signed review succeeds independently of the network. If the
    // node is offline, it remains queued locally for the next explicit sync.
    try {
      await controlApi("/api/commons/sync", { method: "POST", body: JSON.stringify({}) });
    } catch {}
  }
  return result;
}

function createEpistemicBlock(passport) {
  const claims = Array.isArray(passport?.claims) ? passport.claims : [];
  if (!claims.length) return null;

  const block = document.createElement("div");
  block.className = "epistemic-block";
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "epistemic-toggle";
  toggle.textContent = `Inspect ${claims.length} candidate claim${claims.length === 1 ? "" : "s"}`;
  toggle.title = "See which parts of this answer have related sources, human review, or unresolved uncertainty.";

  const content = document.createElement("div");
  content.className = "epistemic-content";
  content.hidden = true;
  toggle.addEventListener("click", () => {
    content.hidden = !content.hidden;
    toggle.textContent = content.hidden
      ? `Inspect ${claims.length} candidate claim${claims.length === 1 ? "" : "s"}`
      : "Hide answer map";
  });

  const intro = document.createElement("p");
  intro.className = "epistemic-caveat";
  intro.textContent = passport.caveat || "This map shows support and disagreement; it is not a truth certificate.";
  content.appendChild(intro);

  claims.forEach((claim) => {
    const card = document.createElement("div");
    card.className = `epistemic-claim epistemic-${claim.status || "open"}`;
    const heading = document.createElement("div");
    heading.className = "epistemic-claim-heading";
    const badge = document.createElement("span");
    badge.className = "epistemic-status";
    badge.textContent = EPISTEMIC_STATUS_LABELS[claim.status] || "Candidate claim";
    const claimText = document.createElement("span");
    claimText.className = "epistemic-claim-text";
    claimText.textContent = claim.text || "";
    heading.appendChild(badge);
    heading.appendChild(claimText);
    card.appendChild(heading);

    (claim.basis || []).forEach((basis) => {
      const line = document.createElement("p");
      line.className = "epistemic-basis";
      line.textContent = basis;
      card.appendChild(line);
    });

    const corrections = claim.reviews?.corrections || [];
    corrections.forEach((correction) => {
      const line = document.createElement("p");
      line.className = "epistemic-correction";
      line.textContent = `${correction.is_local ? "Correction" : "Peer suggestion"}: ${correction.text}`;
      card.appendChild(line);
    });

    const sources = Array.isArray(claim.sources) ? claim.sources : [];
    if (sources.length) {
      const sourceList = document.createElement("div");
      sourceList.className = "epistemic-sources";
      sources.forEach((source) => {
        const url = String(source.source_url || "");
        if (!/^https?:\/\//i.test(url)) return;
        const link = document.createElement("a");
        link.href = url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = source.source_title || new URL(url).hostname;
        link.title = "Related retrieved source; the link does not by itself prove the claim.";
        sourceList.appendChild(link);
      });
      if (sourceList.childNodes.length) card.appendChild(sourceList);
    }

    const actions = document.createElement("div");
    actions.className = "epistemic-actions";
    const supportBtn = document.createElement("button");
    supportBtn.type = "button";
    supportBtn.textContent = "Looks right";
    supportBtn.title = "Record your review on this device. This is human feedback, not automatic proof.";
    const challengeBtn = document.createElement("button");
    challengeBtn.type = "button";
    challengeBtn.textContent = "Challenge / correct";
    actions.appendChild(supportBtn);
    actions.appendChild(challengeBtn);
    card.appendChild(actions);

    const status = document.createElement("span");
    status.className = "epistemic-save-status";
    actions.appendChild(status);

    supportBtn.addEventListener("click", async () => {
      supportBtn.disabled = true;
      challengeBtn.disabled = true;
      status.textContent = "Saving...";
      try {
        await saveClaimReview(claim, "support", "", "", false);
        badge.textContent = EPISTEMIC_STATUS_LABELS.human_reviewed;
        card.className = "epistemic-claim epistemic-human_reviewed";
        status.textContent = "Saved locally.";
      } catch (err) {
        status.textContent = err.message || "Couldn't save review.";
        supportBtn.disabled = false;
        challengeBtn.disabled = false;
      }
    });

    challengeBtn.addEventListener("click", () => {
      supportBtn.hidden = true;
      challengeBtn.hidden = true;
      const form = document.createElement("form");
      form.className = "epistemic-review-form";
      const correction = document.createElement("textarea");
      correction.rows = 2;
      correction.placeholder = "What should Ickle use instead? Leave blank to mark the claim as disputed.";
      const source = document.createElement("input");
      source.type = "url";
      source.placeholder = "Supporting source URL (optional)";
      const shareLabel = document.createElement("label");
      shareLabel.className = "epistemic-share-label";
      const share = document.createElement("input");
      share.type = "checkbox";
      shareLabel.appendChild(share);
      shareLabel.appendChild(document.createTextNode(" Share this signed review with configured peers"));
      const privacy = document.createElement("small");
      privacy.textContent = "Off by default. Once another peer receives a shared event, you cannot delete their copy.";
      const buttons = document.createElement("div");
      buttons.className = "epistemic-form-buttons";
      const save = document.createElement("button");
      save.type = "submit";
      save.textContent = "Save review";
      const cancel = document.createElement("button");
      cancel.type = "button";
      cancel.textContent = "Cancel";
      cancel.addEventListener("click", () => {
        form.remove();
        supportBtn.hidden = false;
        challengeBtn.hidden = false;
      });
      buttons.appendChild(save);
      buttons.appendChild(cancel);
      form.appendChild(correction);
      form.appendChild(source);
      form.appendChild(shareLabel);
      form.appendChild(privacy);
      form.appendChild(buttons);
      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        save.disabled = true;
        cancel.disabled = true;
        status.textContent = "Saving...";
        try {
          const correctionText = correction.value.trim();
          await saveClaimReview(
            claim, correctionText ? "correct" : "dispute", correctionText, source.value.trim(), share.checked
          );
          badge.textContent = correctionText
            ? EPISTEMIC_STATUS_LABELS.corrected
            : EPISTEMIC_STATUS_LABELS.contested;
          card.className = `epistemic-claim epistemic-${correctionText ? "corrected" : "contested"}`;
          status.textContent = share.checked ? "Saved and marked for peer sharing." : "Saved locally.";
          form.remove();
        } catch (err) {
          status.textContent = err.message || "Couldn't save review.";
          save.disabled = false;
          cancel.disabled = false;
        }
      });
      card.appendChild(form);
      correction.focus();
    });

    content.appendChild(card);
  });

  block.appendChild(toggle);
  block.appendChild(content);
  return block;
}

function render() {
  chatArea.innerHTML = "";

  if (!messages.length) {
    const empty = document.createElement("div");
    empty.className = "chat-empty-state";
    empty.innerHTML = `
      <div class="logo">Ickle</div>
      <p>Your local AI assistant</p>
      <p class="sub">Everything runs on this device. Type a message below to get started.</p>
    `;
    chatArea.appendChild(empty);
    return;
  }

  messages.forEach((m, idx) => {
    const row = document.createElement("div");
    row.className = `msg-row ${m.role}`;

    const avatar = document.createElement("div");
    avatar.className = `msg-avatar ${m.role}`;
    avatar.textContent = m.role === "user" ? "U" : "I";

    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";

    if (m.role === "assistant" && m.error) {
      bubble.className += " msg-error";
    }

    if (m.role === "assistant" && m.thinking) {
      bubble.appendChild(createThinkingBlock(m.thinking));
    }

    if (m.role === "assistant" && m.empty) {
      const emptyDiv = document.createElement("div");
      emptyDiv.className = "msg-empty-state";
      emptyDiv.textContent = "Ickle generated no output for this prompt.";
      bubble.appendChild(emptyDiv);
    } else {
      const textDiv = document.createElement("div");
      textDiv.className = "msg-text";
      renderMessageText(textDiv, m.text);
      bubble.appendChild(textDiv);
    }

    if (m.role === "assistant" && m.lowConfidence && !m.error && !m.empty) {
      const badge = document.createElement("span");
      badge.className = "msg-confidence-badge";
      badge.title = "Ickle's own quality/relevance check on this answer looked weak, but this is its real, unedited output.";
      badge.textContent = "Low confidence";
      bubble.appendChild(badge);
    }

    if (m.role === "assistant" && m.epistemics && !m.error && !m.empty) {
      const answerMap = createEpistemicBlock(m.epistemics);
      if (answerMap) bubble.appendChild(answerMap);
    }

    if (m.model) {
      const meta = document.createElement("div");
      meta.className = "msg-meta";
      meta.textContent = `${m.model} · ${timeAgo(m.at)}`;
      bubble.appendChild(meta);
    }

    if (m.role === "assistant" && !m.error && m.text) {
      const promptMsg = messages[idx - 1];
      if (promptMsg && promptMsg.role === "user") {
        bubble.appendChild(createFeedbackRow(promptMsg.text, m.text));
        bubble.appendChild(createSwarmAskRow(promptMsg.text));
      }
    }

    row.appendChild(avatar);
    row.appendChild(bubble);
    chatArea.appendChild(row);
  });

  chatArea.scrollTop = chatArea.scrollHeight;
}

function createFeedbackRow(prompt, response) {
  const wrap = document.createElement("div");
  wrap.className = "msg-feedback";

  const upBtn = document.createElement("button");
  upBtn.type = "button";
  upBtn.className = "msg-feedback-btn";
  upBtn.title = "Good response — use this to help Ickle learn";
  upBtn.setAttribute("aria-label", "Good response");
  upBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M7 22V11M2 13v7a2 2 0 0 0 2 2h12.9a2 2 0 0 0 2-1.7l1.4-9A2 2 0 0 0 18.3 9H14V5a3 3 0 0 0-3-3l-4 9v11"/></svg>';

  const downBtn = document.createElement("button");
  downBtn.type = "button";
  downBtn.className = "msg-feedback-btn";
  downBtn.title = "Not helpful";
  downBtn.setAttribute("aria-label", "Not helpful");
  downBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 2v11M22 11V4a2 2 0 0 0-2-2H7.1a2 2 0 0 0-2 1.7l-1.4 9A2 2 0 0 0 5.7 15H10v4a3 3 0 0 0 3 3l4-9V2"/></svg>';

  const sendFeedback = async (rating) => {
    upBtn.disabled = true;
    downBtn.disabled = true;
    try {
      await api("/api/feedback", {
        method: "POST",
        body: JSON.stringify({ prompt, response, rating }),
      });
      wrap.textContent = "Thanks — that helps Ickle learn.";
    } catch {
      wrap.textContent = "Couldn't save feedback.";
    }
  };

  upBtn.addEventListener("click", () => sendFeedback(5));
  downBtn.addEventListener("click", () => sendFeedback(2));
  wrap.appendChild(upBtn);
  wrap.appendChild(downBtn);
  return wrap;
}

function createSwarmAskRow(prompt) {
  const wrap = document.createElement("div");
  wrap.className = "msg-swarm-ask";

  const askBtn = document.createElement("button");
  askBtn.type = "button";
  askBtn.className = "msg-swarm-ask-btn";
  askBtn.textContent = "Ask the swarm too";
  askBtn.title = "Send this question to peers on the network and see how their answers compare";

  const resultsEl = document.createElement("div");
  resultsEl.className = "msg-swarm-ask-results";
  resultsEl.hidden = true;

  askBtn.addEventListener("click", async () => {
    askBtn.disabled = true;
    askBtn.textContent = "Asking peers...";
    resultsEl.hidden = false;
    resultsEl.textContent = "Waiting on independent perspectives...";
    try {
      const data = await controlApi("/api/swarm/ask", {
        method: "POST",
        body: JSON.stringify({ prompt }),
      });
      resultsEl.innerHTML = "";
      if (!data.peers_asked) {
        resultsEl.textContent = "No peers reachable right now -- add some in the Network tab.";
      } else if (!data.peers_answered) {
        resultsEl.textContent = `Asked ${data.peers_asked} peer(s), but none answered in time.`;
      } else {
        const summary = document.createElement("p");
        summary.className = "hint-text";
        const domainNote = data.domain && data.domain !== "general" ? ` (routed as "${data.domain}")` : "";
        const collective = data.deliberation?.summary || {};
        summary.textContent =
          `${data.peers_answered}/${data.peers_asked} peer(s) answered${domainNote}. ` +
          `${collective.common_claims || 0} repeated claim(s), ${collective.distinct_claims || 0} distinct contribution(s). ` +
          "Agreement is visible, but is not treated as proof.";
        resultsEl.appendChild(summary);
        const commonGround = data.deliberation?.common_ground || [];
        if (commonGround.length) {
          const common = document.createElement("div");
          common.className = "swarm-common-ground";
          const title = document.createElement("strong");
          title.textContent = "Common ground";
          common.appendChild(title);
          commonGround.slice(0, 5).forEach((claim) => {
            const line = document.createElement("p");
            line.textContent = `${claim.peer_count} peers: ${claim.representative}`;
            common.appendChild(line);
          });
          resultsEl.appendChild(common);
        }
        (data.responses || []).forEach((r) => {
          const row = document.createElement("div");
          row.className = "msg-swarm-ask-response";
          const meta = document.createElement("div");
          meta.className = "msg-meta";
          meta.textContent =
            `${r.peer_id.slice(0, 12)}... · answer overlap ${r.consensus_score.toFixed(2)} · ` +
            `your trust ${r.trust_score.toFixed(2)}`;
          const text = document.createElement("div");
          text.className = "msg-text";
          renderMessageText(text, r.response);
          const review = document.createElement("div");
          review.className = "swarm-human-review";
          const helpful = document.createElement("button");
          helpful.type = "button";
          helpful.textContent = "Helpful";
          const unhelpful = document.createElement("button");
          unhelpful.type = "button";
          unhelpful.textContent = "Not helpful";
          const reviewStatus = document.createElement("span");
          const sendPeerReview = async (value) => {
            helpful.disabled = true;
            unhelpful.disabled = true;
            try {
              const saved = await controlApi("/api/swarm/feedback", {
                method: "POST",
                body: JSON.stringify({ peer_id: r.peer_id, prompt, helpful: value }),
              });
              reviewStatus.textContent = `Trust updated to ${Number(saved.trust).toFixed(2)} for ${saved.domain}.`;
            } catch (err) {
              reviewStatus.textContent = err.message || "Couldn't save peer review.";
              helpful.disabled = false;
              unhelpful.disabled = false;
            }
          };
          helpful.addEventListener("click", () => sendPeerReview(true));
          unhelpful.addEventListener("click", () => sendPeerReview(false));
          review.appendChild(helpful);
          review.appendChild(unhelpful);
          review.appendChild(reviewStatus);
          row.appendChild(meta);
          row.appendChild(text);
          row.appendChild(review);
          resultsEl.appendChild(row);
        });
      }
    } catch (err) {
      resultsEl.textContent = err.message || "Couldn't reach the swarm.";
    } finally {
      askBtn.textContent = "Ask the swarm too";
      askBtn.disabled = false;
    }
  });

  wrap.appendChild(askBtn);
  wrap.appendChild(resultsEl);
  return wrap;
}

async function saveMessageToSession(
  role, text, model, thinking, sessionId = activeSessionId, epistemics = null, lowConfidence = false
) {
  if (!sessionId) return;
  try {
    await api(`/api/sessions/${sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({
        role,
        text,
        thinking: thinking || "",
        model: model || "",
        epistemics: epistemics || undefined,
        low_confidence: Boolean(lowConfidence),
      }),
    });
  } catch {}
}

function addMessage(role, text, model, thinking, epistemics = null) {
  messages.push({
    role,
    text: String(text || ""),
    thinking: String(thinking || ""),
    at: new Date().toISOString(),
    model: model || "",
    epistemics: epistemics || null,
  });
  saveMessagesLocal();
  saveMessageToSession(role, text, model, thinking, activeSessionId, epistemics);
  render();
}

function setStatus(online) {
  if (online) {
    statusDot.className = "dot dot-online";
    statusLabel.textContent = "local";
  } else {
    statusDot.className = "dot dot-offline";
    statusLabel.textContent = "offline";
  }
}

function formatTrainingStatus(training) {
  const status = String(training?.status || "unavailable");
  if (status === "running") {
    const step = Number(training.step || 0);
    const total = Number(training.total_steps || 0);
    return total > 0 ? `Running · step ${step}/${total}` : "Running";
  }
  if (status === "stale") return "Stopped unexpectedly";
  if (status === "completed") return "Last run completed";
  if (status === "failed") return "Last run failed";
  if (status === "interrupted") return "Last run stopped";
  return "Idle";
}

async function refreshStatus() {
  try {
    const data = await api("/api/status");
    setStatus(true);
    systemModel.textContent = data.model?.name || data.chat_model || "No model";
    systemTraining.textContent = formatTrainingStatus(data.training);
    if (data.training?.status === "stale") {
      systemDetail.textContent = `The old “running” record is stale (${data.training.stale_reason || "no heartbeat"}); no trainer is active.`;
    } else {
      systemDetail.textContent = "Ickle runs locally on this device. Memory and web access remain under your control.";
    }
  } catch {
    setStatus(false);
    systemModel.textContent = "Unavailable";
    systemTraining.textContent = "Unavailable";
  }
}

async function refreshModels() {
  try {
    const data = await api("/api/models?limit=300&all=1");
    modelList = (data.models || [])
      .filter((m) => !String(m.name || "").endsWith(".checkpoint.pt"))
      .sort((a, b) => Number(b.updated_at || 0) - Number(a.updated_at || 0))
      .slice(0, 20);

    const current = modelSelect.value;
    modelSelect.innerHTML = "";

    if (!modelList.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "No trained model yet";
      modelSelect.appendChild(opt);
      systemDetail.textContent = "No trained model was found yet. Train or import one, then refresh this page.";
      return;
    }

    modelList.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.path || "";
      const mb = Math.round((m.size_bytes || 0) / 1048576 * 10) / 10;
      opt.textContent = `${m.name} (${mb} MB)`;
      modelSelect.appendChild(opt);
    });

    if (current && modelList.some((m) => m.path === current)) {
      modelSelect.value = current;
    } else if (modelList[0]) {
      modelSelect.value = modelList[0].path;
    }

  } catch {
    setStatus(false);
  }
}

function createStreamingRow() {
  let msgDiv = document.getElementById("streaming-row");
  if (msgDiv) return msgDiv;
  msgDiv = document.createElement("div");
  msgDiv.id = "streaming-row";
  msgDiv.className = "msg-row assistant";
  const avatar = document.createElement("div");
  avatar.className = "msg-avatar assistant";
  avatar.textContent = "I";
  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  const textDiv = document.createElement("div");
  textDiv.className = "msg-text";
  textDiv.id = "streaming-response";
  bubble.appendChild(textDiv);
  msgDiv.appendChild(avatar);
  msgDiv.appendChild(bubble);
  chatArea.appendChild(msgDiv);
  return msgDiv;
}

function setSending(value) {
  sending = value;
  sendBtn.classList.toggle("sending", value);
  sendBtn.title = value ? "Stop generating" : "Send message";
}

function cancelStreaming() {
  if (activeStreamController) activeStreamController.abort();
}

async function sendStreaming(text, imageBase64 = null) {
  let thinkingContent = "";
  let responseContent = "";
  let lowConfidence = false;
  let epistemics = null;
  let finished = false;
  const myController = new AbortController();
  activeStreamController = myController;
  // Captured once, up front: the session this stream's response belongs to.
  // activeSessionId is a shared global that switchSession() can reassign
  // while this stream is still in flight -- every mutation below must key
  // off streamSessionId, not the (possibly now-different) live global, or
  // a slow response lands in whatever session the user has since switched
  // to instead of the one that actually asked the question.
  const streamSessionId = activeSessionId;
  const stillOnThisSession = () => activeSessionId === streamSessionId;
  // Guards against a slow/stale stream's finish() clobbering a newer
  // stream's controller/sending indicator after the user switched sessions
  // and immediately sent another message.
  const stillTheActiveStream = () => activeStreamController === myController;

  const finish = (errorText = "") => {
    if (finished) return;
    finished = true;
    if (responseContent) {
      const model = modelSelect.value || "";
      if (stillOnThisSession()) {
        messages.push({
          role: "assistant",
          text: responseContent,
          thinking: thinkingContent,
          at: new Date().toISOString(),
          model: model,
          lowConfidence: lowConfidence,
          epistemics: epistemics,
        });
        saveMessagesLocal();
      }
      saveMessageToSession(
        "assistant", responseContent, model, thinkingContent, streamSessionId, epistemics, lowConfidence
      );
    } else if (errorText) {
      if (stillOnThisSession()) {
        messages.push({
          role: "assistant",
          text: errorText,
          error: true,
          at: new Date().toISOString(),
          model: "",
        });
        saveMessagesLocal();
      }
    } else if (stillOnThisSession()) {
      // The model produced zero tokens -- show that honestly instead of
      // silence or composed placeholder text standing in for a real answer.
      messages.push({
        role: "assistant",
        text: "",
        empty: true,
        at: new Date().toISOString(),
        model: modelSelect.value || "",
      });
      saveMessagesLocal();
    }
    if (stillTheActiveStream()) {
      activeStreamController = null;
      setSending(false);
    }
    if (stillOnThisSession()) {
      render();
      promptInput.focus();
    }
  };

  const handleEvent = (eventName, dataText) => {
    let data = {};
    try { data = dataText ? JSON.parse(dataText) : {}; } catch {}
    // Track thinking/response content regardless (finish() needs the full
    // text to persist to the right session), but skip DOM writes once the
    // user has navigated away from this stream's session -- the streaming
    // row it was updating belongs to a screen that's no longer shown.
    const updateDom = stillOnThisSession();
    if (eventName === "reasoning_start") {
      thinkingContent = "";
      if (updateDom) createStreamingRow();
    } else if (eventName === "reasoning") {
      thinkingContent += data.text || "";
    } else if (eventName === "reasoning_end") {
      thinkingContent = data.text || thinkingContent;
      if (updateDom) {
        const row = createStreamingRow();
        const bubble = row.querySelector(".msg-bubble");
        if (thinkingContent && bubble && !bubble.querySelector(".thinking-block")) {
          bubble.prepend(createThinkingBlock(thinkingContent));
        }
      }
    } else if (eventName === "text") {
      responseContent += data.text || "";
      if (updateDom) {
        createStreamingRow();
        const el = document.getElementById("streaming-response");
        if (el) el.textContent = responseContent;
        chatArea.scrollTop = chatArea.scrollHeight;
      }
    } else if (eventName === "stream_error" || eventName === "error") {
      finish(data.text || "Ickle could not complete that response.");
    } else if (eventName === "done") {
      lowConfidence = Boolean(data.low_confidence);
      epistemics = data.epistemics || null;
      finish();
    }
  };

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: text,
        model: modelSelect.value || "",
        thinking_mode: thinkingToggle.checked,
        enable_memory: memoryToggle.checked,
        enable_web_tools: webToggle.checked,
        agent: agentToggle ? agentToggle.checked : false,
        allow_code_execution: agentToggle && codeExecToggle ? agentToggle.checked && codeExecToggle.checked : false,
        raw_output: rawOutputToggle ? rawOutputToggle.checked : false,
        session_id: streamSessionId || "",
        image_base64: imageBase64 || undefined,
      }),
      signal: myController.signal,
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.error || `Request failed (${response.status})`);
    }
    if (!response.body) throw new Error("Streaming is not supported by this browser.");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const blocks = buffer.split(/\r?\n\r?\n/);
      buffer = blocks.pop() || "";
      for (const block of blocks) {
        let eventName = "message";
        const dataLines = [];
        for (const line of block.split(/\r?\n/)) {
          if (line.startsWith("event:")) eventName = line.slice(6).trim();
          if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
        }
        handleEvent(eventName, dataLines.join("\n"));
      }
      if (done) break;
    }
    finish();
  } catch (error) {
    if (error?.name === "AbortError") {
      finish(responseContent ? "" : "Generation stopped.");
    } else {
      finish(error?.message || "Connection error during streaming.");
    }
  }
}

async function sendMessage() {
  if (sending) {
    cancelStreaming();
    return;
  }
  const text = promptInput.value.trim();
  if (!text) return;

  if (window.location.protocol === "file:") {
    addMessage("user", text);
    addMessage(
      "assistant",
      "Ickle isn't running yet, so this page can't send messages. Start Ickle on this computer, then open http://127.0.0.1:8787 in your browser.",
      "",
    );
    return;
  }

  if (!activeSessionId) {
    await ensureSession();
  }

  setSending(true);
  promptInput.value = "";
  promptInput.style.height = "auto";
  addMessage("user", text);

  const imageToSend = attachedImageBase64;
  clearAttachedImage();
  sendStreaming(text, imageToSend);
}

async function refreshSessions() {
  try {
    const data = await api("/api/sessions");
    const sessions = data.sessions || [];
    renderSessionList(sessions);
    return sessions;
  } catch {}
  return [];
}

function renderSessionList(sessions) {
  sessionList.innerHTML = "";
  if (sessionListEmpty) sessionListEmpty.hidden = sessions.length > 0;

  sessions.forEach((s) => {
    const el = document.createElement("div");
    el.className = "session-item" + (s.id === activeSessionId ? " active" : "");
    el.title = s.title || "Untitled";

    const title = document.createElement("span");
    title.className = "session-title";
    title.textContent = s.title || "Untitled";
    el.appendChild(title);

    const meta = document.createElement("span");
    meta.className = "session-meta";
    meta.textContent = `${s.message_count || 0} msgs`;
    el.appendChild(meta);

    const del = document.createElement("button");
    del.className = "session-delete";
    del.textContent = "×";
    del.title = "Delete this chat";
    del.setAttribute("aria-label", `Delete chat: ${s.title || "Untitled"}`);
    del.onclick = async (e) => {
      e.stopPropagation();
      if (!confirm("Delete this chat?")) return;
      try {
        await api(`/api/sessions/${s.id}`, { method: "DELETE" });
      } catch {
        try {
          await api(`/api/sessions/${s.id}/delete`, { method: "POST" });
        } catch {}
      }
      if (s.id === activeSessionId) {
        messages = [];
        activeSessionId = null;
        saveMessagesLocal();
      }
      await refreshSessions();
      if (!activeSessionId) render();
    };
    el.appendChild(del);

    el.onclick = () => { switchSession(s.id); closeSidebar(); };
    sessionList.appendChild(el);
  });
}

async function switchSession(sessionId) {
  if (sessionId === activeSessionId) return;
  // Stop any in-flight response for the session we're leaving -- otherwise
  // its reply can still arrive after the switch and (without the
  // streamSessionId guards in sendStreaming) would land in the session
  // we're switching to instead of the one that asked for it. Reset the
  // sending indicator synchronously too, rather than waiting for the
  // aborted fetch's rejection to reach sendStreaming's finish() -- that
  // resolves on its own delay, and until it does, sendMessage() reads
  // `sending` as still true and treats the next click as "cancel" instead
  // of actually sending the new message in the session just switched to.
  if (sending) {
    cancelStreaming();
    activeStreamController = null;
    setSending(false);
  }
  activeSessionId = sessionId;
  try {
    const data = await api(`/api/sessions/${sessionId}/messages`);
    messages = (data.messages || []).map((m) => ({
      role: m.role || "user",
      text: m.text || "",
      thinking: m.thinking || "",
      at: m.at || "",
      model: m.model || "",
    }));
    saveMessagesLocal();
  } catch {
    messages = [];
  }
  render();
  await refreshSessions();
}

async function ensureSession() {
  if (activeSessionId) return;
  try {
    const data = await api("/api/sessions", { method: "POST", body: JSON.stringify({}) });
    activeSessionId = data.id;
    await refreshSessions();
  } catch {}
}

async function newSession() {
  // Same in-flight-stream hazard as switchSession(): abort and reset the
  // sending indicator synchronously so it doesn't leak into the new session
  // and doesn't leave sendMessage() reading a stale `sending = true`.
  if (sending) {
    cancelStreaming();
    activeStreamController = null;
    setSending(false);
  }
  messages = [];
  activeSessionId = null;
  saveMessagesLocal();
  render();
  await ensureSession();
  await refreshSessions();
}

function openSidebar() {
  sidebar.classList.add("open");
  if (sidebarScrim) sidebarScrim.hidden = false;
  if (sidebarToggle) sidebarToggle.setAttribute("aria-expanded", "true");
}

function closeSidebar() {
  sidebar.classList.remove("open");
  if (sidebarScrim) sidebarScrim.hidden = true;
  if (sidebarToggle) sidebarToggle.setAttribute("aria-expanded", "false");
}

if (sidebarToggle) {
  sidebarToggle.addEventListener("click", () => {
    if (sidebar.classList.contains("open")) closeSidebar();
    else openSidebar();
  });
}
if (sidebarScrim) sidebarScrim.addEventListener("click", closeSidebar);

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && sidebar.classList.contains("open")) closeSidebar();
});

newSessionBtn.addEventListener("click", () => {
  newSession();
  closeSidebar();
});

promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

function autoResize() {
  promptInput.style.height = "auto";
  promptInput.style.height = Math.min(promptInput.scrollHeight, 160) + "px";
}

promptInput.addEventListener("input", autoResize);

sendBtn.addEventListener("click", sendMessage);

statusButton.addEventListener("click", () => {
  const open = systemPanel.hidden;
  systemPanel.hidden = !open;
  statusButton.setAttribute("aria-expanded", String(open));
});

modelSelect.addEventListener("change", () => {
  try {
    const val = modelSelect.value;
    api("/api/flags", {
      method: "POST",
      body: JSON.stringify({ current_model: val }),
    }).catch(() => {});
  } catch {}
});

// ---------------------------------------------------------------------------
// Train / background-tasks management panel. serve-web starts its own
// embedded control server by default, discovered via /api/control-port in
// init() (see discoverControlPort()) -- so a plain browser tab gets this too,
// not just the desktop app. Only stays hidden if the control API was
// explicitly disabled with `serve-web --no-control`.

const manageButton = $("manage-button");
const manageModal = $("manage-modal");
const manageScrim = $("manage-scrim");
const manageClose = $("manage-close");
const trainActive = $("train-active");
const trainActiveSummary = $("train-active-summary");
const trainActiveDetail = $("train-active-detail");
const trainProgressBar = $("train-progress-bar");
const trainStopBtn = $("train-stop-btn");
const trainForm = $("train-form");
const trainError = $("train-error");
const trainCustomPath = $("train-custom-path");
const trainHfDataset = $("train-hf-dataset");
const trainHfDatasetConfig = $("train-hf-dataset-config");
const trainHfDatasetField = $("train-hf-dataset-field");
const trainNameField = $("train-name-field");
const trainSizeField = $("train-size-field");
const tasksList = $("tasks-list");
const tasksEmpty = $("tasks-empty");

const TASK_TYPE_LABELS = {
  train_model: "Training a model",
  lora_train: "Training a topic add-on",
  continual_guard_step: "Reviewing recent training for regressions",
  evaluate_model: "Checking model quality",
  build_teacher_corpus: "Preparing lesson material",
  build_teacher_preferences: "Preparing preference examples",
  generate_teacher_data: "Asking a teacher AI for training examples",
  train_from_teacher: "Learning from a teacher session",
  learn_wikipedia_topic: "Learning a topic from Wikipedia",
  learn_web_topic: "Learning a topic from the web",
  build_dpo_preferences: "Building preference pairs from your ratings",
  dpo_train: "Aligning the model to your ratings",
};

const TASK_STATUS_LABELS = {
  queued: "Waiting to start",
  running: "In progress",
  completed: "Done",
  failed: "Didn't finish",
  cancelled: "Stopped",
};

let activeTrainTaskId = null;

function friendlyTaskType(taskType) {
  return TASK_TYPE_LABELS[taskType] || taskType.replace(/_/g, " ");
}

function friendlyTaskStatus(status) {
  return TASK_STATUS_LABELS[status] || status;
}

function openManageModal() {
  setCapabilitiesOpen(false);
  manageModal.hidden = false;
  manageScrim.hidden = false;
  document.body.classList.add("manage-open");
  manageClose.focus();
  refreshManagePanel();
}

function closeManageModal() {
  manageModal.hidden = true;
  manageScrim.hidden = true;
  document.body.classList.remove("manage-open");
  if (manageButton) manageButton.focus();
}

const MANAGE_TABS = ["train", "teach", "tasks", "models", "memory", "dashboard", "network", "addons", "sharing", "research", "automation"];
const MANAGE_TAB_REFRESHERS = {
  train: () => { refreshManagePanel(); refreshTrainResourcesTab(); },
  teach: () => refreshTeachTab(),
  tasks: refreshManagePanel,
  models: () => refreshModelsTab(),
  memory: () => refreshMemoryTab(),
  dashboard: () => refreshDashboardTab(),
  network: () => refreshNetworkTab(),
  addons: () => refreshAddonsTab(),
  sharing: () => refreshSharingTab(),
  research: () => refreshResearchTab(),
  automation: () => refreshAutomationTab(),
};

function switchManageTab(tab) {
  let activeButton = null;
  document.querySelectorAll(".manage-tab").forEach((btn) => {
    const active = btn.dataset.tab === tab;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", String(active));
    if (active) activeButton = btn;
  });
  MANAGE_TABS.forEach((t) => {
    const panel = $(`manage-tab-${t}`);
    if (panel) panel.hidden = t !== tab;
  });
  const refresher = MANAGE_TAB_REFRESHERS[tab];
  if (refresher) refresher();
  if (activeButton && window.matchMedia("(max-width: 820px)").matches) {
    const tabStrip = activeButton.closest(".manage-tabs");
    if (tabStrip) {
      window.requestAnimationFrame(() => {
        tabStrip.scrollLeft = Math.max(0, activeButton.offsetLeft - 10);
      });
    }
  }
}

if (manageButton) {
  // Visibility is finalized in init(), after discoverControlPort() resolves --
  // CONTROL_PORT isn't known yet at module-load time when auto-discovered.
  manageButton.addEventListener("click", openManageModal);
  manageClose.addEventListener("click", closeManageModal);
  manageScrim.addEventListener("click", closeManageModal);
  document.querySelectorAll(".manage-tab").forEach((btn) => {
    btn.addEventListener("click", () => switchManageTab(btn.dataset.tab));
    btn.setAttribute("role", "tab");
  });
  document.querySelectorAll('input[name="train-source"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      trainCustomPath.hidden = radio.value !== "custom" || !radio.checked;
      trainHfDataset.hidden = radio.value !== "hf_dataset" || !radio.checked;
      trainHfDatasetConfig.hidden = radio.value !== "hf_dataset" || !radio.checked;
      trainHfDatasetField.hidden = radio.value !== "hf_dataset" || !radio.checked;
    });
  });
  // A handful of well-known Hugging Face datasets don't use a "text" column
  // (the field Ickle assumes by default) -- without this hint, streaming
  // one of them silently reads nothing from every row and fails with
  // "produced too little text", which isn't obvious unless you already know
  // the dataset's schema. This only pre-fills a suggestion; it's still a
  // normal editable field for any dataset not in the list.
  const KNOWN_DATASET_TEXT_FIELDS = {
    "anthropic/hh-rlhf": "chosen",
    "openai/webgpt_comparisons": "answer_0",
    "databricks/databricks-dolly-15k": "response",
  };
  if (trainHfDataset) {
    trainHfDataset.addEventListener("blur", () => {
      const known = KNOWN_DATASET_TEXT_FIELDS[trainHfDataset.value.trim().toLowerCase()];
      if (known && !trainHfDatasetField.value.trim()) trainHfDatasetField.value = known;
    });
  }
  document.querySelectorAll('input[name="train-target"]').forEach((radio) => {
    radio.addEventListener("change", () => updateTrainTargetVisibility());
  });
  document.querySelectorAll('input[name="train-size"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      // Fills in a sensible default for the chosen size; the field stays a
      // normal editable number input, this just saves typing the common case.
      const trainSteps = $("train-steps");
      if (trainSteps && radio.checked) trainSteps.value = TRAIN_SIZE_DEFAULT_STEPS[radio.value] ?? 1200;
    });
  });
  updateTrainTargetVisibility();
}

function updateTrainTargetVisibility() {
  const checked = document.querySelector('input[name="train-target"]:checked');
  const isNew = !checked || checked.value === "new";
  // Naming and model-size only mean anything when starting from scratch --
  // continuing an existing model reuses its own saved architecture
  // (train.py loads config from the init-model checkpoint itself and
  // ignores any --block-size/--n-embd overrides passed alongside it), so
  // showing a "how big" choice there would just be a control that lies
  // about doing something.
  if (trainNameField) trainNameField.hidden = !isNew;
  if (trainSizeField) trainSizeField.hidden = !isNew;
}

function trainSourcePayload(source, customPath, hfDataset, hfDatasetConfig, hfDatasetField) {
  // Plain-language source choices map to a couple of broad, safe presets --
  // no dataset IDs, hyperparameters, or file-format details shown to the user.
  if (source === "wikipedia") {
    return { stream_dataset: "HuggingFaceFW/fineweb", stream_field: "text", stream_max_chars: 2000000 };
  }
  if (source === "conversations") {
    return { stream_dataset: "OpenAssistant/oasst1", stream_field: "text", stream_max_chars: 2000000 };
  }
  if (source === "hf_dataset") {
    const payload = {
      stream_dataset: hfDataset,
      stream_field: (hfDatasetField || "").trim() || "text",
      stream_max_chars: 2000000,
    };
    if (hfDatasetConfig) payload.stream_config = hfDatasetConfig;
    return payload;
  }
  return { data_path: customPath };
}

const TRAIN_SIZE_DEFAULT_STEPS = { quick: 2000, standard: 1200 };

function trainSizePayload(size) {
  // "Standard" leaves model dimensions unset so the server keeps auto-sizing
  // from the resource-budget sliders (the pre-existing behavior). "Quick"
  // pins a small architecture -- confirmed live to train at roughly 2
  // sec/step versus ~73 sec/step for the auto-sized default on a CPU-only
  // 16-core machine (a ~35x difference), because the resource sliders only
  // ever controlled thread/memory ceilings, never model size, which is what
  // actually dominates CPU training speed. Step count now always comes from
  // the explicit #train-steps field (see trainSizeInputs listener below),
  // not baked in here, so it's visible and editable instead of a silent
  // server-side default the UI never showed.
  if (size === "quick") {
    return { block_size: 256, n_embd: 256, n_head: 4, n_layer: 6 };
  }
  return {};
}

function slugifyModelName(name) {
  const slug = name.trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  return slug || "my_ickle";
}

function continuedModelOutPath(sourcePath) {
  // Derives a new file next to the source model instead of either (a)
  // forcing the user to type a name for something they explicitly said
  // isn't a new model, or (b) silently overwriting the model they're
  // actively chatting with while it's mid-retrain. The stamp keeps repeated
  // "keep improving" runs on the same source from colliding.
  const parts = sourcePath.replace(/\\/g, "/").split("/");
  const file = parts.pop();
  const dir = parts.length ? parts.join("/") : "models/candidates";
  const stem = file.replace(/\.pt$/i, "").replace(/\.(best|checkpoint)$/i, "");
  const stamp = new Date().toISOString().replace(/[-:TZ.]/g, "").slice(0, 12);
  return `${dir}/${stem}_continued_${stamp}.pt`;
}

if (trainForm) {
  trainForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    trainError.hidden = true;
    const target = document.querySelector('input[name="train-target"]:checked').value;
    const source = document.querySelector('input[name="train-source"]:checked').value;
    const customPath = trainCustomPath.value.trim();
    const hfDataset = trainHfDataset.value.trim();
    const hfDatasetConfig = trainHfDatasetConfig.value.trim();
    const hfDatasetField = trainHfDatasetField.value.trim();
    if (source === "custom" && !customPath) {
      trainError.textContent = "Enter the path to a text file first.";
      trainError.hidden = false;
      return;
    }
    if (source === "hf_dataset" && !hfDataset) {
      trainError.textContent = "Enter a Hugging Face dataset name first, e.g. wikitext.";
      trainError.hidden = false;
      return;
    }

    const stepsInput = $("train-steps");
    const steps = Math.max(1, parseInt(stepsInput ? stepsInput.value : "", 10) || 1200);

    let taskPayload;
    if (target === "continue") {
      const currentModel = modelSelect.value;
      if (!currentModel) {
        trainError.textContent = "No model is currently selected to continue training.";
        trainError.hidden = false;
        return;
      }
      taskPayload = {
        out_model: continuedModelOutPath(currentModel),
        init_model: currentModel,
        steps,
        ...trainSourcePayload(source, customPath, hfDataset, hfDatasetConfig, hfDatasetField),
      };
    } else {
      const name = $("train-name").value;
      if (!name.trim()) {
        trainError.textContent = "Name this model first.";
        trainError.hidden = false;
        return;
      }
      const size = document.querySelector('input[name="train-size"]:checked').value;
      taskPayload = {
        out_model: `models/candidates/${slugifyModelName(name)}.pt`,
        steps,
        ...trainSourcePayload(source, customPath, hfDataset, hfDatasetConfig, hfDatasetField),
        ...trainSizePayload(size),
      };
    }

    const payload = { task_type: "train_model", payload: taskPayload };
    try {
      await controlApi("/api/tasks", { method: "POST", body: JSON.stringify(payload) });
      refreshManagePanel();
    } catch (err) {
      trainError.textContent = err.message || "Couldn't start training.";
      trainError.hidden = false;
    }
  });
}

if (trainStopBtn) {
  trainStopBtn.addEventListener("click", async () => {
    if (!activeTrainTaskId) return;
    try {
      await controlApi(`/api/tasks/${activeTrainTaskId}/cancel`, { method: "POST" });
      refreshManagePanel();
    } catch (err) {
      // This button previously failed completely silently on error -- a
      // real report of "the stop button doesn't work" traced back to
      // exactly this: the request can fail (server unreachable, task
      // already gone) with zero visible sign anything happened.
      trainError.textContent = err.message || "Couldn't stop training -- try again.";
      trainError.hidden = false;
    }
  });
}

const teachGenerateForm = $("teach-generate-form");
const teachGenerateError = $("teach-generate-error");
const teachStats = $("teach-stats");
const teachTrainBtn = $("teach-train-btn");
const teachTrainError = $("teach-train-error");
const teachDpoBuildBtn = $("teach-dpo-build-btn");
const teachDpoBuildError = $("teach-dpo-build-error");
const teachDpoStats = $("teach-dpo-stats");
const teachDpoTrainBtn = $("teach-dpo-train-btn");
const teachDpoTrainError = $("teach-dpo-train-error");

async function refreshTeachTab() {
  if (!CONTROL_PORT) return;
  try {
    const stats = await controlApi("/api/teach/stats");
    const turns = Number(stats.turn_count || 0);
    teachStats.textContent = turns > 0
      ? `${turns} training example(s) ready from ${stats.session_count || 0} session(s).`
      : "No training examples yet -- generate some above first.";
  } catch {
    teachStats.textContent = "Couldn't load teaching data status.";
  }
  if (teachDpoStats) {
    try {
      const dpoStats = await controlApi("/api/teach/dpo-stats");
      const pairs = Number(dpoStats.pair_count || 0);
      teachDpoStats.textContent = pairs > 0
        ? `${pairs} preference pair(s) ready to align on.`
        : "No preference pairs yet -- build them from your ratings above first.";
    } catch {
      teachDpoStats.textContent = "Couldn't load preference-pair status.";
    }
  }
}

if (teachGenerateForm) {
  teachGenerateForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    teachGenerateError.hidden = true;
    const providerKey = $("teach-provider-key").value.trim();
    const provider = providerKey ? `registry:${providerKey}` : $("teach-provider").value;
    const topic = $("teach-topic").value.trim();
    const count = Math.max(1, Math.min(50, Number($("teach-count").value) || 8));
    if (!topic) {
      teachGenerateError.textContent = "Enter a topic first.";
      teachGenerateError.hidden = false;
      return;
    }
    try {
      await controlApi("/api/tasks", {
        method: "POST",
        body: JSON.stringify({ task_type: "generate_teacher_data", payload: { provider, topic, count } }),
      });
      refreshManagePanel();
      refreshTeachTab();
    } catch (err) {
      teachGenerateError.textContent = err.message || "Couldn't generate training data.";
      teachGenerateError.hidden = false;
    }
  });
}

if (teachTrainBtn) {
  teachTrainBtn.addEventListener("click", async () => {
    teachTrainError.hidden = true;
    try {
      await controlApi("/api/tasks", {
        method: "POST",
        body: JSON.stringify({
          task_type: "train_from_teacher",
          payload: { baseline_model: modelSelect.value || undefined },
        }),
      });
      refreshManagePanel();
    } catch (err) {
      teachTrainError.textContent = err.message || "Couldn't start training.";
      teachTrainError.hidden = false;
    }
  });
}

if (teachDpoBuildBtn) {
  teachDpoBuildBtn.addEventListener("click", async () => {
    teachDpoBuildError.hidden = true;
    try {
      await controlApi("/api/tasks", {
        method: "POST",
        body: JSON.stringify({ task_type: "build_dpo_preferences", payload: {} }),
      });
      refreshManagePanel();
      refreshTeachTab();
    } catch (err) {
      teachDpoBuildError.textContent = err.message || "Couldn't build preference pairs.";
      teachDpoBuildError.hidden = false;
    }
  });
}

if (teachDpoTrainBtn) {
  teachDpoTrainBtn.addEventListener("click", async () => {
    teachDpoTrainError.hidden = true;
    if (!modelSelect.value) {
      teachDpoTrainError.textContent = "Choose a model at the top of the app first.";
      teachDpoTrainError.hidden = false;
      return;
    }
    try {
      await controlApi("/api/tasks", {
        method: "POST",
        body: JSON.stringify({ task_type: "dpo_train", payload: { model: modelSelect.value } }),
      });
      refreshManagePanel();
    } catch (err) {
      teachDpoTrainError.textContent = err.message || "Couldn't start alignment training.";
      teachDpoTrainError.hidden = false;
    }
  });
}

function formatTrainingProgress(fileData) {
  const step = Number(fileData.step || 0);
  const total = Number(fileData.total_steps || 0);
  const pct = total > 0 ? Math.min(100, Math.round((step / total) * 100)) : 0;
  const loss = fileData.val_loss ?? fileData.train_loss;
  let quality = "";
  if (typeof loss === "number") {
    if (loss > 4) quality = "Still learning the basics.";
    else if (loss > 2.5) quality = "Getting the hang of it.";
    else if (loss > 1.5) quality = "Doing well, refining answers.";
    else quality = "Performing strongly.";
  }
  return { pct, quality };
}

async function refreshManagePanel() {
  if (!CONTROL_PORT) return;
  try {
    // Task list comes from the control server; live step/loss numbers come
    // from the chat server's /api/status, which already reads the training
    // status file (see ChatRuntime.get_status -> inspect_training_status) --
    // reusing that instead of adding a second endpoint that duplicates it.
    const [tasksData, statusData] = await Promise.all([controlApi("/api/tasks"), api("/api/status")]);
    renderTasks(tasksData.tasks || []);
    renderTrainActive(tasksData.tasks || [], statusData.training || {});
  } catch (err) {
    tasksEmpty.hidden = false;
    tasksEmpty.textContent = "Couldn't reach the management service.";
    tasksList.innerHTML = "";
  }
}

const quickTaskForm = $("quick-task-form");
if (quickTaskForm) {
  quickTaskForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = $("quick-task-input");
    const resultEl = $("quick-task-result");
    const instruction = input.value.trim();
    if (!instruction) return;
    resultEl.textContent = "Working out what that means...";
    try {
      const result = await controlApi("/api/tasks/infer", {
        method: "POST",
        body: JSON.stringify({ instruction, queue: true }),
      });
      if (!result.inferred) {
        resultEl.textContent = "Couldn't turn that into a task -- try phrasing like 'learn about <topic> from wikipedia/the internet', 'revisit research about <topic>', or 'train model'.";
        return;
      }
      resultEl.textContent = `Queued: ${friendlyTaskType(result.inferred.task_type || "")}`;
      input.value = "";
      refreshManagePanel();
    } catch (err) {
      resultEl.textContent = err.message || "Couldn't create that task.";
    }
  });
}

function renderTrainActive(tasks, liveStatus) {
  const trainTasks = tasks.filter((t) => t.task_type === "train_model");
  const active = trainTasks.find((t) => t.status === "running" || t.status === "queued");
  activeTrainTaskId = active ? active.task_id : null;

  if (!active) {
    trainActive.hidden = true;
    trainForm.hidden = false;
    // The most recent run may have just failed -- without this, the panel
    // silently reverts to a blank form the instant a task fails, with the
    // only trace of what happened buried in the separate Tasks list.
    const mostRecent = trainTasks[0];
    if (mostRecent && mostRecent.status === "failed") {
      trainError.textContent = mostRecent.error
        ? `Training failed: ${mostRecent.error}`
        : "Training failed for an unknown reason.";
      trainError.hidden = false;
    } else if (mostRecent && mostRecent.status === "cancelled") {
      trainError.textContent = "Training was cancelled.";
      trainError.hidden = false;
    }
    return;
  }
  trainActive.hidden = false;
  trainForm.hidden = true;

  // A "queued" task can be a fresh submission, or a failed attempt sitting
  // in its retry backoff with the *same* config that just failed -- e.g. a
  // wrong dataset field name fails identically on every attempt, so without
  // this the error only ever appeared after all retries were exhausted
  // (or never, if the user gave up and cancelled it first).
  if (active.status === "queued" && active.error) {
    trainError.textContent = `Training run hit a problem and will retry: ${active.error}`;
    trainError.hidden = false;
  } else {
    trainError.hidden = true;
  }

  const { pct, quality } = formatTrainingProgress(liveStatus || {});
  trainActiveSummary.textContent = active.status === "queued" ? "Training is queued to start..." : "Training is running...";
  trainProgressBar.style.width = `${pct}%`;
  trainActiveDetail.textContent = quality
    ? `${quality} (${pct}% of this run complete)`
    : `${pct}% of this run complete`;
  renderTrainStats(liveStatus || {});
}

function renderTrainStats(liveStatus) {
  const statsEl = $("train-stats");
  if (!statsEl) return;
  const step = liveStatus.step;
  const total = liveStatus.total_steps;
  const stats = [];
  if (typeof step === "number" && typeof total === "number") stats.push(`Step ${step}/${total}`);
  if (typeof liveStatus.train_loss === "number") stats.push(`loss ${liveStatus.train_loss.toFixed(3)}`);
  if (typeof liveStatus.val_loss === "number") stats.push(`val loss ${liveStatus.val_loss.toFixed(3)}`);
  if (typeof liveStatus.perplexity === "number") stats.push(`perplexity ${liveStatus.perplexity.toFixed(1)}`);
  if (typeof liveStatus.acc_top1 === "number") stats.push(`top-1 acc ${Math.round(liveStatus.acc_top1 * 100)}%`);
  if (typeof liveStatus.acc_top5 === "number") stats.push(`top-5 acc ${Math.round(liveStatus.acc_top5 * 100)}%`);
  if (typeof liveStatus.lr === "number") stats.push(`lr ${liveStatus.lr.toExponential(2)}`);
  if (typeof liveStatus.best_val_loss === "number" && liveStatus.best_val_loss > 0) {
    stats.push(`best val loss ${liveStatus.best_val_loss.toFixed(3)}`);
  }
  if (!stats.length) {
    statsEl.hidden = true;
    return;
  }
  statsEl.hidden = false;
  statsEl.textContent = stats.join(" · ");
}

function renderTasks(tasks) {
  tasksList.innerHTML = "";
  if (!tasks.length) {
    tasksEmpty.hidden = false;
    tasksEmpty.textContent = "No background tasks yet.";
    return;
  }
  tasksEmpty.hidden = true;

  tasks.slice(0, 30).forEach((t) => {
    const row = document.createElement("div");
    row.className = "task-row";

    const main = document.createElement("div");
    main.className = "task-row-main";
    const title = document.createElement("span");
    title.className = "task-row-title";
    title.textContent = friendlyTaskType(String(t.task_type || ""));
    const status = document.createElement("span");
    status.className = `task-row-status task-status-${t.status || "unknown"}`;
    status.textContent = friendlyTaskStatus(String(t.status || ""));
    main.appendChild(title);
    main.appendChild(status);

    if (t.status === "failed" && t.error) {
      const errorEl = document.createElement("span");
      errorEl.className = "task-row-error";
      errorEl.textContent = String(t.error);
      main.appendChild(errorEl);
    }

    row.appendChild(main);

    if (t.status === "running" || t.status === "queued") {
      const cancelBtn = document.createElement("button");
      cancelBtn.className = "task-cancel-btn";
      cancelBtn.textContent = "Cancel";
      cancelBtn.onclick = async () => {
        cancelBtn.disabled = true;
        try {
          await controlApi(`/api/tasks/${t.task_id}/cancel`, { method: "POST" });
          refreshManagePanel();
        } catch (err) {
          cancelBtn.textContent = err.message || "Failed";
          cancelBtn.disabled = false;
        }
      };
      row.appendChild(cancelBtn);
    }

    tasksList.appendChild(row);
  });
}

// --- Models -----------------------------------------------------------

function formatBytes(bytes) {
  const n = Number(bytes || 0);
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`;
  return `${n} B`;
}

async function refreshModelsTab() {
  const list = $("models-list");
  const empty = $("models-empty");
  try {
    const data = await controlApi("/api/models?all=1&limit=100");
    const models = data.models || [];
    list.innerHTML = "";
    empty.hidden = models.length > 0;
    const activePath = modelSelect.value;
    models.forEach((m) => {
      const row = document.createElement("div");
      row.className = "model-row";
      const main = document.createElement("div");
      main.className = "model-row-main";
      const name = document.createElement("span");
      name.className = "model-row-name";
      name.textContent = m.name;
      if (m.path === activePath) {
        const badge = document.createElement("span");
        badge.className = "model-badge";
        badge.textContent = "Active";
        name.appendChild(badge);
      }
      const meta = document.createElement("span");
      meta.className = "model-row-meta";
      const when = m.updated_at ? new Date(m.updated_at * 1000).toLocaleDateString() : "";
      meta.textContent = `${formatBytes(m.size_bytes)} · ${when}`;
      main.appendChild(name);
      main.appendChild(meta);
      row.appendChild(main);

      const useBtn = document.createElement("button");
      useBtn.className = "model-use-btn";
      const isActive = m.path === activePath;
      useBtn.textContent = isActive ? "In use" : "Use this";
      useBtn.disabled = isActive;
      useBtn.onclick = async () => {
        useBtn.disabled = true;
        try {
          await api("/api/flags", { method: "POST", body: JSON.stringify({ current_model: m.path }) });
          // Persisting the flag alone isn't enough: the topbar model picker
          // (#model-select) is what sendMessage() and everything else in the
          // app actually reads to decide which model answers, and setting a
          // value programmatically doesn't fire its own "change" handler --
          // without this, the flag changes server-side but the app keeps
          // chatting with the old model until the user separately touches
          // the topbar dropdown, which is why this looked like it did nothing.
          modelSelect.value = m.path;
          await refreshModels();
          refreshModelsTab();
        } catch (err) {
          useBtn.textContent = err.message || "Failed";
          useBtn.disabled = false;
        }
      };
      row.appendChild(useBtn);
      list.appendChild(row);
    });
  } catch {
    empty.hidden = false;
    empty.textContent = "Couldn't load models.";
  }
}

const modelsCleanupBtn = $("models-cleanup-btn");
if (modelsCleanupBtn) {
  modelsCleanupBtn.addEventListener("click", async () => {
    if (!confirm("This archives old model files to free up space. Recent and active models are kept. Continue?")) return;
    const resultEl = $("models-cleanup-result");
    resultEl.textContent = "Cleaning up...";
    try {
      const result = await controlApi("/api/maintenance/model", { method: "POST", body: JSON.stringify({ apply: true }) });
      const counts = result.status_counts || {};
      resultEl.textContent = `Done: ${counts.done || 0} archived, ${counts.kept || 0} kept.`;
      refreshModelsTab();
    } catch (err) {
      resultEl.textContent = err.message || "Cleanup failed.";
    }
  });
}

const trainingDataCleanupBtn = $("training-data-cleanup-btn");
if (trainingDataCleanupBtn) {
  trainingDataCleanupBtn.addEventListener("click", async () => {
    if (!confirm("This archives old training corpus/checkpoint files to free up space. Continue?")) return;
    const resultEl = $("training-data-cleanup-result");
    resultEl.textContent = "Cleaning up...";
    try {
      const result = await controlApi("/api/maintenance/training", { method: "POST", body: JSON.stringify({ apply: true }) });
      const counts = result.status_counts || {};
      resultEl.textContent = `Done: ${counts.done || 0} archived, ${counts.kept || 0} kept.`;
    } catch (err) {
      resultEl.textContent = err.message || "Cleanup failed.";
    }
  });
}

const workspaceCleanupBtn = $("workspace-cleanup-btn");
if (workspaceCleanupBtn) {
  workspaceCleanupBtn.addEventListener("click", async () => {
    if (!confirm("This clears out old runtime logs and maintenance files. Continue?")) return;
    const resultEl = $("workspace-cleanup-result");
    resultEl.textContent = "Cleaning up...";
    try {
      const result = await controlApi("/api/maintenance/data", { method: "POST", body: JSON.stringify({ apply: true }) });
      const counts = result.status_counts || {};
      resultEl.textContent = `Done: ${counts.done || 0} cleared, ${counts.kept || 0} kept.`;
    } catch (err) {
      resultEl.textContent = err.message || "Cleanup failed.";
    }
  });
}

// --- Training resource budget & autonomous programs -----------------------

function setBudgetSlider(id, value) {
  const input = $(id);
  const label = $(`${id}-value`);
  if (!input) return;
  const pct = typeof value === "number" ? value : 70;
  input.value = pct;
  if (label) label.textContent = `${pct}%`;
}

["budget-cpu", "budget-ram", "budget-gpu"].forEach((id) => {
  const input = $(id);
  if (!input) return;
  input.addEventListener("input", () => {
    const label = $(`${id}-value`);
    if (label) label.textContent = `${input.value}%`;
  });
});

async function refreshTrainResourcesTab() {
  if (!CONTROL_PORT) return;
  try {
    const data = await controlApi("/api/system/resources");
    const budget = data.resource_budget || {};
    const gpuAvailable = Boolean(data.detected && data.detected.gpu && data.detected.gpu.available);
    setBudgetSlider("budget-cpu", budget.cpu_percent);
    setBudgetSlider("budget-ram", budget.ram_percent);
    const gpuRow = $("budget-gpu-row");
    if (gpuRow) gpuRow.hidden = !gpuAvailable;
    if (gpuAvailable) setBudgetSlider("budget-gpu", budget.gpu_percent);
  } catch {}

  try {
    const flags = await controlApi("/api/flags");
    const allowed = Boolean(flags.allow_auto_training_tasks);
    const startBtn = $("program-start-btn");
    if (startBtn) startBtn.disabled = !allowed;
    const note = $("program-disabled-note");
    if (note) note.hidden = allowed;
  } catch {}
}

const budgetSaveBtn = $("budget-save-btn");
if (budgetSaveBtn) {
  budgetSaveBtn.addEventListener("click", async () => {
    const resultEl = $("budget-save-result");
    const gpuRow = $("budget-gpu-row");
    const payload = {
      cpu_percent: Number($("budget-cpu").value),
      ram_percent: Number($("budget-ram").value),
    };
    if (gpuRow && !gpuRow.hidden) payload.gpu_percent = Number($("budget-gpu").value);
    resultEl.textContent = "Saving...";
    try {
      await controlApi("/api/system/resources", { method: "POST", body: JSON.stringify(payload) });
      resultEl.textContent = "Saved -- applies to the next training run.";
    } catch (err) {
      resultEl.textContent = err.message || "Couldn't save.";
    }
  });
}

const programForm = $("program-form");
if (programForm) {
  programForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const errorEl = $("program-error");
    errorEl.hidden = true;
    const topics = $("program-topics").value.split(",").map((t) => t.trim()).filter(Boolean);
    const passes = Math.max(1, Math.min(8, Number($("program-passes").value) || 2));
    const steps = Math.max(100, Number($("program-steps").value) || 1000);
    try {
      await controlApi("/api/programs/research-train", {
        method: "POST",
        body: JSON.stringify({ topics, passes, steps }),
      });
      refreshManagePanel();
    } catch (err) {
      errorEl.textContent = err.message || "Couldn't start the program.";
      errorEl.hidden = false;
    }
  });
}

// --- Memory -------------------------------------------------------------

async function refreshMemoryTab() {
  const summaryEl = $("memory-summary");
  try {
    const summary = await controlApi("/api/memory/summary");
    const factCount = summary.fact_count ?? summary.facts_count ?? summary.total_facts ?? "?";
    summaryEl.textContent = `Ickle remembers ${factCount} thing(s) about you and your conversations, stored only on this device.`;
  } catch {
    summaryEl.textContent = "Couldn't load memory summary.";
  }
  await refreshMemoryFactsList();
}

async function refreshMemoryFactsList() {
  const list = $("memory-facts-list");
  const empty = $("memory-facts-empty");
  if (!list) return;
  try {
    const data = await controlApi("/api/memory/facts?limit=60");
    const facts = data.facts || [];
    list.innerHTML = "";
    empty.hidden = facts.length > 0;
    facts.forEach((f) => {
      const row = document.createElement("div");
      row.className = "task-row";
      const main = document.createElement("div");
      main.className = "task-row-main";
      const text = document.createElement("span");
      text.className = "task-row-title";
      text.textContent = f.fact || f.text || "";
      const meta = document.createElement("span");
      meta.className = "task-row-status";
      meta.textContent = f.category || "general";
      main.appendChild(text);
      main.appendChild(meta);
      row.appendChild(main);
      list.appendChild(row);
    });
  } catch {
    empty.hidden = false;
    empty.textContent = "Couldn't load stored facts.";
  }
}

function renderMemoryResults(items, label) {
  const container = $("memory-results");
  if (!items || !items.length) return;
  const header = document.createElement("div");
  header.className = "task-row-title";
  header.textContent = label;
  container.appendChild(header);
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "task-row";
    const text = document.createElement("span");
    text.textContent = item.fact || item.text || item.finding || JSON.stringify(item);
    row.appendChild(text);
    container.appendChild(row);
  });
}

const memorySearchForm = $("memory-search-form");
if (memorySearchForm) {
  memorySearchForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = $("memory-search-input").value.trim();
    const results = $("memory-results");
    results.innerHTML = "";
    if (!query) return;
    try {
      const data = await controlApi("/api/memory/search", { method: "POST", body: JSON.stringify({ query }) });
      if (!(data.facts || []).length && !(data.research_notes || []).length && !(data.web_facts || []).length) {
        results.innerHTML = '<p class="hint-text">Nothing found for that search.</p>';
        return;
      }
      renderMemoryResults(data.facts, "Facts");
      renderMemoryResults(data.research_notes, "Research notes");
      renderMemoryResults(data.web_facts, "Web facts");
    } catch (err) {
      results.innerHTML = `<p class="hint-text">${err.message || "Search failed."}</p>`;
    }
  });
}

const memoryForgetBtn = $("memory-forget-btn");
if (memoryForgetBtn) {
  memoryForgetBtn.addEventListener("click", async () => {
    if (!confirm("This permanently deletes everything Ickle remembers about you. This cannot be undone. Continue?")) return;
    const result = $("memory-forget-result");
    memoryForgetBtn.disabled = true;
    try {
      await controlApi("/api/memory/clear", { method: "POST", body: JSON.stringify({}) });
      if (result) result.textContent = "Done -- everything Ickle remembered has been deleted.";
      refreshMemoryTab();
    } catch (err) {
      // A destructive action that fails with zero feedback is the worst
      // version of this bug: the confirm dialog told the person it worked.
      if (result) result.textContent = err.message || "Couldn't clear memory -- nothing was deleted.";
    } finally {
      memoryForgetBtn.disabled = false;
    }
  });
}

const memoryAddFactForm = $("memory-add-fact-form");
if (memoryAddFactForm) {
  memoryAddFactForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const factInput = $("memory-add-fact-input");
    const categoryInput = $("memory-add-fact-category");
    const resultEl = $("memory-add-fact-result");
    const fact = factInput.value.trim();
    if (!fact) return;
    try {
      await controlApi("/api/memory/facts", {
        method: "POST",
        body: JSON.stringify({ fact, category: categoryInput.value.trim() || "general" }),
      });
      factInput.value = "";
      categoryInput.value = "";
      resultEl.textContent = "Remembered.";
      refreshMemoryTab();
    } catch (err) {
      resultEl.textContent = err.message || "Couldn't save that.";
    }
  });
}

const memoryExportBtn = $("memory-export-btn");
if (memoryExportBtn) {
  memoryExportBtn.addEventListener("click", async () => {
    const resultEl = $("memory-export-result");
    resultEl.textContent = "Saving...";
    try {
      const result = await controlApi("/api/memory/export/save", { method: "POST", body: JSON.stringify({}) });
      resultEl.textContent = `Saved to ${result.path}`;
    } catch (err) {
      resultEl.textContent = err.message || "Couldn't save the export.";
    }
  });
}

// --- Dashboard ------------------------------------------------------------

function setGauge(barId, textId, pct, text, warnAt = 75, dangerAt = 90) {
  const bar = $(barId);
  const label = $(textId);
  bar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  bar.classList.toggle("gauge-warn", pct >= warnAt && pct < dangerAt);
  bar.classList.toggle("gauge-danger", pct >= dangerAt);
  label.textContent = text;
}

async function refreshDashboardTab() {
  try {
    const stats = await controlApi("/api/system/live");
    if (typeof stats.cpu_percent === "number") {
      setGauge("gauge-cpu", "gauge-cpu-text", stats.cpu_percent, `${Math.round(stats.cpu_percent)}% in use`);
    }
    if (typeof stats.ram_percent === "number") {
      setGauge(
        "gauge-ram", "gauge-ram-text", stats.ram_percent,
        `${formatBytes(stats.ram_used_bytes)} of ${formatBytes(stats.ram_total_bytes)}`
      );
    }
    if (stats.disk_total_bytes) {
      const usedPct = 100 - (stats.disk_free_bytes / stats.disk_total_bytes) * 100;
      setGauge("gauge-disk", "gauge-disk-text", usedPct, `${formatBytes(stats.disk_free_bytes)} free of ${formatBytes(stats.disk_total_bytes)}`, 85, 95);
    }
    const gpuText = $("gauge-gpu-text");
    gpuText.textContent = stats.gpu && stats.gpu.available
      ? `Graphics card in use: ${(stats.gpu.names || []).join(", ")}`
      : "No graphics card in use — running on processor only, which is normal and still works, just slower for training.";
  } catch {}
}

// --- Network (real P2P swarm -- same node the Sharing tab talks to) -------

async function refreshNetworkTab() {
  const summaryEl = $("network-summary");
  const joinToggle = $("network-join-toggle");
  const peersList = $("network-peers-list");
  const peersEmpty = $("network-peers-empty");
  try {
    const data = await controlApi("/api/torickle/status");
    if (joinToggle) joinToggle.checked = Boolean(data.enabled);
    if (networkRefreshBtn) networkRefreshBtn.disabled = !data.enabled;
    const switchLabel = $("network-switch-label");
    if (switchLabel) switchLabel.textContent = data.enabled ? "Leave swarm" : "Join swarm";
    const discovery = data.public_discovery || {};
    const allowedPhases = new Set(["off", "starting", "searching", "listening", "connected", "degraded"]);
    const phase = allowedPhases.has(discovery.phase) ? discovery.phase : "degraded";
    const peerCount = Number(data.peers_known || 0);
    const phaseCopy = {
      off: ["Not participating", "This device is not advertised and makes no discovery requests."],
      starting: ["Starting public discovery", "Preparing the node identity and DHT lookup."],
      searching: ["Searching the public DHT", "Looking for nodes sharing Ickle's versioned network key."],
      listening: ["Public swarm active", "Discovery is working. This may be the first reachable Ickle node on the network."],
      connected: ["Connected to the public swarm", `${peerCount} verified Ickle ${peerCount === 1 ? "peer" : "peers"} available.`],
      degraded: ["Public discovery is limited", discovery.last_error || "The DHT did not answer; UDP may be blocked."],
    }[phase];
    const reachability = data.reachability || "local";
    const reachabilityCopy = reachability === "reachable"
      ? ["Incoming ready", data.port_mapped ? "Router port mapping is active." : "Using the configured public address."]
      : reachability === "outbound-only"
        ? ["Outbound only", "You can reach public peers, but your router did not automatically accept incoming connections."]
        : ["Local only", "The swarm listener is bound to this device."];
    const responded = Number(discovery.nodes_responded || 0);
    const contacted = Number(discovery.nodes_contacted || 0);
    const announced = Number(discovery.announced_to || 0);
    summaryEl.innerHTML = `
      <div class="network-state network-state-${phase}">
        <span class="network-live-dot" aria-hidden="true"></span>
        <div><strong>${phaseCopy[0]}</strong><span>${escapeHtml(phaseCopy[1])}</span></div>
      </div>
      <div class="network-metric"><span>VERIFIED PEERS</span><strong>${peerCount}</strong><small>speaking Ickle protocol</small></div>
      <div class="network-metric"><span>DHT HEALTH</span><strong>${responded}<em> / ${contacted}</em></strong><small>routers answered / contacted</small></div>
      <div class="network-metric"><span>REACHABILITY</span><strong>${reachabilityCopy[0]}</strong><small>${reachabilityCopy[1]}</small></div>
      <div class="network-status-foot">Announced through ${announced} DHT ${announced === 1 ? "node" : "nodes"}${discovery.last_refresh_utc ? ` · refreshed ${escapeHtml(new Date(discovery.last_refresh_utc).toLocaleTimeString())}` : ""}</div>
    `;
    const knownPeers = data.known_peers || [];
    peersList.innerHTML = "";
    peersEmpty.hidden = knownPeers.length > 0;
    knownPeers.forEach((address) => {
      const row = document.createElement("div");
      row.className = "task-row";
      const main = document.createElement("div");
      main.className = "task-row-main";
      const title = document.createElement("span");
      title.className = "task-row-title";
      title.textContent = address;
      main.appendChild(title);
      row.appendChild(main);
      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "danger-button";
      removeBtn.textContent = "Remove";
      removeBtn.addEventListener("click", async () => {
        removeBtn.disabled = true;
        try {
          await controlApi("/api/swarm/peers/remove", { method: "POST", body: JSON.stringify({ address }) });
          refreshNetworkTab();
        } catch (err) {
          removeBtn.textContent = err.message || "Failed";
          removeBtn.disabled = false;
        }
      });
      row.appendChild(removeBtn);
      peersList.appendChild(row);
    });
  } catch {
    summaryEl.innerHTML = '<p class="network-summary-line">Couldn\'t reach the network status service.</p>';
  }

  const coordinatorEl = $("federated-coordinator-status");
  if (coordinatorEl) {
    try {
      const fed = await controlApi("/api/federated/status");
      coordinatorEl.textContent = fed.coordinator_running
        ? `Running: ${fed.active_client_count || 0} device(s) registered, ${fed.completed_rounds || 0} round(s) completed.`
        : "Not running on this device.";
    } catch {
      coordinatorEl.textContent = "Couldn't reach the coordinator status service.";
    }
  }

  await refreshCodistillPanel();
  await refreshCommonsPanel();
  await refreshConsolidationPanel();
  await refreshDisagreementsPanel();
}

// --- Peer teaching / co-distillation ---------------------------------------

async function refreshCodistillPanel() {
  const reportEl = $("codistill-last-report");
  const trustHeading = $("codistill-trust-heading");
  const trustList = $("codistill-trust-list");
  if (!reportEl || !trustList) return;
  try {
    const data = await controlApi("/api/codistill/status");
    const report = data.last_report;
    if (!report) {
      reportEl.textContent = "No teaching round has run yet.";
    } else {
      const when = report.ran_at_utc ? new Date(report.ran_at_utc).toLocaleString() : "";
      reportEl.textContent =
        `Last round (${when}): taught on ${report.probes_taught || 0}/${report.probes_total || 0} probe(s) ` +
        `from ${report.peers_discovered || 0} peer(s) reached. ${data.corpus_pairs || 0} pair(s) waiting to be ` +
        `fed into the anti-forgetting pipeline.`;
    }
    const trust = data.trust || [];
    trustHeading.hidden = trust.length === 0;
    trustList.innerHTML = "";
    trust.forEach((row) => {
      const el = document.createElement("div");
      el.className = "task-row";
      const main = document.createElement("div");
      main.className = "task-row-main";
      const title = document.createElement("span");
      title.className = "task-row-title";
      title.textContent = `${row.peer_id.slice(0, 16)}...`;
      main.appendChild(title);
      const domains = row.domains || {};
      const domainNames = Object.keys(domains).sort();
      if (domainNames.length) {
        const chips = document.createElement("div");
        chips.className = "codistill-domain-chips";
        domainNames.forEach((d) => {
          const chip = document.createElement("span");
          chip.className = "codistill-domain-chip";
          chip.textContent = `${d} ${domains[d].toFixed(2)}`;
          chips.appendChild(chip);
        });
        main.appendChild(chips);
      }
      el.appendChild(main);
      const score = document.createElement("span");
      score.textContent = `overall ${row.trust.toFixed(2)}`;
      el.appendChild(score);
      trustList.appendChild(el);
    });
  } catch {
    reportEl.textContent = "Couldn't reach the co-distillation status service.";
  }
}

const codistillRunBtn = $("codistill-run-btn");
if (codistillRunBtn) {
  codistillRunBtn.addEventListener("click", async () => {
    const statusEl = $("codistill-run-status");
    codistillRunBtn.disabled = true;
    if (statusEl) statusEl.textContent = "Queuing a teaching round -- watch it under Background tasks.";
    try {
      await controlApi("/api/tasks", {
        method: "POST",
        body: JSON.stringify({ task_type: "codistill_round", payload: {} }),
      });
      if (statusEl) statusEl.textContent = "Round queued. Check the Background tasks tab for progress.";
    } catch (err) {
      if (statusEl) statusEl.textContent = err.message || "Couldn't queue a teaching round.";
    } finally {
      codistillRunBtn.disabled = false;
    }
  });
}

const networkJoinToggle = $("network-join-toggle");
if (networkJoinToggle) {
  networkJoinToggle.addEventListener("change", async () => {
    const path = networkJoinToggle.checked ? "/api/swarm/join" : "/api/swarm/leave";
    networkJoinToggle.disabled = true;
    try {
      await controlApi(path, { method: "POST", body: JSON.stringify({}) });
    } catch {
      networkJoinToggle.checked = !networkJoinToggle.checked;
    } finally {
      networkJoinToggle.disabled = false;
    }
    refreshNetworkTab();
  });
}

const networkRefreshBtn = $("network-refresh-btn");
if (networkRefreshBtn) {
  networkRefreshBtn.addEventListener("click", async () => {
    const note = $("network-refresh-note");
    networkRefreshBtn.disabled = true;
    if (note) note.textContent = "Starting a fresh DHT lookup…";
    try {
      await controlApi("/api/swarm/refresh", { method: "POST", body: JSON.stringify({}) });
      if (note) note.textContent = "Discovery refresh started in the background.";
      window.setTimeout(refreshNetworkTab, 1200);
    } catch (err) {
      if (note) note.textContent = err.message || "Couldn't refresh discovery.";
    } finally {
      networkRefreshBtn.disabled = false;
    }
  });
}

const networkAddPeerForm = $("network-add-peer-form");
if (networkAddPeerForm) {
  networkAddPeerForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = $("network-add-peer-input");
    const errorEl = $("network-add-peer-error");
    const address = (input.value || "").trim();
    if (!address) return;
    errorEl.hidden = true;
    try {
      await controlApi("/api/swarm/peers/add", { method: "POST", body: JSON.stringify({ address }) });
      input.value = "";
      refreshNetworkTab();
    } catch (err) {
      errorEl.textContent = err.message || "Couldn't add that peer.";
      errorEl.hidden = false;
    }
  });
}

// --- Epistemic Commons -----------------------------------------------------

async function refreshCommonsPanel() {
  const summary = $("commons-summary");
  const list = $("commons-events-list");
  const empty = $("commons-events-empty");
  if (!summary || !list || !empty) return;
  try {
    const data = await controlApi("/api/commons/status");
    summary.textContent =
      `${data.local_events || 0} review(s) from this device; ${data.peer_events || 0} from ` +
      `${data.peer_authors || 0} peer(s); ${data.shared_events || 0} explicitly shareable.`;
    const remote = (data.events_recent || []).filter((event) => !event.is_local);
    list.innerHTML = "";
    empty.hidden = remote.length > 0;
    remote.forEach((event) => {
      const row = document.createElement("div");
      row.className = "task-row commons-event-row";
      const main = document.createElement("div");
      main.className = "task-row-main";
      const title = document.createElement("span");
      title.className = "task-row-title";
      title.textContent = event.claim_text || "Shared review";
      const detail = document.createElement("span");
      detail.className = "task-row-status";
      const correction = event.correction_text ? ` Suggested: ${event.correction_text}` : "";
      detail.textContent = `${event.relation} by ${String(event.author_peer_id || "peer").slice(0, 12)}...${correction}`;
      main.appendChild(title);
      main.appendChild(detail);
      if (/^https?:\/\//i.test(event.source_url || "")) {
        const source = document.createElement("a");
        source.href = event.source_url;
        source.target = "_blank";
        source.rel = "noopener noreferrer";
        source.textContent = "Review source";
        main.appendChild(source);
      }
      const adopt = document.createElement("button");
      adopt.type = "button";
      adopt.textContent = "Use locally";
      adopt.title = "Adopt this perspective into your local human-reviewed context. It is not automatically treated as true.";
      adopt.addEventListener("click", async () => {
        adopt.disabled = true;
        try {
          await controlApi("/api/commons/adopt", {
            method: "POST",
            body: JSON.stringify({ event_id: event.event_id, shared: false }),
          });
          adopt.textContent = "Adopted";
          refreshCommonsPanel();
        } catch (err) {
          adopt.textContent = err.message || "Failed";
          adopt.disabled = false;
        }
      });
      row.appendChild(main);
      row.appendChild(adopt);
      list.appendChild(row);
    });
  } catch {
    summary.textContent = "Couldn't load the local knowledge commons.";
  }
}

const commonsSyncBtn = $("commons-sync-btn");
if (commonsSyncBtn) {
  commonsSyncBtn.addEventListener("click", async () => {
    const status = $("commons-sync-status");
    commonsSyncBtn.disabled = true;
    if (status) status.textContent = "Syncing signed reviews...";
    try {
      const report = await controlApi("/api/commons/sync", { method: "POST", body: JSON.stringify({}) });
      if (status) {
        status.textContent =
          `Reached ${report.peers_reached || 0}/${report.peers_attempted || 0} peer(s); ` +
          `received ${report.received || 0}, sent ${report.sent || 0}.`;
      }
      await refreshCommonsPanel();
    } catch (err) {
      if (status) status.textContent = err.message || "Couldn't sync shared reviews.";
    } finally {
      commonsSyncBtn.disabled = false;
    }
  });
}

async function refreshConsolidationPanel() {
  const summary = $("consolidation-summary");
  if (!summary) return;
  try {
    const data = await controlApi("/api/consolidation/status");
    const n = data.eligible_corrections || 0;
    summary.textContent =
      n > 0
        ? `${n} adopted correction(s) ready to be folded into the next training step.`
        : "No adopted corrections yet -- correct or adopt a claim above to build this up.";
  } catch {
    summary.textContent = "Couldn't load consolidation status.";
  }
}

const consolidationRunBtn = $("consolidation-run-btn");
if (consolidationRunBtn) {
  consolidationRunBtn.addEventListener("click", async () => {
    const status = $("consolidation-status");
    consolidationRunBtn.disabled = true;
    if (status) status.textContent = "Queuing a guarded training step -- watch it under Background tasks.";
    const stepsInput = $("consolidation-steps");
    const steps = Math.max(1, parseInt(stepsInput ? stepsInput.value : "", 10) || 1200);
    try {
      await controlApi("/api/tasks", {
        method: "POST",
        body: JSON.stringify({ task_type: "continual_guard_step", payload: { steps } }),
      });
      if (status) status.textContent = "Queued. Corrections are included automatically.";
    } catch (err) {
      if (status) status.textContent = err.message || "Couldn't queue consolidation.";
    } finally {
      consolidationRunBtn.disabled = false;
    }
  });
}

async function refreshDisagreementsPanel() {
  const list = $("disagreements-list");
  const empty = $("disagreements-empty");
  if (!list || !empty) return;
  try {
    const data = await controlApi("/api/disagreements/status");
    const top = data.top || [];
    list.innerHTML = "";
    empty.hidden = top.length > 0;
    top.forEach((entry) => {
      const row = document.createElement("div");
      row.className = "task-row commons-event-row";
      const main = document.createElement("div");
      main.className = "task-row-main";
      const title = document.createElement("span");
      title.className = "task-row-title";
      title.textContent = entry.representative || "Disputed claim";
      const detail = document.createElement("span");
      detail.className = "task-row-status";
      detail.textContent = `${entry.peer_count || 0} independent peer(s) disagree -- observed ${entry.times_observed || 1} time(s).`;
      main.appendChild(title);
      main.appendChild(detail);

      const form = document.createElement("div");
      form.className = "epistemic-form-buttons";
      const input = document.createElement("input");
      input.type = "text";
      input.placeholder = "Resolve it: what's actually correct?";
      input.className = "model-picker";
      input.setAttribute("aria-label", `Correction for: ${entry.representative || "claim"}`);
      const resolveBtn = document.createElement("button");
      resolveBtn.type = "button";
      resolveBtn.textContent = "Resolve this";
      resolveBtn.addEventListener("click", async () => {
        const correctionText = input.value.trim();
        if (!correctionText) {
          input.focus();
          return;
        }
        resolveBtn.disabled = true;
        try {
          await api("/api/epistemics/reviews", {
            method: "POST",
            body: JSON.stringify({
              claim_text: entry.representative,
              relation: "correct",
              correction_text: correctionText,
              shared: false,
            }),
          });
          await refreshDisagreementsPanel();
        } catch (err) {
          resolveBtn.textContent = err.message || "Failed";
          resolveBtn.disabled = false;
        }
      });
      form.appendChild(input);
      form.appendChild(resolveBtn);

      row.appendChild(main);
      row.appendChild(form);
      list.appendChild(row);
    });
  } catch {
    empty.hidden = false;
    empty.textContent = "Couldn't load swarm disagreements.";
  }
}

// --- Add-ons (knowledge deltas) --------------------------------------------

async function refreshAddonsTab() {
  const list = $("addons-list");
  const empty = $("addons-empty");
  try {
    const data = await controlApi("/api/deltas");
    const deltas = data.deltas || [];
    list.innerHTML = "";
    empty.hidden = deltas.length > 0;
    deltas.forEach((d) => {
      const row = document.createElement("div");
      row.className = "addon-row";
      const main = document.createElement("div");
      main.className = "addon-row-main";
      const title = document.createElement("span");
      title.textContent = d.domain_description || d.delta_id;
      const meta = document.createElement("span");
      meta.className = "task-row-status";
      meta.textContent = `Version ${d.version || "1.0.0"}`;
      main.appendChild(title);
      main.appendChild(meta);
      row.appendChild(main);

      const toggle = document.createElement("label");
      toggle.className = "addon-toggle";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = d.enabled !== false;
      input.setAttribute("aria-label", `Enable addon: ${d.domain_description || d.delta_id}`);
      input.onchange = async () => {
        try {
          const path = input.checked ? "/api/deltas/enable" : "/api/deltas/disable";
          await controlApi(path, { method: "POST", body: JSON.stringify({ delta_id: d.delta_id }) });
        } catch {
          input.checked = !input.checked;
        }
      };
      const slider = document.createElement("span");
      slider.className = "addon-toggle-slider";
      toggle.appendChild(input);
      toggle.appendChild(slider);
      row.appendChild(toggle);

      const controls = document.createElement("div");
      controls.className = "addon-row-controls";

      const thresholdLabel = document.createElement("label");
      thresholdLabel.className = "addon-threshold-label";
      thresholdLabel.textContent = "Sensitivity";
      const thresholdInput = document.createElement("input");
      thresholdInput.type = "number";
      thresholdInput.min = "0";
      thresholdInput.max = "1";
      thresholdInput.step = "0.05";
      thresholdInput.className = "addon-threshold-input";
      thresholdInput.value = typeof d.activation_threshold === "number" ? d.activation_threshold : 0.6;
      thresholdInput.title = "How closely a question must match this add-on's topic before it activates (0 = always, 1 = only exact matches).";
      thresholdInput.addEventListener("change", async () => {
        try {
          await controlApi("/api/deltas/threshold", {
            method: "POST",
            body: JSON.stringify({ delta_id: d.delta_id, threshold: parseFloat(thresholdInput.value) }),
          });
        } catch {
          thresholdInput.value = d.activation_threshold ?? 0.6;
        }
      });
      thresholdLabel.appendChild(thresholdInput);
      controls.appendChild(thresholdLabel);

      const rollbackBtn = document.createElement("button");
      rollbackBtn.type = "button";
      rollbackBtn.className = "addon-rollback-btn";
      rollbackBtn.textContent = "Roll back";
      rollbackBtn.title = "Revert this add-on to its previous version";
      rollbackBtn.addEventListener("click", async () => {
        rollbackBtn.disabled = true;
        try {
          const result = await controlApi("/api/deltas/rollback", {
            method: "POST",
            body: JSON.stringify({ delta_id: d.delta_id }),
          });
          if (result.ok) {
            refreshAddonsTab();
          } else {
            rollbackBtn.textContent = result.detail || "No earlier version";
            rollbackBtn.disabled = true;
          }
        } catch {
          rollbackBtn.disabled = false;
        }
      });
      controls.appendChild(rollbackBtn);

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "addon-rollback-btn addon-remove-btn";
      removeBtn.textContent = "Remove";
      removeBtn.title = "Permanently delete this add-on and its saved versions";
      removeBtn.addEventListener("click", async () => {
        if (!confirm(`Remove "${d.domain_description || d.delta_id}"? This deletes it and its saved versions permanently.`)) return;
        removeBtn.disabled = true;
        try {
          await controlApi("/api/deltas/remove", {
            method: "POST",
            body: JSON.stringify({ delta_id: d.delta_id }),
          });
          refreshAddonsTab();
        } catch (err) {
          removeBtn.textContent = err.message || "Failed";
          removeBtn.disabled = false;
        }
      });
      controls.appendChild(removeBtn);

      row.appendChild(controls);

      list.appendChild(row);
    });
  } catch {
    empty.hidden = false;
    empty.textContent = "Couldn't load add-ons.";
  }
}

// --- Sharing (torickle) -----------------------------------------------

function renderContributionPanel(contribution) {
  const panel = $("contribution-panel");
  const emptyEl = $("contribution-empty");
  if (!contribution || !contribution.has_history) {
    panel.hidden = true;
    emptyEl.hidden = false;
    return;
  }
  emptyEl.hidden = true;
  panel.hidden = false;

  const contributed = Number(contribution.contributed_score || 0);
  const consumed = Number(contribution.consumed_score || 0);
  const total = contributed + consumed;
  const pct = total > 0 ? Math.round((contributed / total) * 100) : 100;

  $("contribution-ratio").textContent =
    contribution.ratio_display === "unbounded" ? "All giving, no taking" : `${contribution.ratio_display}x`;
  $("contribution-bar").style.width = `${pct}%`;

  const parts = [];
  if (contribution.seed_training_rounds) parts.push(`${contribution.seed_training_rounds} training round(s)`);
  if (contribution.seed_pieces_served) parts.push(`${contribution.seed_pieces_served} update piece(s) served`);
  if (contribution.peer_requests_served) parts.push(`${contribution.peer_requests_served} question(s) answered for peers`);
  if (contribution.peer_requests_consumed) parts.push(`${contribution.peer_requests_consumed} question(s) asked of peers`);
  $("contribution-detail").textContent = parts.length
    ? `You've contributed: ${parts.join(", ")}.`
    : "No network activity yet.";
}

async function refreshSharingTab() {
  const summaryEl = $("sharing-summary");
  const bundlesEl = $("sharing-bundles");
  try {
    const [status, bundlesData, contribution] = await Promise.all([
      controlApi("/api/torickle/status"),
      controlApi("/api/torickle/bundles"),
      controlApi("/api/contribution/status").catch(() => null),
    ]);
    bundlesEl.innerHTML = "";
    renderContributionPanel(contribution);
    if (!status.active) {
      summaryEl.innerHTML = '<p class="sharing-summary-line">Sharing is not active right now.</p>';
      return;
    }
    summaryEl.innerHTML = `<p class="sharing-summary-line">Connected to ${status.peers_known || 0} peer(s). Sharing ${status.bundles_served || 0} update(s) with the network.</p>`;
    (bundlesData.bundles || []).forEach((b) => {
      const row = document.createElement("div");
      row.className = "task-row";
      const main = document.createElement("div");
      main.className = "task-row-main";
      const title = document.createElement("span");
      title.className = "task-row-title";
      title.textContent = `Update ${String(b.bundle_id || "").slice(0, 12)}`;
      const status2 = document.createElement("span");
      status2.className = "task-row-status";
      status2.textContent = `${b.piece_count || 0} piece(s), ${formatBytes(b.total_bytes)}`;
      main.appendChild(title);
      main.appendChild(status2);
      row.appendChild(main);
      bundlesEl.appendChild(row);
    });
  } catch {
    summaryEl.innerHTML = '<p class="sharing-summary-line">Couldn\'t reach the sharing service.</p>';
  }
}

// --- Research memory --------------------------------------------------

async function refreshResearchTab() {
  await refreshResearchSessions();
}

async function refreshResearchSessions() {
  const list = $("research-sessions-list");
  const empty = $("research-sessions-empty");
  if (!list) return;
  try {
    const data = await controlApi("/api/research/sessions?limit=20");
    const sessions = data.sessions || [];
    list.innerHTML = "";
    empty.hidden = sessions.length > 0;
    sessions.forEach((s) => {
      const row = document.createElement("div");
      row.className = "task-row";
      const main = document.createElement("div");
      main.className = "task-row-main";
      const title = document.createElement("span");
      title.className = "task-row-title";
      title.textContent = s.topic || s.session_id || "Untitled";
      const meta = document.createElement("span");
      meta.className = "task-row-status";
      meta.textContent = `${s.note_count || 0} note(s) · ${timeAgo(s.updated_at_utc)}`;
      main.appendChild(title);
      main.appendChild(meta);
      row.appendChild(main);
      list.appendChild(row);
    });
  } catch {
    empty.hidden = false;
    empty.textContent = "Couldn't load research sessions.";
  }
}

const researchSearchForm = $("research-search-form");
if (researchSearchForm) {
  researchSearchForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = $("research-search-input").value.trim();
    const results = $("research-results");
    const empty = $("research-results-empty");
    results.innerHTML = "";
    if (!query) return;
    try {
      const data = await controlApi(`/api/research/find?query=${encodeURIComponent(query)}&limit=20`);
      const notes = data.notes || [];
      empty.hidden = notes.length > 0;
      notes.forEach((n) => {
        const row = document.createElement("div");
        row.className = "task-row";
        const main = document.createElement("div");
        main.className = "task-row-main";
        const title = document.createElement("span");
        title.className = "task-row-title";
        title.textContent = n.finding || n.question || "";
        const meta = document.createElement("span");
        meta.className = "task-row-status";
        meta.textContent = [n.topic, n.source_title].filter(Boolean).join(" · ");
        main.appendChild(title);
        main.appendChild(meta);
        row.appendChild(main);
        results.appendChild(row);
      });
    } catch (err) {
      empty.hidden = false;
      empty.textContent = err.message || "Search failed.";
    }
  });
}

// --- Automation (runtime flags) ----------------------------------------

const AUTOMATION_FLAG_FIELDS = {
  "automation-chat-enabled": "chat_enabled",
  "automation-worker-enabled": "background_task_worker_enabled",
  "automation-auto-training": "allow_auto_training_tasks",
  "automation-parallel-training": "parallel_training_enabled",
};

async function refreshAutomationTab() {
  const errorEl = $("automation-error");
  if (errorEl) errorEl.hidden = true;
  try {
    const flags = await controlApi("/api/flags");
    Object.entries(AUTOMATION_FLAG_FIELDS).forEach(([elId, flagKey]) => {
      const el = $(elId);
      if (el) el.checked = Boolean(flags[flagKey]);
    });
    const maxParallel = $("automation-max-parallel");
    if (maxParallel) maxParallel.value = Number(flags.max_parallel_training_tasks || 2);
  } catch {
    if (errorEl) {
      errorEl.textContent = "Couldn't load automation settings.";
      errorEl.hidden = false;
    }
  }
}

async function saveAutomationFlag(flagKey, value) {
  const savedNote = $("automation-saved-note");
  const errorEl = $("automation-error");
  try {
    await controlApi("/api/flags", { method: "POST", body: JSON.stringify({ [flagKey]: value }) });
    if (savedNote) savedNote.textContent = "Saved.";
  } catch (err) {
    if (errorEl) {
      errorEl.textContent = err.message || "Couldn't save that setting.";
      errorEl.hidden = false;
    }
  }
}

Object.entries(AUTOMATION_FLAG_FIELDS).forEach(([elId, flagKey]) => {
  const el = $(elId);
  if (!el) return;
  el.addEventListener("change", () => saveAutomationFlag(flagKey, el.checked));
});

const automationMaxParallel = $("automation-max-parallel");
if (automationMaxParallel) {
  automationMaxParallel.addEventListener("change", () => {
    const value = Math.max(1, Math.min(4, Number(automationMaxParallel.value) || 1));
    automationMaxParallel.value = value;
    saveAutomationFlag("max_parallel_training_tasks", value);
  });
}

const ONBOARDING_SEEN_KEY = "ickle_onboarding_seen";

function maybeShowOnboardingBanner() {
  const banner = $("onboarding-banner");
  if (!banner) return;
  if (localStorage.getItem(ONBOARDING_SEEN_KEY)) return;
  banner.hidden = false;
}

function dismissOnboardingBanner() {
  const banner = $("onboarding-banner");
  if (banner) banner.hidden = true;
  localStorage.setItem(ONBOARDING_SEEN_KEY, "1");
}

const onboardingDismissBtn = $("onboarding-dismiss");
const onboardingLearnMoreBtn = $("onboarding-learn-more");
if (onboardingDismissBtn) onboardingDismissBtn.addEventListener("click", dismissOnboardingBanner);
if (onboardingLearnMoreBtn) {
  onboardingLearnMoreBtn.addEventListener("click", () => {
    dismissOnboardingBanner();
    if (manageButton && !manageButton.hidden) {
      openManageModal();
      switchManageTab("network");
    }
  });
}

let manageInterval = null;
function startManagePolling() {
  if (!CONTROL_PORT || manageInterval) return;
  manageInterval = setInterval(() => {
    if (!manageModal.hidden) {
      const activeTab = document.querySelector(".manage-tab.active")?.dataset.tab;
      if (activeTab === "network") refreshNetworkTab();
      else refreshManagePanel();
    }
  }, 4000);
}

function setCapabilitiesOpen(open) {
  if (!capabilitiesPanel || !capabilitiesButton) return;
  capabilitiesPanel.hidden = !open;
  capabilitiesButton.setAttribute("aria-expanded", String(open));
}

function syncCapabilitiesState() {
  const switches = [memoryToggle, webToggle, thinkingToggle, agentToggle, codeExecToggle, rawOutputToggle];
  const enabled = switches.filter((control) => control && control.checked).length;
  if (capabilitiesCount) capabilitiesCount.textContent = String(enabled);
  if (agentCapabilities) agentCapabilities.hidden = !(agentToggle && agentToggle.checked);
  if (!agentToggle?.checked && codeExecToggle) codeExecToggle.checked = false;
  if (capabilitiesButton) capabilitiesButton.classList.toggle("agent-active", Boolean(agentToggle?.checked));
}

if (capabilitiesButton && capabilitiesPanel) {
  capabilitiesButton.addEventListener("click", (event) => {
    event.stopPropagation();
    setCapabilitiesOpen(capabilitiesPanel.hidden);
  });
  capabilitiesPanel.addEventListener("click", (event) => event.stopPropagation());
  if (capabilitiesClose) capabilitiesClose.addEventListener("click", () => setCapabilitiesOpen(false));
  document.addEventListener("click", () => setCapabilitiesOpen(false));
}

if (agentToggle && codeExecToggleLabel) {
  const syncCodeExecVisibility = () => {
    syncCapabilitiesState();
  };
  agentToggle.addEventListener("change", syncCodeExecVisibility);
  syncCodeExecVisibility();
}

[memoryToggle, webToggle, thinkingToggle, rawOutputToggle, codeExecToggle].forEach((control) => {
  if (control) control.addEventListener("change", syncCapabilitiesState);
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (capabilitiesPanel && !capabilitiesPanel.hidden) {
    setCapabilitiesOpen(false);
    capabilitiesButton?.focus();
  } else if (manageModal && !manageModal.hidden) {
    closeManageModal();
  }
});

// ---------------------------------------------------------------------------

async function init() {
  if (window.location.protocol !== "file:") {
    await discoverControlPort();
    if (manageButton && CONTROL_PORT) {
      manageButton.hidden = false;
      maybeShowOnboardingBanner();
      // ?manage=train (or any manage-tab data-tab value) opens straight to
      // that tab instead of the chat screen -- e.g. for jumping straight to
      // training progress after starting a run.
      const requestedManageTab = new URLSearchParams(window.location.search).get("manage");
      if (requestedManageTab && MANAGE_TABS.includes(requestedManageTab)) {
        openManageModal();
        switchManageTab(requestedManageTab);
      }
    }

    await Promise.all([refreshModels(), refreshStatus()]);
    const sessions = await refreshSessions();

    if (!activeSessionId && sessions.length > 0) {
      await switchSession(sessions[0].id);
    }

    if (!activeSessionId) {
      await ensureSession();
    }

    if (activeSessionId) {
      try {
        const data = await api(`/api/sessions/${activeSessionId}/messages`);
        messages = (data.messages || []).map((m) => ({
          role: m.role || "user",
          text: m.text || "",
          thinking: m.thinking || "",
          at: m.at || "",
          model: m.model || "",
        }));
        saveMessagesLocal();
      } catch {}
    } else {
      messages = loadMessagesLocal();
    }
    render();

    setInterval(refreshModels, 15000);
    setInterval(refreshStatus, 15000);
    startManagePolling();
  } else {
    setStatus(false);
  }

  promptInput.focus();
}

init();
