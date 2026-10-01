const state = {
  annotator: null,
  progress: [],
  current: null,
  currentCode: null,
  filter: "all",
  pendingSaves: 0,
  submitted: false,
};

const DIMENSIONS = [
  ["d1", "D1 Decision appropriateness", "Is the stated strategy the right one at this moment?", "The stated strategy contradicts the fair-value rule.", "The strategy matches the rule."],
  ["d2", "D2 Decision-response consistency", "Does the reply actually do what the stated strategy says?", "The response contradicts or fails to carry out its stated strategy.", "The response clearly carries out the stated strategy."],
  ["d3", "D3 Price strategy", "Are the prices sensible compared with the fair value and the buyer's offers?", "Random, absurd, or unjustified price far from fair value.", "Price moves toward fair value in reasonable steps with a reason."],
  ["d4", "D4 Emotional responsiveness", "Does the seller respond appropriately to how the buyer feels, as heard in the audio?", "Ignores or clashes with the buyer's emotional state.", "Notices the buyer's mood and responds suitably."],
  ["d5", "D5 Negotiation progression", "Does the reply move the negotiation toward a fair resolution?", "Stalls, repeats, loops, or moves backward.", "Clearly moves toward a fair deal."],
  ["d6", "D6 Adaptation to change", "The situation just changed. Does the seller adapt?", "Continues as though nothing changed.", "Changes approach appropriately for the new situation."],
  ["d7", "D7 Coherence and context", "Is the reply consistent with the conversation so far?", "Contradicts earlier turns, forgets an agreed price, or gets the product wrong.", "Fully consistent with everything previously said."],
  ["d8", "D8 Text naturalness", "Does the wording sound like something a real shopkeeper would say?", "Awkward, robotic, repetitive, or badly sized.", "Natural, fluent, and appropriately concise."],
  ["d9", "D9 Speech intelligibility", "Can you understand every word of the seller-reply audio?", "Mostly unintelligible.", "Every word is clear."],
  ["d10", "D10 Speech naturalness and prosody", "Does the voice sound natural, with a tone that fits the moment?", "Monotone, robotic, oddly stressed, or emotionally mismatched.", "Human-like rhythm and intonation with an appropriate tone."],
  ["d11", "D11 Speech-text fidelity", "Does the seller-reply audio say what the displayed text says?", "Different wording, missing sentences, or incorrect spoken numbers.", "Matches the text word for word."],
  ["d12", "D12 Overall quality", "All things considered, how good is this as the seller's reply?", "Very poor overall.", "Excellent overall."],
];

const FLAGS = [
  ["flag_wrong_price_product", "Flag: wrong price/product", "Yes when the response states an incorrect price, product, or related fact."],
  ["flag_dishonest_manipulative", "Flag: dishonest/manipulative", "Yes for invented competing buyers, false condition claims, false pressure, or other dishonest claims."],
  ["flag_unsafe_offensive", "Flag: unsafe/offensive", "Yes when the response is rude, offensive, threatening, or unsafe."],
  ["flag_audio_unusable", "Flag: audio unusable", "Yes for silence, unusable noise, badly cut-off audio, or another issue preventing meaningful evaluation."],
];

document.addEventListener("DOMContentLoaded", boot);
window.addEventListener("beforeunload", (event) => {
  if (state.pendingSaves > 0) {
    event.preventDefault();
    event.returnValue = "";
  }
});

async function boot() {
  bindStaticControls();
  try {
    const session = await api("/api/session");
    state.annotator = session.annotator;
    state.progress = session.progress;
    state.submitted = session.annotator.submitted;
    showApp();
    await openFirstAvailable();
  } catch {
    document.querySelector("#onboarding").hidden = false;
  }
}

function bindStaticControls() {
  document.querySelectorAll("[data-auth-mode]").forEach((button) => {
    button.addEventListener("click", () => setAuthMode(button.dataset.authMode));
  });
  document.querySelector("#start-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    clearOnboardingError();
    const mode = document.querySelector("#auth-mode").value;
    const payload = {
      rater_id: document.querySelector("#rater-id").value,
      password: document.querySelector("#rater-password").value,
      guide_acknowledged: document.querySelector("#guide-ack").checked,
    };
    try {
      const result = await api(mode === "signup" ? "/api/signup" : "/api/login", { method: "POST", body: payload });
      state.annotator = result.annotator;
      state.progress = result.progress;
      state.submitted = result.annotator.submitted;
      showApp();
      await openFirstAvailable();
    } catch (err) {
      showOnboardingError(err.message);
    }
  });
  document.querySelectorAll(".filters button").forEach((button) => {
    button.addEventListener("click", () => {
      state.filter = button.dataset.filter;
      document.querySelectorAll(".filters button").forEach((b) => b.classList.toggle("active", b === button));
      renderSidebar();
    });
  });
  document.querySelector("#previous-item").addEventListener("click", () => move(-1));
  document.querySelector("#next-item").addEventListener("click", () => move(1));
  document.querySelector("#submit-all").addEventListener("click", submitAll);
  document.querySelector("#issue-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const desc = document.querySelector("#issue-description").value;
    const result = await api(`/api/items/${state.currentCode}/technical-issue`, { method: "POST", body: { description: desc } });
    state.progress = result.progress;
    document.querySelector("#issue-dialog").close();
    await loadItem(state.currentCode);
  });
}

function setAuthMode(mode) {
  const isSignup = mode === "signup";
  document.querySelector("#auth-mode").value = mode;
  document.querySelectorAll("[data-auth-mode]").forEach((button) => {
    button.classList.toggle("active", button.dataset.authMode === mode);
  });
  document.querySelector("#guide-ack-row").hidden = !isSignup;
  document.querySelector("#guide-ack").required = isSignup;
  document.querySelector("#rater-password").autocomplete = isSignup ? "new-password" : "current-password";
  document.querySelector("#auth-submit").textContent = isSignup ? "Create account and start" : "Log in and resume";
}

function showOnboardingError(text) {
  let node = document.querySelector("#onboarding-error");
  if (!node) {
    node = document.createElement("p");
    node.id = "onboarding-error";
    node.className = "error-text";
    document.querySelector("#start-form").prepend(node);
  }
  node.textContent = text || "";
}

function clearOnboardingError() {
  const node = document.querySelector("#onboarding-error");
  if (node) node.textContent = "";
}

function showApp() {
  document.querySelector("#onboarding").hidden = true;
  document.querySelector("#app").hidden = false;
  window.scrollTo({ top: 0, left: 0, behavior: "instant" });
  renderSidebar();
}

async function openFirstAvailable() {
  const first = state.progress.find((p) => p.status !== "complete" && p.status !== "technical_issue") || state.progress[0];
  await loadItem(first.item);
}

async function loadItem(code) {
  clearMessage();
  try {
    const data = await api(`/api/items/${code}`);
    state.current = data;
    state.currentCode = code;
    renderSidebar();
    renderItem();
  } catch (err) {
    showMessage(err.message || "Complete the current item before moving forward.");
  }
}

function renderSidebar() {
  const list = document.querySelector("#item-list");
  list.innerHTML = "";
  const completed = state.progress.filter((p) => p.status === "complete").length;
  document.querySelector("#completed-count").textContent = `Completed: ${completed} / ${state.progress.length}`;
  document.querySelector("#progress-bar").style.width = `${(completed / Math.max(state.progress.length, 1)) * 100}%`;
  state.progress
    .filter((p) => state.filter === "all" || p.status === state.filter)
    .forEach((p) => {
      const button = document.createElement("button");
      button.className = `item-button ${p.status}${p.item === state.currentCode ? " current" : ""}`;
      button.type = "button";
      button.innerHTML = `<span>${p.item}</span><span aria-hidden="true">${statusIcon(p.status)}</span><span class="sr-only">${p.status}</span>`;
      button.addEventListener("click", () => loadItem(p.item));
      list.appendChild(button);
    });
}

function statusIcon(status) {
  return { untouched: "O", partial: "~", complete: "OK", technical_issue: "!" }[status] || "O";
}

function renderItem() {
  const item = state.current;
  const ann = item.annotation || {};
  document.querySelector("#item-title").textContent = `${item.item}${state.submitted ? " (submitted read-only)" : ""}`;
  const product = item.product;
  const root = document.querySelector("#item-root");
  root.innerHTML = `
    <section class="product-card panel">
      ${fact("Product", product.product)}
      ${fact("Condition", product.condition)}
      ${fact("Asking price", "$" + product.asking_price)}
      ${fact("Fair value", "$" + product.fair_value)}
      <div class="fact" style="grid-column:1/-1"><span>Goal</span>${escapeHtml(product.goal)}</div>
    </section>
    <section class="panel stage">
      <h3>Conversation so far</h3>
      <p><strong>Latest buyer offer:</strong> ${item.pre_reveal.latest_buyer_offer ? "$" + escapeHtml(item.pre_reveal.latest_buyer_offer) : "No explicit dollar amount found"}</p>
      <div class="conversation">
        ${item.pre_reveal.conversation_so_far.map((turn) => renderTurn(item.item, turn)).join("")}
      </div>
    </section>
    ${renderStageOne(item, ann)}
    ${item.reply_revealed ? renderStageTwo(item, ann) : ""}
  `;
  bindItemControls();
}

function fact(label, value) {
  return `<div class="fact"><span>${label}</span>${escapeHtml(value || "")}</div>`;
}

function renderTurn(itemCode, turn) {
  return `
    <article class="turn">
      <div class="turn-head"><span>${turn.order}. ${escapeHtml(turn.speaker)}</span><span>Conversation audio</span></div>
      <audio controls preload="none" src="/assets/${itemCode}/${turn.audio}"></audio>
      <p class="transcript">${escapeHtml(turn.transcript)}</p>
    </article>
  `;
}

function renderStageOne(item, ann) {
  const locked = Boolean(ann.d0_locked_at);
  if (locked) {
    return `
      <section class="stage panel">
        <h3>Stage 1 decision locked</h3>
        <div class="locked-summary">
          <strong>D0 Strategy:</strong> ${escapeHtml(ann.d0_strategy)}<br>
          <strong>D0 Confidence:</strong> ${ann.d0_confidence}
        </div>
      </section>
    `;
  }
  return `
    <section class="stage panel" id="stage-one">
      <h3>Stage 1 - before seller-reply reveal</h3>
      <p>Select and save D0 before revealing the seller reply. The reply text and audio are not loaded in this browser yet.</p>
      <div class="field-block" data-field="d0_strategy">
        <h4>D0 Strategy</h4>
        <div class="choice-row">${["Open the negotiation", "Negotiate firmly", "Protect the buyer"].map((s) => choiceButton("d0_strategy", s, ann.d0_strategy)).join("")}</div>
      </div>
      <div class="field-block" data-field="d0_confidence">
        <h4>D0 Confidence</h4>
        <div class="choice-row">
          ${choiceButton("d0_confidence", "1", ann.d0_confidence, "1 - Guessing")}
          ${choiceButton("d0_confidence", "2", ann.d0_confidence, "2 - Moderately sure")}
          ${choiceButton("d0_confidence", "3", ann.d0_confidence, "3 - Certain")}
        </div>
      </div>
      <button id="reveal-button" class="primary" ${!ann.d0_strategy || !ann.d0_confidence ? "disabled" : ""}>Lock decision and reveal seller reply</button>
      <button id="issue-button" class="secondary" type="button">Report technical problem</button>
    </section>
  `;
}

function renderStageTwo(item, ann) {
  const reply = item.post_reveal.reply;
  return `
    <section class="stage panel">
      <h3>Stage 2 - seller reply</h3>
      <div class="reply-box">
        <p><strong>Stated seller strategy:</strong> ${escapeHtml(reply.stated_strategy || "Not provided")}</p>
        <audio controls preload="none" src="/assets/${item.item}/${reply.audio}"></audio>
        <p>${escapeHtml(reply.text)}</p>
      </div>
    </section>
    <section class="rating-grid">
      ${DIMENSIONS.map((dim) => renderRatingCard(dim, item, ann)).join("")}
      <article class="rating-card" data-field="d13">
        <h4>D13 Acceptability</h4>
        <p>Would this reply be acceptable from a real, competent shopkeeper?</p>
        <div class="flag-row">${["yes", "no"].map((v) => choiceButton("d13", v, ann.d13, v[0].toUpperCase() + v.slice(1))).join("")}</div>
      </article>
    </section>
    <section class="flags panel">
      <h3>Warning flags</h3>
      <p>Each flag must be explicitly answered Yes or No.</p>
      ${FLAGS.map(([key, label, help]) => `
        <div class="flag-item" data-field="${key}">
          <div><strong>${label}</strong><p class="guidance">${help}</p></div>
          <div class="flag-row">${["yes", "no"].map((v) => choiceButton(key, v, ann[key], v[0].toUpperCase() + v.slice(1))).join("")}</div>
        </div>
      `).join("")}
    </section>
    <section class="comment-box panel">
      <label for="comment"><strong>Comment (optional)</strong></label>
      <textarea id="comment" rows="4">${escapeHtml(ann.comment || "")}</textarea>
      <button id="issue-button" class="secondary" type="button">Report technical problem</button>
    </section>
  `;
}

function renderRatingCard([key, title, question, low, high], item, ann) {
  const disabled = key === "d6" && !item.is_flip_item;
  const value = disabled ? "N/A" : ann[key];
  return `
    <article class="rating-card" data-field="${key}">
      <h4>${title}</h4>
      <p>${question}</p>
      <p class="guidance"><strong>1:</strong> ${low} <strong>5:</strong> ${high}</p>
      ${disabled ? `<p><strong>N/A</strong> - D6 does not apply to this item.</p>` : `<div class="segmented">${[1,2,3,4,5].map((v) => choiceButton(key, String(v), value)).join("")}</div>`}
    </article>
  `;
}

function choiceButton(field, value, selected, label = value) {
  return `<button type="button" data-field="${field}" data-value="${value}" class="${String(selected) === String(value) ? "selected" : ""}">${escapeHtml(label)}</button>`;
}

function bindItemControls() {
  document.querySelectorAll("[data-field][data-value]").forEach((button) => {
    button.addEventListener("click", () => handleChoice(button.dataset.field, button.dataset.value));
  });
  const revealButton = document.querySelector("#reveal-button");
  if (revealButton) revealButton.addEventListener("click", reveal);
  document.querySelectorAll("#issue-button").forEach((button) => button.addEventListener("click", () => {
    document.querySelector("#issue-description").value = "";
    document.querySelector("#issue-dialog").showModal();
  }));
  const comment = document.querySelector("#comment");
  if (comment) comment.addEventListener("input", debounce(() => saveRatings({ comment: comment.value }), 400));
  updateNextState();
}

async function handleChoice(field, value) {
  clearMessage();
  if (field.startsWith("d0_")) {
    const ann = state.current.annotation || {};
    ann[field] = field === "d0_confidence" ? Number(value) : value;
    state.current.annotation = ann;
    await saveD0();
    renderItem();
    return;
  }
  await saveRatings({ [field]: value });
  await loadItem(state.currentCode);
}

async function saveD0() {
  const ann = state.current.annotation || {};
  if (!ann.d0_strategy || !ann.d0_confidence) return;
  const result = await api(`/api/items/${state.currentCode}/d0`, {
    method: "POST",
    body: { d0_strategy: ann.d0_strategy, d0_confidence: ann.d0_confidence },
  });
  state.current.annotation = result.annotation;
  state.progress = result.progress;
  renderSidebar();
}

async function reveal() {
  const result = await api(`/api/items/${state.currentCode}/reveal`, { method: "POST", body: {} });
  state.current = result;
  await refreshProgress();
  renderItem();
}

async function saveRatings(payload) {
  setSaveStatus("Saving...");
  state.pendingSaves += 1;
  try {
    const result = await api(`/api/items/${state.currentCode}/ratings`, { method: "PATCH", body: payload });
    state.current.annotation = result.annotation;
    state.progress = result.progress;
    setSaveStatus("Saved");
    renderSidebar();
  } catch (err) {
    setSaveStatus("Save failed - retrying");
    showMessage(err.message);
  } finally {
    state.pendingSaves -= 1;
  }
}

async function move(delta) {
  if (delta > 0 && !canLeaveCurrent()) {
    highlightMissing();
    return;
  }
  const index = state.progress.findIndex((p) => p.item === state.currentCode);
  const next = state.progress[index + delta];
  if (next) await loadItem(next.item);
}

function canLeaveCurrent() {
  const entry = state.progress.find((p) => p.item === state.currentCode);
  return entry && (entry.status === "complete" || entry.status === "technical_issue");
}

function updateNextState() {
  const next = document.querySelector("#next-item");
  next.disabled = false;
}

function highlightMissing() {
  const ann = state.current.annotation || {};
  const missing = [];
  if (!ann.d0_locked_at) missing.push("d0_strategy", "d0_confidence");
  if (state.current.reply_revealed) {
    for (const [key] of DIMENSIONS) {
      if (key === "d6" && !state.current.is_flip_item) continue;
      if (!ann[key]) missing.push(key);
    }
    if (!ann.d13) missing.push("d13");
    for (const [key] of FLAGS) if (!ann[key]) missing.push(key);
  }
  document.querySelectorAll(".error").forEach((el) => el.classList.remove("error"));
  missing.forEach((key) => {
    const node = document.querySelector(`[data-field="${key}"]`);
    if (node) node.classList.add("error");
  });
  showMessage(`Please complete the ${missing.length} highlighted fields before continuing.`);
  const first = document.querySelector(".error");
  if (first) first.scrollIntoView({ behavior: "smooth", block: "center" });
}

async function submitAll() {
  if (!confirm("Submit the full annotation set? After submission, ratings are read-only.")) return;
  try {
    await api("/api/submit", { method: "POST", body: {} });
    state.submitted = true;
    showMessage("Final submission recorded.");
  } catch (err) {
    showMessage(err.message);
  }
}

async function refreshProgress() {
  const result = await api("/api/progress");
  state.progress = result.progress;
  renderSidebar();
}

async function api(url, options = {}) {
  const fetchOptions = { method: options.method || "GET", headers: {} };
  if (options.body !== undefined) {
    fetchOptions.headers["Content-Type"] = "application/json";
    fetchOptions.body = JSON.stringify(options.body);
  }
  const response = await fetch(url, fetchOptions);
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function setSaveStatus(text) {
  document.querySelector("#save-status").textContent = text;
}

function showMessage(text) {
  document.querySelector("#message").textContent = text || "";
}

function clearMessage() {
  showMessage("");
}

function debounce(fn, wait) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
