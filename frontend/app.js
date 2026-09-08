const chat = document.getElementById("chat");
const queryInput = document.getElementById("query");
const sendBtn = document.getElementById("sendBtn");
const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");

function addMsg(role, text, sources) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  const p = document.createElement("p");
  p.textContent = text;
  div.appendChild(p);
  if (sources && sources.length) {
    const s = document.createElement("div");
    s.className = "sources";
    s.textContent = "来源：" + sources.map(x => x.document + "#" + x.chunk_id).join("、");
    div.appendChild(s);
  }
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

async function send() {
  const query = queryInput.value.trim();
  if (!query) return;
  addMsg("user", query);
  queryInput.value = "";
  sendBtn.disabled = true;
  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await resp.json();
    addMsg("bot", data.answer, data.sources);
  } catch (e) {
    addMsg("bot", "请求失败：" + e.message);
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", send);
queryInput.addEventListener("keydown", e => {
  if (e.key === "Enter") send();
});

uploadBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", async () => {
  if (!fileInput.files.length) return;
  const form = new FormData();
  for (const f of fileInput.files) form.append("files", f);
  uploadBtn.disabled = true;
  uploadBtn.textContent = "上传中…";
  try {
    const resp = await fetch("/api/documents", { method: "POST", body: form });
    const data = await resp.json();
    addMsg("bot", data.error || `上传完成：新增 ${data.added_chunks} 个文本块（共 ${data.total_chunks} 块）`);
  } catch (e) {
    addMsg("bot", "上传失败：" + e.message);
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = "上传文档";
    fileInput.value = "";
  }
});
