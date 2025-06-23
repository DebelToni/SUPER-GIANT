//---------------------------------------------------------------
// Change this if your API host/port differs
//---------------------------------------------------------------
const apiBase = window.API_BASE || "http://localhost:8000";
// const apiBase = "http://localhost:8000";

const term   = document.querySelector("#terminal");
const form   = document.querySelector("#runForm");
const input  = document.querySelector("#cmdInput");
const cancel = document.querySelector("#cancelBtn");

let currentJob = null;
let es         = null;

// Convenience
const scrollBottom = () => term.scrollTop = term.scrollHeight;

// Render helpers
const write = (html) => { term.insertAdjacentHTML("beforeend", html); scrollBottom(); };
const echoCmd = (cmd)  => write(`<span class="cmd">$ ${cmd}\n</span>`);
const print   = (text)=> write(`<span class="out">${text}</span>`);
const replaceLast = (text) => {
  // replace last line (for \r progress bars)
  const lines = term.innerHTML.split("\n");
  lines[lines.length-1] = `<span class="out">${text}</span>`;
  term.innerHTML = lines.join("\n");
  scrollBottom();
};

// Submit --------------------------------------------------------------------
form.addEventListener("submit", async (e) => {
  e.preventDefault();
   const cmd = input.value.trim();
   if (!cmd) return;
   input.value = "";
   if (cmd.toLowerCase() === "clear") {
     term.innerHTML = "";
     return;
   }
   echoCmd(cmd);
  // kill existing stream if any
  if (es) es.close();

  const res = await fetch(`${apiBase}/command`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({cmd})
  });
  const {job_id} = await res.json();
  currentJob = job_id;
  cancel.disabled = false;
  listen(job_id);
});

// SSE -----------------------------------------------------------------------
// function listen(job_id) {
//   es = new EventSource(`${apiBase}/events/${job_id}`);
//
//   es.addEventListener("message", (evt) => {
//     if (evt.data.endsWith("\r")) {          // progress update
//       replaceLast(evt.data.replace(/\r$/, ""));
//     } else {
//       print(evt.data);
//       if (!evt.data.endsWith("\n")) print("\n");
//     }
//   });
//
//   es.addEventListener("done", () => {
//     cancel.disabled = true;
//     es.close();
//   });
// }

// swap EventSource for fetch + stream
async function listen(jobId) {
  const res = await fetch(
    `${apiBase}/events/${jobId}`,
    { headers: { "ngrok-skip-browser-warning": "true" } }
  );
  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");

  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();                // keep incomplete chunk
    for (const line of lines) print(line + "\n");
  }
  cancel.disabled = true;
}


// Cancel --------------------------------------------------------------------
cancel.addEventListener("click", async () => {
  if (!currentJob) return;
  await fetch(`${apiBase}/command/${currentJob}/cancel`, {method:"POST"});
});

