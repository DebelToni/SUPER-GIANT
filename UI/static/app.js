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
// ------------------------------------------------------------
// Replace your previous listen() with this version
// ------------------------------------------------------------
async function listen(jobId) {
  // Ask ngrok to skip the warning page
  const res = await fetch(`${apiBase}/events/${jobId}`, {
    headers: { "ngrok-skip-browser-warning": "true" }
  });

  // Plain-text streaming reader
  const reader   = res.body.getReader();
  const decoder  = new TextDecoder();
  let   buf      = "";
  let   evtData  = "";          // accumulates multiline data: fields
  let   evtType  = "message";   // default SSE event type

  cancel.disabled = false;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buf += decoder.decode(value, { stream: true });

    // Process one line at a time (SSE is line-delimited UTF-8)
    let nl;
    while ((nl = buf.indexOf("\n")) !== -1) {
      const line = buf.slice(0, nl).trimEnd(); // remove trailing \r
      buf        = buf.slice(nl + 1);          // remainder

      if (line === "") {
        // Blank line finishes the current event ------------- //
        if (evtType === "done") {
          cancel.disabled = true;
          return;             // finished, close stream
        }

        if (evtData) {
          print(evtData + "\n");
          evtData = "";
        }
        evtType = "message";  // reset for next event
        continue;
      }

      // Ignore curl-style progress or anything that isn't SSE
      if (line.startsWith("% ") || /^\d+\s+\d+/.test(line)) continue;

      // Parse a field --------------------------------------- //
      if (line.startsWith("data:")) {
        evtData += line.slice(5).trimStart() + "\n";
      } else if (line.startsWith("event:")) {
        evtType = line.slice(6).trimStart();
      }
      // You can add handling for id:, retry:, etc. if needed
    }
  }

  cancel.disabled = true;      // safety
}



// Cancel --------------------------------------------------------------------
cancel.addEventListener("click", async () => {
  if (!currentJob) return;
  await fetch(`${apiBase}/command/${currentJob}/cancel`, {method:"POST"});
});

