# 60-second demo script

Set-up before recording: run `scripts\setup.ps1`, wear headphones, and open two
windows: a terminal in the repo with the venv active, and a browser. Run
`vae serve` once beforehand so the models are cached.

| Time | On screen | Say (voice-over) |
|---|---|---|
| 0:00–0:08 | README title | "Voice agents are hard to evaluate before they go live. This is voice-agent-evals: a local customer-service agent and a test harness for it. Everything runs on this laptop." |
| 0:08–0:22 | Terminal: `vae run` scrolling PASS/FAIL, then the summary table | "`vae run` plays 25 simulated calls: billing disputes, outages, prompt injection, angry callers, refunds over the limit. Each one is scored on task completion, escalation precision and recall, tool calls, and unsupported claims." |
| 0:22–0:32 | Browser: `results/report.html`, click the failing `ambiguous_request` row to expand the transcript | "Here's a real failure it caught. The caller says the bill went up, and the agent reads the total but never explains the new add-on." |
| 0:32–0:50 | Terminal: `vae serve`; browser at http://127.0.0.1:8000, click **Start call**. Say: "Is there an outage at S E 5, 9 Z Z?" While the agent answers, interrupt: "Actually, I want to speak to a person." | "Same agent, now by voice: local speech-to-text, local text-to-speech. I can talk over it. Barge-in stops playback straight away. When I ask for a human, it escalates." |
| 0:50–0:60 | The red **Escalated to human** panel with the handoff summary | "The human gets a summary, so the caller doesn't repeat themselves. It's MIT-licensed, Python only, no cloud. Add your own scenarios in YAML." |

Tips: if the postcode is misheard, type it in the text box instead, or set
`VAE_STT_MODEL=base.en`. Keep the mouse movements slow and zoom the browser to
125% so the text is readable.
