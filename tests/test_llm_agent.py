

def test_chat_client_caps_reply_length():
    from voice_agent_evals.agent.llm import ChatClient

    sent = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    client = ChatClient("http://x/v1", "k", "m", max_tokens=123)
    client._http.post = lambda url, json: (sent.update(json), _Resp())[1]
    client.chat([{"role": "user", "content": "hi"}])
    assert sent["max_tokens"] == 123
