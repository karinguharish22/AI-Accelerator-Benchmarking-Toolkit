import asyncio
import json

import httpx
import pytest

from benchmarks.load_test import RequestResult, parse_sse_line, percentile, send_request, summarize


def test_parse_sse_line():
    assert parse_sse_line('data: {"a": 1}') == {"a": 1}
    assert parse_sse_line("data: [DONE]") is None
    assert parse_sse_line(": keep-alive") is None
    assert parse_sse_line("") is None


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([5], 99) == 5
    assert percentile([], 50) is None


def test_tpot_excludes_first_token():
    r = RequestResult(ok=True, ttft_s=0.1, e2e_s=1.1, output_tokens=11)
    assert r.tpot_s == pytest.approx(0.1)
    assert RequestResult(ok=True, ttft_s=0.1, e2e_s=0.2, output_tokens=1).tpot_s is None


def test_summarize_counts_failures_and_throughput():
    results = [
        RequestResult(ok=True, ttft_s=0.1, e2e_s=1.0, output_tokens=100, prompt_tokens=50),
        RequestResult(ok=True, ttft_s=0.3, e2e_s=2.0, output_tokens=100, prompt_tokens=50),
        RequestResult(ok=False, error="boom"),
    ]
    s = summarize(results, concurrency=2, wall_s=2.0)
    assert s["succeeded"] == 2 and s["failed"] == 1
    assert s["output_tokens_per_s"] == pytest.approx(100.0)
    assert s["ttft_ms_p50"] == pytest.approx(200.0)
    assert s["mean_prompt_tokens"] == 50


def _sse(events):
    lines = [f"data: {json.dumps(e)}\n\n" for e in events] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


def test_send_request_against_fake_streaming_server():
    def handler(request):
        body = json.loads(request.content)
        assert body["stream"] is True and body["ignore_eos"] is True
        events = [{"choices": [{"text": "tok"}]} for _ in range(5)]
        events.append({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 5}})
        return httpx.Response(200, content=_sse(events), headers={"content-type": "text/event-stream"})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await send_request(client, "http://fake/v1/completions", "m", "hello", 5, 10)

    r = asyncio.run(go())
    assert r.ok
    assert r.output_tokens == 5 and r.prompt_tokens == 12
    assert r.ttft_s is not None and r.e2e_s >= r.ttft_s


def test_send_request_reports_http_errors():
    async def go():
        transport = httpx.MockTransport(lambda req: httpx.Response(503, text="overloaded"))
        async with httpx.AsyncClient(transport=transport) as client:
            return await send_request(client, "http://fake/v1/completions", "m", "hello", 5, 10)

    r = asyncio.run(go())
    assert not r.ok and "503" in r.error
