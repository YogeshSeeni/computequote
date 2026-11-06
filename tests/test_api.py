from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

SCENARIO = {
    "model": "llama-3-70b",
    "gpu": "h100-sxm-80gb",
    "n_gpus": 1024,
    "seq_len": 8192,
    "global_batch_tokens": 8 * 2**20,
    "total_tokens": 15 * 10**12,
}


def test_healthz() -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


def test_presets() -> None:
    body = client.get("/v1/presets").json()
    assert {m["slug"] for m in body["models"]} >= {"llama-3-70b", "llama-2-7b"}
    assert any(g["slug"] == "h100-sxm-80gb" for g in body["gpus"])
    assert body["models"][0]["spec"]["n_params"] > 0


def test_prices_filter_and_staleness_fields() -> None:
    body = client.get("/v1/prices", params={"gpu": "a100-sxm-80gb"}).json()
    assert body["observations"]
    assert all(o["gpu_slug"] == "a100-sxm-80gb" for o in body["observations"])
    assert {"is_stale", "price_usd_per_gpu_hour", "source_url"} <= set(body["observations"][0])


def test_estimate_train_happy_path() -> None:
    r = client.post("/v1/estimate/train", json=SCENARIO)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["best"]["feasible"] is True
    assert body["best"]["perf"]["step_time_s"] > 0
    assert body["cost"]["gpu_hours"] > 0
    assert body["cost"]["by_provider"]
    assert body["cost"]["min_cost_usd"] == body["cost"]["by_provider"][0]["cost_usd"]
    assert body["job"]["n_params"] > 70e9
    assert "model=llama-3-70b" in body["scenario_query"]
    assert body["search_time_ms"] < 1000
    assert len(body["alternatives"]) <= 4


def test_estimate_train_custom_model_spec() -> None:
    custom = dict(SCENARIO)
    custom["model"] = {
        "name": "custom 1B",
        "n_layers": 16,
        "d_model": 2048,
        "n_heads": 16,
        "n_kv_heads": 4,
        "d_ff": 8192,
        "vocab_size": 32000,
    }
    custom["n_gpus"] = 8
    r = client.post("/v1/estimate/train", json=custom)
    assert r.status_code == 200, r.text
    assert r.json()["job"]["model_slug"] is None


def test_unknown_preset_is_404() -> None:
    assert client.post("/v1/estimate/train", json={**SCENARIO, "model": "nope"}).status_code == 404
    assert client.post("/v1/estimate/train", json={**SCENARIO, "gpu": "nope"}).status_code == 404


def test_bad_batch_is_422() -> None:
    r = client.post("/v1/estimate/train", json={**SCENARIO, "global_batch_tokens": 8193})
    assert r.status_code == 422


def test_configs_search() -> None:
    r = client.post("/v1/configs/search", json={**SCENARIO, "top_k": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 < len(body["feasible"]) <= 3
    assert all(c["reason"] for c in body["infeasible"])
