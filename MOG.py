import asyncio
import base64
import binascii
import contextlib
import logging
import os
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from jito_searcher_client import get_async_searcher_client
from jito_searcher_client.convert import versioned_tx_to_protobuf_packet
from jito_searcher_client.generated.bundle_pb2 import Bundle
from jito_searcher_client.generated.searcher_pb2 import (
    SendBundleRequest,
    SubscribeBundleResultsRequest,
)
from jito_searcher_client.generated.searcher_pb2_grpc import SearcherServiceStub
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction

log = logging.getLogger("jito-gateway")

LANDED_OUTCOMES = {"processed", "finalized"}
FAILED_OUTCOMES = {"rejected", "dropped"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JITO_")

    block_engine_url: str = "amsterdam.mainnet.block-engine.jito.wtf"
    keypair_path: str = "~/.config/solana/id.json"
    cache_size: int = 10_000
    sent_capacity: int = 10_000


class BundleResultCache:
    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._items: OrderedDict[str, str] = OrderedDict()

    def put(self, bundle_id: str, kind: str) -> None:
        self._items[bundle_id] = kind
        self._items.move_to_end(bundle_id)
        while len(self._items) > self._capacity:
            self._items.popitem(last=False)

    def get(self, bundle_id: str) -> str | None:
        kind = self._items.get(bundle_id)
        if kind is not None:
            self._items.move_to_end(bundle_id)
        return kind


class LandTracker:
    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._sent: OrderedDict[str, str | None] = OrderedDict()
        self.sent = 0
        self.landed = 0
        self.failed = 0

    def track(self, bundle_id: str) -> None:
        self.sent += 1
        self._sent[bundle_id] = None
        self._sent.move_to_end(bundle_id)
        while len(self._sent) > self._capacity:
            self._sent.popitem(last=False)

    def observe(self, bundle_id: str, outcome: str) -> None:
        if self._sent.get(bundle_id, "") is not None:
            return
        if outcome in LANDED_OUTCOMES:
            self._sent[bundle_id] = outcome
            self.landed += 1
        elif outcome in FAILED_OUTCOMES:
            self._sent[bundle_id] = outcome
            self.failed += 1

    def rate(self) -> float:
        return self.landed / self.sent if self.sent else 0.0


async def consume_results(
    client: SearcherServiceStub, cache: BundleResultCache, lands: LandTracker
) -> None:
    while True:
        try:
            stream = client.SubscribeBundleResults(SubscribeBundleResultsRequest())
            async for result in stream:
                outcome = result.WhichOneof("result") or "none"
                cache.put(result.bundle_id, outcome)
                lands.observe(result.bundle_id, outcome)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("bundle result stream failed, retrying")
            await asyncio.sleep(2)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    with open(os.path.expanduser(settings.keypair_path)) as f:
        keypair = Keypair.from_json(f.read())
    client = await get_async_searcher_client(settings.block_engine_url, keypair)
    cache = BundleResultCache(settings.cache_size)
    lands = LandTracker(settings.sent_capacity)

    app.state.client = client
    app.state.results = cache
    app.state.lands = lands
    watcher = asyncio.create_task(consume_results(client, cache, lands))
    try:
        yield
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher

app = FastAPI(title="Jito gateway", lifespan=lifespan)


def get_client(request: Request) -> SearcherServiceStub:
    return request.app.state.client


def get_lands(request: Request) -> LandTracker:
    return request.app.state.lands


class BundleIn(BaseModel):
    txs: list[str] = Field(min_length=1, max_length=5, description="base64 transactions")


class BundleOut(BaseModel):
    bundle_id: str


class LandRateOut(BaseModel):
    sent: int
    landed: int
    failed: int
    rate: float


@app.post("/bundles", response_model=BundleOut)
async def submit_bundle(
    body: BundleIn,
    client: SearcherServiceStub = Depends(get_client),
    lands: LandTracker = Depends(get_lands),
) -> BundleOut:
    try:
        txs = [VersionedTransaction.from_bytes(base64.b64decode(t, validate=True)) for t in body.txs]
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(422, f"invalid transaction: {exc}") from exc

    bundle = Bundle(header=None, packets=[versioned_tx_to_protobuf_packet(tx) for tx in txs])
    response = await client.SendBundle(SendBundleRequest(bundle=bundle))
    lands.track(response.uuid)
    return BundleOut(bundle_id=response.uuid)


@app.get("/bundles/{bundle_id}")
async def bundle_status(bundle_id: str, request: Request) -> dict[str, str]:
    kind = request.app.state.results.get(bundle_id)
    if kind is None:
        raise HTTPException(404, "unknown bundle")
    return {"bundle_id": bundle_id, "result": kind}


@app.get("/land-rate", response_model=LandRateOut)
async def land_rate(lands: LandTracker = Depends(get_lands)) -> LandRateOut:
    return LandRateOut(sent=lands.sent, landed=lands.landed, failed=lands.failed, rate=lands.rate())
