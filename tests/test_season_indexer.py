"""Tests for SeasonIndexer class."""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from custom_components.discrete_state_forecaster.model.time_indexers.season_indexer import (
    SeasonIndexer,
)


class TestSeasonIndexerInitialization:
    @pytest.mark.asyncio
    async def test_initialization(self) -> None:
        indexer = SeasonIndexer()
        assert indexer.name == "season"


class TestSeasonIndexerKey:
    @pytest.mark.asyncio
    async def test_spring(self) -> None:
        indexer = SeasonIndexer()
        ts = datetime(2026, 3, 15, 12, 0, 0)
        assert await indexer.key(ts) == "spring"

    @pytest.mark.asyncio
    async def test_summer(self) -> None:
        indexer = SeasonIndexer()
        ts = datetime(2026, 6, 15, 12, 0, 0)
        assert await indexer.key(ts) == "summer"

    @pytest.mark.asyncio
    async def test_autumn(self) -> None:
        indexer = SeasonIndexer()
        ts = datetime(2026, 9, 15, 12, 0, 0)
        assert await indexer.key(ts) == "autumn"

    @pytest.mark.asyncio
    async def test_winter_december(self) -> None:
        indexer = SeasonIndexer()
        ts = datetime(2026, 12, 15, 12, 0, 0)
        assert await indexer.key(ts) == "winter"

    @pytest.mark.asyncio
    async def test_winter_january(self) -> None:
        indexer = SeasonIndexer()
        ts = datetime(2026, 1, 15, 12, 0, 0)
        assert await indexer.key(ts) == "winter"

    @pytest.mark.asyncio
    async def test_leap_year_february(self) -> None:
        indexer = SeasonIndexer()
        ts = datetime(2024, 2, 29, 12, 0, 0)
        assert await indexer.key(ts) == "winter"


class TestSeasonIndexerNextBoundary:
    @pytest.mark.asyncio
    async def test_next_boundary_from_february(self) -> None:
        indexer = SeasonIndexer()
        ts = datetime(2026, 2, 15, 10, 30, 0)
        next_bound = await indexer.next_boundary(ts)
        assert next_bound == datetime(2026, 3, 1, 0, 0, 0)

    @pytest.mark.asyncio
    async def test_next_boundary_from_may(self) -> None:
        indexer = SeasonIndexer()
        ts = datetime(2026, 5, 31, 23, 59, 59)
        next_bound = await indexer.next_boundary(ts)
        assert next_bound == datetime(2026, 6, 1, 0, 0, 0)

    @pytest.mark.asyncio
    async def test_next_boundary_from_december(self) -> None:
        indexer = SeasonIndexer()
        ts = datetime(2026, 12, 15, 8, 0, 0)
        next_bound = await indexer.next_boundary(ts)
        assert next_bound == datetime(2027, 3, 1, 0, 0, 0)

    @pytest.mark.asyncio
    async def test_next_boundary_preserves_tzinfo(self) -> None:
        indexer = SeasonIndexer()
        tz = ZoneInfo("Europe/Budapest")
        ts = datetime(2026, 5, 31, 23, 30, 0, tzinfo=tz)
        next_bound = await indexer.next_boundary(ts)
        assert next_bound.tzinfo == tz
        assert next_bound == datetime(2026, 6, 1, 0, 0, 0, tzinfo=tz)
