from pathlib import Path

import pytest

from npc.cli import init_campaign
from npc.config import load_config


@pytest.fixture
def campaign(tmp_path: Path):
    campaign_dir = tmp_path / "campaign"
    init_campaign(campaign_dir)
    return campaign_dir


@pytest.fixture
def config(campaign):
    return load_config(campaign)
