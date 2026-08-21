from dataclasses import dataclass

from src.tools.firefox_reader import read_url_text


@dataclass
class MinecraftTopic:
    name: str
    url: str


BEGINNER_TOPICS = {
    "day1": MinecraftTopic("First Day Survival", "https://minecraft.wiki/w/Tutorials/Beginner%27s_guide"),
    "crafting": MinecraftTopic("Crafting Basics", "https://minecraft.wiki/w/Crafting"),
    "mining": MinecraftTopic("Mining", "https://minecraft.wiki/w/Tutorials/Mining"),
    "food": MinecraftTopic("Hunger and Food", "https://minecraft.wiki/w/Hunger"),
    "shelter": MinecraftTopic("Shelter", "https://minecraft.wiki/w/Tutorials/Shelters"),
}


def list_beginner_topics() -> list[str]:
    return list(BEGINNER_TOPICS.keys())


def fetch_minecraft_topic(topic_key: str, timeout_ms: int = 15_000, max_chars: int = 4_000) -> str:
    topic = BEGINNER_TOPICS.get(topic_key)
    if not topic:
        valid = ", ".join(sorted(BEGINNER_TOPICS.keys()))
        raise ValueError(f"Unknown topic '{topic_key}'. Valid topics: {valid}")
    return read_url_text(topic.url, timeout_ms=timeout_ms, max_chars=max_chars)
