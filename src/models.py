from dataclasses import dataclass


@dataclass
class OCRWord:
    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int
    block_num: int
    par_num: int
    line_num: int
    word_num: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


@dataclass
class TextBlock:
    page_index: int
    block_id: int
    source_text: str
    translated_text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float = 0.0
    source_line_height: float = 0.0
    color_rgb: tuple[int, int, int] | None = None
    is_heading: bool = False

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height
