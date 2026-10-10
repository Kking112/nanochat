"""Display-only streaming think-tag styling; conversation tokens stay unchanged."""

from nanochat.reasoning import THINK_END, THINK_START


class ThinkRenderer:
    def __init__(self, interactive: bool) -> None:
        self.interactive = interactive
        self.pending = ""
        self.thinking = False

    def feed(self, chunk: str) -> str:
        if not self.interactive:
            return chunk
        self.pending += chunk
        output: list[str] = []
        while self.pending:
            tag = THINK_END if self.thinking else THINK_START
            if self.pending.startswith(tag):
                self.pending = self.pending[len(tag):]
                self.thinking = not self.thinking
                output.append("\033[2m" + tag if self.thinking else tag + "\033[0m")
            elif tag.startswith(self.pending):
                break  # The next streamed chunk may complete this delimiter.
            else:
                output.append(self.pending[0])
                self.pending = self.pending[1:]
        return "".join(output)

    def finish(self) -> str:
        result = self.pending + ("\033[0m" if self.thinking else "")
        self.pending = ""
        self.thinking = False
        return result
