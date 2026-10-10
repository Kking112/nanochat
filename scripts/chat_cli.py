"""
New and upgraded chat mode because a lot of the code has changed since the last one.

Intended to be run single GPU only atm:
python -m scripts.chat_cli
"""
import argparse
import sys
from collections.abc import Sequence

from nanochat.checkpoint_manager import CHECKPOINT_SOURCES, load_model
from nanochat.common import autodetect_device_type, compute_init
from nanochat.engine import Engine
from nanochat.reasoning_display import ThinkRenderer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Chat with the model')
    parser.add_argument('-i', '--source', type=str, default="sft", choices=tuple(CHECKPOINT_SOURCES), help="Checkpoint source")
    parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
    parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
    parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt the model, get a single response back')
    parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
    parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Completion budget (reasoning: 1024; other sources: 256)")
    return parser

def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_new_tokens is None:
        args.max_new_tokens = 1024 if args.source.startswith("reason_") else 256
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    # Init the model and tokenizer

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _ddp, _ddp_rank, _ddp_local_rank, _ddp_world_size, device = compute_init(device_type)
    model, tokenizer, _meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

    # Special tokens for the chat state machine
    bos = tokenizer.get_bos_token_id()
    user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
    assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

    # Create Engine for efficient generation
    engine = Engine(model, tokenizer)

    print("\nNanoChat Interactive Mode")
    print("-" * 50)
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to start a new conversation")
    print("-" * 50)

    conversation_tokens = [bos]

    while True:

        if args.prompt:
            # Get the prompt from the launch command
            user_input = args.prompt
        else:
            # Get the prompt interactively from the console
            try:
                user_input = input("\nUser: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

        # Handle special commands
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if user_input.lower() == 'clear':
            conversation_tokens = [bos]
            print("Conversation cleared.")
            continue

        if not user_input:
            continue

        next_prompt = [user_start, *tokenizer.encode(user_input), user_end, assistant_start]
        if len(conversation_tokens) + len(next_prompt) >= model.config.sequence_len:
            print("Conversation exceeds the model context. Use 'clear' or a shorter prompt.")
            if args.prompt:
                break
            continue

        # Add User message to the conversation
        conversation_tokens.append(user_start)
        conversation_tokens.extend(tokenizer.encode(user_input))
        conversation_tokens.append(user_end)

        # Kick off the assistant
        conversation_tokens.append(assistant_start)
        generate_kwargs = {
            "num_samples": 1,
            "max_tokens": min(args.max_new_tokens, model.config.sequence_len - len(conversation_tokens)),
            "temperature": args.temperature,
            "top_k": args.top_k,
        }
        response_tokens = []
        renderer = ThinkRenderer(sys.stdout.isatty())
        print("\nAssistant: ", end="", flush=True)
        for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
            token = token_column[0] # pop the batch dimension (num_samples=1)
            response_tokens.append(token)
            token_text = tokenizer.decode([token])
            print(renderer.feed(token_text), end="", flush=True)
            if token in (assistant_end, bos):
                break
        print(renderer.finish())
        # we have to ensure that the assistant end token is the last token
        # so even if generation ends due to max tokens, we have to append it to the end
        if not response_tokens or response_tokens[-1] not in (assistant_end, bos):
            response_tokens.append(assistant_end)
        conversation_tokens.extend(response_tokens)

        # In the prompt mode, we only want a single response and exit
        if args.prompt:
            break


if __name__ == "__main__":
    main()
