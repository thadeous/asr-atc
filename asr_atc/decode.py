from typing import Dict, List


def prepare_vocab_for_decode(ordered_vocab: List) -> List[str]:
    # create a vocabulary expected by the ctc decoder
    labels = [k for k, v in ordered_vocab]
    labels = [k if k != '|' else ' ' for k in labels]
    labels = [k if k != '<s>' else '<' for k in labels]
    labels = [k if k != '</s>' else '>' for k in labels]
    labels = [k if k != '<unk>' else '*' for k in labels]
    labels = [k if k != '<pad>' else '' for k in labels]
    return labels
