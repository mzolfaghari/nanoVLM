import torch


class BaseCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _pad_batch(self, batch, max_length):
        batch["input_ids"] = [torch.nn.functional.pad(ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id) for ids in batch["input_ids"]]
        batch["labels"]    = [torch.nn.functional.pad(labels, (max_length - len(labels), 0), value=self.tokenizer.pad_token_id) for labels in batch["labels"]]
        batch["attention_mask"] = [torch.nn.functional.pad(attention_mask, (max_length - len(attention_mask), 0), value=0) for attention_mask in batch["attention_mask"]]

    def prepare_batch(self, batch, max_length=None):
        # 1) Handle empty
        if not batch:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}

        # 2) Drop None rows
        batch = [s for s in batch if s is not None]
        if not batch:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}

        # batch is a list of dicts, each containing "input_ids", "attention_mask", "labels", "images"
        # let's convert it to a dict of lists of tensors
        batch = {k: [item[k] for item in batch] for k in batch[0]}

        if max_length is not None:
            batch = self._discard_samples_that_are_too_long(batch, max_length)

        if len(batch["input_ids"]) == 0:
            return batch

        # Pad samples to max length
        if max_length is not None:
            max_len = max_length
        else:
            max_len = max(map(len, batch["input_ids"]))
        self._pad_batch(batch, max_len) #  dictionaries in Python are mutable and passed by reference

        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "images": batch["images"],
            "labels": torch.stack(batch["labels"]),
        }

    def _discard_samples_that_are_too_long(self, batch, max_length):
        filtered = [
            (ids, label, attn, img)
            for ids, label, attn, img in zip(batch["input_ids"], batch["labels"], batch["attention_mask"], batch["images"])
            if len(ids) <= max_length
        ]
        if not filtered:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}
        batch_token_ids, batch_labels, batch_attentions, batch_images = zip(*filtered)
        return {"input_ids": list(batch_token_ids), "labels": list(batch_labels), "attention_mask": list(batch_attentions), "images": list(batch_images)}


class VQACollator(BaseCollator):  # Visual Question Answering Collator
    def __init__(self, tokenizer, max_length):
        self.max_length = max_length
        super().__init__(tokenizer)

    def _pad_batch(self, batch, max_length):  # Reimplementing to use -100 as the pad value for labels, so that it's ignored by the loss
        batch["input_ids"] = [torch.nn.functional.pad(ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id) for ids in batch["input_ids"]]
        batch["labels"]    = [torch.nn.functional.pad(labels, (max_length - len(labels), 0), value=-100) for labels in batch["labels"]]
        batch["attention_mask"] = [torch.nn.functional.pad(attention_mask, (max_length - len(attention_mask), 0), value=0) for attention_mask in batch["attention_mask"]]

    def __call__(self, batch):
        batch = self.prepare_batch(batch, max_length=self.max_length)
        return batch


class DistillCollator(VQACollator):
    """
    Extends VQACollator with the extra fields required for online distillation.

    Extra fields added to each batch:
      - answer_mask:       BoolTensor [B, T_seq]  — True at answer-token positions
                           in the student sequence.  Used to gather student logits
                           and to mask the KD loss.
      - raw_images:        List[List[PIL.Image]]  — original images before student
                           preprocessing, passed verbatim to the teacher processor.
      - raw_answers:       List[str]              — ground-truth answer text for each
                           sample, used by the teacher to build its prompt.
      - raw_conversations: List[List[dict]]       — prompt-only message lists (no
                           answer) for each sample, used by the teacher.

    Important: ConstantLengthDataset packing must NOT be used with this collator.
    Each sample must be independent so that teacher sequence positions can be
    unambiguously aligned with student answer positions.
    """

    def _pad_batch(self, batch, max_length):
        # Pad tensors (same as VQACollator)
        batch["input_ids"] = [
            torch.nn.functional.pad(ids, (max_length - len(ids), 0),
                                    value=self.tokenizer.pad_token_id)
            for ids in batch["input_ids"]
        ]
        batch["labels"] = [
            torch.nn.functional.pad(labels, (max_length - len(labels), 0),
                                    value=-100)
            for labels in batch["labels"]
        ]
        batch["attention_mask"] = [
            torch.nn.functional.pad(am, (max_length - len(am), 0), value=0)
            for am in batch["attention_mask"]
        ]
        # answer_mask pads with False (these positions are never answer tokens)
        batch["answer_mask"] = [
            torch.nn.functional.pad(am.long(), (max_length - len(am), 0),
                                    value=0).bool()
            for am in batch["answer_mask"]
        ]

    def _discard_samples_that_are_too_long(self, batch, max_length):
        """Override to also filter answer_mask and raw_* fields."""
        keys = list(batch.keys())
        rows = list(zip(*[batch[k] for k in keys]))
        # input_ids is always the first key; filter by its length
        id_idx = keys.index("input_ids")
        filtered = [row for row in rows if len(row[id_idx]) <= max_length]
        if not filtered:
            return {k: [] for k in keys}
        transposed = list(zip(*filtered))
        return {k: list(v) for k, v in zip(keys, transposed)}

    def __call__(self, batch):
        # Drop None entries (samples that failed processing in the dataset)
        batch = [s for s in batch if s is not None]
        if not batch:
            return {
                "input_ids": torch.zeros(0), "labels": torch.zeros(0),
                "attention_mask": torch.zeros(0), "images": [],
                "answer_mask": torch.zeros(0, dtype=torch.bool),
                "raw_images": [], "raw_answers": [], "raw_conversations": [],
            }

        # Separate out the non-tensor fields before passing to prepare_batch
        raw_images        = [s["raw_images"]       for s in batch]
        raw_answers       = [s["raw_answer"]        for s in batch]
        raw_conversations = [s["raw_conversation"]  for s in batch]

        # Build the tensor-only batch dict for prepare_batch
        tensor_batch = [
            {k: v for k, v in s.items()
             if k not in ("raw_images", "raw_answer", "raw_conversation")}
            for s in batch
        ]

        result = self.prepare_batch(tensor_batch, max_length=self.max_length)

        # Stack answer_mask (prepare_batch already handles it via _pad_batch)
        if isinstance(result.get("answer_mask"), list) and result["answer_mask"]:
            result["answer_mask"] = torch.stack(result["answer_mask"])

        # Attach raw fields (lists — intentionally not stacked into tensors)
        result["raw_images"]        = raw_images
        result["raw_answers"]       = raw_answers
        result["raw_conversations"] = raw_conversations

        return result
