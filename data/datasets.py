import torch
from PIL import Image
from torch.utils.data import Dataset
from data.processors import get_image_string
import logging


class BaseDataset(Dataset):
    def __init__(self, dataset, tokenizer, image_processor, mp_image_token_length, relevance_min_rating=1, image_correspondence_min_rating=1, visual_dependency_min_rating=1, formatting_min_rating=1, max_chat_tokens=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.relevance_min_rating = relevance_min_rating
        self.image_correspondence_min_rating = image_correspondence_min_rating
        self.visual_dependency_min_rating = visual_dependency_min_rating
        self.formatting_min_rating = formatting_min_rating
        tok_max = getattr(self.tokenizer, "model_max_length", None)
        if tok_max is None or tok_max > 1_000_000:
            tok_max = 8192
        if max_chat_tokens is not None:
            self.max_chat_tokens = min(int(max_chat_tokens), int(tok_max))
        else:
            self.max_chat_tokens = int(tok_max)
        self.prefix_len = self._get_prefix_len()

    def __len__(self):
        return len(self.dataset)

    def _get_prefix_len(self):
        random_string_5_letters = "xzyvd"
        random_string_chat_templated = self.tokenizer.apply_chat_template([{"role": "assistant", "content": random_string_5_letters}], tokenize=False, add_special_tokens=False)
        random_string_location = random_string_chat_templated.find(random_string_5_letters)
        return len(self.tokenizer.encode(random_string_chat_templated[:random_string_location]))

    def _get_messages(self, item, splitted_image_counts):
        messages = []
        for index, text in enumerate(item['texts']):
            try:
                if item.get('relevance_ratings') is not None and item['relevance_ratings'][index] is not None and item['relevance_ratings'][index] < self.relevance_min_rating:
                    continue
                if item.get('image_correspondence_ratings') is not None and item['image_correspondence_ratings'][index] is not None and item['image_correspondence_ratings'][index] < self.image_correspondence_min_rating:
                    continue
                if item.get('visual_dependency_ratings') is not None and item['visual_dependency_ratings'][index] is not None and item['visual_dependency_ratings'][index] < self.visual_dependency_min_rating:
                    continue
                if item.get('formatting_ratings') is not None and item['formatting_ratings'][index] is not None and item['formatting_ratings'][index] < self.formatting_min_rating:
                    continue
            except Exception as e:
                logging.warning(f"Error processing item: {item}, index: {index}: {e}")

            messages.append({"role": "user", "content": text['user']})
            messages.append({"role": "assistant", "content": text['assistant']})

        if len(messages) == 0:
            return messages

        # Safety check to ensure no image tokens are present in the text before adding them.
        for msg in messages:
            if self.tokenizer.image_token in msg["content"]:
                logging.warning(f"Found and removed an image token in the {msg['role']} text before adding the image string.")
                msg["content"] = msg["content"].replace(self.tokenizer.image_token, "")

        if len(splitted_image_counts) > 0:
            image_string = get_image_string(self.tokenizer, splitted_image_counts, self.mp_image_token_length)
            messages[0]["content"] = image_string + messages[0]["content"]

        return messages

    def _process_images(self, images):
        processed_images = []
        splitted_image_counts = []
        for image in images:
            if isinstance(image, Image.Image):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                processed_image, splitted_image_count = self.image_processor(image)
                if not hasattr(self.tokenizer, "global_image_token") and splitted_image_count[0]*splitted_image_count[1] == len(processed_image) - 1:
                    # If the tokenizer doesn't have a global image token, but the processor generated it, remove it
                    processed_image = processed_image[1:]
                processed_images.append(processed_image)
                splitted_image_counts.append(splitted_image_count)
            else:
                raise ValueError(f"Error processing image: {image}")
        return processed_images, splitted_image_counts


    def _prepare_inputs_and_loss_mask(self, messages):
        conv_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
        )
        max_len = self.max_chat_tokens
        if len(conv_ids["input_ids"]) > max_len:
            logging.warning(
                "Dropping sample: conversation still longer than max_chat_tokens after trimming (%s tokens > %s).",
                len(conv_ids["input_ids"]),
                max_len,
            )
            return None, None, None

        mask = [0] * len(conv_ids["input_ids"])

        # Locate each assistant turn and flip its mask to 1
        cursor = 0
        for msg in messages:
            segment_ids = self.tokenizer.apply_chat_template(
                [msg], tokenize=True, add_special_tokens=False
            )
            seg_len = len(segment_ids)

            if msg["role"] == "assistant":
                start = cursor + self.prefix_len
                end   = cursor + seg_len
                mask[start:end] = [1] * (end - start)  # attend to these tokens

            cursor += seg_len
        
        return torch.tensor(conv_ids["input_ids"]), torch.tensor(mask).to(torch.bool), torch.tensor(conv_ids["attention_mask"])


class VQADataset(BaseDataset):  # Visual Question Answering Dataset
    def iter_for_worker(self):  # with iterable datasets, each worker gets different shards
        for data in self.dataset:
            yield self._process_data(data)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self._process_data(item)

    def _process_data(self, item):
        # Handle images (should be a list)
        if item['images'] is None:
            images_data = []
        else:
            images_data = item['images']
            if not isinstance(images_data, list):
                images_data = [images_data]

        # Keep raw PIL images before student preprocessing (needed by teacher).
        # We convert to RGB here so the teacher processor doesn't have to.
        raw_pil_images = []
        for img in images_data:
            if isinstance(img, Image.Image):
                raw_pil_images.append(img.convert("RGB") if img.mode != "RGB" else img)

        processed_images = []
        splitted_image_counts = []
        if images_data: # Only process if there are images
            processed_images, splitted_image_counts = self._process_images(images_data)

        messages = self._get_messages(item, splitted_image_counts)

        if len(messages) == 0:
            return None

        packed = self._prepare_inputs_and_loss_mask(messages)
        if packed[0] is None:
            return None
        input_ids, mask, attention_mask = packed
        labels = self._get_labels(input_ids, mask)

        # answer_mask: True at positions the student is trained to predict.
        # Matches labels != -100 (after the roll-by-1 shift in _get_labels).
        answer_mask = (labels != -100)

        # Build raw conversation fields for the teacher.
        # raw_conversations: the prompt turns (all but the last assistant turn).
        # raw_answer:        the final assistant response text.
        texts = item.get("texts", [])
        if texts:
            last_text = texts[-1]
            raw_answer = last_text.get("assistant", "")
            # Build prompt messages (user turns only up to the last one)
            raw_conversation = []
            for t in texts[:-1]:
                raw_conversation.append({"role": "user",      "content": t.get("user", "")})
                raw_conversation.append({"role": "assistant", "content": t.get("assistant", "")})
            # Add the final user turn (without the answer)
            raw_conversation.append({"role": "user", "content": last_text.get("user", "")})
        else:
            raw_answer = ""
            raw_conversation = []

        return {
            "images":            processed_images,
            "input_ids":         input_ids,
            "attention_mask":    attention_mask,
            "labels":            labels,
            # ── Distillation fields ──────────────────────────────────────────
            "answer_mask":       answer_mask,       # bool [T_seq]
            "raw_images":        raw_pil_images,    # List[PIL.Image]
            "raw_answer":        raw_answer,        # str
            "raw_conversation":  raw_conversation,  # List[dict]
        }

    def _get_labels(self, input_ids, mask):
        labels = input_ids.clone().masked_fill(~mask, -100)
        labels = labels.roll(-1) # Shift labels for causal LM
        labels[-1] = -100 # Last token has no target
        
        return labels
