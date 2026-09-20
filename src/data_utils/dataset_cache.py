import pandas as pd
import torch
from torch.utils.data import Dataset


class DualTokenizerCollateWithTeacher:
    def __init__(self, tok_student, task: str, max_len: int):
        self.ts = tok_student
        self.task = task
        self.max_len = max_len

    def __call__(self, batch):
        samples, teacher_cls = zip(*batch)
        teacher_cls = torch.stack(teacher_cls, dim=0)  # [B, d_t]

        if self.task == "single_cls":
            s1s, ys = zip(*samples)
            s_enc = self.ts(list(s1s), max_length=self.max_len, truncation=True,
                            padding=True, return_tensors="pt",
                            return_special_tokens_mask=True)
            out = {
                "input_ids_stu": s_enc["input_ids"],
                "attention_mask_stu": s_enc["attention_mask"],
                "special_tokens_mask_stu": s_enc["special_tokens_mask"],
                "teacher_cls": teacher_cls,
                "labels": torch.tensor(ys, dtype=torch.long),
            }
            if "token_type_ids" in s_enc:
                out["token_type_ids_stu"] = s_enc["token_type_ids"]
            return out

        # ---------- pair ----------
        s1s, s2s = zip(*samples)

        s1_enc = self.ts(list(s1s), max_length=self.max_len, truncation=True,
                         padding=True, return_tensors="pt",
                         return_special_tokens_mask=True)
        s2_enc = self.ts(list(s2s), max_length=self.max_len, truncation=True,
                         padding=True, return_tensors="pt",
                         return_special_tokens_mask=True)

        out = {
            "input_ids1_stu": s1_enc["input_ids"],
            "attention_mask1_stu": s1_enc["attention_mask"],
            "special_tokens_mask1_stu": s1_enc["special_tokens_mask"],
            "input_ids2_stu": s2_enc["input_ids"],
            "attention_mask2_stu": s2_enc["attention_mask"],
            "special_tokens_mask2_stu": s2_enc["special_tokens_mask"],
            "teacher_cls": teacher_cls,
        }

        if "token_type_ids" in s1_enc:
            out["token_type_ids1_stu"] = s1_enc["token_type_ids"]
        if "token_type_ids" in s2_enc:
            out["token_type_ids2_stu"] = s2_enc["token_type_ids"]

        return out
    
class TextPairWithFusionTarget(Dataset):
    """Raw text samples paired with a precomputed [num_sides, d_S] target."""

    def __init__(self, df: pd.DataFrame, task: str, targets: torch.Tensor):
        self.task = task
        self.targets = targets  # [N, num_sides, d_S]

        if task == "single_cls":
            self.samples = [(t, None, int(y)) for t, y in zip(df["text"].astype(str),
                                                              df["label"].astype(int))]
        elif task == "pair_cls":
            labels = (
                df["label"].astype(int).tolist()
                if "label" in df.columns
                else [None] * len(df)
            )
            self.samples = [
                (a, b, label)
                for a, b, label in zip(
                    df["premise"].astype(str),
                    df["hypothesis"].astype(str),
                    labels,
                )
            ]
        else:
            self.samples = [
                (a, b, None)
                for a, b in zip(
                    df["sentence1"].astype(str), df["sentence2"].astype(str)
                )
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]


class DualTokenizerCollateWithFusionTarget:
    """Student-only collate that carries the cached fused target per sample."""

    def __init__(self, tok_student, task: str, max_len: int, num_sides: int = 1):
        self.ts = tok_student
        self.task = task
        self.max_len = max_len
        self.num_sides = num_sides

    def _encode(self, texts):
        return self.ts(
            list(texts),
            max_length=self.max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )

    def __call__(self, batch):
        samples, targets = zip(*batch)
        targets = torch.stack(targets, dim=0)  # [B, S, d_S]

        text1s = [sample[0] for sample in samples]
        text2s = [sample[1] for sample in samples]

        encoder1 = self._encode(text1s)
        out = {
            "input_ids1_stu": encoder1["input_ids"],
            "attention_mask1_stu": encoder1["attention_mask"],
            "special_tokens_mask1_stu": encoder1["special_tokens_mask"],
            "target1": targets[:, 0, :],
        }
        if "token_type_ids" in encoder1:
            out["token_type_ids1_stu"] = encoder1["token_type_ids"]

        has_second = all(text is not None for text in text2s)
        if has_second:
            encoder2 = self._encode([str(text) for text in text2s])
            out["input_ids2_stu"] = encoder2["input_ids"]
            out["attention_mask2_stu"] = encoder2["attention_mask"]
            out["special_tokens_mask2_stu"] = encoder2["special_tokens_mask"]
            if "token_type_ids" in encoder2:
                out["token_type_ids2_stu"] = encoder2["token_type_ids"]
            if self.num_sides >= 2:
                out["target2"] = targets[:, 1, :]

        labels = [sample[2] for sample in samples]
        has_labels = all(label is not None for label in labels)
        if has_labels:
            dtype = torch.float32 if self.task == "pair_reg" else torch.long
            out["labels"] = torch.tensor(labels, dtype=dtype)
        return out


class TextPairWithTeacher(Dataset):
    def __init__(self, df: pd.DataFrame, task: str, teacher_cls: torch.Tensor):
        self.task = task
        self.teacher_cls = teacher_cls   # [N, d_t]

        if task == "single_cls":
            self.samples = [(t, int(y)) for t, y in zip(df["text"].astype(str),
                                                        df["label"].astype(int))]
        elif task == "pair_cls":
            self.samples = [(a, b) for a,b in zip(df["premise"].astype(str),
                                                  df["hypothesis"].astype(str))]
        else:
            self.samples = [(a, b) for a,b in zip(df["sentence1"].astype(str),
                                                  df["sentence2"].astype(str))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tcls = self.teacher_cls[idx]   # lấy đúng teacher CLS của sample này
        return item, tcls
