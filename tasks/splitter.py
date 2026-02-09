"""Task 4: Dataset Splitter - Train/Val/Test split with labeler confidence."""

import logging
from typing import Dict, Tuple

import pandas as pd
from pathlib import Path
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from .utils import resolve_path, build_local_filename, md_table

logger = logging.getLogger(__name__)


class DatasetSplitter:
    """Train/Val/Test split with labeler confidence-based test selection."""

    def __init__(self, cfg: DictConfig, script_dir: Path):
        self.cfg = cfg
        self.csv_path = script_dir / cfg.target.csv_file
        self.output_dir = resolve_path(cfg.paths.output_dir, script_dir)
        self.image_dir = resolve_path(cfg.paths.image_dir, script_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ratios = cfg.split.ratios
        self.labeler_confidence = dict(cfg.split.labeler_confidence)
        self.seed = cfg.split.random_seed
        self.label_col = cfg.split.label_column
        self.labeler_col = cfg.split.labeler_column
        self.output_format = cfg.download.format.lower() if hasattr(cfg.download, 'format') else 'jpg'

        self.df = pd.read_csv(self.csv_path)

    def _normalize_confidence(self) -> Dict[str, float]:
        total = sum(self.labeler_confidence.values())
        if total == 0:
            n = len(self.labeler_confidence)
            return {k: 1.0 / n for k in self.labeler_confidence}
        return {k: v / total for k, v in self.labeler_confidence.items()}

    def _stratified_sample(self, df: pd.DataFrame, n: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Stratified sample n rows; returns (sampled, remaining)."""
        if n <= 0:
            return pd.DataFrame(columns=df.columns), df
        if n >= len(df):
            return df, pd.DataFrame(columns=df.columns)
        try:
            remaining, sampled = train_test_split(
                df, test_size=n / len(df), stratify=df[self.label_col], random_state=self.seed)
            return sampled, remaining
        except ValueError:
            sampled = df.sample(n=min(n, len(df)), random_state=self.seed)
            return sampled, df.drop(sampled.index)

    def split(self) -> Dict[str, pd.DataFrame]:
        """Perform the dataset split."""
        total = len(self.df)
        total_ratio = self.ratios.train + self.ratios.val + self.ratios.test
        test_size = int(round(total * self.ratios.test / total_ratio))
        val_size = int(round(total * self.ratios.val / total_ratio))

        logger.info(f"Splitting {total} samples -> train/val/test "
                     f"(target: {total - test_size - val_size}/{val_size}/{test_size})")

        # Step 1: Test set by labeler confidence
        norm_conf = self._normalize_confidence()
        test_parts, remain_parts = [], []

        for labeler in self.df[self.labeler_col].unique():
            ldf = self.df[self.df[self.labeler_col] == labeler]
            weight = norm_conf.get(labeler, 0.0)
            n_test = min(int(round(test_size * weight)), len(ldf))
            if n_test > 0:
                test_part, remain_part = self._stratified_sample(ldf, n_test)
                test_parts.append(test_part)
                remain_parts.append(remain_part)
            else:
                remain_parts.append(ldf)

        test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=self.df.columns)
        remaining = pd.concat(remain_parts, ignore_index=True) if remain_parts else self.df.copy()

        # Step 2: Val from remaining
        if len(remaining) > 0 and val_size > 0:
            val_ratio = min(val_size / len(remaining), 0.99)
            try:
                train_df, val_df = train_test_split(
                    remaining, test_size=val_ratio, stratify=remaining[self.label_col],
                    random_state=self.seed)
            except ValueError:
                train_df, val_df = train_test_split(
                    remaining, test_size=val_ratio, random_state=self.seed)
        else:
            train_df, val_df = remaining, pd.DataFrame(columns=self.df.columns)

        return {k: v.reset_index(drop=True) for k, v in
                [('train', train_df), ('val', val_df), ('test', test_df)]}

    def _get_local_image_path(self, row) -> str:
        filename = build_local_filename(
            row['image_url'], row.get(self.label_col, 'unknown'),
            self.output_format, row.get('creative_id'))
        return str(self.image_dir.resolve() / filename)

    def save_splits(self, splits: Dict[str, pd.DataFrame]):
        """Save split CSVs with local image paths."""
        for name, df in splits.items():
            df_out = df.copy()
            df_out['local_image_path'] = df_out.apply(self._get_local_image_path, axis=1)
            path = self.output_dir / f'{name}.csv'
            df_out.to_csv(path, index=False)
            logger.info(f"  {name}: {len(df_out)} samples -> {path}")

    def verify_and_report(self, splits: Dict[str, pd.DataFrame]):
        """Verify splits and generate markdown report."""
        total = sum(len(df) for df in splits.values())
        total_ratio = self.ratios.train + self.ratios.val + self.ratios.test
        norm_conf = self._normalize_confidence()
        global_dist = self.df[self.label_col].value_counts(normalize=True)
        labels = list(global_dist.index)
        warnings_list = []

        # 1. Split Ratios
        ratio_rows = []
        for name in ['train', 'val', 'test']:
            expected = getattr(self.ratios, name) / total_ratio
            actual = len(splits[name]) / total if total > 0 else 0
            diff = actual - expected
            status = "OK" if abs(diff) < 0.02 else "WARN"
            if abs(diff) >= 0.02:
                warnings_list.append(f"Split ratio '{name}' deviates by {diff*100:+.1f}%")
            ratio_rows.append([name, f"{expected*100:.1f}%", f"{actual*100:.1f}%",
                                f"{diff*100:+.1f}%", len(splits[name]), status])

        # 2. Labeler Confidence (test set)
        test_df = splits['test']
        conf_rows = []
        if len(test_df) > 0:
            test_counts = test_df[self.labeler_col].value_counts()
            for labeler in sorted(set(norm_conf) | set(test_counts.index)):
                exp = norm_conf.get(labeler, 0.0)
                cnt = test_counts.get(labeler, 0)
                act = cnt / len(test_df)
                diff = act - exp
                tol = 0.05 if cnt > 10 else 0.15
                status = "OK" if abs(diff) < tol else "WARN"
                if abs(diff) >= tol:
                    warnings_list.append(f"Labeler '{labeler}' test: expected {exp*100:.1f}%, got {act*100:.1f}%")
                conf_rows.append([labeler, f"{exp*100:.1f}%", f"{act*100:.1f}%",
                                   f"{diff*100:+.1f}%", cnt, status])

        # 3. Label Distribution
        dist_rows = []
        for name in ['train', 'val', 'test']:
            df = splits[name]
            if len(df) == 0:
                continue
            split_dist = df[self.label_col].value_counts(normalize=True)
            row = [name]
            max_diff = 0
            for lbl in labels:
                s = split_dist.get(lbl, 0)
                g = global_dist.get(lbl, 0)
                max_diff = max(max_diff, abs(s - g))
                row.append(f"{s*100:.1f}%")
            status = "OK" if max_diff < 0.03 else "WARN"
            if max_diff >= 0.03:
                warnings_list.append(f"Label distribution in '{name}' deviates by {max_diff*100:.1f}%")
            row.extend([f"{max_diff*100:.1f}%", status])
            dist_rows.append(row)

        # Build markdown
        lines = [
            "# Dataset Split Verification Report",
            f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Config: `{self.cfg.target.csv_file}`\n\n---\n",
            "## 1. Split Ratio Verification\n",
            md_table(['Split', 'Expected', 'Actual', 'Diff', 'Count', 'Status'], ratio_rows),
            "\n## 2. Labeler Confidence in Test Set\n",
            md_table(['Labeler', 'Expected', 'Actual', 'Diff', 'Count', 'Status'], conf_rows),
            "\n## 3. Label Distribution (Stratification)\n",
            md_table(['Split'] + [str(l) for l in labels] + ['Max Diff', 'Status'], dist_rows),
            "\n## 4. Detailed Statistics\n",
        ]

        for name in ['train', 'val', 'test']:
            df = splits[name]
            lines.append(f"\n### {name.upper()} ({len(df)} samples)\n")
            if len(df) > 0:
                lc = df[self.label_col].value_counts()
                lines.append(md_table(['Label', 'Count', '%'],
                             [[l, c, f"{c/len(df)*100:.1f}%"] for l, c in lc.items()]))
                lines.append("")

        lines.append("\n## 5. Warnings\n")
        if warnings_list:
            for w in warnings_list:
                lines.append(f"- {w}")
        else:
            lines.append("All checks passed!")

        report_path = self.output_dir / 'split_verification_report.md'
        report_path.write_text('\n'.join(lines), encoding='utf-8')

        # Console summary
        for name in ['train', 'val', 'test']:
            logger.info(f"  {name}: {len(splits[name])} samples")
        if warnings_list:
            for w in warnings_list:
                logger.warning(f"  {w}")
        else:
            logger.info("  All verification checks passed!")
        logger.info(f"  Report: {report_path}")

    def run(self):
        """Run the complete split pipeline."""
        logger.info("=" * 60)
        logger.info("Task 4: Dataset Split")
        logger.info("=" * 60)
        splits = self.split()
        self.save_splits(splits)
        self.verify_and_report(splits)
        logger.info("Dataset split completed!")
        return splits
