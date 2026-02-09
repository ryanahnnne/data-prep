"""Task 2: EDA (Exploratory Data Analysis) for Classification model training."""

import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from omegaconf import DictConfig

from .utils import resolve_path, get_filename_from_url, md_table, save_fig

logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class ClassificationEDA:
    """Perform EDA for Classification model training."""

    def __init__(self, cfg: DictConfig, script_dir: Path):
        self.cfg = cfg
        self.csv_path = script_dir / cfg.target.csv_file
        self.image_dir = resolve_path(cfg.paths.image_dir, script_dir)
        self.output_dir = resolve_path(cfg.paths.output_dir, script_dir)
        self.class_name = cfg.target.class_name
        self.class_name_kr = cfg.target.class_name_kr

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = pd.read_csv(self.csv_path)
        self.df['filename'] = self.df['image_url'].apply(get_filename_from_url)
        self._merge_download_results()

        self.image_metadata = None
        self.df_merged = None

    def _merge_download_results(self):
        """Merge local path from download_results.csv."""
        results_path = self.image_dir / 'download_results.csv'
        if not results_path.exists():
            self.df['local_path'] = None
            return
        try:
            rdf = pd.read_csv(results_path)
            if 'path' in rdf.columns and 'creative_id' in rdf.columns:
                path_map = (rdf[rdf['path'].notna()]
                            .drop_duplicates(subset='creative_id', keep='first')
                            [['creative_id', 'path']]
                            .rename(columns={'path': 'local_path'}))
                self.df = self.df.merge(path_map, on='creative_id', how='left')
            else:
                self.df['local_path'] = None
        except Exception:
            self.df['local_path'] = None

    def _resolve_image_path(self, row) -> Optional[Path]:
        """Resolve image path: prefer local_path, fallback to image_dir search."""
        local_path = row.get('local_path')
        if local_path and pd.notna(local_path):
            p = Path(local_path)
            if p.exists():
                return p

        filename = row.get('filename')
        if not filename:
            return None
        base = os.path.splitext(filename)[0]
        for ext in [os.path.splitext(filename)[1], '.jpg', '.jpeg', '.png']:
            p = self.image_dir / f"{base}{ext}"
            if p.exists():
                return p
        return None

    def analyze_csv_data(self) -> dict:
        """Analyze CSV data statistics."""
        return {
            'total_samples': len(self.df),
            'unique_images': self.df['creative_id'].nunique(),
            'labelers': self.df['labeler'].unique().tolist(),
            'label_distribution': self.df['label'].value_counts().to_dict(),
            'label_meta_distribution': self.df['label_meta'].value_counts().to_dict(),
            'duplicate_creative_ids': self.df['creative_id'].duplicated().sum()
        }

    def analyze_images(self) -> pd.DataFrame:
        """Analyze image properties."""
        logger.info("Analyzing image properties...")
        image_data = []

        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Analyzing images"):
            image_path = self._resolve_image_path(row)
            info = {'creative_id': row['creative_id'], 'filename': row['filename']}

            if image_path is None:
                info.update(exists=False, width=None, height=None, aspect_ratio=None,
                            file_size_kb=None, format=None, mode=None, is_corrupt=True)
            else:
                try:
                    with Image.open(image_path) as img:
                        w, h = img.size
                        info.update(exists=True, width=w, height=h,
                                    aspect_ratio=round(w / h, 2) if h > 0 else None,
                                    file_size_kb=round(os.path.getsize(image_path) / 1024, 2),
                                    format=img.format, mode=img.mode, is_corrupt=False)
                except Exception:
                    info.update(exists=True, width=None, height=None, aspect_ratio=None,
                                file_size_kb=round(os.path.getsize(image_path) / 1024, 2) if image_path.exists() else None,
                                format=None, mode=None, is_corrupt=True)
            image_data.append(info)

        self.image_metadata = pd.DataFrame(image_data)
        meta = self.image_metadata.drop(columns=['filename'], errors='ignore')
        self.df_merged = self.df.merge(meta, on='creative_id', how='left')
        return self.image_metadata

    def _get_valid_images(self, df=None):
        """Get valid (exists & not corrupt) images from a dataframe."""
        if df is None:
            df = self.image_metadata
        if df is None:
            return pd.DataFrame()
        return df[(df['exists'] == True) & (df['is_corrupt'] == False)]

    def detect_outliers(self) -> dict:
        """Detect outliers using IQR, z-score, and domain thresholds."""
        outliers = {}
        valid = self._get_valid_images()
        if len(valid) == 0:
            return outliers

        oc = self.cfg.eda.outlier
        iqr_mult = oc.iqr_multiplier
        iqr_hard_range = getattr(oc, 'iqr_hard_range', None) or {}
        z_thresh = getattr(oc, 'z_score_threshold', None)
        min_possible = {'width': 1, 'height': 1, 'file_size_kb': 0, 'aspect_ratio': 0.01}

        for col in ['width', 'height', 'file_size_kb', 'aspect_ratio']:
            data = valid[col].dropna()
            if len(data) == 0:
                continue

            Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
            IQR = Q3 - Q1
            median = data.median()

            lo = max(Q1 - iqr_mult * IQR, min_possible.get(col, 0))
            hi = Q3 + iqr_mult * IQR
            if IQR <= 1e-6:
                lo, hi = min(lo, median - 1), max(hi, median + 1)

            if col in iqr_hard_range:
                hard = iqr_hard_range[col]
                if hard and len(hard) >= 2:
                    if hard[0] is not None:
                        lo = min(lo, hard[0])
                    if hard[1] is not None:
                        hi = max(hi, hard[1])
            lo = min(lo, hi - 1e-6)

            mask = (data < lo) | (data > hi)
            ids = valid.loc[data[mask].index, 'creative_id'].tolist()
            outliers[col] = {
                'count': len(ids), 'lower_bound': round(float(lo), 2),
                'upper_bound': round(float(hi), 2), 'Q1': round(float(Q1), 2),
                'Q3': round(float(Q3), 2), 'median': round(float(median), 2),
                'IQR': round(float(IQR), 2), 'outlier_ids': ids[:20],
            }

            if z_thresh and z_thresh > 0:
                std = data.std()
                if std and std > 1e-9:
                    z = ((data - data.mean()) / std).abs()
                    z_ids = valid.loc[data[z > z_thresh].index, 'creative_id'].tolist()
                    if z_ids:
                        outliers[f'{col}_zscore'] = {
                            'count': len(z_ids), 'z_threshold': z_thresh,
                            'mean': round(float(data.mean()), 2),
                            'std': round(float(std), 2), 'outlier_ids': z_ids[:20],
                        }

        missing = self.image_metadata[self.image_metadata['exists'] == False]
        outliers['missing_images'] = {'count': len(missing), 'ids': missing['creative_id'].tolist()[:20]}
        corrupt = self.image_metadata[self.image_metadata['is_corrupt'] == True]
        outliers['corrupt_images'] = {'count': len(corrupt), 'ids': corrupt['creative_id'].tolist()[:20]}

        extreme_ar = valid[(valid['aspect_ratio'] < oc.extreme_aspect_ratio_min) |
                           (valid['aspect_ratio'] > oc.extreme_aspect_ratio_max)]
        outliers['extreme_aspect_ratio'] = {'count': len(extreme_ar), 'ids': extreme_ar['creative_id'].tolist()[:20]}

        small = valid[valid['width'] < oc.small_image_threshold]
        outliers['very_small_images'] = {'count': len(small), 'ids': small['creative_id'].tolist()[:20]}

        large = valid[(valid['width'] > oc.large_image_threshold) |
                      (valid['height'] > oc.large_image_threshold)]
        outliers['very_large_images'] = {'count': len(large), 'ids': large['creative_id'].tolist()[:20]}

        return outliers

    def create_visualizations(self):
        """Create EDA visualizations."""
        logger.info("Creating visualizations...")
        sns.set_style("whitegrid")
        viz = self.cfg.eda.visualization
        dpi = viz.dpi

        # 1. Label Distribution
        fig, axes = plt.subplots(1, 2, figsize=tuple(viz.figsize_label))
        for ax, col, title in [(axes[0], 'label', f'Label Distribution ({self.class_name})'),
                                (axes[1], 'label_meta', 'Label Meta Distribution')]:
            counts = self.df[col].value_counts()
            palette = {'label': ['#2ecc71' if str(l).upper() == 'TRUE' else '#e74c3c' for l in counts.index],
                        'label_meta': ['#3498db', '#9b59b6', '#e67e22'][:len(counts)]}
            colors = palette.get(col, sns.color_palette('Set2', len(counts)))
            ax.bar(counts.index, counts.values, color=colors, edgecolor='black')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(col.replace('_', ' ').title())
            ax.set_ylabel('Count')
            for i, (lbl, cnt) in enumerate(counts.items()):
                ax.annotate(f'{cnt}\n({cnt/len(self.df)*100:.1f}%)',
                            xy=(i, cnt), ha='center', va='bottom', fontsize=11)
        save_fig(fig, self.output_dir / 'label_distribution.png', dpi)

        # 2. Labeler Analysis
        fig, axes = plt.subplots(1, 2, figsize=tuple(viz.figsize_labeler))
        labeler_counts = self.df['labeler'].value_counts()
        axes[0].barh(labeler_counts.index, labeler_counts.values, color='#3498db', edgecolor='black')
        axes[0].set_title('Samples per Labeler', fontsize=14, fontweight='bold')
        for i, cnt in enumerate(labeler_counts.values):
            axes[0].annotate(f'{cnt}', xy=(cnt, i), va='center', fontsize=10)

        crosstab = pd.crosstab(self.df['labeler'], self.df['label'], normalize='index') * 100
        crosstab.plot(kind='barh', stacked=True, ax=axes[1],
                      color=['#e74c3c', '#2ecc71'], edgecolor='black')
        axes[1].set_title('Label Distribution by Labeler (%)', fontsize=14, fontweight='bold')
        axes[1].legend(title='Label', loc='lower right')
        save_fig(fig, self.output_dir / 'labeler_analysis.png', dpi)

        # 3 & 4. Image Properties
        valid = self._get_valid_images()
        if len(valid) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=tuple(viz.figsize_image_props))
        axes[0, 0].scatter(valid['width'], valid['height'], alpha=0.3, s=10, c='#3498db')
        axes[0, 0].set_title('Image Resolution Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Width (px)'); axes[0, 0].set_ylabel('Height (px)')

        axes[0, 1].hist(valid['aspect_ratio'].dropna(), bins=50, color='#9b59b6', edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Aspect Ratio Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].axvline(x=1.0, color='red', linestyle='--', label='Square (1:1)')
        axes[0, 1].legend()

        axes[1, 0].hist(valid['file_size_kb'].dropna(), bins=50, color='#e67e22', edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('File Size Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('File Size (KB)')

        fmt_counts = valid['format'].value_counts()
        axes[1, 1].pie(fmt_counts.values, labels=fmt_counts.index, autopct='%1.1f%%',
                        colors=sns.color_palette('Set2', len(fmt_counts)))
        axes[1, 1].set_title('Image Format Distribution', fontsize=14, fontweight='bold')
        save_fig(fig, self.output_dir / 'image_properties.png', dpi)

        # Label vs Image Properties
        if self.df_merged is not None and 'label' in self.df_merged.columns:
            merged_valid = self._get_valid_images(self.df_merged).copy()
            if len(merged_valid) > 0:
                fig, axes = plt.subplots(1, 3, figsize=tuple(viz.figsize_label_vs_props))
                palette = {}
                for lbl in merged_valid['label'].unique():
                    u = str(lbl).upper()
                    palette[lbl] = '#2ecc71' if u == 'TRUE' else '#e74c3c' if u == 'FALSE' else '#3498db'

                merged_valid['resolution'] = merged_valid['width'] * merged_valid['height']
                for ax, col, title in [(axes[0], 'file_size_kb', 'File Size by Label'),
                                        (axes[1], 'aspect_ratio', 'Aspect Ratio by Label'),
                                        (axes[2], 'resolution', 'Resolution (WxH) by Label')]:
                    sns.boxplot(data=merged_valid, x='label', y=col, hue='label',
                                ax=ax, palette=palette, legend=False)
                    ax.set_title(title, fontsize=14, fontweight='bold')
                save_fig(fig, self.output_dir / 'label_vs_image_properties.png', dpi)

        logger.info(f"Visualizations saved to: {self.output_dir}")

    def _image_gallery(self, creative_ids: list, title: str, max_images: int = 10) -> str:
        """Generate markdown image gallery."""
        lines = [f"\n**{title}** (showing up to {max_images} samples):\n"]
        shown = 0
        for cid in creative_ids[:max_images * 2]:
            if shown >= max_images:
                break
            row = self.df[self.df['creative_id'] == cid]
            if len(row) == 0:
                continue
            img_path = self._resolve_image_path(row.iloc[0])
            if img_path is None:
                continue
            try:
                rel = os.path.relpath(img_path, self.output_dir)
            except ValueError:
                rel = str(img_path)
            label = row.iloc[0]['label']
            lines.extend([
                f"| creative_id: `{cid}` | label: `{label}` |",
                "| :---: | :---: |",
                f"| ![{cid}]({rel}) | |", ""
            ])
            shown += 1
        if shown == 0:
            lines.append("*No images found*\n")
        return "\n".join(lines)

    def generate_report(self) -> str:
        """Generate EDA report in Markdown."""
        stats = self.analyze_csv_data()
        outliers = self.detect_outliers()
        total = stats['total_samples']
        report = []
        r = report.append

        r(f"# EDA Report: {self.class_name} ({self.class_name_kr})")
        r(f"\n**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        r(f"**Config:** `{self.cfg.target.csv_file}`\n\n---\n")

        # 1. Basic Statistics
        r("## 1. Basic Statistics\n")
        r(md_table(['Metric', 'Value'], [
            ['Total samples', total],
            ['Unique images', stats['unique_images']],
            ['Labelers', ', '.join(stats['labelers'])],
            ['Duplicate creative_ids', stats['duplicate_creative_ids']],
        ]))

        # 2. Label Distribution
        r("\n## 2. Label Distribution\n")
        label_dist = stats['label_distribution']
        r(md_table(['Label', 'Count', 'Percentage'],
                   [[l, c, f"{c/total*100:.1f}%"] for l, c in label_dist.items()]))
        values = list(label_dist.values())
        if len(values) == 2:
            ratio = max(values) / min(values)
            r(f"\n**Class Imbalance Ratio:** {ratio:.2f}:1")
            if ratio > 3:
                r("\n> Warning: Significant class imbalance detected!")
        r("\n![Label Distribution](label_distribution.png)")

        # 3. Label Meta Distribution
        r("\n## 3. Label Meta Distribution\n")
        r(md_table(['Label Meta', 'Count', 'Percentage'],
                   [[m, c, f"{c/total*100:.1f}%"] for m, c in stats['label_meta_distribution'].items()]))

        # 4. Labeler Statistics
        r("\n## 4. Labeler Statistics\n")
        ls = self.df.groupby('labeler').agg(
            total=('creative_id', 'count'),
            true_count=('label', lambda x: (x.astype(str).str.upper() == 'TRUE').sum())
        )
        ls['true_rate'] = ls['true_count'] / ls['total'] * 100
        r(md_table(['Labeler', 'Samples', 'TRUE Count', 'TRUE Rate'],
                   [[lb, int(row['total']), int(row['true_count']), f"{row['true_rate']:.1f}%"]
                    for lb, row in ls.iterrows()]))
        if len(ls) > 1 and ls['true_rate'].max() - ls['true_rate'].min() > 20:
            r("\n> Warning: Potential labeler bias detected!")
        r("\n![Labeler Analysis](labeler_analysis.png)")

        # 5. Image Statistics
        if self.image_metadata is not None:
            valid = self._get_valid_images()
            r("\n## 5. Image Statistics\n")
            r(md_table(['Metric', 'Value'], [
                ['Total images analyzed', len(self.image_metadata)],
                ['Valid images', len(valid)],
                ['Missing images', (self.image_metadata['exists'] == False).sum()],
                ['Corrupt images', self.image_metadata['is_corrupt'].sum()],
            ]))

            if len(valid) > 0:
                r("\n### Resolution Statistics\n")
                r(md_table(['Dimension', 'Min', 'Max', 'Mean', 'Median'],
                           [[d, f"{valid[d].min():.0f}", f"{valid[d].max():.0f}",
                             f"{valid[d].mean():.0f}", f"{valid[d].median():.0f}"]
                            for d in ['width', 'height']] +
                           [['Aspect Ratio', f"{valid['aspect_ratio'].min():.2f}",
                             f"{valid['aspect_ratio'].max():.2f}",
                             f"{valid['aspect_ratio'].mean():.2f}",
                             f"{valid['aspect_ratio'].median():.2f}"],
                            ['File Size (KB)', f"{valid['file_size_kb'].min():.1f}",
                             f"{valid['file_size_kb'].max():.1f}",
                             f"{valid['file_size_kb'].mean():.1f}",
                             f"{valid['file_size_kb'].median():.1f}"]]))

                r("\n### Image Formats\n")
                r(md_table(['Format', 'Count'],
                           [[f, c] for f, c in valid['format'].value_counts().items()]))
                r("\n### Color Modes\n")
                r(md_table(['Mode', 'Count'],
                           [[m, c] for m, c in valid['mode'].value_counts().items()]))
                r("\n![Image Properties](image_properties.png)")
                r("\n![Label vs Image Properties](label_vs_image_properties.png)")

        # 6. Outlier Analysis
        r("\n## 6. Outlier Analysis\n")
        if outliers:
            iqr_out = {k: v for k, v in outliers.items() if 'lower_bound' in v and not k.endswith('_zscore')}
            if iqr_out:
                r("### IQR-based Outliers\n")
                r(md_table(['Metric', 'Count', 'Normal Range', 'Q1', 'Q3', 'Median', 'IQR'],
                           [[n, v['count'], f"[{v['lower_bound']}, {v['upper_bound']}]",
                             v['Q1'], v['Q3'], v['median'], v['IQR']]
                            for n, v in iqr_out.items()]))

            z_out = {k: v for k, v in outliers.items() if k.endswith('_zscore') and v.get('count', 0) > 0}
            if z_out:
                r("\n### Z-Score Extreme Values\n")
                r(md_table(['Metric', 'Count', '|z| threshold', 'Mean', 'Std'],
                           [[n, v['count'], v.get('z_threshold', ''), v.get('mean', ''), v.get('std', '')]
                            for n, v in z_out.items()]))

            sample_count = self.cfg.eda.report.outlier_sample_count
            for key, title in [('very_small_images', 'Very Small Images'),
                                ('very_large_images', 'Very Large Images'),
                                ('extreme_aspect_ratio', 'Extreme Aspect Ratio')]:
                if key in outliers and outliers[key]['count'] > 0:
                    info = outliers[key]
                    r(f"\n### {title}\n\n**Count:** {info['count']}\n")
                    if info.get('ids'):
                        r(self._image_gallery(info['ids'], f"Sample {title}", sample_count))

        # 7. Recommendations
        r("\n## 7. Recommendations\n")
        recs = []
        if len(values) == 2 and max(values) / min(values) > 2:
            recs.append("Address class imbalance using weighted loss, SMOTE, or undersampling")
        if self.image_metadata is not None:
            valid = self._get_valid_images()
            if len(valid) > 0:
                if valid['width'].std() > 200 or valid['height'].std() > 200:
                    recs.append("Standardize image sizes (high resolution variance detected)")
                if outliers.get('very_small_images', {}).get('count', 0) > 0:
                    recs.append("Review/remove very small images")
                if outliers.get('corrupt_images', {}).get('count', 0) > 0:
                    recs.append("Remove or re-download corrupt images")
        recs.append("Apply data augmentation for better generalization")
        recs.append("Use stratified split to maintain label distribution")
        if len(stats['labelers']) > 1:
            recs.append("Consider inter-annotator agreement analysis")
        for rec in recs:
            r(f"- {rec}")

        # 8. Sample Images
        r("\n## 8. Sample Images\n")
        sample_count = self.cfg.eda.report.sample_images_count
        for label_val, label_name in [('TRUE', 'TRUE Label'), ('FALSE', 'FALSE Label')]:
            ids = self.df[self.df['label'].astype(str).str.upper() == label_val]['creative_id'].tolist()
            if ids:
                r(self._image_gallery(ids[:10], f"{label_name} Samples", sample_count))

        report_text = "\n".join(report)
        fmt = self.cfg.eda.report.format
        report_path = self.output_dir / f'eda_report.{fmt}'
        report_path.write_text(report_text, encoding='utf-8')
        logger.info(f"EDA Report saved to: {report_path}")
        return report_text

    def run(self):
        """Run the complete EDA pipeline."""
        logger.info("=" * 60)
        logger.info(f"Task 2: EDA - {self.class_name} ({self.class_name_kr})")
        logger.info("=" * 60)

        self.analyze_images()
        self.create_visualizations()
        self.generate_report()

        if self.df_merged is not None:
            drop_cols = ['class_name', 'image_url', 'label_meta', 'filename',
                         'file_size_kb', 'mode', 'is_corrupt', 'format', 'exists']
            df_save = self.df_merged.drop(
                columns=[c for c in drop_cols if c in self.df_merged.columns])
            merged_path = self.output_dir / 'dataset_with_image_metadata.csv'
            df_save.to_csv(merged_path, index=False)
            logger.info(f"Merged dataset saved to: {merged_path}")

        logger.info("EDA Complete!")
