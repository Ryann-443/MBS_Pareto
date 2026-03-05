"""
Pareto Ranking Module

Objectives (after normalization, all are maximized):
  - greenness    : Ecological benefit  (originally maximize)
  - generation   : Power generation    (originally maximize)
  - accessibility: Accessibility cost  (originally minimize; inverted)
"""

import warnings
from typing import Dict, List

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class MultiObjectiveEvaluator:
    """
    Pareto-based multi-objective evaluator for PV site selection.

    The evaluator normalizes, discretizes, and ranks grid cells using fast
    non-dominated sorting and crowding distance.
    """

    def __init__(
        self,
        discretization_bins: int = 10,
        maximize_cols: list = None,
        minimize_cols: list = None,
        column_mapping: dict = None,
    ):
        """
        Initialize the evaluator.

        Parameters:
            discretization_bins: Number of quantile bins for discretization.
                Reduces excessive stratification in Pareto fronts.
            maximize_cols: Objective columns to maximize (higher is better).
            minimize_cols: Objective columns to minimize (lower is better).
            column_mapping: Dict mapping raw column names to standardized names.
        """
        self.discretization_bins = discretization_bins
        self.maximize_cols = maximize_cols or ["greenness", "generation"]
        self.minimize_cols = minimize_cols or ["accessibility"]
        self.all_numeric_cols = self.maximize_cols + self.minimize_cols
        self.column_mapping = column_mapping or {
            "Predicted_Probability": "greenness",
            "pvout_mean": "generation",
            "access": "accessibility",
            "unsuit": "unsuitability",
        }
        self.data = None
        self.normalized_data = None
        self.valid_data = None

    # -----------------------------------------------------------------
    # Data loading and preprocessing
    # -----------------------------------------------------------------

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load a GeoPackage file and preprocess it.

        Parameters:
            file_path: Path to the input GeoPackage.

        Returns:
            Preprocessed DataFrame (also stored in ``self.data``).
        """
        gdf = gpd.read_file(file_path)
        self.data = pd.DataFrame(gdf.drop(columns="geometry"))
        print(f"Loaded {len(self.data)} grid cells")

        # Validate required columns
        raw_required = list(self.column_mapping.keys())
        missing = [c for c in raw_required if c not in self.data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self._preprocess_data()
        return self.data

    def _preprocess_data(self) -> None:
        """
        Rename columns, fill missing values, flag exclusions, then
        normalize and discretize objective columns.
        """
        # Rename columns to standardized names
        self.data.rename(columns=self.column_mapping, inplace=True)

        # Fill missing values with column median
        for col in self.all_numeric_cols:
            self.data[col].fillna(self.data[col].median(), inplace=True)

        # Ensure unsuitability is integer (0 or 1)
        self.data["unsuitability"] = self.data["unsuitability"].astype(int)

        # Hard constraint: flag grids to exclude from ranking
        self.data["exclude_from_calculation"] = (
            (self.data["greenness"] == 0)
            | (self.data["generation"] == 0)
            | (self.data["unsuitability"] == 1)
        )

        self.valid_data = self.data[~self.data["exclude_from_calculation"]].copy()
        if len(self.valid_data) == 0:
            raise ValueError("No valid data available for ranking")

        # Normalize and discretize
        self._normalize_and_discretize()

        print("Preprocessing complete")
        print(f"  Total grids: {len(self.data)}")
        print(f"  Excluded:    {self.data['exclude_from_calculation'].sum()}")
        print(f"  Valid:       {len(self.valid_data)}")

    def _normalize_and_discretize(self) -> None:
        """
        Normalize objectives to [0, 1] and discretize into quantile bins.

        - Maximize objectives: standard min-max scaling.
        - Minimize objectives: inverted min-max scaling so that higher
          normalized values always indicate better performance.
        - All objectives are then discretized into ``discretization_bins``
          quantile-based bins to reduce noise and excessive Pareto layers.
        """
        self.normalized_data = self.data.copy()
        valid_idx = ~self.data["exclude_from_calculation"]

        # 1. Standard min-max normalization for maximize objectives
        if self.maximize_cols:
            scaler = MinMaxScaler()
            self.normalized_data.loc[valid_idx, self.maximize_cols] = (
                scaler.fit_transform(self.data.loc[valid_idx, self.maximize_cols])
            )

        # 2. Inverted min-max normalization for minimize objectives
        #    Formula: (max - value) / (max - min)
        for col in self.minimize_cols:
            vals = self.data.loc[valid_idx, col]
            lo, hi = vals.min(), vals.max()
            if hi != lo:
                self.normalized_data.loc[valid_idx, col] = (hi - vals) / (hi - lo)
            else:
                self.normalized_data.loc[valid_idx, col] = 0.5

        # 3. Quantile-based discretization for all objectives
        for col in self.all_numeric_cols:
            vals = self.normalized_data.loc[valid_idx, col]
            try:
                binned = pd.qcut(
                    vals, q=self.discretization_bins,
                    labels=False, duplicates="drop",
                )
                if binned.nunique() > 1:
                    binned = binned / binned.max()  # Rescale to [0, 1]
                else:
                    binned = pd.Series(0.5, index=binned.index)
                self.normalized_data.loc[valid_idx, col] = binned
            except ValueError:
                # Fall back to continuous values if binning fails
                # (e.g., too many duplicate values)
                pass

        # Set excluded rows to 0
        self.normalized_data.loc[~valid_idx, self.all_numeric_cols] = 0

    # -----------------------------------------------------------------
    # Pareto sorting algorithms
    # -----------------------------------------------------------------

    def _compute_objectives(self) -> np.ndarray:
        """
        Build the (N, 3) objective matrix from normalized data.

        Column order: [greenness, generation, accessibility].
        After normalization, all three are "higher is better".
        """
        n = len(self.normalized_data)
        objectives = np.zeros((n, 3))
        objectives[:, 0] = self.normalized_data["greenness"].values
        objectives[:, 1] = self.normalized_data["generation"].values
        objectives[:, 2] = self.normalized_data["accessibility"].values
        return objectives

    @staticmethod
    def _dominates(obj_a: np.ndarray, obj_b: np.ndarray) -> bool:
        """
        Check if solution A Pareto-dominates solution B.

        A dominates B iff:
          - A is no worse than B in all objectives, AND
          - A is strictly better than B in at least one objective.

        All objectives are treated as maximization targets.
        """
        at_least_one_better = False
        for i in range(len(obj_a)):
            if obj_a[i] < obj_b[i]:
                return False
            if obj_a[i] > obj_b[i]:
                at_least_one_better = True
        return at_least_one_better

    def _fast_non_dominated_sort(
        self, objectives: np.ndarray
    ) -> List[List[int]]:
        """
        NSGA-II fast non-dominated sorting.

        Parameters:
            objectives: (N, M) array of objective values.

        Returns:
            List of fronts, where each front is a list of row indices.
            Front 0 is the best (non-dominated) front.
        """
        n = len(objectives)
        domination_count = np.zeros(n, dtype=int)
        dominated_by: List[List[int]] = [[] for _ in range(n)]
        fronts: List[List[int]] = [[]]

        # Pairwise dominance comparison
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(objectives[i], objectives[j]):
                    dominated_by[i].append(j)
                elif self._dominates(objectives[j], objectives[i]):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                fronts[0].append(i)

        # Build subsequent fronts
        idx = 0
        while fronts[idx]:
            next_front: List[int] = []
            for i in fronts[idx]:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            idx += 1
            fronts.append(next_front)

        # Remove trailing empty front
        return fronts[:-1]

    @staticmethod
    def _crowding_distance(
        front: List[int], objectives: np.ndarray
    ) -> np.ndarray:
        """
        Compute crowding distance for solutions in a single Pareto front.

        Boundary solutions receive infinite distance.

        Parameters:
            front: Indices of solutions belonging to this front.
            objectives: Full (N, M) objective matrix.

        Returns:
            Array of crowding distances (length = len(front)).
        """
        n = len(front)
        distances = np.zeros(n)
        if n <= 2:
            distances[:] = float("inf")
            return distances

        n_obj = objectives.shape[1]
        for m in range(n_obj):
            # Sort by objective m
            sorted_idx = np.argsort(
                [objectives[front[k], m] for k in range(n)]
            )
            # Boundary solutions get infinite distance
            distances[sorted_idx[0]] = float("inf")
            distances[sorted_idx[-1]] = float("inf")
            obj_range = (
                objectives[front[sorted_idx[-1]], m]
                - objectives[front[sorted_idx[0]], m]
            )
            if obj_range == 0:
                continue
            for k in range(1, n - 1):
                distances[sorted_idx[k]] += (
                    objectives[front[sorted_idx[k + 1]], m]
                    - objectives[front[sorted_idx[k - 1]], m]
                ) / obj_range

        return distances

    # -----------------------------------------------------------------
    # Main evaluation entry point
    # -----------------------------------------------------------------

    def evaluate_all_grids(self) -> pd.DataFrame:
        """
        Run Pareto ranking on all valid grids.

        Returns:
            DataFrame with added columns:
              - pareto_rank: Pareto front level (1 = best).
              - crowding_distance: Crowding distance within each front.
              - ecological_benefit: Normalized ecological objective.
              - power_generation: Normalized generation objective.
              - accessibility_benefit: Normalized (inverted) accessibility.
        """
        if self.data is None:
            raise ValueError("Call load_data() first")

        print("Running Pareto sorting...")

        valid_mask = ~self.data["exclude_from_calculation"].values
        objectives_full = self._compute_objectives()

        # Non-dominated sorting on the valid subset only
        valid_indices = np.where(valid_mask)[0]
        objectives_valid = objectives_full[valid_mask]
        fronts_valid = self._fast_non_dominated_sort(objectives_valid)

        # Map local indices back to global indices
        fronts = [[valid_indices[i] for i in f] for f in fronts_valid]

        # Assign Pareto ranks and crowding distances
        n = len(self.data)
        pareto_rank = np.zeros(n, dtype=int)
        crowding_dist = np.zeros(n)

        for rank, front in enumerate(fronts, start=1):
            for idx in front:
                pareto_rank[idx] = rank
            if front:
                dists = self._crowding_distance(front, objectives_full)
                for k, idx in enumerate(front):
                    crowding_dist[idx] = dists[k]

        # Build result DataFrame
        results = self.data.copy()
        results["pareto_rank"] = pareto_rank
        results["crowding_distance"] = crowding_dist
        results["ecological_benefit"] = objectives_full[:, 0]
        results["power_generation"] = objectives_full[:, 1]
        results["accessibility_benefit"] = objectives_full[:, 2]

        # Sort: best rank first, then highest crowding distance
        results.sort_values(
            ["pareto_rank", "crowding_distance"],
            ascending=[True, False],
            inplace=True,
        )

        print(f"Done: {len(fronts)} Pareto fronts generated")
        if fronts:
            print(f"  Front 1 contains {len(fronts[0])} solutions")

        return results

    # -----------------------------------------------------------------
    # Reporting and visualization
    # -----------------------------------------------------------------

    def summary_report(self, results: pd.DataFrame) -> Dict:
        """
        Generate a summary dictionary from the evaluation results.

        Parameters:
            results: Output from ``evaluate_all_grids()``.

        Returns:
            Dictionary with key statistics.
        """
        valid = results[~results["exclude_from_calculation"]]
        return {
            "discretization_bins": self.discretization_bins,
            "total_grids": len(results),
            "excluded_grids": int(results["exclude_from_calculation"].sum()),
            "valid_grids": len(valid),
            "pareto_fronts": int(results["pareto_rank"].max()),
            "first_front_size": int((results["pareto_rank"] == 1).sum()),
            "avg_ecological_benefit": float(valid["ecological_benefit"].mean()),
            "avg_power_generation": float(valid["power_generation"].mean()),
            "avg_accessibility_benefit": float(
                valid["accessibility_benefit"].mean()
            ),
        }

    def plot_results(
        self, results: pd.DataFrame, save_path: str = None
    ) -> None:
        """
        Visualize Pareto ranking results with three panels:
          1. Bar chart of front sizes.
          2. 3-D scatter of all three objectives.
          3. Histogram of crowding distances.

        Parameters:
            results: Output from ``evaluate_all_grids()``.
            save_path: Optional path to save the figure (PNG, 300 dpi).
        """
        valid = results[~results["exclude_from_calculation"]].copy()

        fig = plt.figure(figsize=(15, 12))

        # --- Panel 1: Pareto front size distribution ---
        ax1 = fig.add_subplot(221)
        counts = valid["pareto_rank"].value_counts().sort_index()
        counts = counts[counts.index > 0]
        ax1.bar(counts.index, counts.values, alpha=0.7, color="lightgreen")
        ax1.set_title(f"Pareto front distribution ({len(counts)} fronts)")
        ax1.set_xlabel("Pareto rank")
        ax1.set_ylabel("Number of grid cells")

        # --- Panel 2: 3-D scatter plot ---
        ax2 = fig.add_subplot(222, projection="3d")
        sc = ax2.scatter(
            valid["ecological_benefit"],
            valid["power_generation"],
            valid["accessibility_benefit"],
            c=valid["pareto_rank"],
            cmap="viridis_r",
            alpha=0.6,
        )
        ax2.set_xlabel("Ecological benefit")
        ax2.set_ylabel("Power generation")
        ax2.set_zlabel("Accessibility benefit")
        ax2.set_title("3-D Pareto front")
        plt.colorbar(sc, ax=ax2, label="Pareto rank")

        # Highlight front-1 solutions
        front1 = valid[valid["pareto_rank"] == 1]
        if len(front1):
            ax2.scatter(
                front1["ecological_benefit"],
                front1["power_generation"],
                front1["accessibility_benefit"],
                color="red", s=100, alpha=0.8,
                label=f"Front 1 ({len(front1)} solutions)",
            )
            ax2.legend()

        # --- Panel 3: Crowding distance histogram ---
        ax3 = fig.add_subplot(223)
        finite = valid["crowding_distance"][
            valid["crowding_distance"] != float("inf")
        ]
        if len(finite):
            ax3.hist(finite, bins=30, alpha=0.7, color="orange")
            ax3.set_title("Crowding distance distribution")
            ax3.set_xlabel("Crowding distance")
            ax3.set_ylabel("Frequency")
            ax3.axvline(
                finite.mean(), color="r", ls="--",
                label=f"Mean = {finite.mean():.2f}",
            )
            ax3.axvline(
                finite.median(), color="g", ls="--",
                label=f"Median = {finite.median():.2f}",
            )
            ax3.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def get_front_solutions(
        results: pd.DataFrame, front_level: int = 1
    ) -> pd.DataFrame:
        """
        Extract solutions at a specific Pareto front level.

        Parameters:
            results: Full result DataFrame from ``evaluate_all_grids()``.
            front_level: Pareto rank to extract (1 = best front).

        Returns:
            Filtered DataFrame sorted by crowding distance (descending).
        """
        valid = results[~results["exclude_from_calculation"]]
        front = valid[valid["pareto_rank"] == front_level].copy()
        return front.sort_values("crowding_distance", ascending=False)
