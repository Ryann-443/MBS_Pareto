"""
Main Entry Point

Orchestrates the full pipeline:
  1. Suitability identification
  2. Pareto non-dominated sorting (Pareto ranking)
  3. Result export and merging
"""

import config
from src.suitability_identification import add_unsuitability_to_grid, validate_result
from src.pareto_ranking import MultiObjectiveEvaluator
from src.export_results import merge_grid_with_ranking


def main():
    """Run the full multi-objective Pareto ranking pipeline."""

    # ==================================================================
    # Step 1: Suitability identification
    # ==================================================================

    suitable, unsuitable, errors = add_unsuitability_to_grid(
        grid_path=config.GRID_INPUT_PATH,
        suitability_raster_path=config.SUITABILITY_RASTER_PATH,
        output_path=config.GRID_WITH_UNSUIT_PATH,
    )
    print(f"\n  Suitable: {suitable}")
    print(f"  Unsuitable: {unsuitable}")
    print(f"  Errors: {errors}")

    validate_result(config.GRID_INPUT_PATH, config.GRID_WITH_UNSUIT_PATH)

    # ==================================================================
    # Step 2: Pareto ranking
    # ==================================================================

    evaluator = MultiObjectiveEvaluator(
        discretization_bins=config.DISCRETIZATION_BINS,
        maximize_cols=config.MAXIMIZE_OBJECTIVES,
        minimize_cols=config.MINIMIZE_OBJECTIVES,
        column_mapping=config.COLUMN_MAPPING,
    )
    evaluator.load_data(config.GRID_WITH_UNSUIT_PATH)
    results = evaluator.evaluate_all_grids()

    # Print summary report
    report = evaluator.summary_report(results)
    print("\n--- Summary Report ---")
    for key, val in report.items():
        print(f"  {key}: {val}")

    # Generate visualization
    evaluator.plot_results(results)

    # Save ranking results to Excel
    results.to_excel(config.RANKING_EXCEL_PATH, index=False)
    print(f"\nRanking results saved to: {config.RANKING_EXCEL_PATH}")

    # Display top solutions from Pareto front 1
    front1 = evaluator.get_front_solutions(results, front_level=1)
    if len(front1):
        print("\n--- Top Pareto Front-1 Solutions ---")
        cols = [
            "GridID", "pareto_rank",
            "ecological_benefit", "power_generation",
            "accessibility_benefit", "crowding_distance",
        ]
        print(front1.head(10)[cols].to_string(index=False))

    # ==================================================================
    # Step 3: Merge ranking into geospatial grid
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 3: Export merged geospatial grid")
    print("=" * 60)

    merge_grid_with_ranking(
        grid_path=config.GRID_WITH_UNSUIT_PATH,
        excel_path=config.RANKING_EXCEL_PATH,
        output_path=config.FINAL_GRID_PATH,
    )

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
